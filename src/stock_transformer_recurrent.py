from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .fixed_state import EVENT_COUNT, STATE_WIDTH
from .recurrent_compiled import CompiledRecurrentGovernanceTransformer, RecurrentCompiledConfig
from .recurrent_softmax import SlotSoftmaxConfig, SoftmaxSlotGather


@dataclass(frozen=True)
class StockGatherConfig:
    score_gap: float = 8.0
    dtype_name: str = "float32"
    logit_margin: float = 16.0

    @property
    def dtype(self) -> torch.dtype:
        return {
            "float64": torch.float64,
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[self.dtype_name]


class StockSoftmaxGather(nn.Module):
    """Standard nn.MultiheadAttention implementation of fixed-slot retrieval.

    This module uses ordinary Q/K/V projections and softmax attention. It only
    implements the slot-gather part of the recurrent machine; governance
    arithmetic remains outside this stock attention layer.
    """

    def __init__(self, slot_count: int, score_gap: float = 8.0, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.slot_count = int(slot_count)
        self.embed_dim = 2 * self.slot_count
        self.bit_dim = self.slot_count
        self.score_gap = float(score_gap)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.slot_count,
            dropout=0.0,
            bias=True,
            batch_first=True,
        )
        self.reset_synthesized_weights(dtype)

    def reset_synthesized_weights(self, dtype: torch.dtype = torch.float32) -> None:
        with torch.no_grad():
            for param in self.attn.parameters():
                param.zero_()
            scale = self.score_gap * math.sqrt(2.0)
            e = self.embed_dim
            for slot in range(self.slot_count):
                head_offset = 2 * slot
                # Query bias gives the compute token q=[1,0] in every head.
                self.attn.in_proj_bias[head_offset] = 1.0
                # Key projection maps slot identity j to key=[gap*sqrt(2),0]
                # in head j, yielding attention score `gap` after scaling.
                self.attn.in_proj_weight[e + head_offset, slot] = scale
                # Value projection copies the token's scalar bit value into the
                # first channel of every head.
                self.attn.in_proj_weight[2 * e + head_offset, self.bit_dim] = 1.0
                # Output projection maps each head's first value channel to the
                # corresponding retrieved slot coordinate.
                self.attn.out_proj.weight[slot, head_offset] = 1.0
        self.to(dtype=dtype)

    def _tokens(self, slots: Tensor) -> Tensor:
        squeeze = False
        if slots.ndim == 1:
            slots = slots.unsqueeze(0)
            squeeze = True
        if slots.ndim != 2 or slots.shape[1] != self.slot_count:
            raise ValueError(f"slots must have shape [batch,{self.slot_count}]")
        batch = slots.shape[0]
        tokens = torch.zeros((batch, self.slot_count + 1, self.embed_dim), dtype=slots.dtype, device=slots.device)
        eye = torch.eye(self.slot_count, dtype=slots.dtype, device=slots.device)
        tokens[:, 1:, : self.slot_count] = eye.unsqueeze(0)
        tokens[:, 1:, self.bit_dim] = slots
        return tokens.squeeze(0) if squeeze else tokens

    def _mask(self, device: torch.device) -> Tensor:
        # The compute token should attend only to data slots, not itself. Other
        # query rows are irrelevant because only output token 0 is read.
        mask = torch.zeros((self.slot_count + 1, self.slot_count + 1), dtype=torch.bool, device=device)
        mask[0, 0] = True
        return mask

    def forward(self, slots: Tensor, return_weights: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        squeeze = False
        if slots.ndim == 1:
            slots = slots.unsqueeze(0)
            squeeze = True
        dtype = self.attn.in_proj_weight.dtype
        slots = slots.to(dtype=dtype)
        tokens = self._tokens(slots)
        out, weights = self.attn(tokens, tokens, tokens, attn_mask=self._mask(slots.device), need_weights=return_weights, average_attn_weights=False)
        gathered = out[:, 0, : self.slot_count]
        if squeeze:
            gathered = gathered.squeeze(0)
            if return_weights:
                weights = weights.squeeze(0)
        if return_weights:
            return gathered, weights
        return gathered


class StockGatherRecurrentModel(nn.Module):
    """Stock MHA slot retrieval plus the existing custom recurrent circuit.

    This is intentionally not claimed as a full stock-transformer governance
    model: only the retrieval operation runs through ordinary transformer layer
    code. The transition arithmetic remains the custom hard/discrete circuit.
    """

    def __init__(self, config: StockGatherConfig | None = None):
        super().__init__()
        self.config = config or StockGatherConfig()
        self.input_width = STATE_WIDTH + EVENT_COUNT
        self.gather = StockSoftmaxGather(self.input_width, self.config.score_gap, dtype=self.config.dtype)
        self.compiled = CompiledRecurrentGovernanceTransformer(
            RecurrentCompiledConfig(dtype=self.config.dtype, logit_margin=self.config.logit_margin)
        )

    def _event_onehot(self, event_ids: Tensor, device: torch.device) -> Tensor:
        if event_ids.ndim == 2 and event_ids.shape[1] == EVENT_COUNT:
            return event_ids.to(device=device, dtype=self.config.dtype)
        if event_ids.ndim == 0:
            event_ids = event_ids.reshape(1)
        return torch.nn.functional.one_hot(event_ids.to(device=device, dtype=torch.long), num_classes=EVENT_COUNT).to(self.config.dtype)

    def forward(self, state_bits: Tensor, event_ids: Tensor) -> tuple[Tensor, Tensor]:
        squeeze = False
        if state_bits.ndim == 1:
            state_bits = state_bits.unsqueeze(0)
            squeeze = True
        event_onehot = self._event_onehot(event_ids, state_bits.device)
        slots = torch.cat([state_bits.to(self.config.dtype), event_onehot], dim=-1)
        retrieved = self.gather(slots)
        next_bits, logits = self.compiled(retrieved[:, :STATE_WIDTH], retrieved[:, STATE_WIDTH:])
        if squeeze:
            return next_bits.squeeze(0), logits.squeeze(0)
        return next_bits, logits


def save_stock_model(path: str | Path, model: StockGatherRecurrentModel) -> None:
    payload = {"config": asdict(model.config), "state_dict": model.state_dict()}
    torch.save(payload, path)


def load_stock_model(path: str | Path, device: torch.device | str = "cpu") -> StockGatherRecurrentModel:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = StockGatherRecurrentModel(StockGatherConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"])
    return model.to(device)


def compare_stock_gather_to_reference(slots: Tensor, score_gap: float, dtype: torch.dtype = torch.float32) -> dict[str, Any]:
    device = slots.device
    stock = StockSoftmaxGather(slots.shape[-1], score_gap, dtype=dtype).to(device)
    direct = SoftmaxSlotGather(slots.shape[-1], SlotSoftmaxConfig(score_gap=score_gap, dtype=dtype)).to(device)
    with torch.no_grad():
        stock_out = stock(slots.to(dtype))
        direct_out = direct(slots.to(dtype))
    diff = torch.abs(stock_out - direct_out)
    return {
        "checked": int(slots.shape[0] if slots.ndim == 2 else 1),
        "max_abs_diff": float(diff.max().detach().cpu().item()),
        "passed": bool(torch.allclose(stock_out, direct_out, atol=1e-5 if dtype == torch.float32 else 1e-8, rtol=1e-5)),
        "score_gap": float(score_gap),
        "dtype": str(dtype),
        "slot_count": int(slots.shape[-1]),
        "parameter_count": sum(p.numel() for p in stock.parameters()),
    }
