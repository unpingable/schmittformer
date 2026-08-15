from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from .compiled_bits import bits_to_int_tensor, int_tensor_to_bits
from .compiled_counter import CompiledSaturatingDecrementer
from .fixed_state import (
    EVENT_COUNT,
    OUTPUT_COUNT,
    STATE_WIDTH,
    GovernanceEvent,
    GovernanceOutput,
    GovernanceState,
    decode_state_bits,
    decode_states_bits,
    encode_state_bits,
    encode_states_bits,
    event_tensor,
)
from .recurrent_compiled import CompiledRecurrentGovernanceTransformer, RecurrentCompiledConfig
from .recurrent_reference import invariant_violations, transition
from .softmax_attention import EFFECTIVELY_HARD, NUMERIC_FAILURE, PASS_EXACT, SEMANTIC_FAILURE


@dataclass(frozen=True)
class SlotSoftmaxConfig:
    score_gap: float = 16.0
    dtype: torch.dtype = torch.float32
    logit_margin: float = 16.0


def slot_softmax_weights(width: int, score_gap: float, dtype: torch.dtype, device: torch.device) -> Tensor:
    scores = torch.zeros((width, width), dtype=dtype, device=device)
    idx = torch.arange(width, device=device)
    scores[idx, idx] = float(score_gap)
    return torch.softmax(scores, dim=-1)


def classify_slot_softmax(exact: bool, finite: bool, effectively_hard: bool) -> str:
    if not finite:
        return NUMERIC_FAILURE
    if exact and effectively_hard:
        return EFFECTIVELY_HARD
    if exact:
        return PASS_EXACT
    return SEMANTIC_FAILURE


def slot_attention_stats(weights: Tensor) -> dict[str, Any]:
    diag = torch.diagonal(weights, dim1=-2, dim2=-1)
    off = weights.clone()
    idx = torch.arange(weights.shape[-1], device=weights.device)
    off[idx, idx] = 0
    losing = off.sum(dim=-1)
    return {
        "min_correct_mass": float(diag.min().detach().cpu().item()),
        "max_losing_mass": float(losing.max().detach().cpu().item()),
        "max_losing_weight": float(off.max().detach().cpu().item()),
        "effectively_hard": bool((off == 0).all().detach().cpu().item()),
        "finite": bool(torch.isfinite(weights).all().detach().cpu().item()),
    }


def slot_leakage_bound(width: int, score_gap: float) -> float:
    z = (int(width) - 1) * math.exp(-float(score_gap))
    return z / (1.0 + z)


class SoftmaxSlotGather(nn.Module):
    """Finite-temperature softmax slot retrieval over a fixed-width vector.

    Head j has score `gap` for slot j and 0 for all other slots. The operation is
    ordinary softmax-weighted value averaging; no hardmax or post-softmax zeroing
    is used.
    """

    def __init__(self, width: int, config: SlotSoftmaxConfig | None = None):
        super().__init__()
        self.width = int(width)
        self.config = config or SlotSoftmaxConfig()

    def forward(self, slots: Tensor, return_debug: bool = False) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        squeeze = False
        if slots.ndim == 1:
            slots = slots.unsqueeze(0)
            squeeze = True
        if slots.ndim != 2 or slots.shape[1] != self.width:
            raise ValueError(f"slots must have shape [batch,{self.width}]")
        dtype = self.config.dtype
        slots = slots.to(dtype)
        weights = slot_softmax_weights(self.width, self.config.score_gap, dtype, slots.device)
        retrieved = slots @ weights.T
        if squeeze:
            retrieved_out = retrieved.squeeze(0)
        else:
            retrieved_out = retrieved
        if not return_debug:
            return retrieved_out
        debug = {
            "weights": weights,
            "input_error": torch.abs(retrieved - slots),
            "retrieved_margin": torch.abs(retrieved - 0.5),
        }
        if squeeze:
            debug = {k: (v.squeeze(0) if v.ndim > 1 and v.shape[0] == 1 else v) for k, v in debug.items()}
        return retrieved_out, debug


class SoftmaxSaturatingDecrementer(nn.Module):
    def __init__(self, width: int, config: SlotSoftmaxConfig | None = None):
        super().__init__()
        self.width = int(width)
        self.config = config or SlotSoftmaxConfig()
        self.gather = SoftmaxSlotGather(width, self.config)
        self.counter = CompiledSaturatingDecrementer(width, dtype=self.config.dtype)

    def forward(self, bits: Tensor, return_debug: bool = False) -> dict[str, Tensor] | tuple[dict[str, Tensor], dict[str, Tensor]]:
        retrieved, debug = self.gather(bits, return_debug=True)
        out = self.counter(retrieved)
        if return_debug:
            debug["output_margin"] = torch.minimum(
                torch.abs(out["decremented_bits"] - 0.5).min(dim=-1).values if out["decremented_bits"].ndim == 2 else torch.abs(out["decremented_bits"] - 0.5).min().reshape(()),
                torch.minimum(torch.abs(out["is_zero"] - 0.5), torch.abs(out["is_nonzero"] - 0.5)),
            )
            return out, debug
        return out


class SoftmaxRecurrentGovernanceTransformer(nn.Module):
    """Finite-softmax slot retrieval followed by the existing hard/discrete recurrent circuit.

    This removes hard slot selection/retrieval from the recurrent input boundary.
    The Boolean/arithmetic transition remains the hard/discrete tensor circuit,
    which is why the stock-transformer claim is not made here.
    """

    def __init__(self, config: SlotSoftmaxConfig | None = None):
        super().__init__()
        self.config = config or SlotSoftmaxConfig()
        self.input_width = STATE_WIDTH + EVENT_COUNT
        self.gather = SoftmaxSlotGather(self.input_width, self.config)
        self.compiled = CompiledRecurrentGovernanceTransformer(
            RecurrentCompiledConfig(dtype=self.config.dtype, logit_margin=self.config.logit_margin)
        )

    @property
    def physical_input_width(self) -> int:
        return self.input_width

    def _event_onehot(self, event_ids: Tensor, device: torch.device) -> Tensor:
        if event_ids.ndim == 2 and event_ids.shape[1] == EVENT_COUNT:
            return event_ids.to(device=device, dtype=self.config.dtype)
        if event_ids.ndim == 0:
            event_ids = event_ids.reshape(1)
        return torch.nn.functional.one_hot(event_ids.to(device=device, dtype=torch.long), num_classes=EVENT_COUNT).to(self.config.dtype)

    def forward(self, state_bits: Tensor, event_ids: Tensor, return_debug: bool = False) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, Tensor, dict[str, Tensor]]:
        squeeze = False
        if state_bits.ndim == 1:
            state_bits = state_bits.unsqueeze(0)
            squeeze = True
        if state_bits.ndim != 2 or state_bits.shape[1] != STATE_WIDTH:
            raise ValueError(f"state_bits must have shape [batch,{STATE_WIDTH}]")
        event_onehot = self._event_onehot(event_ids, state_bits.device)
        if event_onehot.shape[0] != state_bits.shape[0]:
            raise ValueError("event batch size mismatch")
        slots = torch.cat([state_bits.to(self.config.dtype), event_onehot], dim=-1)
        retrieved, gather_debug = self.gather(slots, return_debug=True)
        retrieved_state = retrieved[:, :STATE_WIDTH]
        retrieved_event = retrieved[:, STATE_WIDTH:]
        next_bits, logits = self.compiled(retrieved_state, retrieved_event)
        if squeeze:
            next_bits = next_bits.squeeze(0)
            logits = logits.squeeze(0)
        if return_debug:
            debug = {
                "weights": gather_debug["weights"],
                "input_error": gather_debug["input_error"],
                "retrieved_margin": gather_debug["retrieved_margin"],
                "next_bit_margin": torch.abs(next_bits - 0.5),
                "logit_margin": logits - torch.topk(logits, k=2, dim=-1).values[..., 1:2],
            }
            return next_bits, logits, debug
        return next_bits, logits


def _counter_values(width: int, exhaustive: bool, random_samples: int, seed: int, device: torch.device) -> Tensor:
    max_value = 1 << width
    if exhaustive:
        return torch.arange(max_value, dtype=torch.long, device=device)
    gen = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
    gen.manual_seed(seed)
    random_values = torch.randint(0, max_value, (random_samples,), dtype=torch.long, device=device, generator=gen)
    boundaries = torch.tensor([0, 1, 2, 3, 15, 16, 31, 32, 127, 128, 255, 256, 4095, 4096, 32767, 32768, max_value - 2, max_value - 1], dtype=torch.long, device=device)
    boundaries = boundaries[(boundaries >= 0) & (boundaries < max_value)]
    return torch.cat([boundaries, random_values], dim=0).unique()


def verify_softmax_counter(
    width: int,
    score_gap: float,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    exhaustive: bool = True,
    random_samples: int = 100000,
    batch_size: int = 65536,
    seed: int = 1234,
) -> dict[str, Any]:
    device = torch.device(device)
    config = SlotSoftmaxConfig(score_gap=score_gap, dtype=dtype)
    model = SoftmaxSaturatingDecrementer(width, config).to(device)
    values = _counter_values(width, exhaustive, random_samples, seed, device)
    failures: list[dict[str, Any]] = []
    checked = 0
    min_output_margin = float("inf")
    min_retrieved_margin = float("inf")
    max_input_error = 0.0
    max_output_error = 0.0
    weights = slot_softmax_weights(width, score_gap, dtype, device)
    stats = slot_attention_stats(weights)
    start = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for start_idx in range(0, values.numel(), batch_size):
            chunk = values[start_idx : start_idx + batch_size]
            bits = int_tensor_to_bits(chunk, width, dtype=dtype)
            out, debug = model(bits, return_debug=True)
            actual = bits_to_int_tensor(out["decremented_bits"])
            expected = torch.clamp(chunk - 1, min=0)
            expected_bits = int_tensor_to_bits(expected, width, dtype=dtype)
            expected_zero = (chunk == 0).to(dtype)
            expected_nonzero = (chunk != 0).to(dtype)
            bad = (actual != expected) | (out["is_zero"].round().to(torch.long) != expected_zero.to(torch.long)) | (out["is_nonzero"].round().to(torch.long) != expected_nonzero.to(torch.long))
            if bad.any():
                idxs = torch.nonzero(bad, as_tuple=False).flatten()[:20]
                for idx in idxs.detach().cpu().tolist():
                    failures.append({"input": int(chunk[idx].detach().cpu().item()), "expected": int(expected[idx].detach().cpu().item()), "actual": int(actual[idx].detach().cpu().item())})
            margins = torch.minimum(
                torch.abs(out["decremented_bits"] - 0.5).min(),
                torch.minimum(torch.abs(out["is_zero"] - 0.5).min(), torch.abs(out["is_nonzero"] - 0.5).min()),
            )
            min_output_margin = min(min_output_margin, float(margins.detach().cpu().item()))
            min_retrieved_margin = min(min_retrieved_margin, float(debug["retrieved_margin"].min().detach().cpu().item()))
            max_input_error = max(max_input_error, float(debug["input_error"].max().detach().cpu().item()))
            max_output_error = max(max_output_error, float(torch.abs(out["decremented_bits"] - expected_bits).max().detach().cpu().item()))
            checked += int(chunk.numel())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.time() - start
    finite = bool(stats["finite"])
    effectively_hard = bool(stats["effectively_hard"])
    exact = not failures
    result: dict[str, Any] = {
        "width": width,
        "score_gap": float(score_gap),
        "device": str(device),
        "dtype": str(dtype),
        "exhaustive": bool(exhaustive),
        "checked": checked,
        "passed": exact,
        "classification": classify_slot_softmax(exact, finite, effectively_hard),
        "failures": failures[:20],
        "elapsed_seconds": elapsed,
        "values_per_second": checked / elapsed if elapsed else None,
        "min_output_margin_to_half": min_output_margin,
        "min_retrieved_margin_to_half": min_retrieved_margin,
        "max_input_error_before_decode": max_input_error,
        "max_output_error_before_decode": max_output_error,
        "attention": stats,
        "leakage_bound": slot_leakage_bound(width, score_gap),
    }
    if device.type == "cuda":
        result["peak_gpu_memory_mib"] = float(torch.cuda.max_memory_allocated(device) / 2**20)
    return result


def compare_softmax_reference(states: Sequence[GovernanceState], events: Sequence[int | GovernanceEvent], score_gap: float, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> dict[str, Any]:
    device = torch.device(device)
    model = SoftmaxRecurrentGovernanceTransformer(SlotSoftmaxConfig(score_gap=score_gap, dtype=dtype)).to(device)
    state_bits = encode_states_bits(states, device=device, dtype=dtype)
    event_ids = event_tensor(events, device=device)
    failures: list[dict[str, Any]] = []
    invariant_count = 0
    with torch.no_grad():
        next_bits, logits, debug = model(state_bits, event_ids, return_debug=True)
    actual_states = decode_states_bits(next_bits)
    actual_outputs = logits.argmax(dim=-1).detach().cpu().tolist()
    expected_bits_rows = []
    output_margins = []
    for index, (state, event, actual_state, actual_output) in enumerate(zip(states, events, actual_states, actual_outputs)):
        expected = transition(state, event)
        invariant_count += len(invariant_violations(state, event, expected))
        expected_bits_rows.append(encode_state_bits(expected.next_state, device=device, dtype=dtype))
        row_logits = logits[index]
        correct = row_logits[int(expected.output)]
        runner = torch.cat([row_logits[: int(expected.output)], row_logits[int(expected.output) + 1 :]]).max()
        output_margins.append(float((correct - runner).detach().cpu().item()))
        if actual_state != expected.next_state or int(actual_output) != expected.output:
            failures.append({"index": index, "state": state.to_json(), "event": int(GovernanceEvent(int(event))), "expected": expected.to_json(), "actual_state": actual_state.to_json(), "actual_output": int(actual_output)})
            if len(failures) >= 20:
                break
    expected_bits = torch.stack(expected_bits_rows, dim=0) if expected_bits_rows else torch.empty((0, STATE_WIDTH), device=device, dtype=dtype)
    bit_error = torch.abs(next_bits[: expected_bits.shape[0]] - expected_bits).max() if expected_bits.numel() else torch.tensor(0.0, device=device)
    weights = slot_softmax_weights(STATE_WIDTH + EVENT_COUNT, score_gap, dtype, device)
    stats = slot_attention_stats(weights)
    exact = not failures
    finite = bool(stats["finite"]) and bool(torch.isfinite(next_bits).all().detach().cpu().item()) and bool(torch.isfinite(logits).all().detach().cpu().item())
    return {
        "checked": len(states),
        "passed": exact,
        "classification": classify_slot_softmax(exact, finite, bool(stats["effectively_hard"])),
        "failures": failures,
        "reference_invariant_violations": invariant_count,
        "score_gap": float(score_gap),
        "device": str(device),
        "dtype": str(dtype),
        "attention": stats,
        "leakage_bound": slot_leakage_bound(STATE_WIDTH + EVENT_COUNT, score_gap),
        "min_next_bit_margin_to_half": float(torch.abs(next_bits - 0.5).min().detach().cpu().item()),
        "max_next_bit_error_before_decode": float(bit_error.detach().cpu().item()),
        "min_output_margin": min(output_margins) if output_margins else None,
        "max_input_error_before_transition": float(debug["input_error"].max().detach().cpu().item()),
    }
