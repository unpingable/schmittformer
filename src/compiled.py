from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from .reference import HIGH_THRESHOLD, LOW_THRESHOLD, State, run_hysteresis


RESET = 0
NEUTRAL = 1
SET = 2


@dataclass(frozen=True)
class CompiledConfig:
    low: int = LOW_THRESHOLD
    high: int = HIGH_THRESHOLD
    event_margin: float = 8.0
    state_logit_margin: float = 16.0
    score_gap: float = 16.0
    ineligible_score: float = -1.0e4
    attention: str = "hard"
    dtype: torch.dtype = torch.float32


class CompiledHysteresisTransformer(nn.Module):
    """Synthesized causal controller for the hysteresis task.

    The computation is:

    1. deterministic token embedding/table maps each input to RESET, NEUTRAL,
       or SET logits;
    2. causal recency attention selects the latest non-neutral event in the
       prefix, including an explicit initial-state event;
    3. selected event values are projected to OFF/ON logits.

    No learned parameters are used. The transition is represented as attention
    over append-only history, not as a Python loop over mutable state.
    """

    def __init__(self, config: CompiledConfig | None = None):
        super().__init__()
        self.config = config or CompiledConfig()
        if self.config.attention not in {"hard", "soft"}:
            raise ValueError("attention must be 'hard' or 'soft'")
        self.register_buffer(
            "event_logit_table",
            self._build_event_logit_table(),
            persistent=True,
        )
        self.register_buffer(
            "state_logit_table",
            self._build_state_logit_table(),
            persistent=True,
        )

    def _build_event_logit_table(self) -> Tensor:
        table = torch.full((10, 3), -self.config.event_margin, dtype=self.config.dtype)
        for x in range(10):
            if x <= self.config.low:
                event = RESET
            elif x >= self.config.high:
                event = SET
            else:
                event = NEUTRAL
            table[x, event] = self.config.event_margin
        return table

    def _build_state_logit_table(self) -> Tensor:
        margin = self.config.state_logit_margin
        table = torch.zeros((3, 2), dtype=self.config.dtype)
        table[RESET] = torch.tensor([margin, -margin], dtype=self.config.dtype)
        table[SET] = torch.tensor([-margin, margin], dtype=self.config.dtype)
        return table

    def event_logits(self, input_ids: Tensor) -> Tensor:
        flat = input_ids.to(torch.long).reshape(-1)
        logits = self.event_logit_table.index_select(0, flat)
        return logits.reshape(*input_ids.shape, 3)

    def event_classes(self, input_ids: Tensor) -> Tensor:
        return self.event_logits(input_ids).argmax(dim=-1)

    def _initial_event_class(
        self,
        initial_state: int | State | Tensor,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        if isinstance(initial_state, Tensor):
            states = initial_state.to(device=device, dtype=torch.long).reshape(-1)
            if states.numel() == 1:
                states = states.expand(batch_size)
            if states.numel() != batch_size:
                raise ValueError("initial_state tensor must have shape [batch] or [1]")
        else:
            states = torch.full(
                (batch_size,),
                int(State(int(initial_state))),
                dtype=torch.long,
                device=device,
            )
        return torch.where(
            states == int(State.ON),
            torch.full_like(states, SET),
            torch.full_like(states, RESET),
        )

    def forward(
        self,
        input_ids: Tensor,
        initial_state: int | State | Tensor = State.OFF,
        return_debug: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        squeeze = False
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
            squeeze = True
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [time] or [batch, time]")

        input_ids = input_ids.to(torch.long)
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if seq_len == 0:
            empty = torch.empty(
                (batch_size, 0, 2),
                dtype=self.state_logit_table.dtype,
                device=device,
            )
            if squeeze:
                empty = empty.squeeze(0)
            return (empty, {}) if return_debug else empty

        token_events = self.event_classes(input_ids)
        initial_event = self._initial_event_class(initial_state, batch_size, device)
        all_events = torch.cat([initial_event[:, None], token_events], dim=1)

        values = self.state_logit_table.index_select(0, all_events.reshape(-1))
        values = values.reshape(batch_size, seq_len + 1, 2)

        event_is_real = all_events != NEUTRAL
        positions = torch.arange(seq_len + 1, device=device, dtype=values.dtype)
        key_scores = positions[None, :] * self.config.score_gap
        key_scores = key_scores.expand(batch_size, -1).masked_fill(
            ~event_is_real,
            self.config.ineligible_score,
        )

        query_positions = torch.arange(1, seq_len + 1, device=device)
        key_positions = torch.arange(seq_len + 1, device=device)
        causal = key_positions[None, :] <= query_positions[:, None]
        scores = key_scores[:, None, :].expand(batch_size, seq_len, seq_len + 1)
        scores = scores.masked_fill(~causal[None, :, :], self.config.ineligible_score)

        if self.config.attention == "hard":
            selected_indices = scores.argmax(dim=-1)
            logits = values.gather(
                1,
                selected_indices[:, :, None].expand(batch_size, seq_len, 2),
            )
            attention_weights = torch.nn.functional.one_hot(
                selected_indices,
                num_classes=seq_len + 1,
            ).to(values.dtype)
        else:
            attention_weights = torch.softmax(scores, dim=-1)
            selected_indices = attention_weights.argmax(dim=-1)
            logits = attention_weights @ values

        if squeeze:
            logits = logits.squeeze(0)
            selected_indices = selected_indices.squeeze(0)
            attention_weights = attention_weights.squeeze(0)
            all_events = all_events.squeeze(0)

        debug = {
            "token_events": token_events.squeeze(0) if squeeze else token_events,
            "all_events": all_events,
            "attention_indices": selected_indices,
            "attention_weights": attention_weights,
        }
        return (logits, debug) if return_debug else logits


def predict_compiled(
    inputs: Sequence[int],
    initial_state: int | State = State.OFF,
    attention: str = "hard",
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> list[int]:
    model = CompiledHysteresisTransformer(
        CompiledConfig(attention=attention, dtype=dtype)
    ).to(device)
    with torch.no_grad():
        tokens = torch.tensor(list(inputs), dtype=torch.long, device=device)
        logits = model(tokens, initial_state=initial_state)
        return [int(x) for x in logits.argmax(dim=-1).cpu().tolist()]


def verify_reachable_transitions(
    attention: str = "hard",
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    model = CompiledHysteresisTransformer(
        CompiledConfig(attention=attention, dtype=dtype)
    ).to(device)
    failures: list[dict[str, Any]] = []
    with torch.no_grad():
        for initial in (State.OFF, State.ON):
            for previous in (State.OFF, State.ON):
                for x in range(10):
                    prefix = [7] if previous == State.ON else [0]
                    if initial == State.ON:
                        prefix = [0] if previous == State.OFF else [7]
                    sequence = prefix + [x]
                    expected = run_hysteresis(sequence, initial_state=initial)[-1]
                    actual = int(
                        model(
                            torch.tensor(sequence, dtype=torch.long, device=device),
                            initial_state=initial,
                        ).argmax(dim=-1)[-1].item()
                    )
                    if actual != expected:
                        failures.append(
                            {
                                "initial_state": int(initial),
                                "previous_state": int(previous),
                                "input": x,
                                "expected": expected,
                                "actual": actual,
                                "sequence": sequence,
                            }
                        )
    return {
        "attention": attention,
        "dtype": str(dtype),
        "checked": 40,
        "failures": failures,
        "passed": not failures,
    }
