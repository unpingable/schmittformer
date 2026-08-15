from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from .reference import State, transition, run_hysteresis
from .softmax_attention import SoftmaxAttentionConfig, attention_stats_to_json, latest_state_attention


@dataclass(frozen=True)
class SoftmaxHysteresisConfig:
    state_record_gap: float = 2.0
    non_state_penalty: float = 4.0
    dtype: torch.dtype = torch.float32

    def attention_config(self) -> SoftmaxAttentionConfig:
        return SoftmaxAttentionConfig(self.state_record_gap, self.non_state_penalty, self.dtype)


class SoftmaxHysteresisTransformer(nn.Module):
    input_vocab_size = 10

    def __init__(self, config: SoftmaxHysteresisConfig | None = None):
        super().__init__()
        self.config = config or SoftmaxHysteresisConfig()
        self.state_token_offset = self.input_vocab_size
        table = torch.empty((2, self.input_vocab_size), dtype=torch.long)
        for state in (State.OFF, State.ON):
            for x in range(self.input_vocab_size):
                table[int(state), x] = int(transition(state, x))
        self.register_buffer("transition_table", table, persistent=True)

    @property
    def num_states(self) -> int:
        return 2

    def input_token(self, x: int) -> int:
        if not 0 <= int(x) < 10:
            raise ValueError("hysteresis input must be 0..9")
        return int(x)

    def state_token(self, state: int | State) -> int:
        return self.state_token_offset + int(State(int(state)))

    def encode_history_from_reference(
        self,
        inputs: Sequence[int],
        initial_state: int | State = State.OFF,
    ) -> list[int]:
        state = State(int(initial_state))
        tokens = [self.state_token(state)]
        for x in inputs:
            tokens.append(self.input_token(x))
            state = transition(state, x)
            tokens.append(self.state_token(state))
        return tokens

    def next_state_logits(
        self,
        token_history: Tensor,
        return_debug: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        squeeze = False
        if token_history.ndim == 1:
            token_history = token_history.unsqueeze(0)
            squeeze = True
        tokens = token_history.to(torch.long)
        current_input = tokens[:, -1]
        if ((current_input < 0) | (current_input >= self.input_vocab_size)).any():
            raise ValueError("last token must be a hysteresis input token")
        masses, debug = latest_state_attention(
            tokens,
            self.state_token_offset,
            self.num_states,
            self.config.attention_config(),
        )
        batch = tokens.shape[0]
        logits = torch.zeros((batch, self.num_states), dtype=self.config.dtype, device=tokens.device)
        for state_id in range(self.num_states):
            next_ids = self.transition_table[state_id, current_input]
            logits.scatter_add_(1, next_ids[:, None], masses[:, state_id : state_id + 1])
        runner = logits.masked_fill(torch.nn.functional.one_hot(logits.argmax(dim=-1), self.num_states).bool(), float("-inf")).max(dim=-1).values
        debug["state_masses"] = masses
        debug["next_logits"] = logits
        debug["decision_margin"] = logits.max(dim=-1).values - runner
        if squeeze:
            logits = logits.squeeze(0)
            debug = {k: v.squeeze(0) for k, v in debug.items()}
        return (logits, debug) if return_debug else logits

    def decode_from_tokens(
        self,
        state_record_history: Sequence[int],
        suffix_inputs: Sequence[int],
        device: str | torch.device = "cpu",
    ) -> tuple[list[int], list[int]]:
        tokens = [int(x) for x in state_record_history]
        outputs: list[int] = []
        for x in suffix_inputs:
            tokens.append(self.input_token(x))
            tensor = torch.tensor(tokens, dtype=torch.long, device=device)
            with torch.no_grad():
                next_state = int(self.next_state_logits(tensor).argmax(dim=-1).item())
            tokens.append(self.state_token(next_state))
            outputs.append(next_state)
        return outputs, tokens

    def decode_inputs(
        self,
        inputs: Sequence[int],
        initial_state: int | State = State.OFF,
        device: str | torch.device = "cpu",
    ) -> tuple[list[int], list[int]]:
        tokens = [self.state_token(initial_state)]
        return self.decode_from_tokens(tokens, inputs, device=device)


def verify_softmax_hysteresis_transitions(
    state_record_gap: float,
    non_state_penalty: float,
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    model = SoftmaxHysteresisTransformer(SoftmaxHysteresisConfig(state_record_gap, non_state_penalty, dtype))
    failures = []
    min_margin = float("inf")
    max_stale = 0.0
    max_non_state = 0.0
    effectively_hard = True
    finite = True
    histories = {
        State.OFF: [[], [4, 5, 6], [7, 3], [7, 3, 5, 6]],
        State.ON: [[7], [7, 4, 5, 6], [0, 7], [0, 7, 5, 6]],
    }
    for state, prefixes in histories.items():
        for prefix in prefixes:
            if (run_hysteresis(prefix)[-1] if prefix else 0) != int(state):
                continue
            tokens = model.encode_history_from_reference(prefix)
            for x in range(10):
                expected = int(transition(state, x))
                logits, debug = model.next_state_logits(torch.tensor([*tokens, x], dtype=torch.long), return_debug=True)
                actual = int(logits.argmax(dim=-1).item())
                runner = logits[1 - expected].item()
                margin = float(logits[expected].item() - runner)
                min_margin = min(min_margin, margin)
                max_stale = max(max_stale, float(debug["stale_state_mass"].item()))
                max_non_state = max(max_non_state, float(debug["non_state_mass"].item()))
                effectively_hard = effectively_hard and bool(debug["effectively_hard"].item())
                finite = finite and bool(debug["finite"].item())
                if actual != expected:
                    failures.append({"state": int(state), "prefix": prefix, "input": x, "expected": expected, "actual": actual, "margin": margin})
    return {
        "passed": not failures,
        "failures": failures,
        "min_decision_margin": min_margin,
        "max_stale_state_mass": max_stale,
        "max_non_state_mass": max_non_state,
        "effectively_hard": effectively_hard,
        "finite": finite,
    }
