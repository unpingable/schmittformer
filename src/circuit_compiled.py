from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from .circuit_reference import (
    EVENTS,
    Event,
    CircuitState,
    Mode,
    collect_equivalent_histories,
    initial_state,
    invariant_violations,
    reachable_graph,
    run_controller,
    state_id_maps,
    transition,
)


@dataclass(frozen=True)
class CompiledCircuitConfig:
    state_logit_margin: float = 32.0
    ineligible_score: float = -1.0e6
    score_gap: float = 16.0
    dtype: torch.dtype = torch.float32


class CompiledCircuitBreakerTransformer(nn.Module):
    """Autoregressive hard-attention compiled circuit-breaker controller.

    Runtime token history alternates state records and input records:

        STATE(s0), INPUT(x1), STATE(s1), INPUT(x2), STATE(s2), ...

    On a new input token, hard attention selects the latest state token in the
    causal history. A deterministic transition lookup maps (selected state,
    current input) to the next complete logical state token. The Python driver
    only performs greedy autoregressive decoding by appending the emitted state
    token; it does not update controller state itself.
    """

    input_vocab_size = 3

    def __init__(self, config: CompiledCircuitConfig | None = None):
        super().__init__()
        self.config = config or CompiledCircuitConfig()
        states, state_to_id = state_id_maps()
        self.states = states
        self.state_to_id = state_to_id
        self.state_token_offset = self.input_vocab_size
        transition_table = torch.empty((len(states), len(EVENTS)), dtype=torch.long)
        mode_table = torch.empty((len(states),), dtype=torch.long)
        for state, state_id in state_to_id.items():
            mode_table[state_id] = state.mode
            for event in EVENTS:
                transition_table[state_id, int(event)] = state_to_id[transition(state, event)]
        self.register_buffer("transition_table", transition_table, persistent=True)
        self.register_buffer("mode_table", mode_table, persistent=True)

    @property
    def num_states(self) -> int:
        return len(self.states)

    @property
    def vocab_size(self) -> int:
        return self.input_vocab_size + self.num_states

    def input_token(self, event: int | Event) -> int:
        return int(Event(int(event)))

    def state_token(self, state: CircuitState | int) -> int:
        state_id = self.state_to_id[state] if isinstance(state, CircuitState) else int(state)
        return self.state_token_offset + state_id

    def token_state_id(self, token: Tensor) -> Tensor:
        return token.to(torch.long) - self.state_token_offset

    def state_from_id(self, state_id: int) -> CircuitState:
        return self.states[int(state_id)]

    def encode_history_from_reference(
        self,
        inputs: Sequence[int | Event],
        start: CircuitState | None = None,
    ) -> list[int]:
        state = start or initial_state()
        tokens = [self.state_token(state)]
        for event in inputs:
            tokens.append(self.input_token(event))
            state = transition(state, event)
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
        if token_history.ndim != 2:
            raise ValueError("token_history must have shape [time] or [batch, time]")
        tokens = token_history.to(torch.long)
        batch_size, seq_len = tokens.shape
        if seq_len < 2:
            raise ValueError("history must include at least an initial state token and input token")
        current_input = tokens[:, -1]
        if ((current_input < 0) | (current_input >= self.input_vocab_size)).any():
            raise ValueError("last token must be an input token")

        is_state = tokens >= self.state_token_offset
        positions = torch.arange(seq_len, device=tokens.device, dtype=self.config.dtype)
        scores = positions[None, :].expand(batch_size, -1) * self.config.score_gap
        scores = scores.masked_fill(~is_state, self.config.ineligible_score)
        selected = scores.argmax(dim=-1)
        selected_tokens = tokens.gather(1, selected[:, None]).squeeze(1)
        state_ids = selected_tokens - self.state_token_offset
        next_ids = self.transition_table[state_ids, current_input]

        logits = torch.full(
            (batch_size, self.num_states),
            -self.config.state_logit_margin,
            dtype=self.config.dtype,
            device=tokens.device,
        )
        logits.scatter_(1, next_ids[:, None], self.config.state_logit_margin)
        debug = {"selected_index": selected, "selected_state_id": state_ids, "next_state_id": next_ids}
        if squeeze:
            logits = logits.squeeze(0)
            debug = {k: v.squeeze(0) for k, v in debug.items()}
        return (logits, debug) if return_debug else logits

    def decode_from_tokens(
        self,
        state_record_history: Sequence[int],
        suffix_inputs: Sequence[int | Event],
        device: str | torch.device = "cpu",
    ) -> tuple[list[int], list[int], list[int]]:
        tokens = [int(x) for x in state_record_history]
        state_ids: list[int] = []
        mode_ids: list[int] = []
        for event in suffix_inputs:
            tokens.append(self.input_token(event))
            tensor = torch.tensor(tokens, dtype=torch.long, device=device)
            with torch.no_grad():
                next_id = int(self.next_state_logits(tensor).argmax(dim=-1).item())
            tokens.append(self.state_token(next_id))
            state_ids.append(next_id)
            mode_ids.append(int(self.mode_table[next_id].item()))
        return state_ids, mode_ids, tokens

    def decode_inputs(
        self,
        inputs: Sequence[int | Event],
        start: CircuitState | None = None,
        device: str | torch.device = "cpu",
    ) -> tuple[list[CircuitState], list[int]]:
        state = start or initial_state()
        tokens = [self.state_token(state)]
        state_ids, _, tokens = self.decode_from_tokens(tokens, inputs, device=device)
        return [self.state_from_id(i) for i in state_ids], tokens


def predict_compiled_circuit_modes(inputs: Sequence[int | Event]) -> list[int]:
    model = CompiledCircuitBreakerTransformer()
    states, _ = model.decode_inputs(inputs)
    return [state.mode for state in states]


def verify_compiled_transition_graph(max_histories_per_state: int = 8) -> dict[str, Any]:
    model = CompiledCircuitBreakerTransformer()
    groups = collect_equivalent_histories(max_per_state=max_histories_per_state)
    failures: list[dict[str, Any]] = []
    checked = 0
    histories_checked = 0
    for state, histories in groups.items():
        expected_state_id = model.state_to_id[state]
        for history in histories:
            tokens = model.encode_history_from_reference(history)
            assert tokens[-1] == model.state_token(expected_state_id)
            histories_checked += 1
            for event in EVENTS:
                checked += 1
                next_expected = transition(state, event)
                expected_id = model.state_to_id[next_expected]
                logits, debug = model.next_state_logits(
                    torch.tensor([*tokens, model.input_token(event)], dtype=torch.long),
                    return_debug=True,
                )
                actual_id = int(logits.argmax(dim=-1).item())
                if actual_id != expected_id:
                    failures.append(
                        {
                            "state": state.to_json(),
                            "history": list(history),
                            "event": int(event),
                            "expected_state": next_expected.to_json(),
                            "expected_id": expected_id,
                            "actual_id": actual_id,
                            "debug": {k: int(v.item()) for k, v in debug.items()},
                        }
                    )
    return {
        "passed": not failures,
        "reachable_states": model.num_states,
        "histories_checked": histories_checked,
        "transitions_checked": checked,
        "failures": failures[:20],
    }


def verify_compiled_history_equivalence(
    max_histories_per_state: int = 8,
    suffixes: Sequence[Sequence[int | Event]] | None = None,
) -> dict[str, Any]:
    model = CompiledCircuitBreakerTransformer()
    groups = collect_equivalent_histories(max_per_state=max_histories_per_state)
    suffixes = suffixes or [
        [Event.SUCCESS, Event.SUCCESS],
        [Event.FAILURE],
        [Event.UNKNOWN] * 9,
        [Event.FAILURE, Event.FAILURE, Event.FAILURE, Event.UNKNOWN, Event.UNKNOWN, Event.UNKNOWN, Event.UNKNOWN, Event.UNKNOWN, Event.SUCCESS, Event.SUCCESS],
        [Event.SUCCESS, Event.UNKNOWN, Event.SUCCESS, Event.FAILURE, Event.UNKNOWN],
    ]
    violations: list[dict[str, Any]] = []
    groups_checked = 0
    comparisons = 0
    for state, histories in groups.items():
        if len(histories) < 2:
            continue
        groups_checked += 1
        for suffix in suffixes:
            reference_outputs = None
            reference_history = None
            for history in histories:
                tokens = model.encode_history_from_reference(history)
                state_ids, _, _ = model.decode_from_tokens(tokens, suffix)
                outputs = tuple(state_ids)
                if reference_outputs is None:
                    reference_outputs = outputs
                    reference_history = history
                else:
                    comparisons += 1
                    if outputs != reference_outputs:
                        violations.append(
                            {
                                "state": state.to_json(),
                                "suffix": [int(Event(x)) for x in suffix],
                                "history_a": list(reference_history or ()),
                                "history_b": list(history),
                                "outputs_a": list(reference_outputs),
                                "outputs_b": list(outputs),
                            }
                        )
    return {
        "passed": not violations,
        "groups_checked": groups_checked,
        "comparisons": comparisons,
        "history_equivalence_violations": len(violations),
        "examples": violations[:20],
    }


def evaluate_compiled_long_traces() -> dict[str, Any]:
    scenarios = {
        "healthy_4096": [Event.SUCCESS, Event.UNKNOWN, Event.SUCCESS, Event.SUCCESS] * 1024,
        "sustained_unknown_1024": [Event.UNKNOWN] * 1024,
        "sustained_failure_256": [Event.FAILURE] * 256,
        "repeated_trip_recover_1280": [Event.FAILURE, Event.FAILURE, Event.FAILURE, Event.UNKNOWN, Event.UNKNOWN, Event.UNKNOWN, Event.UNKNOWN, Event.UNKNOWN, Event.SUCCESS, Event.SUCCESS] * 128,
        "threshold_edge_1024": [Event.SUCCESS, Event.FAILURE, Event.SUCCESS, Event.FAILURE, Event.UNKNOWN, Event.SUCCESS, Event.FAILURE, Event.UNKNOWN] * 128,
    }
    model = CompiledCircuitBreakerTransformer()
    out: dict[str, Any] = {}
    start_time = time.time()
    total_tokens = 0
    for name, inputs in scenarios.items():
        expected = run_controller(inputs)
        actual, _ = model.decode_inputs(inputs)
        total_tokens += len(inputs)
        first_bad = next((i for i, (a, b) in enumerate(zip(expected, actual)) if a != b), None)
        invariant_count = 0
        prev = initial_state()
        for event, pred in zip(inputs, actual):
            invariant_count += len(invariant_violations(prev, event, pred))
            prev = pred
        out[name] = {
            "length": len(inputs),
            "exact_state": first_bad is None,
            "first_divergence": first_bad,
            "invariant_violations": invariant_count,
        }
    elapsed = time.time() - start_time
    out["throughput"] = {
        "tokens": total_tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": total_tokens / elapsed if elapsed else None,
    }
    return out
