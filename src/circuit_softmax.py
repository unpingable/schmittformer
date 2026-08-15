from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from .circuit_compiled import CompiledCircuitBreakerTransformer
from .circuit_reference import (
    EVENTS,
    Event,
    CircuitState,
    collect_equivalent_histories,
    initial_state,
    invariant_violations,
    run_controller,
    state_id_maps,
    transition,
)
from .softmax_attention import SoftmaxAttentionConfig, latest_state_attention


@dataclass(frozen=True)
class SoftmaxCircuitConfig:
    state_record_gap: float = 2.0
    non_state_penalty: float = 4.0
    dtype: torch.dtype = torch.float32

    def attention_config(self) -> SoftmaxAttentionConfig:
        return SoftmaxAttentionConfig(self.state_record_gap, self.non_state_penalty, self.dtype)


class SoftmaxCircuitBreakerTransformer(nn.Module):
    input_vocab_size = 3

    def __init__(self, config: SoftmaxCircuitConfig | None = None):
        super().__init__()
        self.config = config or SoftmaxCircuitConfig()
        states, state_to_id = state_id_maps()
        self.states = states
        self.state_to_id = state_to_id
        self.state_token_offset = self.input_vocab_size
        table = torch.empty((len(states), len(EVENTS)), dtype=torch.long)
        mode_table = torch.empty((len(states),), dtype=torch.long)
        for state, state_id in state_to_id.items():
            mode_table[state_id] = state.mode
            for event in EVENTS:
                table[state_id, int(event)] = state_to_id[transition(state, event)]
        self.register_buffer("transition_table", table, persistent=True)
        self.register_buffer("mode_table", mode_table, persistent=True)

    @property
    def num_states(self) -> int:
        return len(self.states)

    def input_token(self, event: int | Event) -> int:
        return int(Event(int(event)))

    def state_token(self, state: CircuitState | int) -> int:
        state_id = self.state_to_id[state] if isinstance(state, CircuitState) else int(state)
        return self.state_token_offset + state_id

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
        tokens = token_history.to(torch.long)
        current_input = tokens[:, -1]
        if ((current_input < 0) | (current_input >= self.input_vocab_size)).any():
            raise ValueError("last token must be an event input token")
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
        winners = logits.argmax(dim=-1)
        runner = logits.masked_fill(torch.nn.functional.one_hot(winners, self.num_states).bool(), float("-inf")).max(dim=-1).values
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


def verify_softmax_circuit_transition_graph(
    state_record_gap: float,
    non_state_penalty: float,
    dtype: torch.dtype = torch.float32,
    max_histories_per_state: int = 8,
) -> dict[str, Any]:
    model = SoftmaxCircuitBreakerTransformer(SoftmaxCircuitConfig(state_record_gap, non_state_penalty, dtype))
    hard = CompiledCircuitBreakerTransformer()
    groups = collect_equivalent_histories(max_per_state=max_histories_per_state)
    failures = []
    checked = 0
    histories_checked = 0
    min_margin = float("inf")
    max_stale = 0.0
    max_non_state = 0.0
    effectively_hard = True
    finite = True
    for state, histories in groups.items():
        for history in histories:
            tokens = model.encode_history_from_reference(history)
            histories_checked += 1
            for event in EVENTS:
                checked += 1
                expected = transition(state, event)
                expected_id = model.state_to_id[expected]
                logits, debug = model.next_state_logits(torch.tensor([*tokens, model.input_token(event)], dtype=torch.long), return_debug=True)
                actual_id = int(logits.argmax(dim=-1).item())
                hard_states, _, _ = hard.decode_from_tokens(tokens, [event])
                hard_id = hard_states[-1]
                margin = float(debug["decision_margin"].item())
                min_margin = min(min_margin, margin)
                max_stale = max(max_stale, float(debug["stale_state_mass"].item()))
                max_non_state = max(max_non_state, float(debug["non_state_mass"].item()))
                effectively_hard = effectively_hard and bool(debug["effectively_hard"].item())
                finite = finite and bool(debug["finite"].item())
                if actual_id != expected_id or actual_id != hard_id:
                    failures.append({
                        "state": state.to_json(),
                        "history": list(history),
                        "event": int(event),
                        "expected_id": expected_id,
                        "hard_id": hard_id,
                        "actual_id": actual_id,
                        "margin": margin,
                        "correct_mass": float(debug["correct_mass"].item()),
                        "stale_state_mass": float(debug["stale_state_mass"].item()),
                        "non_state_mass": float(debug["non_state_mass"].item()),
                    })
    return {
        "passed": not failures,
        "reachable_states": model.num_states,
        "histories_checked": histories_checked,
        "transitions_checked": checked,
        "failures": failures[:20],
        "min_decision_margin": min_margin,
        "max_stale_state_mass": max_stale,
        "max_non_state_mass": max_non_state,
        "effectively_hard": effectively_hard,
        "finite": finite,
    }


def verify_softmax_circuit_history_equivalence(
    state_record_gap: float,
    non_state_penalty: float,
    dtype: torch.dtype = torch.float32,
    max_histories_per_state: int = 8,
    suffixes: Sequence[Sequence[int | Event]] | None = None,
) -> dict[str, Any]:
    model = SoftmaxCircuitBreakerTransformer(SoftmaxCircuitConfig(state_record_gap, non_state_penalty, dtype))
    groups = collect_equivalent_histories(max_per_state=max_histories_per_state)
    suffixes = suffixes or [[0, 0], [1], [2] * 9, [1, 1, 1, 2, 2, 2, 2, 2, 0, 0], [0, 2, 0, 1]]
    semantic_violations = []
    latent_diffs = []
    comparisons = 0
    groups_checked = 0
    for state, histories in groups.items():
        if len(histories) < 2:
            continue
        groups_checked += 1
        for suffix in suffixes:
            base_outputs = None
            base_masses = None
            base_history = None
            for history in histories:
                tokens = model.encode_history_from_reference(history)
                masses_over_suffix = []
                outputs, _, _ = model.decode_from_tokens(tokens, suffix)
                probe_tokens = list(tokens)
                for event in suffix:
                    probe_tokens.append(model.input_token(event))
                    _, debug = model.next_state_logits(torch.tensor(probe_tokens, dtype=torch.long), return_debug=True)
                    masses_over_suffix.append(debug["state_masses"].detach().cpu())
                    next_id = int(debug["next_logits"].argmax(dim=-1).item())
                    probe_tokens.append(model.state_token(next_id))
                if base_outputs is None:
                    base_outputs = tuple(outputs)
                    base_masses = masses_over_suffix
                    base_history = history
                else:
                    comparisons += 1
                    if tuple(outputs) != base_outputs:
                        semantic_violations.append({
                            "state": state.to_json(),
                            "suffix": [int(Event(x)) for x in suffix],
                            "history_a_len": len(base_history or ()),
                            "history_b_len": len(history),
                            "outputs_a": list(base_outputs),
                            "outputs_b": outputs,
                        })
                    max_diff = 0.0
                    for a, b in zip(base_masses or [], masses_over_suffix):
                        max_diff = max(max_diff, float((a - b).abs().max().item()))
                    latent_diffs.append(max_diff)
    return {
        "passed": not semantic_violations,
        "groups_checked": groups_checked,
        "comparisons": comparisons,
        "history_equivalence_violations": len(semantic_violations),
        "examples": semantic_violations[:20],
        "max_latent_state_mass_diff": max(latent_diffs) if latent_diffs else 0.0,
        "mean_latent_state_mass_diff": sum(latent_diffs) / len(latent_diffs) if latent_diffs else 0.0,
    }


def evaluate_softmax_circuit_long_traces(
    state_record_gap: float,
    non_state_penalty: float,
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    scenarios = {
        "healthy_4096": [Event.SUCCESS, Event.UNKNOWN, Event.SUCCESS, Event.SUCCESS] * 1024,
        "sustained_unknown_1024": [Event.UNKNOWN] * 1024,
        "sustained_failure_256": [Event.FAILURE] * 256,
        "repeated_trip_recover_1280": [Event.FAILURE, Event.FAILURE, Event.FAILURE, Event.UNKNOWN, Event.UNKNOWN, Event.UNKNOWN, Event.UNKNOWN, Event.UNKNOWN, Event.SUCCESS, Event.SUCCESS] * 128,
        "threshold_edge_1024": [Event.SUCCESS, Event.FAILURE, Event.SUCCESS, Event.FAILURE, Event.UNKNOWN, Event.SUCCESS, Event.FAILURE, Event.UNKNOWN] * 128,
    }
    model = SoftmaxCircuitBreakerTransformer(SoftmaxCircuitConfig(state_record_gap, non_state_penalty, dtype))
    out = {}
    for name, inputs in scenarios.items():
        expected = run_controller(inputs)
        actual, _ = model.decode_inputs(inputs)
        first_bad = next((i for i, (a, b) in enumerate(zip(expected, actual)) if a != b), None)
        invariant_count = 0
        prev = initial_state()
        for event, pred in zip(inputs, actual):
            invariant_count += len(invariant_violations(prev, event, pred))
            prev = pred
        out[name] = {"length": len(inputs), "exact_state": first_bad is None, "first_divergence": first_bad, "invariant_violations": invariant_count}
    return out
