from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from .governance_reference import (
    EVENTS,
    EVENT_NAMES,
    OUTPUTS,
    OUTPUT_NAMES,
    Event,
    GovernanceState,
    Output,
    collect_equivalent_histories,
    initial_state,
    invariant_violations,
    output_sequence,
    run_kernel,
    state_id_maps,
    transition,
)


@dataclass(frozen=True)
class CompiledGovernanceConfig:
    logit_margin: float = 32.0
    ineligible_score: float = -1.0e6
    score_gap: float = 16.0
    dtype: torch.dtype = torch.float32


class CompiledGovernanceTransformer(nn.Module):
    """Hard-attention compiled finite governance kernel.

    Histories alternate:

        STATE(s0), EVENT(e1), STATE(s1), EVENT(e2), STATE(s2), ...

    On an event token, the compiled computation retrieves the latest state
    record, applies deterministic transition/output lookup tables, and emits
    logits for both the next state token and the observable governance output.
    """

    input_vocab_size = len(EVENTS)

    def __init__(self, config: CompiledGovernanceConfig | None = None):
        super().__init__()
        self.config = config or CompiledGovernanceConfig()
        states, state_to_id = state_id_maps()
        self.states = states
        self.state_to_id = state_to_id
        self.state_token_offset = self.input_vocab_size
        transition_table = torch.empty((len(states), len(EVENTS)), dtype=torch.long)
        output_table = torch.empty((len(states), len(EVENTS)), dtype=torch.long)
        for state, state_id in state_to_id.items():
            for event in EVENTS:
                result = transition(state, event)
                transition_table[state_id, int(event)] = state_to_id[result.next_state]
                output_table[state_id, int(event)] = result.output
        self.register_buffer("transition_table", transition_table, persistent=True)
        self.register_buffer("output_table", output_table, persistent=True)

    @property
    def num_states(self) -> int:
        return len(self.states)

    @property
    def num_outputs(self) -> int:
        return len(OUTPUTS)

    @property
    def vocab_size(self) -> int:
        return self.input_vocab_size + self.num_states

    def input_token(self, event: int | Event) -> int:
        return int(Event(int(event)))

    def state_token(self, state: GovernanceState | int) -> int:
        state_id = self.state_to_id[state] if isinstance(state, GovernanceState) else int(state)
        return self.state_token_offset + state_id

    def state_from_id(self, state_id: int) -> GovernanceState:
        return self.states[int(state_id)]

    def encode_history_from_reference(
        self,
        events: Sequence[int | Event],
        start: GovernanceState | None = None,
    ) -> list[int]:
        state = start or initial_state()
        tokens = [self.state_token(state)]
        for event in events:
            tokens.append(self.input_token(event))
            state = transition(state, event).next_state
            tokens.append(self.state_token(state))
        return tokens

    def next_logits(
        self,
        token_history: Tensor,
        return_debug: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, dict[str, Tensor]]:
        squeeze = False
        if token_history.ndim == 1:
            token_history = token_history.unsqueeze(0)
            squeeze = True
        if token_history.ndim != 2:
            raise ValueError("token_history must have shape [time] or [batch, time]")
        tokens = token_history.to(torch.long)
        batch_size, seq_len = tokens.shape
        if seq_len < 2:
            raise ValueError("history must include an initial state token and an event token")
        current_event = tokens[:, -1]
        if ((current_event < 0) | (current_event >= self.input_vocab_size)).any():
            raise ValueError("last token must be an event token")

        is_state = tokens >= self.state_token_offset
        positions = torch.arange(seq_len, device=tokens.device, dtype=self.config.dtype)
        scores = positions[None, :].expand(batch_size, -1) * self.config.score_gap
        scores = scores.masked_fill(~is_state, self.config.ineligible_score)
        selected = scores.argmax(dim=-1)
        selected_tokens = tokens.gather(1, selected[:, None]).squeeze(1)
        state_ids = selected_tokens - self.state_token_offset
        next_ids = self.transition_table[state_ids, current_event]
        output_ids = self.output_table[state_ids, current_event]

        state_logits = torch.full(
            (batch_size, self.num_states),
            -self.config.logit_margin,
            dtype=self.config.dtype,
            device=tokens.device,
        )
        output_logits = torch.full(
            (batch_size, self.num_outputs),
            -self.config.logit_margin,
            dtype=self.config.dtype,
            device=tokens.device,
        )
        state_logits.scatter_(1, next_ids[:, None], self.config.logit_margin)
        output_logits.scatter_(1, output_ids[:, None], self.config.logit_margin)
        debug = {
            "selected_index": selected,
            "selected_state_id": state_ids,
            "next_state_id": next_ids,
            "output_id": output_ids,
        }
        if squeeze:
            state_logits = state_logits.squeeze(0)
            output_logits = output_logits.squeeze(0)
            debug = {key: value.squeeze(0) for key, value in debug.items()}
        if return_debug:
            return state_logits, output_logits, debug
        return state_logits, output_logits

    def decode_from_tokens(
        self,
        state_record_history: Sequence[int],
        suffix_events: Sequence[int | Event],
        device: str | torch.device = "cpu",
    ) -> tuple[list[int], list[int], list[int]]:
        tokens = [int(x) for x in state_record_history]
        state_ids: list[int] = []
        output_ids: list[int] = []
        for event in suffix_events:
            tokens.append(self.input_token(event))
            tensor = torch.tensor(tokens, dtype=torch.long, device=device)
            with torch.no_grad():
                state_logits, output_logits = self.next_logits(tensor)
                next_id = int(state_logits.argmax(dim=-1).item())
                output_id = int(output_logits.argmax(dim=-1).item())
            tokens.append(self.state_token(next_id))
            state_ids.append(next_id)
            output_ids.append(output_id)
        return state_ids, output_ids, tokens

    def decode_events(
        self,
        events: Sequence[int | Event],
        start: GovernanceState | None = None,
        device: str | torch.device = "cpu",
    ) -> tuple[list[GovernanceState], list[int], list[int]]:
        state = start or initial_state()
        tokens = [self.state_token(state)]
        state_ids, output_ids, tokens = self.decode_from_tokens(tokens, events, device=device)
        return [self.state_from_id(state_id) for state_id in state_ids], output_ids, tokens


def verify_compiled_governance_transition_graph(max_histories_per_state: int = 8) -> dict[str, Any]:
    model = CompiledGovernanceTransformer()
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
                expected = transition(state, event)
                expected_id = model.state_to_id[expected.next_state]
                state_logits, output_logits, debug = model.next_logits(
                    torch.tensor([*tokens, model.input_token(event)], dtype=torch.long),
                    return_debug=True,
                )
                actual_state_id = int(state_logits.argmax(dim=-1).item())
                actual_output_id = int(output_logits.argmax(dim=-1).item())
                if actual_state_id != expected_id or actual_output_id != expected.output:
                    failures.append(
                        {
                            "state": state.to_json(),
                            "history": list(history),
                            "event": EVENT_NAMES[Event(event)],
                            "event_id": int(event),
                            "expected_state_id": expected_id,
                            "actual_state_id": actual_state_id,
                            "expected_output": OUTPUT_NAMES[expected.output_enum],
                            "expected_output_id": expected.output,
                            "actual_output": OUTPUT_NAMES[Output(actual_output_id)],
                            "actual_output_id": actual_output_id,
                            "debug": {key: int(value.item()) for key, value in debug.items()},
                        }
                    )
    return {
        "passed": not failures,
        "reachable_states": model.num_states,
        "histories_checked": histories_checked,
        "transitions_checked": checked,
        "failures": failures[:20],
    }


def verify_compiled_governance_history_equivalence(
    max_histories_per_state: int = 8,
    suffixes: Sequence[Sequence[int | Event]] | None = None,
) -> dict[str, Any]:
    model = CompiledGovernanceTransformer()
    groups = collect_equivalent_histories(max_per_state=max_histories_per_state)
    suffixes = suffixes or [
        [Event.PROPOSE_A],
        [Event.PROPOSE_B],
        [Event.CLAIM_EVIDENCE, Event.PROPOSE_A],
        [Event.LEASE_TICK, Event.LEASE_TICK, Event.LEASE_TICK, Event.PROPOSE_A],
        [Event.ACTION_RESULT_AMBIGUOUS, Event.PROPOSE_A, Event.CLAIM_RECOVERY_SUCCESS, Event.SETTLE_FAILURE],
        [Event.SETTLE_SUCCESS, Event.FUND_BUDGET, Event.PROPOSE_B],
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
                state_ids, output_ids, _ = model.decode_from_tokens(tokens, suffix)
                outputs = (tuple(state_ids), tuple(output_ids))
                if reference_outputs is None:
                    reference_outputs = outputs
                    reference_history = history
                else:
                    comparisons += 1
                    if outputs != reference_outputs:
                        violations.append(
                            {
                                "state": state.to_json(),
                                "suffix": [int(Event(event)) for event in suffix],
                                "history_a": list(reference_history or ()),
                                "history_b": list(history),
                                "outputs_a": [list(reference_outputs[0]), list(reference_outputs[1])],
                                "outputs_b": [list(outputs[0]), list(outputs[1])],
                            }
                        )
    return {
        "passed": not violations,
        "groups_checked": groups_checked,
        "comparisons": comparisons,
        "history_equivalence_violations": len(violations),
        "examples": violations[:20],
    }


def evaluate_compiled_governance_long_traces() -> dict[str, Any]:
    scenarios = {
        "healthy_claim_spam_4096": [Event.CLAIM_EVIDENCE, Event.NOOP, Event.MALFORMED_REQUEST, Event.NOOP] * 1024,
        "repeated_admit_settle_512": [
            Event.ISSUE_AUTH_A,
            Event.QUALIFY_EVIDENCE,
            Event.FUND_BUDGET,
            Event.PROPOSE_A,
            Event.ACTION_RESULT_SUCCESS,
            Event.PROPOSE_A,
            Event.ACTION_RESULT_FAILURE,
            Event.FUND_BUDGET,
        ]
        * 64,
        "ambiguous_cycles_1024": [
            Event.ISSUE_AUTH_B,
            Event.QUALIFY_EVIDENCE,
            Event.FUND_BUDGET,
            Event.PROPOSE_B,
            Event.ACTION_RESULT_AMBIGUOUS,
            Event.PROPOSE_B,
            Event.CLAIM_RECOVERY_SUCCESS,
            Event.SETTLE_FAILURE,
        ]
        * 128,
        "lease_edges_768": [
            Event.ISSUE_AUTH_A,
            Event.QUALIFY_EVIDENCE,
            Event.FUND_BUDGET,
            Event.LEASE_TICK,
            Event.PROPOSE_A,
            Event.ACTION_RESULT_SUCCESS,
            Event.LEASE_TICK,
            Event.LEASE_TICK,
            Event.PROPOSE_A,
            Event.ISSUE_AUTH_A,
            Event.PROPOSE_A,
            Event.ACTION_RESULT_AMBIGUOUS,
            Event.SETTLE_SUCCESS,
        ]
        * 59,
    }
    model = CompiledGovernanceTransformer()
    out: dict[str, Any] = {}
    for name, events in scenarios.items():
        expected_states, expected_results = run_kernel(events)
        actual_states, actual_outputs, _ = model.decode_events(events)
        first_state_bad = next((i for i, (a, b) in enumerate(zip(expected_states, actual_states)) if a != b), None)
        expected_outputs = [result.output for result in expected_results]
        first_output_bad = next((i for i, (a, b) in enumerate(zip(expected_outputs, actual_outputs)) if a != b), None)
        invariant_count = 0
        prev = initial_state()
        for event, state, output in zip(events, actual_states, actual_outputs):
            expected_like = transition(prev, event)
            predicted = type(expected_like)(state, output, expected_like.refusal_reason, expected_like.admitted_action)
            invariant_count += len(invariant_violations(prev, event, predicted))
            prev = state
        out[name] = {
            "length": len(events),
            "exact_state": first_state_bad is None,
            "exact_output": first_output_bad is None,
            "first_state_divergence": first_state_bad,
            "first_output_divergence": first_output_bad,
            "invariant_violations": invariant_count,
        }
    return out


def predict_compiled_governance_outputs(events: Sequence[int | Event]) -> list[int]:
    model = CompiledGovernanceTransformer()
    _, outputs, _ = model.decode_events(events)
    return outputs

