from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from .compiled_bits import (
    hard_and,
    hard_and_many,
    hard_eq_bits,
    hard_mux,
    hard_not,
    hard_one_hot,
    hard_or,
    hard_or_many,
    int_tensor_to_bits,
    saturating_decrement_bits,
)
from .fixed_state import (
    BUDGET_BITS,
    EVENT_COUNT,
    LEASE_BITS,
    OUTPUT_COUNT,
    STATE_WIDTH,
    Authority,
    GovernanceEvent,
    GovernanceOutput,
    GovernanceState,
    Occurrence,
    Settlement,
    decode_state_bits,
    decode_states_bits,
    encode_state_bits,
    encode_states_bits,
    event_tensor,
)
from .recurrent_reference import invariant_violations, transition


@dataclass(frozen=True)
class RecurrentCompiledConfig:
    dtype: torch.dtype = torch.float32
    logit_margin: float = 16.0


@dataclass(frozen=True)
class CompiledStepResult:
    next_state_bits: Tensor
    output_logits: Tensor
    debug: dict[str, Tensor]


class CompiledRecurrentGovernanceTransformer(nn.Module):
    """Fixed-width synthesized recurrent governance transition.

    Input is only `(state_t bits, event_t)`. No prior execution history or KV
    cache is supplied. The computation is an unrolled hard/discrete Boolean
    tensor circuit over fixed slots, shaped like a transformer-program MLP block:
    exact threshold gates, hard multiplexers, and deterministic output logits.
    """

    def __init__(self, config: RecurrentCompiledConfig | None = None):
        super().__init__()
        self.config = config or RecurrentCompiledConfig()
        self._constant_cache: dict[tuple[int, str, int | None, str], dict[str, Tensor]] = {}

    @property
    def physical_input_width(self) -> int:
        return STATE_WIDTH + EVENT_COUNT

    @property
    def physical_output_width(self) -> int:
        return STATE_WIDTH + OUTPUT_COUNT

    def _split_state(self, state_bits: Tensor) -> dict[str, Tensor]:
        pos = 0
        authority = state_bits[:, pos]
        pos += 1
        lease = state_bits[:, pos : pos + LEASE_BITS]
        pos += LEASE_BITS
        budget = state_bits[:, pos : pos + BUDGET_BITS]
        pos += BUDGET_BITS
        occurrence = state_bits[:, pos : pos + 2]
        pos += 2
        settlement = state_bits[:, pos : pos + 2]
        return {"authority": authority, "lease": lease, "budget": budget, "occurrence": occurrence, "settlement": settlement}

    def _event_flags(self, event_ids: Tensor, dtype: torch.dtype) -> dict[GovernanceEvent, Tensor]:
        if event_ids.ndim == 2 and event_ids.shape[1] == EVENT_COUNT:
            onehot = event_ids.to(dtype)
        else:
            onehot = torch.nn.functional.one_hot(event_ids.to(torch.long), num_classes=EVENT_COUNT).to(dtype)
        return {event: onehot[:, int(event)] for event in GovernanceEvent}

    def _state_bits_from_fields(self, authority: Tensor, lease: Tensor, budget: Tensor, occurrence: Tensor, settlement: Tensor) -> Tensor:
        return torch.cat([authority[:, None], lease, budget, occurrence, settlement], dim=-1)

    def _constant_state_bits(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> dict[str, Tensor]:
        key = (batch_size, device.type, device.index, str(dtype))
        cached = self._constant_cache.get(key)
        if cached is not None:
            return cached
        zeros_lease = torch.zeros((batch_size, LEASE_BITS), dtype=dtype, device=device)
        ones_lease = torch.ones((batch_size, LEASE_BITS), dtype=dtype, device=device)
        one_lease = int_tensor_to_bits(torch.ones(batch_size, dtype=torch.long, device=device), LEASE_BITS, dtype=dtype)
        zeros_budget = torch.zeros((batch_size, BUDGET_BITS), dtype=dtype, device=device)
        ones_budget = torch.ones((batch_size, BUDGET_BITS), dtype=dtype, device=device)
        one_budget = int_tensor_to_bits(torch.ones(batch_size, dtype=torch.long, device=device), BUDGET_BITS, dtype=dtype)
        idle = int_tensor_to_bits(torch.full((batch_size,), int(Occurrence.IDLE), dtype=torch.long, device=device), 2, dtype=dtype)
        in_flight = int_tensor_to_bits(torch.full((batch_size,), int(Occurrence.IN_FLIGHT), dtype=torch.long, device=device), 2, dtype=dtype)
        ambiguous = int_tensor_to_bits(torch.full((batch_size,), int(Occurrence.AMBIGUOUS), dtype=torch.long, device=device), 2, dtype=dtype)
        none = int_tensor_to_bits(torch.full((batch_size,), int(Settlement.NONE), dtype=torch.long, device=device), 2, dtype=dtype)
        success = int_tensor_to_bits(torch.full((batch_size,), int(Settlement.SUCCESS), dtype=torch.long, device=device), 2, dtype=dtype)
        failure = int_tensor_to_bits(torch.full((batch_size,), int(Settlement.FAILURE), dtype=torch.long, device=device), 2, dtype=dtype)
        safe = self._state_bits_from_fields(
            torch.zeros(batch_size, dtype=dtype, device=device),
            zeros_lease,
            zeros_budget,
            idle,
            none,
        )
        out = {
            "zeros_lease": zeros_lease,
            "ones_lease": ones_lease,
            "one_lease": one_lease,
            "zeros_budget": zeros_budget,
            "ones_budget": ones_budget,
            "one_budget": one_budget,
            "idle": idle,
            "in_flight": in_flight,
            "ambiguous": ambiguous,
            "none": none,
            "success": success,
            "failure": failure,
            "safe_state": safe,
        }
        self._constant_cache[key] = out
        return out

    def forward(self, state_bits: Tensor, event_ids: Tensor, return_debug: bool = False) -> Tensor | tuple[Tensor, Tensor] | CompiledStepResult:
        squeeze = False
        if state_bits.ndim == 1:
            state_bits = state_bits.unsqueeze(0)
            if not (event_ids.ndim == 2 and event_ids.shape[1] == EVENT_COUNT):
                event_ids = event_ids.reshape(1)
            squeeze = True
        if state_bits.ndim != 2 or state_bits.shape[1] != STATE_WIDTH:
            raise ValueError(f"state_bits must have shape [batch,{STATE_WIDTH}]")
        if event_ids.ndim == 0:
            event_ids = event_ids.reshape(1)
        if event_ids.shape[0] != state_bits.shape[0]:
            raise ValueError("event batch size mismatch")

        dtype = self.config.dtype
        state_bits = state_bits.to(dtype)
        batch_size = state_bits.shape[0]
        device = state_bits.device
        fields = self._split_state(state_bits)
        c = self._constant_state_bits(batch_size, device, dtype)
        e = self._event_flags(event_ids.to(device), dtype)

        occurrence = fields["occurrence"]
        settlement = fields["settlement"]
        is_idle = hard_eq_bits(occurrence, [0, 0])
        is_in_flight = hard_eq_bits(occurrence, [1, 0])
        is_ambiguous = hard_eq_bits(occurrence, [0, 1])
        valid_occurrence = hard_or_many(torch.stack([is_idle, is_in_flight, is_ambiguous], dim=-1))
        is_settlement_none = hard_eq_bits(settlement, [0, 0])
        is_settlement_success = hard_eq_bits(settlement, [1, 0])
        is_settlement_failure = hard_eq_bits(settlement, [0, 1])
        valid_settlement = hard_or_many(torch.stack([is_settlement_none, is_settlement_success, is_settlement_failure], dim=-1))
        valid_state = hard_and(valid_occurrence, valid_settlement)
        invalid_state = hard_not(valid_state)

        lease_dec, lease_zero, lease_nonzero = saturating_decrement_bits(fields["lease"])
        budget_dec, budget_zero, budget_nonzero = saturating_decrement_bits(fields["budget"])
        del lease_zero, budget_zero
        authority_valid = fields["authority"]
        authority_invalid = hard_not(authority_valid)

        proposal = e[GovernanceEvent.PROPOSE_ACTION]
        result_success = e[GovernanceEvent.RESULT_SUCCESS]
        result_failure = e[GovernanceEvent.RESULT_FAILURE]
        result_ambiguous = e[GovernanceEvent.RESULT_AMBIGUOUS]
        settle_success = e[GovernanceEvent.SETTLE_SUCCESS]
        settle_failure = e[GovernanceEvent.SETTLE_FAILURE]
        any_settle = hard_or(settle_success, settle_failure)

        propose_and_valid = hard_and(proposal, valid_state)
        propose_idle = hard_and(propose_and_valid, is_idle)
        admit = hard_and_many(torch.stack([propose_idle, authority_valid, lease_nonzero, budget_nonzero], dim=-1))
        refuse_ambiguous = hard_and(propose_and_valid, is_ambiguous)
        refuse_in_flight = hard_and(propose_and_valid, is_in_flight)
        refuse_no_authority = hard_and_many(torch.stack([propose_idle, authority_invalid], dim=-1))
        refuse_expired = hard_and_many(torch.stack([propose_idle, authority_valid, hard_not(lease_nonzero)], dim=-1))
        refuse_budget = hard_and_many(torch.stack([propose_idle, authority_valid, lease_nonzero, hard_not(budget_nonzero)], dim=-1))

        authority = fields["authority"]
        authority = hard_mux(e[GovernanceEvent.GRANT_AUTHORITY], torch.ones_like(authority), authority)
        authority = hard_mux(e[GovernanceEvent.REVOKE_AUTHORITY], torch.zeros_like(authority), authority)

        lease = fields["lease"]
        lease = hard_mux(e[GovernanceEvent.TICK], lease_dec, lease)
        lease = hard_mux(e[GovernanceEvent.RENEW_LEASE_MAX], c["ones_lease"], lease)
        lease = hard_mux(e[GovernanceEvent.RENEW_LEASE_ONE], c["one_lease"], lease)

        budget = fields["budget"]
        budget = hard_mux(admit, budget_dec, budget)
        budget = hard_mux(e[GovernanceEvent.RESET_BUDGET_MAX], c["ones_budget"], budget)
        budget = hard_mux(e[GovernanceEvent.RESET_BUDGET_ONE], c["one_budget"], budget)

        occurrence_next = occurrence
        settlement_next = settlement
        success_from_inflight = hard_and(is_in_flight, result_success)
        failure_from_inflight = hard_and(is_in_flight, result_failure)
        ambiguous_from_inflight = hard_and(is_in_flight, result_ambiguous)
        settle_from_ambiguous = hard_and(is_ambiguous, any_settle)
        settle_success_from_ambiguous = hard_and(is_ambiguous, settle_success)
        settle_failure_from_ambiguous = hard_and(is_ambiguous, settle_failure)

        occurrence_next = hard_mux(admit, c["in_flight"], occurrence_next)
        occurrence_next = hard_mux(success_from_inflight, c["idle"], occurrence_next)
        occurrence_next = hard_mux(failure_from_inflight, c["idle"], occurrence_next)
        occurrence_next = hard_mux(ambiguous_from_inflight, c["ambiguous"], occurrence_next)
        occurrence_next = hard_mux(settle_from_ambiguous, c["idle"], occurrence_next)

        settlement_next = hard_mux(admit, c["none"], settlement_next)
        settlement_next = hard_mux(success_from_inflight, c["success"], settlement_next)
        settlement_next = hard_mux(failure_from_inflight, c["failure"], settlement_next)
        settlement_next = hard_mux(ambiguous_from_inflight, c["none"], settlement_next)
        settlement_next = hard_mux(settle_success_from_ambiguous, c["success"], settlement_next)
        settlement_next = hard_mux(settle_failure_from_ambiguous, c["failure"], settlement_next)

        next_state = self._state_bits_from_fields(authority, lease, budget, occurrence_next, settlement_next)
        next_state = hard_mux(invalid_state, c["safe_state"], next_state)

        output = hard_one_hot(int(GovernanceOutput.NO_ACTION), OUTPUT_COUNT, (batch_size,), device, dtype)
        output = hard_mux(admit, hard_one_hot(int(GovernanceOutput.ADMIT_ACTION), OUTPUT_COUNT, (batch_size,), device, dtype), output)
        output = hard_mux(refuse_budget, hard_one_hot(int(GovernanceOutput.REFUSE_BUDGET), OUTPUT_COUNT, (batch_size,), device, dtype), output)
        output = hard_mux(refuse_expired, hard_one_hot(int(GovernanceOutput.REFUSE_EXPIRED), OUTPUT_COUNT, (batch_size,), device, dtype), output)
        output = hard_mux(refuse_no_authority, hard_one_hot(int(GovernanceOutput.REFUSE_NO_AUTHORITY), OUTPUT_COUNT, (batch_size,), device, dtype), output)
        output = hard_mux(refuse_in_flight, hard_one_hot(int(GovernanceOutput.REFUSE_IN_FLIGHT), OUTPUT_COUNT, (batch_size,), device, dtype), output)
        output = hard_mux(refuse_ambiguous, hard_one_hot(int(GovernanceOutput.REFUSE_AMBIGUOUS), OUTPUT_COUNT, (batch_size,), device, dtype), output)
        output = hard_mux(invalid_state, hard_one_hot(int(GovernanceOutput.REFUSE_INVALID_STATE), OUTPUT_COUNT, (batch_size,), device, dtype), output)
        logits = output * self.config.logit_margin + (1.0 - output) * (-self.config.logit_margin)

        debug = {
            "valid_state": valid_state,
            "is_idle": is_idle,
            "is_in_flight": is_in_flight,
            "is_ambiguous": is_ambiguous,
            "lease_nonzero": lease_nonzero,
            "budget_nonzero": budget_nonzero,
            "admit": admit,
            "refuse_ambiguous": refuse_ambiguous,
            "refuse_in_flight": refuse_in_flight,
            "refuse_no_authority": refuse_no_authority,
            "refuse_expired": refuse_expired,
            "refuse_budget": refuse_budget,
        }
        if squeeze:
            next_state = next_state.squeeze(0)
            logits = logits.squeeze(0)
            debug = {k: v.squeeze(0) for k, v in debug.items()}
        if return_debug:
            return CompiledStepResult(next_state, logits, debug)
        return next_state, logits


def compiled_step(
    state: GovernanceState,
    event: int | GovernanceEvent,
    model: CompiledRecurrentGovernanceTransformer | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[GovernanceState, int]:
    model = model or CompiledRecurrentGovernanceTransformer(RecurrentCompiledConfig(dtype=dtype)).to(device)
    state_bits = encode_state_bits(state, device=torch.device(device), dtype=dtype)
    event_ids = event_tensor([event], device=torch.device(device))
    with torch.no_grad():
        next_bits, logits = model(state_bits, event_ids)
    return decode_state_bits(next_bits), int(logits.argmax(dim=-1).detach().cpu().item())


def run_compiled_trace(
    events: Sequence[int | GovernanceEvent],
    start: GovernanceState,
    model: CompiledRecurrentGovernanceTransformer | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[list[GovernanceState], list[int]]:
    device = torch.device(device)
    model = model or CompiledRecurrentGovernanceTransformer(RecurrentCompiledConfig(dtype=dtype)).to(device)
    state_bits = encode_state_bits(start, device=device, dtype=dtype)
    states: list[GovernanceState] = []
    outputs: list[int] = []
    with torch.no_grad():
        for event in events:
            state_bits, logits = model(state_bits, event_tensor([event], device=device))
            state = decode_state_bits(state_bits)
            states.append(state)
            outputs.append(int(logits.argmax(dim=-1).detach().cpu().item()))
    return states, outputs


def compare_compiled_reference(states: Sequence[GovernanceState], events: Sequence[int | GovernanceEvent], device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> dict[str, Any]:
    device = torch.device(device)
    model = CompiledRecurrentGovernanceTransformer(RecurrentCompiledConfig(dtype=dtype)).to(device)
    state_bits = encode_states_bits(states, device=device, dtype=dtype)
    event_ids = event_tensor(events, device=device)
    with torch.no_grad():
        next_bits, logits = model(state_bits, event_ids)
    actual_states = decode_states_bits(next_bits)
    actual_outputs = logits.argmax(dim=-1).detach().cpu().tolist()
    failures: list[dict[str, Any]] = []
    invariant_count = 0
    for index, (state, event, actual_state, actual_output) in enumerate(zip(states, events, actual_states, actual_outputs)):
        expected = transition(state, event)
        invariant_count += len(invariant_violations(state, event, expected))
        if actual_state != expected.next_state or int(actual_output) != expected.output:
            failures.append(
                {
                    "index": index,
                    "state": state.to_json(),
                    "event": int(GovernanceEvent(int(event))),
                    "expected": expected.to_json(),
                    "actual_state": actual_state.to_json(),
                    "actual_output": int(actual_output),
                }
            )
            if len(failures) >= 20:
                break
    return {
        "checked": len(states),
        "passed": not failures,
        "failures": failures,
        "reference_invariant_violations": invariant_count,
        "device": str(device),
        "dtype": str(dtype),
    }


def random_valid_states(count: int, seed: int = 1234) -> list[GovernanceState]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    authority = torch.randint(0, 2, (count,), generator=generator)
    lease = torch.randint(0, 1 << LEASE_BITS, (count,), generator=generator)
    budget = torch.randint(0, 1 << BUDGET_BITS, (count,), generator=generator)
    occurrence = torch.randint(0, 3, (count,), generator=generator)
    settlement = torch.randint(0, 3, (count,), generator=generator)
    return [
        GovernanceState(int(a), int(l), int(b), int(o), int(s))
        for a, l, b, o, s in zip(authority.tolist(), lease.tolist(), budget.tolist(), occurrence.tolist(), settlement.tolist())
    ]


def random_events(count: int, seed: int = 5678) -> list[GovernanceEvent]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    ids = torch.randint(0, EVENT_COUNT, (count,), generator=generator).tolist()
    return [GovernanceEvent(int(x)) for x in ids]


def invalid_state_cases(dtype: torch.dtype = torch.float32) -> list[dict[str, Any]]:
    device = torch.device("cpu")
    model = CompiledRecurrentGovernanceTransformer(RecurrentCompiledConfig(dtype=dtype)).to(device)
    base = encode_state_bits(GovernanceState(int(Authority.VALID), 10, 10, int(Occurrence.IDLE), int(Settlement.NONE)), dtype=dtype)
    cases = []
    # occurrence = 3 is invalid
    occ_bad = base.clone()
    occ_bad[1 + LEASE_BITS + BUDGET_BITS : 1 + LEASE_BITS + BUDGET_BITS + 2] = torch.tensor([1.0, 1.0], dtype=dtype)
    # settlement = 3 is invalid
    set_bad = base.clone()
    set_bad[-2:] = torch.tensor([1.0, 1.0], dtype=dtype)
    for name, bits in [("invalid_occurrence_3", occ_bad), ("invalid_settlement_3", set_bad)]:
        with torch.no_grad():
            next_bits, logits = model(bits, event_tensor([GovernanceEvent.PROPOSE_ACTION]))
        cases.append(
            {
                "case": name,
                "output": int(logits.argmax(dim=-1).item()),
                "next_state": decode_state_bits(next_bits).to_json(),
            }
        )
    return cases
