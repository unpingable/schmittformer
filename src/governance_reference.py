from __future__ import annotations

from collections import deque
from itertools import product
from dataclasses import dataclass
from enum import IntEnum
from random import Random
from typing import Any, Sequence


class ProposalKind(IntEnum):
    NONE = 0
    WORK_A = 1
    WORK_B = 2


class PreconditionBasis(IntEnum):
    NONE = 0
    P0 = 1
    P1 = 2


class ProgramCounter(IntEnum):
    OBSERVATION_REQUIRED = 0
    PROPOSAL_RECORDED = 1
    STANDING_REQUIRED = 2
    ADMISSIBLE_PENDING_AUTHORIZATION = 3
    AUTHORIZATION_CONSUMED = 4
    DISPATCHED = 5
    RECONCILIATION_REQUIRED = 6
    SETTLED_OBSERVATION_REQUIRED = 7
    HALTED = 8
    COMPLETED = 9


class SettlementOutcome(IntEnum):
    NONE = 0
    SUCCESS = 1
    FAILURE = 2


class Event(IntEnum):
    NOOP = 0
    CLAIM_AUTHORITY_RECORD = 1
    PROPOSE_INITIAL_A_P0 = 2
    PROPOSE_INITIAL_A_P1 = 3
    PROPOSE_INITIAL_B_P0 = 4
    PROPOSE_RETRY_A_P0 = 5
    PROPOSE_RETRY_A_P1 = 6
    PROPOSE_RETRY_B_P0 = 7
    PROPOSE_SUCCESSOR_A_P0 = 8
    PROPOSE_SUCCESSOR_B_P0 = 9
    PROPOSE_STALE_OBSERVATION = 10
    PROPOSE_CONTRADICTORY_OBSERVATION = 11
    REQUIRE_STANDING = 12
    RECORD_ADMISSIBLE_CURRENT = 13
    RECORD_ADMISSIBLE_ABSENT_STANDING = 14
    RECORD_ADMISSIBLE_REVOKED_STANDING = 15
    RECORD_ADMISSIBLE_EXPIRED_STANDING = 16
    RECORD_ADMISSIBLE_INADMISSIBLE = 17
    RECORD_ADMISSIBLE_STALE_OBSERVATION = 18
    RECORD_ADMISSIBLE_CONTRADICTION = 19
    TICK = 20
    CONSUME_AUTH_CURRENT = 21
    CONSUME_AUTH_EXPIRED_STANDING = 22
    CONSUME_AUTH_STALE_OBSERVATION = 23
    CONSUME_AUTH_INADMISSIBLE = 24
    ACCEPT_DOCKET_CUSTODY = 25
    RECORD_SETTLEMENT_SUCCESS = 26
    RECORD_SETTLEMENT_FAILURE = 27
    REQUIRE_RECONCILIATION = 28
    RECORD_RECONCILED_SUCCESS = 29
    RECORD_RECONCILED_FAILURE = 30
    OPEN_CONTINUATION = 31
    NOTE_PROBE = 32
    HALT = 33
    ESCALATE = 34
    HUMAN_RETURN = 35
    HUMAN_TERMINATE = 36
    COMPLETE = 37
    MALFORMED = 38


class RefusalReason(IntEnum):
    ILLEGAL_TRANSITION = 0
    BINDING_MISMATCH = 1
    STALE_OBSERVATION = 2
    CONTRADICTION = 3
    ABSENT_STANDING = 4
    STANDING_NOT_CURRENT = 5
    INADMISSIBLE_EXACT_WORK = 6
    BUDGET_EXHAUSTED = 7
    RETRY_PRECONDITIONS_CHANGED = 8
    RETRY_PROPOSAL_CHANGED = 9
    SUCCESSOR_PROPOSAL_REUSED = 10
    OCCURRENCE_REUSED = 11
    UNRESOLVED_ATTEMPT = 12
    HUMAN_DECISION_REQUIRED = 13
    MALFORMED = 14


class Output(IntEnum):
    NO_OUTPUT = 0
    CLAIM_IGNORED = 1
    PROPOSAL_RECORDED = 2
    STANDING_REQUIRED = 3
    ADMISSIBLE_RECORDED = 4
    AUTHORIZATION_CONSUMED_A = 5
    AUTHORIZATION_CONSUMED_B = 6
    DOCKET_CUSTODY_ACCEPTED = 7
    SETTLED_SUCCESS = 8
    SETTLED_FAILURE = 9
    RECONCILIATION_REQUIRED = 10
    CONTINUATION_OPENED = 11
    PROBE_NOTED = 12
    HALTED = 13
    ESCALATED = 14
    HUMAN_RETURNED = 15
    COMPLETED = 16
    TICKED = 17
    REFUSE_ILLEGAL_TRANSITION = 18
    REFUSE_BINDING_MISMATCH = 19
    REFUSE_STALE_OBSERVATION = 20
    REFUSE_CONTRADICTION = 21
    REFUSE_ABSENT_STANDING = 22
    REFUSE_STANDING_NOT_CURRENT = 23
    REFUSE_INADMISSIBLE_EXACT_WORK = 24
    REFUSE_BUDGET_EXHAUSTED = 25
    REFUSE_RETRY_PRECONDITIONS_CHANGED = 26
    REFUSE_RETRY_PROPOSAL_CHANGED = 27
    REFUSE_SUCCESSOR_PROPOSAL_REUSED = 28
    REFUSE_OCCURRENCE_REUSED = 29
    REFUSE_UNRESOLVED_ATTEMPT = 30
    REFUSE_HUMAN_DECISION_REQUIRED = 31
    REFUSE_MALFORMED = 32


RETRY_LIMIT = 2
PROBE_LIMIT = 1
ESCALATION_LIMIT = 1
STANDING_LEASE_MAX = 2

PROPOSAL_NAMES = {value: value.name for value in ProposalKind}
PRECONDITION_NAMES = {value: value.name for value in PreconditionBasis}
PROGRAM_COUNTER_NAMES = {value: value.name for value in ProgramCounter}
SETTLEMENT_OUTCOME_NAMES = {value: value.name for value in SettlementOutcome}
EVENT_NAMES = {value: value.name for value in Event}
REFUSAL_NAMES = {value: value.name for value in RefusalReason}
OUTPUT_NAMES = {value: value.name for value in Output}

EVENTS: tuple[Event, ...] = tuple(Event)
OUTPUTS: tuple[Output, ...] = tuple(Output)

REFUSAL_OUTPUTS: dict[RefusalReason, Output] = {
    RefusalReason.ILLEGAL_TRANSITION: Output.REFUSE_ILLEGAL_TRANSITION,
    RefusalReason.BINDING_MISMATCH: Output.REFUSE_BINDING_MISMATCH,
    RefusalReason.STALE_OBSERVATION: Output.REFUSE_STALE_OBSERVATION,
    RefusalReason.CONTRADICTION: Output.REFUSE_CONTRADICTION,
    RefusalReason.ABSENT_STANDING: Output.REFUSE_ABSENT_STANDING,
    RefusalReason.STANDING_NOT_CURRENT: Output.REFUSE_STANDING_NOT_CURRENT,
    RefusalReason.INADMISSIBLE_EXACT_WORK: Output.REFUSE_INADMISSIBLE_EXACT_WORK,
    RefusalReason.BUDGET_EXHAUSTED: Output.REFUSE_BUDGET_EXHAUSTED,
    RefusalReason.RETRY_PRECONDITIONS_CHANGED: Output.REFUSE_RETRY_PRECONDITIONS_CHANGED,
    RefusalReason.RETRY_PROPOSAL_CHANGED: Output.REFUSE_RETRY_PROPOSAL_CHANGED,
    RefusalReason.SUCCESSOR_PROPOSAL_REUSED: Output.REFUSE_SUCCESSOR_PROPOSAL_REUSED,
    RefusalReason.OCCURRENCE_REUSED: Output.REFUSE_OCCURRENCE_REUSED,
    RefusalReason.UNRESOLVED_ATTEMPT: Output.REFUSE_UNRESOLVED_ATTEMPT,
    RefusalReason.HUMAN_DECISION_REQUIRED: Output.REFUSE_HUMAN_DECISION_REQUIRED,
    RefusalReason.MALFORMED: Output.REFUSE_MALFORMED,
}

SAFE_HALT_SOURCES = {
    ProgramCounter.OBSERVATION_REQUIRED,
    ProgramCounter.PROPOSAL_RECORDED,
    ProgramCounter.STANDING_REQUIRED,
    ProgramCounter.ADMISSIBLE_PENDING_AUTHORIZATION,
    ProgramCounter.RECONCILIATION_REQUIRED,
    ProgramCounter.SETTLED_OBSERVATION_REQUIRED,
}


@dataclass(frozen=True, order=True)
class GovernanceState:
    pc: int = int(ProgramCounter.OBSERVATION_REQUIRED)
    proposal: int = int(ProposalKind.NONE)
    preconditions: int = int(PreconditionBasis.NONE)
    has_prior: int = 0
    prior_proposal: int = int(ProposalKind.NONE)
    prior_preconditions: int = int(PreconditionBasis.NONE)
    retries_used: int = 0
    probes_used: int = 0
    escalations_used: int = 0
    standing_lease: int = 0
    settlement_outcome: int = int(SettlementOutcome.NONE)
    halted_unresolved_attempt: int = 0

    def __post_init__(self) -> None:
        pc = ProgramCounter(int(self.pc))
        proposal = ProposalKind(int(self.proposal))
        preconditions = PreconditionBasis(int(self.preconditions))
        has_prior = int(self.has_prior)
        prior_proposal = ProposalKind(int(self.prior_proposal))
        prior_preconditions = PreconditionBasis(int(self.prior_preconditions))
        retries = int(self.retries_used)
        probes = int(self.probes_used)
        escalations = int(self.escalations_used)
        lease = int(self.standing_lease)
        outcome = SettlementOutcome(int(self.settlement_outcome))
        halted_unresolved = int(self.halted_unresolved_attempt)
        if has_prior not in (0, 1) or halted_unresolved not in (0, 1):
            raise ValueError("boolean fields must be 0 or 1")
        if not (0 <= retries <= RETRY_LIMIT and 0 <= probes <= PROBE_LIMIT and 0 <= escalations <= ESCALATION_LIMIT):
            raise ValueError("budget counters out of range")
        if not (0 <= lease <= STANDING_LEASE_MAX):
            raise ValueError("standing lease out of range")
        if has_prior == 0 and (prior_proposal != ProposalKind.NONE or prior_preconditions != PreconditionBasis.NONE):
            raise ValueError("prior fields require has_prior")
        if has_prior == 1 and (prior_proposal == ProposalKind.NONE or prior_preconditions == PreconditionBasis.NONE):
            raise ValueError("has_prior requires prior proposal and preconditions")
        proposal_states = {
            ProgramCounter.PROPOSAL_RECORDED,
            ProgramCounter.STANDING_REQUIRED,
            ProgramCounter.ADMISSIBLE_PENDING_AUTHORIZATION,
            ProgramCounter.AUTHORIZATION_CONSUMED,
            ProgramCounter.DISPATCHED,
            ProgramCounter.RECONCILIATION_REQUIRED,
            ProgramCounter.SETTLED_OBSERVATION_REQUIRED,
        }
        if pc in proposal_states and (proposal == ProposalKind.NONE or preconditions == PreconditionBasis.NONE):
            raise ValueError("proposal-bearing state needs proposal and preconditions")
        if pc not in proposal_states and pc not in (ProgramCounter.HALTED, ProgramCounter.COMPLETED):
            if proposal != ProposalKind.NONE or preconditions != PreconditionBasis.NONE:
                raise ValueError("state cannot retain current proposal")
        if pc == ProgramCounter.ADMISSIBLE_PENDING_AUTHORIZATION:
            if lease == 0:
                raise ValueError("admissible pending authorization requires a live lease")
        elif lease != 0:
            raise ValueError("standing lease belongs only to admissible pending authorization")
        if pc == ProgramCounter.SETTLED_OBSERVATION_REQUIRED:
            if outcome == SettlementOutcome.NONE:
                raise ValueError("settled state requires a known outcome")
        elif outcome != SettlementOutcome.NONE:
            raise ValueError("settlement outcome belongs only to settled state")
        if pc != ProgramCounter.HALTED and halted_unresolved:
            raise ValueError("unresolved halted flag belongs only to halted state")
        object.__setattr__(self, "pc", int(pc))
        object.__setattr__(self, "proposal", int(proposal))
        object.__setattr__(self, "preconditions", int(preconditions))
        object.__setattr__(self, "has_prior", has_prior)
        object.__setattr__(self, "prior_proposal", int(prior_proposal))
        object.__setattr__(self, "prior_preconditions", int(prior_preconditions))
        object.__setattr__(self, "retries_used", retries)
        object.__setattr__(self, "probes_used", probes)
        object.__setattr__(self, "escalations_used", escalations)
        object.__setattr__(self, "standing_lease", lease)
        object.__setattr__(self, "settlement_outcome", int(outcome))
        object.__setattr__(self, "halted_unresolved_attempt", halted_unresolved)

    @property
    def pc_enum(self) -> ProgramCounter:
        return ProgramCounter(self.pc)

    @property
    def proposal_enum(self) -> ProposalKind:
        return ProposalKind(self.proposal)

    @property
    def preconditions_enum(self) -> PreconditionBasis:
        return PreconditionBasis(self.preconditions)

    @property
    def prior_proposal_enum(self) -> ProposalKind:
        return ProposalKind(self.prior_proposal)

    @property
    def prior_preconditions_enum(self) -> PreconditionBasis:
        return PreconditionBasis(self.prior_preconditions)

    @property
    def settlement_outcome_enum(self) -> SettlementOutcome:
        return SettlementOutcome(self.settlement_outcome)

    def to_json(self) -> dict[str, Any]:
        return {
            "pc": PROGRAM_COUNTER_NAMES[self.pc_enum],
            "pc_id": self.pc,
            "proposal": PROPOSAL_NAMES[self.proposal_enum],
            "proposal_id": self.proposal,
            "preconditions": PRECONDITION_NAMES[self.preconditions_enum],
            "preconditions_id": self.preconditions,
            "has_prior": bool(self.has_prior),
            "prior_proposal": PROPOSAL_NAMES[self.prior_proposal_enum],
            "prior_proposal_id": self.prior_proposal,
            "prior_preconditions": PRECONDITION_NAMES[self.prior_preconditions_enum],
            "prior_preconditions_id": self.prior_preconditions,
            "retries_used": self.retries_used,
            "retry_limit": RETRY_LIMIT,
            "probes_used": self.probes_used,
            "probe_limit": PROBE_LIMIT,
            "escalations_used": self.escalations_used,
            "escalation_limit": ESCALATION_LIMIT,
            "standing_lease": self.standing_lease,
            "settlement_outcome": SETTLEMENT_OUTCOME_NAMES[self.settlement_outcome_enum],
            "settlement_outcome_id": self.settlement_outcome,
            "halted_unresolved_attempt": bool(self.halted_unresolved_attempt),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "GovernanceState":
        return cls(
            pc=int(payload.get("pc_id", ProgramCounter[payload["pc"]].value)),
            proposal=int(payload.get("proposal_id", ProposalKind[payload["proposal"]].value)),
            preconditions=int(payload.get("preconditions_id", PreconditionBasis[payload["preconditions"]].value)),
            has_prior=int(payload["has_prior"]),
            prior_proposal=int(payload.get("prior_proposal_id", ProposalKind[payload["prior_proposal"]].value)),
            prior_preconditions=int(payload.get("prior_preconditions_id", PreconditionBasis[payload["prior_preconditions"]].value)),
            retries_used=int(payload["retries_used"]),
            probes_used=int(payload["probes_used"]),
            escalations_used=int(payload["escalations_used"]),
            standing_lease=int(payload["standing_lease"]),
            settlement_outcome=int(payload.get("settlement_outcome_id", SettlementOutcome[payload["settlement_outcome"]].value)),
            halted_unresolved_attempt=int(payload["halted_unresolved_attempt"]),
        )


@dataclass(frozen=True)
class TransitionResult:
    next_state: GovernanceState
    output: int
    refusal_reason: int | None = None
    admitted_action: int | None = None

    @property
    def output_enum(self) -> Output:
        return Output(self.output)

    def to_json(self) -> dict[str, Any]:
        return {
            "next_state": self.next_state.to_json(),
            "output": OUTPUT_NAMES[self.output_enum],
            "output_id": self.output,
            "refusal_reason": None if self.refusal_reason is None else REFUSAL_NAMES[RefusalReason(self.refusal_reason)],
            "refusal_reason_id": self.refusal_reason,
            "admitted_action": None if self.admitted_action is None else PROPOSAL_NAMES[ProposalKind(self.admitted_action)],
            "admitted_action_id": self.admitted_action,
        }


def initial_state() -> GovernanceState:
    return GovernanceState()


def normalize_event(symbol: int | str | Event) -> Event:
    if isinstance(symbol, str):
        return Event[symbol]
    return Event(int(symbol))


def refusal(state: GovernanceState, reason: RefusalReason) -> TransitionResult:
    return TransitionResult(state, int(REFUSAL_OUTPUTS[reason]), int(reason), None)


def proposal_event(event: Event) -> tuple[str, ProposalKind, PreconditionBasis] | None:
    table = {
        Event.PROPOSE_INITIAL_A_P0: ("initial", ProposalKind.WORK_A, PreconditionBasis.P0),
        Event.PROPOSE_INITIAL_A_P1: ("initial", ProposalKind.WORK_A, PreconditionBasis.P1),
        Event.PROPOSE_INITIAL_B_P0: ("initial", ProposalKind.WORK_B, PreconditionBasis.P0),
        Event.PROPOSE_RETRY_A_P0: ("retry", ProposalKind.WORK_A, PreconditionBasis.P0),
        Event.PROPOSE_RETRY_A_P1: ("retry", ProposalKind.WORK_A, PreconditionBasis.P1),
        Event.PROPOSE_RETRY_B_P0: ("retry", ProposalKind.WORK_B, PreconditionBasis.P0),
        Event.PROPOSE_SUCCESSOR_A_P0: ("successor", ProposalKind.WORK_A, PreconditionBasis.P0),
        Event.PROPOSE_SUCCESSOR_B_P0: ("successor", ProposalKind.WORK_B, PreconditionBasis.P0),
    }
    return table.get(event)


def state_with(
    state: GovernanceState,
    *,
    pc: ProgramCounter | None = None,
    proposal: ProposalKind | None = None,
    preconditions: PreconditionBasis | None = None,
    has_prior: int | None = None,
    prior_proposal: ProposalKind | None = None,
    prior_preconditions: PreconditionBasis | None = None,
    retries_used: int | None = None,
    probes_used: int | None = None,
    escalations_used: int | None = None,
    standing_lease: int | None = None,
    settlement_outcome: SettlementOutcome | None = None,
    halted_unresolved_attempt: int | None = None,
) -> GovernanceState:
    return GovernanceState(
        pc=state.pc if pc is None else pc,
        proposal=state.proposal if proposal is None else proposal,
        preconditions=state.preconditions if preconditions is None else preconditions,
        has_prior=state.has_prior if has_prior is None else has_prior,
        prior_proposal=state.prior_proposal if prior_proposal is None else prior_proposal,
        prior_preconditions=state.prior_preconditions if prior_preconditions is None else prior_preconditions,
        retries_used=state.retries_used if retries_used is None else retries_used,
        probes_used=state.probes_used if probes_used is None else probes_used,
        escalations_used=state.escalations_used if escalations_used is None else escalations_used,
        standing_lease=state.standing_lease if standing_lease is None else standing_lease,
        settlement_outcome=state.settlement_outcome if settlement_outcome is None else settlement_outcome,
        halted_unresolved_attempt=state.halted_unresolved_attempt if halted_unresolved_attempt is None else halted_unresolved_attempt,
    )


def clear_current_work(state: GovernanceState, *, pc: ProgramCounter) -> GovernanceState:
    return state_with(
        state,
        pc=pc,
        proposal=ProposalKind.NONE,
        preconditions=PreconditionBasis.NONE,
        standing_lease=0,
        settlement_outcome=SettlementOutcome.NONE,
        halted_unresolved_attempt=0,
    )


def record_proposal(state: GovernanceState, klass: str, proposal: ProposalKind, preconditions: PreconditionBasis) -> TransitionResult:
    if state.pc_enum != ProgramCounter.OBSERVATION_REQUIRED:
        return refusal(state, RefusalReason.ILLEGAL_TRANSITION)
    if klass == "initial":
        if state.has_prior:
            return refusal(state, RefusalReason.BINDING_MISMATCH)
    elif klass == "retry":
        if not state.has_prior:
            return refusal(state, RefusalReason.BINDING_MISMATCH)
        if state.retries_used >= RETRY_LIMIT:
            return refusal(state, RefusalReason.BUDGET_EXHAUSTED)
        if proposal != state.prior_proposal_enum:
            return refusal(state, RefusalReason.RETRY_PROPOSAL_CHANGED)
        if preconditions != state.prior_preconditions_enum:
            return refusal(state, RefusalReason.RETRY_PRECONDITIONS_CHANGED)
        state = state_with(state, retries_used=state.retries_used + 1)
    elif klass == "successor":
        if not state.has_prior:
            return refusal(state, RefusalReason.BINDING_MISMATCH)
        if proposal == state.prior_proposal_enum:
            return refusal(state, RefusalReason.SUCCESSOR_PROPOSAL_REUSED)
    else:
        raise AssertionError(f"unknown proposal class: {klass}")
    next_state = state_with(
        state,
        pc=ProgramCounter.PROPOSAL_RECORDED,
        proposal=proposal,
        preconditions=preconditions,
        standing_lease=0,
        settlement_outcome=SettlementOutcome.NONE,
        halted_unresolved_attempt=0,
    )
    return TransitionResult(next_state, int(Output.PROPOSAL_RECORDED))


def transition(state: GovernanceState, symbol: int | str | Event) -> TransitionResult:
    event = normalize_event(symbol)

    if event == Event.NOOP:
        return TransitionResult(state, int(Output.NO_OUTPUT))
    if event == Event.CLAIM_AUTHORITY_RECORD:
        return TransitionResult(state, int(Output.CLAIM_IGNORED))
    if event == Event.MALFORMED:
        return refusal(state, RefusalReason.MALFORMED)

    proposed = proposal_event(event)
    if proposed is not None:
        return record_proposal(state, *proposed)
    if event == Event.PROPOSE_STALE_OBSERVATION:
        if state.pc_enum == ProgramCounter.OBSERVATION_REQUIRED:
            return refusal(state, RefusalReason.STALE_OBSERVATION)
        return refusal(state, RefusalReason.ILLEGAL_TRANSITION)
    if event == Event.PROPOSE_CONTRADICTORY_OBSERVATION:
        if state.pc_enum == ProgramCounter.OBSERVATION_REQUIRED:
            return refusal(state, RefusalReason.CONTRADICTION)
        return refusal(state, RefusalReason.ILLEGAL_TRANSITION)

    if event == Event.REQUIRE_STANDING:
        if state.pc_enum != ProgramCounter.PROPOSAL_RECORDED:
            return refusal(state, RefusalReason.ILLEGAL_TRANSITION)
        return TransitionResult(state_with(state, pc=ProgramCounter.STANDING_REQUIRED), int(Output.STANDING_REQUIRED))

    if event in {
        Event.RECORD_ADMISSIBLE_CURRENT,
        Event.RECORD_ADMISSIBLE_ABSENT_STANDING,
        Event.RECORD_ADMISSIBLE_REVOKED_STANDING,
        Event.RECORD_ADMISSIBLE_EXPIRED_STANDING,
        Event.RECORD_ADMISSIBLE_INADMISSIBLE,
        Event.RECORD_ADMISSIBLE_STALE_OBSERVATION,
        Event.RECORD_ADMISSIBLE_CONTRADICTION,
    }:
        if state.pc_enum != ProgramCounter.STANDING_REQUIRED:
            return refusal(state, RefusalReason.ILLEGAL_TRANSITION)
        if event == Event.RECORD_ADMISSIBLE_CURRENT:
            next_state = state_with(
                state,
                pc=ProgramCounter.ADMISSIBLE_PENDING_AUTHORIZATION,
                standing_lease=STANDING_LEASE_MAX,
            )
            return TransitionResult(next_state, int(Output.ADMISSIBLE_RECORDED))
        if event == Event.RECORD_ADMISSIBLE_ABSENT_STANDING:
            return refusal(state, RefusalReason.ABSENT_STANDING)
        if event in (Event.RECORD_ADMISSIBLE_REVOKED_STANDING, Event.RECORD_ADMISSIBLE_EXPIRED_STANDING):
            return refusal(state, RefusalReason.STANDING_NOT_CURRENT)
        if event == Event.RECORD_ADMISSIBLE_INADMISSIBLE:
            return refusal(state, RefusalReason.INADMISSIBLE_EXACT_WORK)
        if event == Event.RECORD_ADMISSIBLE_STALE_OBSERVATION:
            return refusal(state, RefusalReason.STALE_OBSERVATION)
        if event == Event.RECORD_ADMISSIBLE_CONTRADICTION:
            return refusal(state, RefusalReason.CONTRADICTION)

    if event == Event.TICK:
        if state.pc_enum == ProgramCounter.ADMISSIBLE_PENDING_AUTHORIZATION and state.standing_lease > 1:
            return TransitionResult(state_with(state, standing_lease=state.standing_lease - 1), int(Output.TICKED))
        if state.pc_enum == ProgramCounter.ADMISSIBLE_PENDING_AUTHORIZATION and state.standing_lease == 1:
            # The finite model makes expiry visible by returning to standing-required.
            # A later spend must be preceded by a fresh admissibility transition.
            return TransitionResult(state_with(state, pc=ProgramCounter.STANDING_REQUIRED, standing_lease=0), int(Output.TICKED))
        return TransitionResult(state, int(Output.NO_OUTPUT))

    if event in {
        Event.CONSUME_AUTH_CURRENT,
        Event.CONSUME_AUTH_EXPIRED_STANDING,
        Event.CONSUME_AUTH_STALE_OBSERVATION,
        Event.CONSUME_AUTH_INADMISSIBLE,
    }:
        if state.pc_enum != ProgramCounter.ADMISSIBLE_PENDING_AUTHORIZATION:
            return refusal(state, RefusalReason.ILLEGAL_TRANSITION)
        if event == Event.CONSUME_AUTH_CURRENT:
            if state.standing_lease <= 0:
                return refusal(state, RefusalReason.STANDING_NOT_CURRENT)
            output = Output.AUTHORIZATION_CONSUMED_A if state.proposal_enum == ProposalKind.WORK_A else Output.AUTHORIZATION_CONSUMED_B
            next_state = state_with(state, pc=ProgramCounter.AUTHORIZATION_CONSUMED, standing_lease=0)
            return TransitionResult(next_state, int(output), None, int(state.proposal_enum))
        if event == Event.CONSUME_AUTH_EXPIRED_STANDING:
            return refusal(state, RefusalReason.STANDING_NOT_CURRENT)
        if event == Event.CONSUME_AUTH_STALE_OBSERVATION:
            return refusal(state, RefusalReason.STALE_OBSERVATION)
        if event == Event.CONSUME_AUTH_INADMISSIBLE:
            return refusal(state, RefusalReason.INADMISSIBLE_EXACT_WORK)

    if event == Event.ACCEPT_DOCKET_CUSTODY:
        if state.pc_enum != ProgramCounter.AUTHORIZATION_CONSUMED:
            return refusal(state, RefusalReason.ILLEGAL_TRANSITION)
        return TransitionResult(state_with(state, pc=ProgramCounter.DISPATCHED), int(Output.DOCKET_CUSTODY_ACCEPTED))

    if event in (Event.RECORD_SETTLEMENT_SUCCESS, Event.RECORD_SETTLEMENT_FAILURE):
        if state.pc_enum != ProgramCounter.DISPATCHED:
            return refusal(state, RefusalReason.ILLEGAL_TRANSITION)
        outcome = SettlementOutcome.SUCCESS if event == Event.RECORD_SETTLEMENT_SUCCESS else SettlementOutcome.FAILURE
        output = Output.SETTLED_SUCCESS if outcome == SettlementOutcome.SUCCESS else Output.SETTLED_FAILURE
        next_state = state_with(state, pc=ProgramCounter.SETTLED_OBSERVATION_REQUIRED, settlement_outcome=outcome)
        return TransitionResult(next_state, int(output))

    if event == Event.REQUIRE_RECONCILIATION:
        if state.pc_enum != ProgramCounter.DISPATCHED:
            return refusal(state, RefusalReason.ILLEGAL_TRANSITION)
        return TransitionResult(state_with(state, pc=ProgramCounter.RECONCILIATION_REQUIRED), int(Output.RECONCILIATION_REQUIRED))

    if event in (Event.RECORD_RECONCILED_SUCCESS, Event.RECORD_RECONCILED_FAILURE):
        if state.pc_enum != ProgramCounter.RECONCILIATION_REQUIRED:
            return refusal(state, RefusalReason.ILLEGAL_TRANSITION)
        outcome = SettlementOutcome.SUCCESS if event == Event.RECORD_RECONCILED_SUCCESS else SettlementOutcome.FAILURE
        output = Output.SETTLED_SUCCESS if outcome == SettlementOutcome.SUCCESS else Output.SETTLED_FAILURE
        next_state = state_with(state, pc=ProgramCounter.SETTLED_OBSERVATION_REQUIRED, settlement_outcome=outcome)
        return TransitionResult(next_state, int(output))

    if event == Event.OPEN_CONTINUATION:
        if state.pc_enum != ProgramCounter.SETTLED_OBSERVATION_REQUIRED:
            return refusal(state, RefusalReason.ILLEGAL_TRANSITION)
        next_state = GovernanceState(
            pc=ProgramCounter.OBSERVATION_REQUIRED,
            proposal=ProposalKind.NONE,
            preconditions=PreconditionBasis.NONE,
            has_prior=1,
            prior_proposal=state.proposal,
            prior_preconditions=state.preconditions,
            retries_used=state.retries_used,
            probes_used=state.probes_used,
            escalations_used=state.escalations_used,
        )
        return TransitionResult(next_state, int(Output.CONTINUATION_OPENED))

    if event == Event.NOTE_PROBE:
        if state.pc_enum not in (ProgramCounter.OBSERVATION_REQUIRED, ProgramCounter.SETTLED_OBSERVATION_REQUIRED):
            return refusal(state, RefusalReason.ILLEGAL_TRANSITION)
        if state.probes_used >= PROBE_LIMIT:
            return refusal(state, RefusalReason.BUDGET_EXHAUSTED)
        return TransitionResult(state_with(state, probes_used=state.probes_used + 1), int(Output.PROBE_NOTED))

    if event in (Event.HALT, Event.ESCALATE):
        if state.pc_enum not in SAFE_HALT_SOURCES:
            return refusal(state, RefusalReason.ILLEGAL_TRANSITION)
        escalations = state.escalations_used
        output = Output.HALTED
        if event == Event.ESCALATE:
            if state.escalations_used >= ESCALATION_LIMIT:
                return refusal(state, RefusalReason.BUDGET_EXHAUSTED)
            escalations += 1
            output = Output.ESCALATED
        next_state = state_with(
            state,
            pc=ProgramCounter.HALTED,
            escalations_used=escalations,
            standing_lease=0,
            settlement_outcome=SettlementOutcome.NONE,
            halted_unresolved_attempt=int(state.pc_enum == ProgramCounter.RECONCILIATION_REQUIRED),
        )
        return TransitionResult(next_state, int(output))

    if event == Event.HUMAN_RETURN:
        if state.pc_enum != ProgramCounter.HALTED:
            return refusal(state, RefusalReason.ILLEGAL_TRANSITION)
        if state.halted_unresolved_attempt:
            return refusal(state, RefusalReason.UNRESOLVED_ATTEMPT)
        next_state = GovernanceState(
            pc=ProgramCounter.OBSERVATION_REQUIRED,
            has_prior=state.has_prior,
            prior_proposal=state.prior_proposal,
            prior_preconditions=state.prior_preconditions,
            retries_used=state.retries_used,
            probes_used=state.probes_used,
            escalations_used=state.escalations_used,
        )
        return TransitionResult(next_state, int(Output.HUMAN_RETURNED))

    if event in (Event.HUMAN_TERMINATE, Event.COMPLETE):
        if event == Event.COMPLETE:
            if state.pc_enum != ProgramCounter.OBSERVATION_REQUIRED:
                return refusal(state, RefusalReason.ILLEGAL_TRANSITION)
            return TransitionResult(clear_current_work(state, pc=ProgramCounter.COMPLETED), int(Output.COMPLETED))
        if state.pc_enum != ProgramCounter.HALTED:
            return refusal(state, RefusalReason.ILLEGAL_TRANSITION)
        if state.halted_unresolved_attempt:
            return refusal(state, RefusalReason.UNRESOLVED_ATTEMPT)
        return TransitionResult(clear_current_work(state, pc=ProgramCounter.COMPLETED), int(Output.COMPLETED))

    raise AssertionError(f"unhandled governance event: {event}")


def run_kernel(
    events: Sequence[int | str | Event],
    start: GovernanceState | None = None,
    include_initial: bool = False,
) -> tuple[list[GovernanceState], list[TransitionResult]]:
    state = start or initial_state()
    states: list[GovernanceState] = [state] if include_initial else []
    results: list[TransitionResult] = []
    for event in events:
        result = transition(state, event)
        results.append(result)
        state = result.next_state
        states.append(state)
    return states, results


def output_sequence(events: Sequence[int | str | Event], start: GovernanceState | None = None) -> list[int]:
    return [result.output for result in run_kernel(events, start=start)[1]]


def state_sort_key(state: GovernanceState) -> tuple[Any, ...]:
    return (
        state.pc,
        state.proposal,
        state.preconditions,
        state.has_prior,
        state.prior_proposal,
        state.prior_preconditions,
        state.retries_used,
        state.probes_used,
        state.escalations_used,
        state.standing_lease,
        state.settlement_outcome,
        state.halted_unresolved_attempt,
    )


def enumerate_reachable_states() -> tuple[list[GovernanceState], dict[GovernanceState, tuple[int, ...]]]:
    start = initial_state()
    seen: dict[GovernanceState, tuple[int, ...]] = {start: ()}
    queue: deque[GovernanceState] = deque([start])
    while queue:
        state = queue.popleft()
        history = seen[state]
        for event in EVENTS:
            next_state = transition(state, event).next_state
            if next_state not in seen:
                seen[next_state] = (*history, int(event))
                queue.append(next_state)
    states = sorted(seen, key=state_sort_key)
    return states, seen


def syntactic_states() -> list[GovernanceState]:
    states: list[GovernanceState] = []
    for values in product(
        [int(value) for value in ProgramCounter],
        [int(value) for value in ProposalKind],
        [int(value) for value in PreconditionBasis],
        (0, 1),
        [int(value) for value in ProposalKind],
        [int(value) for value in PreconditionBasis],
        range(RETRY_LIMIT + 1),
        range(PROBE_LIMIT + 1),
        range(ESCALATION_LIMIT + 1),
        range(STANDING_LEASE_MAX + 1),
        [int(value) for value in SettlementOutcome],
        (0, 1),
    ):
        try:
            states.append(GovernanceState(*values))
        except ValueError:
            continue
    return sorted(states, key=state_sort_key)


def state_id_maps() -> tuple[list[GovernanceState], dict[GovernanceState, int]]:
    states, _ = enumerate_reachable_states()
    return states, {state: i for i, state in enumerate(states)}


def reachable_graph() -> dict[str, Any]:
    states, canonical = enumerate_reachable_states()
    syntactic = syntactic_states()
    state_to_id = {state: i for i, state in enumerate(states)}
    transitions = []
    admitted = 0
    refusals = 0
    for state in states:
        for event in EVENTS:
            result = transition(state, event)
            if result.admitted_action is not None:
                admitted += 1
            if result.refusal_reason is not None:
                refusals += 1
            transitions.append(
                {
                    "from": state_to_id[state],
                    "event": EVENT_NAMES[Event(event)],
                    "event_id": int(event),
                    "to": state_to_id[result.next_state],
                    "output": OUTPUT_NAMES[result.output_enum],
                    "output_id": result.output,
                    "refusal_reason": None if result.refusal_reason is None else REFUSAL_NAMES[RefusalReason(result.refusal_reason)],
                    "admitted_action": None if result.admitted_action is None else PROPOSAL_NAMES[ProposalKind(result.admitted_action)],
                }
            )
    return {
        "source_model": "ag-ng finite abstraction",
        "syntactic_states": len(syntactic),
        "reachable_states": len(states),
        "event_alphabet_size": len(EVENTS),
        "reachable_transitions": len(transitions),
        "admitted_transitions": admitted,
        "refusal_transitions": refusals,
        "states": [state.to_json() for state in states],
        "canonical_histories": {str(state_to_id[state]): list(history) for state, history in canonical.items()},
        "transitions": transitions,
    }


def transition_label(_before: GovernanceState, result: TransitionResult) -> str:
    if result.admitted_action is not None:
        return f"AG_ISSUE_{ProposalKind(result.admitted_action).name}"
    if result.refusal_reason is not None:
        return f"REFUSE_{RefusalReason(result.refusal_reason).name}"
    return result.output_enum.name


def invariant_violations(
    before: GovernanceState,
    event: int | str | Event,
    result: TransitionResult,
) -> list[str]:
    event = normalize_event(event)
    after = result.next_state
    violations: list[str] = []
    if before.retries_used > RETRY_LIMIT or before.probes_used > PROBE_LIMIT or before.escalations_used > ESCALATION_LIMIT:
        violations.append("budget_counter_out_of_range")
    if after.retries_used > RETRY_LIMIT or after.probes_used > PROBE_LIMIT or after.escalations_used > ESCALATION_LIMIT:
        violations.append("budget_counter_out_of_range_after")
    if result.admitted_action is not None:
        if before.pc_enum != ProgramCounter.ADMISSIBLE_PENDING_AUTHORIZATION:
            violations.append("authority_burn_without_admissible_pending")
        if before.standing_lease <= 0:
            violations.append("authority_burn_without_live_standing")
        if after.pc_enum != ProgramCounter.AUTHORIZATION_CONSUMED:
            violations.append("authority_burn_did_not_enter_consumed_state")
    if before.pc_enum in (ProgramCounter.AUTHORIZATION_CONSUMED, ProgramCounter.DISPATCHED, ProgramCounter.RECONCILIATION_REQUIRED):
        if result.admitted_action is not None:
            violations.append("second_authority_burn")
    if before.pc_enum == ProgramCounter.DISPATCHED and event == Event.REQUIRE_RECONCILIATION:
        if after.pc_enum != ProgramCounter.RECONCILIATION_REQUIRED:
            violations.append("ambiguous_attempt_did_not_require_reconciliation")
    if before.pc_enum == ProgramCounter.RECONCILIATION_REQUIRED:
        if event == Event.OPEN_CONTINUATION and result.refusal_reason != int(RefusalReason.ILLEGAL_TRANSITION):
            violations.append("blind_retry_after_ambiguity")
        if event == Event.ACCEPT_DOCKET_CUSTODY and result.refusal_reason != int(RefusalReason.ILLEGAL_TRANSITION):
            violations.append("second_docket_attempt_after_ambiguity")
    if before.pc_enum == ProgramCounter.HALTED and before.halted_unresolved_attempt:
        if event in (Event.HUMAN_RETURN, Event.HUMAN_TERMINATE) and result.refusal_reason != int(RefusalReason.UNRESOLVED_ATTEMPT):
            violations.append("unresolved_attempt_human_resume")
    if event == Event.CLAIM_AUTHORITY_RECORD and before != after:
        violations.append("claim_created_authority")
    if event == Event.TICK and before.pc_enum == ProgramCounter.ADMISSIBLE_PENDING_AUTHORIZATION:
        if after.pc_enum == ProgramCounter.AUTHORIZATION_CONSUMED:
            violations.append("tick_consumed_authority")
        if after.standing_lease >= before.standing_lease:
            violations.append("tick_did_not_reduce_lease")
    if result.refusal_reason is not None and after != before:
        violations.append("refusal_changed_state")
    if before.pc_enum == ProgramCounter.COMPLETED and after != before:
        violations.append("completed_not_terminal")
    return violations


def collect_equivalent_histories(
    max_per_state: int = 8,
    seed: int = 17,
    random_sequences: int = 6000,
    random_length: int = 96,
) -> dict[GovernanceState, list[tuple[int, ...]]]:
    states, canonical = enumerate_reachable_states()
    groups: dict[GovernanceState, list[tuple[int, ...]]] = {state: [history] for state, history in canonical.items()}
    neutral_patterns = [
        (Event.NOOP,),
        (Event.CLAIM_AUTHORITY_RECORD,),
        (Event.MALFORMED,),
        (Event.NOOP, Event.CLAIM_AUTHORITY_RECORD, Event.MALFORMED),
    ]
    for state in states:
        base = canonical[state]
        for repeats in (1, 3, 11, 37):
            for pattern in neutral_patterns:
                if len(groups[state]) >= max_per_state:
                    break
                suffix = tuple(int(x) for x in pattern) * repeats
                candidate = (*base, *suffix)
                reached = run_kernel(candidate)[0][-1]
                if reached == state and candidate not in groups[state]:
                    groups[state].append(candidate)
            if len(groups[state]) >= max_per_state:
                break
    rng = Random(seed)
    weighted_events = (
        [Event.NOOP] * 10
        + [Event.CLAIM_AUTHORITY_RECORD] * 8
        + [Event.PROPOSE_INITIAL_A_P0, Event.PROPOSE_INITIAL_B_P0, Event.PROPOSE_RETRY_A_P0, Event.PROPOSE_SUCCESSOR_B_P0]
        + [Event.REQUIRE_STANDING, Event.RECORD_ADMISSIBLE_CURRENT, Event.CONSUME_AUTH_CURRENT]
        + [Event.ACCEPT_DOCKET_CUSTODY, Event.RECORD_SETTLEMENT_SUCCESS, Event.REQUIRE_RECONCILIATION, Event.RECORD_RECONCILED_FAILURE]
        + [Event.OPEN_CONTINUATION, Event.NOTE_PROBE, Event.HALT, Event.ESCALATE]
        + [Event.TICK, Event.MALFORMED]
        + [Event.RECORD_ADMISSIBLE_EXPIRED_STANDING, Event.CONSUME_AUTH_EXPIRED_STANDING]
    )
    for _ in range(random_sequences):
        history = tuple(int(rng.choice(weighted_events)) for _ in range(random_length))
        reached = run_kernel(history)[0][-1]
        bucket = groups[reached]
        if len(bucket) < max_per_state and history not in bucket:
            bucket.append(history)
    return groups


def adversarial_sequences() -> dict[str, list[int]]:
    return {
        "authority_claim_is_not_authority": [int(Event.CLAIM_AUTHORITY_RECORD), int(Event.ACCEPT_DOCKET_CUSTODY), int(Event.CONSUME_AUTH_CURRENT)],
        "normal_success_path": [
            int(Event.PROPOSE_INITIAL_A_P0),
            int(Event.REQUIRE_STANDING),
            int(Event.RECORD_ADMISSIBLE_CURRENT),
            int(Event.CONSUME_AUTH_CURRENT),
            int(Event.ACCEPT_DOCKET_CUSTODY),
            int(Event.RECORD_SETTLEMENT_SUCCESS),
        ],
        "expired_before_burn": [
            int(Event.PROPOSE_INITIAL_A_P0),
            int(Event.REQUIRE_STANDING),
            int(Event.RECORD_ADMISSIBLE_CURRENT),
            int(Event.TICK),
            int(Event.TICK),
            int(Event.CONSUME_AUTH_CURRENT),
        ],
        "standing_revoked_before_burn": [
            int(Event.PROPOSE_INITIAL_A_P0),
            int(Event.REQUIRE_STANDING),
            int(Event.RECORD_ADMISSIBLE_CURRENT),
            int(Event.CONSUME_AUTH_EXPIRED_STANDING),
        ],
        "one_use_burn": [
            int(Event.PROPOSE_INITIAL_A_P0),
            int(Event.REQUIRE_STANDING),
            int(Event.RECORD_ADMISSIBLE_CURRENT),
            int(Event.CONSUME_AUTH_CURRENT),
            int(Event.CONSUME_AUTH_CURRENT),
            int(Event.ACCEPT_DOCKET_CUSTODY),
            int(Event.ACCEPT_DOCKET_CUSTODY),
        ],
        "ambiguous_no_blind_retry": [
            int(Event.PROPOSE_INITIAL_A_P0),
            int(Event.REQUIRE_STANDING),
            int(Event.RECORD_ADMISSIBLE_CURRENT),
            int(Event.CONSUME_AUTH_CURRENT),
            int(Event.ACCEPT_DOCKET_CUSTODY),
            int(Event.REQUIRE_RECONCILIATION),
            int(Event.OPEN_CONTINUATION),
            int(Event.ACCEPT_DOCKET_CUSTODY),
            int(Event.RECORD_RECONCILED_FAILURE),
            int(Event.OPEN_CONTINUATION),
        ],
        "retry_same_preconditions_only": [
            int(Event.PROPOSE_INITIAL_A_P0),
            int(Event.REQUIRE_STANDING),
            int(Event.RECORD_ADMISSIBLE_CURRENT),
            int(Event.CONSUME_AUTH_CURRENT),
            int(Event.ACCEPT_DOCKET_CUSTODY),
            int(Event.RECORD_SETTLEMENT_FAILURE),
            int(Event.OPEN_CONTINUATION),
            int(Event.PROPOSE_RETRY_A_P1),
            int(Event.PROPOSE_RETRY_A_P0),
        ],
        "successor_cannot_reuse_proposal": [
            int(Event.PROPOSE_INITIAL_A_P0),
            int(Event.REQUIRE_STANDING),
            int(Event.RECORD_ADMISSIBLE_CURRENT),
            int(Event.CONSUME_AUTH_CURRENT),
            int(Event.ACCEPT_DOCKET_CUSTODY),
            int(Event.RECORD_SETTLEMENT_SUCCESS),
            int(Event.OPEN_CONTINUATION),
            int(Event.PROPOSE_SUCCESSOR_A_P0),
            int(Event.PROPOSE_SUCCESSOR_B_P0),
        ],
        "probe_budget_exhaustion": [int(Event.NOTE_PROBE), int(Event.NOTE_PROBE)],
        "halted_unresolved_blocks_human_return": [
            int(Event.PROPOSE_INITIAL_A_P0),
            int(Event.REQUIRE_STANDING),
            int(Event.RECORD_ADMISSIBLE_CURRENT),
            int(Event.CONSUME_AUTH_CURRENT),
            int(Event.ACCEPT_DOCKET_CUSTODY),
            int(Event.REQUIRE_RECONCILIATION),
            int(Event.HALT),
            int(Event.HUMAN_RETURN),
        ],
        "long_claim_spam": [int(Event.CLAIM_AUTHORITY_RECORD), int(Event.NOOP), int(Event.MALFORMED), int(Event.ACCEPT_DOCKET_CUSTODY)] * 512,
    }
