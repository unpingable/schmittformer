from __future__ import annotations

from typing import Iterable, Sequence

from .fixed_state import (
    Authority,
    GovernanceEvent,
    GovernanceOutput,
    GovernanceState,
    Occurrence,
    Settlement,
    TransitionResult,
    configured_state,
    initial_state,
)


def transition(state: GovernanceState, event: int | GovernanceEvent) -> TransitionResult:
    event = GovernanceEvent(int(event))
    authority = Authority(int(state.authority))
    lease = int(state.lease_remaining)
    budget = int(state.action_budget)
    occurrence = Occurrence(int(state.occurrence))
    settlement = Settlement(int(state.settlement))
    output = GovernanceOutput.NO_ACTION

    if event == GovernanceEvent.TICK:
        lease = max(lease - 1, 0)
    elif event == GovernanceEvent.GRANT_AUTHORITY:
        authority = Authority.VALID
    elif event == GovernanceEvent.REVOKE_AUTHORITY:
        authority = Authority.INVALID
    elif event == GovernanceEvent.RENEW_LEASE_MAX:
        lease = 0xFFFF
    elif event == GovernanceEvent.RENEW_LEASE_ONE:
        lease = 1
    elif event == GovernanceEvent.RESET_BUDGET_MAX:
        budget = 0xFF
    elif event == GovernanceEvent.RESET_BUDGET_ONE:
        budget = 1
    elif event == GovernanceEvent.PROPOSE_ACTION:
        if occurrence == Occurrence.AMBIGUOUS:
            output = GovernanceOutput.REFUSE_AMBIGUOUS
        elif occurrence == Occurrence.IN_FLIGHT:
            output = GovernanceOutput.REFUSE_IN_FLIGHT
        elif authority != Authority.VALID:
            output = GovernanceOutput.REFUSE_NO_AUTHORITY
        elif lease <= 0:
            output = GovernanceOutput.REFUSE_EXPIRED
        elif budget <= 0:
            output = GovernanceOutput.REFUSE_BUDGET
        else:
            output = GovernanceOutput.ADMIT_ACTION
            budget = max(budget - 1, 0)
            occurrence = Occurrence.IN_FLIGHT
            settlement = Settlement.NONE
    elif event == GovernanceEvent.RESULT_SUCCESS:
        if occurrence == Occurrence.IN_FLIGHT:
            occurrence = Occurrence.IDLE
            settlement = Settlement.SUCCESS
    elif event == GovernanceEvent.RESULT_FAILURE:
        if occurrence == Occurrence.IN_FLIGHT:
            occurrence = Occurrence.IDLE
            settlement = Settlement.FAILURE
    elif event == GovernanceEvent.RESULT_AMBIGUOUS:
        if occurrence == Occurrence.IN_FLIGHT:
            occurrence = Occurrence.AMBIGUOUS
            settlement = Settlement.NONE
    elif event == GovernanceEvent.SETTLE_SUCCESS:
        if occurrence == Occurrence.AMBIGUOUS:
            occurrence = Occurrence.IDLE
            settlement = Settlement.SUCCESS
    elif event == GovernanceEvent.SETTLE_FAILURE:
        if occurrence == Occurrence.AMBIGUOUS:
            occurrence = Occurrence.IDLE
            settlement = Settlement.FAILURE
    elif event == GovernanceEvent.NOOP:
        pass
    else:
        raise AssertionError(event)

    return TransitionResult(
        GovernanceState(int(authority), lease, budget, int(occurrence), int(settlement)),
        int(output),
    )


def run_reference(
    events: Sequence[int | GovernanceEvent],
    start: GovernanceState | None = None,
) -> tuple[list[GovernanceState], list[int]]:
    state = start or initial_state()
    states: list[GovernanceState] = []
    outputs: list[int] = []
    for event in events:
        result = transition(state, event)
        state = result.next_state
        states.append(state)
        outputs.append(result.output)
    return states, outputs


def invariant_violations(before: GovernanceState, event: int | GovernanceEvent, result: TransitionResult) -> list[str]:
    event = GovernanceEvent(int(event))
    after = result.next_state
    violations: list[str] = []
    if not (0 <= after.lease_remaining <= 0xFFFF):
        violations.append("lease_range")
    if not (0 <= after.action_budget <= 0xFF):
        violations.append("budget_range")
    if before.lease_remaining == 0 and event == GovernanceEvent.TICK and after.lease_remaining != 0:
        violations.append("lease_underflow")
    if event == GovernanceEvent.PROPOSE_ACTION and result.output == int(GovernanceOutput.ADMIT_ACTION):
        if before.authority != int(Authority.VALID):
            violations.append("authority_bypass")
        if before.lease_remaining <= 0:
            violations.append("lease_bypass")
        if before.action_budget <= 0:
            violations.append("budget_bypass")
        if before.action_budget - after.action_budget != 1:
            violations.append("budget_not_consumed")
    if before.action_budget == 0 and event == GovernanceEvent.PROPOSE_ACTION and result.output == int(GovernanceOutput.ADMIT_ACTION):
        violations.append("zero_budget_admit")
    if before.authority == int(Authority.INVALID) and event == GovernanceEvent.PROPOSE_ACTION and result.output == int(GovernanceOutput.ADMIT_ACTION):
        violations.append("invalid_authority_admit")
    if before.occurrence == int(Occurrence.AMBIGUOUS) and event == GovernanceEvent.PROPOSE_ACTION and result.output == int(GovernanceOutput.ADMIT_ACTION):
        violations.append("blind_retry")
    if before.occurrence == int(Occurrence.AMBIGUOUS) and event == GovernanceEvent.PROPOSE_ACTION and result.output != int(GovernanceOutput.REFUSE_AMBIGUOUS):
        violations.append("ambiguous_refusal_precedence")
    return violations


def scenario_traces() -> dict[str, list[GovernanceEvent]]:
    return {
        "admit_then_ambiguous_then_settle": [
            GovernanceEvent.GRANT_AUTHORITY,
            GovernanceEvent.RENEW_LEASE_ONE,
            GovernanceEvent.RESET_BUDGET_ONE,
            GovernanceEvent.PROPOSE_ACTION,
            GovernanceEvent.TICK,
            GovernanceEvent.RESULT_AMBIGUOUS,
            GovernanceEvent.PROPOSE_ACTION,
            GovernanceEvent.SETTLE_FAILURE,
            GovernanceEvent.PROPOSE_ACTION,
        ],
        "lease_ordering_propose_then_tick": [
            GovernanceEvent.GRANT_AUTHORITY,
            GovernanceEvent.RENEW_LEASE_ONE,
            GovernanceEvent.RESET_BUDGET_ONE,
            GovernanceEvent.PROPOSE_ACTION,
            GovernanceEvent.TICK,
        ],
        "lease_ordering_tick_then_propose": [
            GovernanceEvent.GRANT_AUTHORITY,
            GovernanceEvent.RENEW_LEASE_ONE,
            GovernanceEvent.RESET_BUDGET_ONE,
            GovernanceEvent.TICK,
            GovernanceEvent.PROPOSE_ACTION,
        ],
        "revoke_while_in_flight_then_result": [
            GovernanceEvent.GRANT_AUTHORITY,
            GovernanceEvent.RENEW_LEASE_MAX,
            GovernanceEvent.RESET_BUDGET_ONE,
            GovernanceEvent.PROPOSE_ACTION,
            GovernanceEvent.REVOKE_AUTHORITY,
            GovernanceEvent.RESULT_SUCCESS,
            GovernanceEvent.PROPOSE_ACTION,
        ],
        "proposal_spam_after_expiry": [
            GovernanceEvent.GRANT_AUTHORITY,
            GovernanceEvent.RENEW_LEASE_ONE,
            GovernanceEvent.RESET_BUDGET_MAX,
            GovernanceEvent.TICK,
            *([GovernanceEvent.PROPOSE_ACTION] * 8),
        ],
    }


def final_state_after(events: Iterable[int | GovernanceEvent], start: GovernanceState | None = None) -> GovernanceState:
    state = start or initial_state()
    for event in events:
        state = transition(state, event).next_state
    return state


def budget_exhaustion_trace(initial_budget: int) -> tuple[list[GovernanceEvent], GovernanceState]:
    state = configured_state(lease=0xFFFF, budget=initial_budget, authority=Authority.VALID)
    events: list[GovernanceEvent] = []
    for _ in range(initial_budget + 2):
        events.extend([GovernanceEvent.PROPOSE_ACTION, GovernanceEvent.RESULT_SUCCESS])
    return events, state
