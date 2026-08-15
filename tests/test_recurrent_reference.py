from src.fixed_state import Authority, GovernanceEvent, GovernanceOutput, Occurrence, Settlement, GovernanceState, configured_state
from src.recurrent_reference import transition


def test_reference_admission_consumes_budget_and_enters_inflight() -> None:
    state = configured_state(lease=1, budget=1, authority=Authority.VALID)
    result = transition(state, GovernanceEvent.PROPOSE_ACTION)
    assert result.output == int(GovernanceOutput.ADMIT_ACTION)
    assert result.next_state.action_budget == 0
    assert result.next_state.occurrence == int(Occurrence.IN_FLIGHT)
    assert result.next_state.settlement == int(Settlement.NONE)


def test_reference_saturating_tick() -> None:
    state = configured_state(lease=0, budget=1, authority=Authority.VALID)
    assert transition(state, GovernanceEvent.TICK).next_state.lease_remaining == 0


def test_reference_ambiguous_blocks_retry_until_settlement() -> None:
    state = GovernanceState(int(Authority.VALID), 10, 1, int(Occurrence.AMBIGUOUS), int(Settlement.NONE))
    refused = transition(state, GovernanceEvent.PROPOSE_ACTION)
    assert refused.output == int(GovernanceOutput.REFUSE_AMBIGUOUS)
    settled = transition(state, GovernanceEvent.SETTLE_FAILURE)
    assert settled.next_state.occurrence == int(Occurrence.IDLE)
    assert settled.next_state.settlement == int(Settlement.FAILURE)
