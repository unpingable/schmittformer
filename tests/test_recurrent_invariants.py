from src.fixed_state import Authority, GovernanceEvent, GovernanceOutput, Occurrence, Settlement, GovernanceState
from src.recurrent_compiled import compare_compiled_reference


def test_recurrent_policy_invariants_on_selected_states() -> None:
    states = [
        GovernanceState(int(Authority.INVALID), 10, 10, int(Occurrence.IDLE), int(Settlement.NONE)),
        GovernanceState(int(Authority.VALID), 0, 10, int(Occurrence.IDLE), int(Settlement.NONE)),
        GovernanceState(int(Authority.VALID), 10, 0, int(Occurrence.IDLE), int(Settlement.NONE)),
        GovernanceState(int(Authority.VALID), 10, 10, int(Occurrence.IN_FLIGHT), int(Settlement.NONE)),
        GovernanceState(int(Authority.VALID), 10, 10, int(Occurrence.AMBIGUOUS), int(Settlement.NONE)),
    ]
    events = [GovernanceEvent.PROPOSE_ACTION] * len(states)
    report = compare_compiled_reference(states, events)
    assert report["passed"], report["failures"]
