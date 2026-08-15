from src.fixed_state import Authority, GovernanceEvent, GovernanceOutput, Occurrence, Settlement, GovernanceState, configured_state
from src.recurrent_compiled import compare_compiled_reference, compiled_step
from src.recurrent_reference import scenario_traces, transition


def test_compiled_matches_reference_on_edge_states_all_events() -> None:
    states = [
        GovernanceState(int(Authority.INVALID), 0, 0, int(Occurrence.IDLE), int(Settlement.NONE)),
        configured_state(lease=1, budget=1, authority=Authority.VALID),
        configured_state(lease=65535, budget=255, authority=Authority.VALID),
        GovernanceState(int(Authority.VALID), 0, 1, int(Occurrence.IDLE), int(Settlement.SUCCESS)),
        GovernanceState(int(Authority.VALID), 5, 0, int(Occurrence.IDLE), int(Settlement.FAILURE)),
        GovernanceState(int(Authority.VALID), 5, 2, int(Occurrence.IN_FLIGHT), int(Settlement.NONE)),
        GovernanceState(int(Authority.VALID), 5, 2, int(Occurrence.AMBIGUOUS), int(Settlement.NONE)),
    ]
    expanded_states = []
    events = []
    for state in states:
        for event in GovernanceEvent:
            expanded_states.append(state)
            events.append(event)
    report = compare_compiled_reference(expanded_states, events)
    assert report["passed"], report["failures"]


def test_compiled_single_step_matches_reference() -> None:
    state = configured_state(lease=1, budget=1, authority=Authority.VALID)
    actual_state, actual_output = compiled_step(state, GovernanceEvent.PROPOSE_ACTION)
    expected = transition(state, GovernanceEvent.PROPOSE_ACTION)
    assert actual_state == expected.next_state
    assert actual_output == expected.output


def test_compiled_scenarios_match_reference() -> None:
    for events in scenario_traces().values():
        state = GovernanceState(int(Authority.INVALID), 0, 0, int(Occurrence.IDLE), int(Settlement.NONE))
        for event in events:
            actual_state, actual_output = compiled_step(state, event)
            expected = transition(state, event)
            assert actual_state == expected.next_state
            assert actual_output == expected.output
            state = actual_state
