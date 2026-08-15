from src.governance_reference import EVENTS, enumerate_reachable_states, reachable_graph, state_id_maps, transition


def test_reachable_graph_counts_are_stable_for_current_semantic_kernel():
    graph = reachable_graph()
    assert graph["reachable_states"] == 912
    assert graph["event_alphabet_size"] == 39
    assert graph["reachable_transitions"] == 35568
    assert graph["admitted_transitions"] == 144
    assert graph["refusal_transitions"] == 30724


def test_every_transition_stays_inside_reachable_graph():
    states, _ = enumerate_reachable_states()
    reachable = set(states)
    for state in states:
        for event in EVENTS:
            assert transition(state, event).next_state in reachable


def test_state_id_maps_cover_reachable_states_once():
    states, state_to_id = state_id_maps()
    assert len(states) == len(state_to_id) == 912
    assert set(state_to_id.values()) == set(range(len(states)))
