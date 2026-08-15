from src.circuit_reference import Event, enumerate_reachable_states, reachable_graph, transition


def test_reachable_graph_counts() -> None:
    graph = reachable_graph()
    assert graph["syntactic_normalized_states"] == 80
    assert graph["reachable_states"] == 48
    assert graph["reachable_transitions"] == 48 * 3


def test_all_reference_transitions_stay_reachable() -> None:
    states, _ = enumerate_reachable_states()
    reachable = set(states)
    for state in states:
        for event in Event:
            assert transition(state, event) in reachable
