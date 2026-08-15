from src.reference import State, run_hysteresis, state_labels, transition, transition_table


def test_example_trace() -> None:
    inputs = [2, 5, 8, 6, 4, 3, 5, 7]
    states = run_hysteresis(inputs)
    assert state_labels(states) == "OONNNOON"


def test_deadband_retains_initial_state() -> None:
    inputs = [4, 5, 6, 5, 4, 6]
    assert run_hysteresis(inputs, State.OFF) == [0] * len(inputs)
    assert run_hysteresis(inputs, State.ON) == [1] * len(inputs)


def test_transition_table_has_hysteresis() -> None:
    table = transition_table()
    for x in (4, 5, 6):
        assert table[(0, x)] == 0
        assert table[(1, x)] == 1
    assert transition(State.OFF, 7) == State.ON
    assert transition(State.ON, 3) == State.OFF
