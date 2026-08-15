from src.circuit_reference import Event, Mode, closed_state, half_open_state, initial_state, open_state, run_controller, transition


def test_unknown_does_not_change_closed_failure_window() -> None:
    state = closed_state([Event.FAILURE, Event.SUCCESS])
    assert transition(state, Event.UNKNOWN) == state


def test_closed_trips_on_three_failures_in_five_relevant_outcomes() -> None:
    states = run_controller([Event.FAILURE, Event.SUCCESS, Event.FAILURE, Event.UNKNOWN, Event.SUCCESS, Event.FAILURE])
    assert states[-1] == open_state(4)


def test_open_cooldown_uses_full_ticks_before_half_open() -> None:
    state = open_state(2)
    assert transition(state, Event.SUCCESS) == open_state(1)
    assert transition(open_state(1), Event.FAILURE) == open_state(0)
    assert transition(open_state(0), Event.FAILURE) == half_open_state(0, 3)


def test_half_open_unknown_preserves_recovery_progress() -> None:
    state = half_open_state(1, 2)
    assert transition(state, Event.UNKNOWN) == state
    assert transition(state, Event.SUCCESS).mode == Mode.CLOSED


def test_half_open_failure_reopens() -> None:
    assert transition(half_open_state(1, 2), Event.FAILURE) == open_state(4)
