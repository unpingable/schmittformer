from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from enum import IntEnum
from itertools import product
from typing import Any, Iterable, Sequence


class Event(IntEnum):
    SUCCESS = 0
    FAILURE = 1
    UNKNOWN = 2


class Mode(IntEnum):
    CLOSED = 0
    OPEN = 1
    HALF_OPEN = 2


EVENT_NAMES = {Event.SUCCESS: "SUCCESS", Event.FAILURE: "FAILURE", Event.UNKNOWN: "UNKNOWN"}
MODE_NAMES = {Mode.CLOSED: "CLOSED", Mode.OPEN: "OPEN", Mode.HALF_OPEN: "HALF_OPEN"}
EVENTS: tuple[Event, ...] = (Event.SUCCESS, Event.FAILURE, Event.UNKNOWN)
RELEVANT_EVENTS: tuple[Event, ...] = (Event.SUCCESS, Event.FAILURE)
FAILURE_WINDOW = 5
FAILURE_THRESHOLD = 3
OPEN_COOLDOWN = 4
HALF_OPEN_SUCCESS_TARGET = 2
HALF_OPEN_PROBE_BUDGET = 3


@dataclass(frozen=True, order=True)
class CircuitState:
    mode: int
    failure_window: tuple[int, ...] = ()
    cooldown_remaining: int = 0
    consecutive_successes: int = 0
    probe_budget: int = 0

    def __post_init__(self) -> None:
        mode = Mode(int(self.mode))
        object.__setattr__(self, "mode", int(mode))
        object.__setattr__(self, "failure_window", tuple(int(Event(x)) for x in self.failure_window))
        object.__setattr__(self, "cooldown_remaining", int(self.cooldown_remaining))
        object.__setattr__(self, "consecutive_successes", int(self.consecutive_successes))
        object.__setattr__(self, "probe_budget", int(self.probe_budget))

    @property
    def mode_enum(self) -> Mode:
        return Mode(self.mode)

    def to_json(self) -> dict[str, Any]:
        return {
            "mode": MODE_NAMES[self.mode_enum],
            "mode_id": self.mode,
            "failure_window": [EVENT_NAMES[Event(x)] for x in self.failure_window],
            "failure_window_ids": list(self.failure_window),
            "cooldown_remaining": self.cooldown_remaining,
            "consecutive_successes": self.consecutive_successes,
            "probe_budget": self.probe_budget,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "CircuitState":
        if "mode_id" in payload:
            mode = int(payload["mode_id"])
        else:
            mode = Mode[payload["mode"]].value
        if "failure_window_ids" in payload:
            window = tuple(int(x) for x in payload["failure_window_ids"])
        else:
            window = tuple(Event[x].value for x in payload.get("failure_window", []))
        return cls(
            mode=mode,
            failure_window=window,
            cooldown_remaining=int(payload.get("cooldown_remaining", 0)),
            consecutive_successes=int(payload.get("consecutive_successes", 0)),
            probe_budget=int(payload.get("probe_budget", 0)),
        )


def normalize_event(symbol: int | str | Event) -> Event:
    if isinstance(symbol, str):
        return Event[symbol]
    return Event(int(symbol))


def closed_state(window: Sequence[int | Event] = ()) -> CircuitState:
    return CircuitState(Mode.CLOSED, tuple(int(Event(x)) for x in window), 0, 0, 0)


def open_state(cooldown_remaining: int = OPEN_COOLDOWN) -> CircuitState:
    return CircuitState(Mode.OPEN, (), int(cooldown_remaining), 0, 0)


def half_open_state(
    consecutive_successes: int = 0,
    probe_budget: int = HALF_OPEN_PROBE_BUDGET,
) -> CircuitState:
    return CircuitState(Mode.HALF_OPEN, (), 0, int(consecutive_successes), int(probe_budget))


def initial_state() -> CircuitState:
    return closed_state(())


def transition(state: CircuitState, symbol: int | str | Event) -> CircuitState:
    event = normalize_event(symbol)
    mode = state.mode_enum

    if mode == Mode.CLOSED:
        if event == Event.UNKNOWN:
            return state
        window = (*state.failure_window, int(event))[-FAILURE_WINDOW:]
        if sum(x == int(Event.FAILURE) for x in window) >= FAILURE_THRESHOLD:
            return open_state(OPEN_COOLDOWN)
        return closed_state(window)

    if mode == Mode.OPEN:
        # Cooldown is checked at the start of the tick. Inputs observed while OPEN
        # never count as recovery probes. A state with cooldown 0 uses one input
        # tick to enter HALF_OPEN; that input is ignored for recovery purposes.
        if state.cooldown_remaining > 0:
            return open_state(state.cooldown_remaining - 1)
        return half_open_state()

    if mode == Mode.HALF_OPEN:
        if event == Event.UNKNOWN:
            return state
        budget = state.probe_budget - 1
        if event == Event.FAILURE:
            return open_state(OPEN_COOLDOWN)
        successes = state.consecutive_successes + 1
        if successes >= HALF_OPEN_SUCCESS_TARGET:
            return closed_state(())
        if budget <= 0:
            return open_state(OPEN_COOLDOWN)
        return half_open_state(successes, budget)

    raise AssertionError(f"unhandled mode: {mode}")


def run_controller(
    inputs: Sequence[int | str | Event],
    start: CircuitState | None = None,
    include_initial: bool = False,
) -> list[CircuitState]:
    state = start or initial_state()
    states: list[CircuitState] = [state] if include_initial else []
    for symbol in inputs:
        state = transition(state, symbol)
        states.append(state)
    return states


def mode_sequence(inputs: Sequence[int | str | Event], start: CircuitState | None = None) -> list[int]:
    return [s.mode for s in run_controller(inputs, start=start)]


def state_sort_key(state: CircuitState) -> tuple[Any, ...]:
    return (
        state.mode,
        len(state.failure_window),
        state.failure_window,
        state.cooldown_remaining,
        state.consecutive_successes,
        state.probe_budget,
    )


def syntactic_states() -> list[CircuitState]:
    states: list[CircuitState] = []
    for length in range(FAILURE_WINDOW + 1):
        for window in product((int(Event.SUCCESS), int(Event.FAILURE)), repeat=length):
            states.append(closed_state(window))
    for cooldown in range(OPEN_COOLDOWN + 1):
        states.append(open_state(cooldown))
    for successes in range(HALF_OPEN_SUCCESS_TARGET + 1):
        for budget in range(HALF_OPEN_PROBE_BUDGET + 1):
            states.append(half_open_state(successes, budget))
    return sorted(set(states), key=state_sort_key)


def enumerate_reachable_states() -> tuple[list[CircuitState], dict[CircuitState, tuple[int, ...]]]:
    start = initial_state()
    seen: dict[CircuitState, tuple[int, ...]] = {start: ()}
    queue: deque[CircuitState] = deque([start])
    while queue:
        state = queue.popleft()
        history = seen[state]
        for event in EVENTS:
            next_state = transition(state, event)
            if next_state not in seen:
                seen[next_state] = (*history, int(event))
                queue.append(next_state)
    states = sorted(seen, key=state_sort_key)
    return states, seen


def reachable_graph() -> dict[str, Any]:
    states, canonical = enumerate_reachable_states()
    state_to_id = {state: i for i, state in enumerate(states)}
    transitions = []
    for state in states:
        for event in EVENTS:
            next_state = transition(state, event)
            transitions.append(
                {
                    "from": state_to_id[state],
                    "event": EVENT_NAMES[event],
                    "event_id": int(event),
                    "to": state_to_id[next_state],
                    "from_mode": MODE_NAMES[state.mode_enum],
                    "to_mode": MODE_NAMES[next_state.mode_enum],
                }
            )
    return {
        "syntactic_normalized_states": len(syntactic_states()),
        "reachable_states": len(states),
        "reachable_transitions": len(transitions),
        "states": [state.to_json() | {"id": state_to_id[state]} for state in states],
        "canonical_histories": {
            str(state_to_id[state]): list(canonical[state]) for state in states
        },
        "transitions": transitions,
    }


def state_id_maps() -> tuple[list[CircuitState], dict[CircuitState, int]]:
    states, _ = enumerate_reachable_states()
    return states, {state: i for i, state in enumerate(states)}


def collect_equivalent_histories(max_per_state: int = 8) -> dict[CircuitState, list[tuple[int, ...]]]:
    states, canonical = enumerate_reachable_states()
    trip_recover_cycle = (
        int(Event.FAILURE),
        int(Event.FAILURE),
        int(Event.FAILURE),
        int(Event.UNKNOWN),
        int(Event.UNKNOWN),
        int(Event.UNKNOWN),
        int(Event.UNKNOWN),
        int(Event.UNKNOWN),
        int(Event.SUCCESS),
        int(Event.SUCCESS),
    )
    edge_cycle = (
        int(Event.SUCCESS),
        int(Event.FAILURE),
        int(Event.SUCCESS),
        int(Event.FAILURE),
        int(Event.UNKNOWN),
        int(Event.FAILURE),
        int(Event.SUCCESS),
        int(Event.UNKNOWN),
        int(Event.SUCCESS),
        int(Event.UNKNOWN),
        int(Event.UNKNOWN),
        int(Event.UNKNOWN),
        int(Event.UNKNOWN),
        int(Event.UNKNOWN),
        int(Event.SUCCESS),
        int(Event.SUCCESS),
    )
    prefixes = [
        (),
        (int(Event.UNKNOWN),) * 7,
        (int(Event.UNKNOWN),) * 37,
        trip_recover_cycle,
        (int(Event.UNKNOWN),) * 11 + trip_recover_cycle,
        trip_recover_cycle + (int(Event.UNKNOWN),) * 13 + edge_cycle,
        edge_cycle + trip_recover_cycle,
        (int(Event.SUCCESS),) * 25 + trip_recover_cycle + (int(Event.UNKNOWN),) * 19,
    ]
    groups: dict[CircuitState, list[tuple[int, ...]]] = {}
    for state in states:
        base = canonical[state]
        histories: list[tuple[int, ...]] = []
        for prefix in prefixes:
            candidate = (*prefix, *base)
            reached = run_controller(candidate)[-1] if candidate else initial_state()
            if reached == state and candidate not in histories:
                histories.append(candidate)
            if len(histories) >= max_per_state:
                break
        groups[state] = histories
    return groups


def transition_label(previous: CircuitState, next_state: CircuitState) -> str:
    return f"{MODE_NAMES[previous.mode_enum]}->{MODE_NAMES[next_state.mode_enum]}"


def invariant_violations(
    previous: CircuitState,
    event: int | Event,
    next_state: CircuitState,
) -> list[str]:
    event = normalize_event(event)
    violations: list[str] = []
    expected = transition(previous, event)
    if next_state != expected:
        violations.append("transition_mismatch")
    if previous.mode_enum == Mode.OPEN and next_state.mode_enum == Mode.CLOSED:
        violations.append("forbidden_OPEN_to_CLOSED")
    if previous.mode_enum == Mode.CLOSED and next_state.mode_enum == Mode.HALF_OPEN:
        violations.append("forbidden_CLOSED_to_HALF_OPEN")
    if previous.mode_enum == Mode.HALF_OPEN and next_state.mode_enum == Mode.CLOSED:
        if transition(previous, event).mode_enum != Mode.CLOSED:
            violations.append("half_open_closed_without_recovery_condition")
    if previous.mode_enum == Mode.HALF_OPEN and event == Event.FAILURE:
        if next_state.mode_enum != Mode.OPEN:
            violations.append("half_open_failure_did_not_reopen")
    if previous.mode_enum == Mode.HALF_OPEN and event == Event.UNKNOWN:
        if next_state.consecutive_successes > previous.consecutive_successes:
            violations.append("unknown_increased_recovery_progress")
        if next_state.probe_budget < previous.probe_budget:
            violations.append("unknown_consumed_probe_budget")
    if previous.mode_enum == Mode.OPEN and previous.cooldown_remaining > 0:
        if next_state.mode_enum != Mode.OPEN:
            violations.append("cooldown_shortened_mode")
        elif next_state.cooldown_remaining != previous.cooldown_remaining - 1:
            violations.append("cooldown_shortened_count")
    if previous.mode_enum == Mode.CLOSED and event == Event.UNKNOWN:
        if next_state.failure_window != previous.failure_window:
            violations.append("unknown_altered_failure_window")
    if next_state.probe_budget < 0:
        violations.append("negative_probe_budget")
    return violations


def summarize_state(state: CircuitState) -> str:
    if state.mode_enum == Mode.CLOSED:
        chars = "".join("F" if x == int(Event.FAILURE) else "S" for x in state.failure_window)
        return f"CLOSED[{chars}]"
    if state.mode_enum == Mode.OPEN:
        return f"OPEN[c={state.cooldown_remaining}]"
    return f"HALF_OPEN[s={state.consecutive_successes},b={state.probe_budget}]"
