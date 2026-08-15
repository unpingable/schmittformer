from __future__ import annotations

from enum import IntEnum
from itertools import product
from typing import Iterable, Iterator, Sequence


class State(IntEnum):
    OFF = 0
    ON = 1


ALPHABET: tuple[int, ...] = tuple(range(10))
LOW_THRESHOLD = 3
HIGH_THRESHOLD = 7


def normalize_state(state: int | State) -> State:
    try:
        return State(int(state))
    except ValueError as exc:
        raise ValueError(f"invalid state {state!r}; expected 0/OFF or 1/ON") from exc


def validate_input_symbol(x: int) -> int:
    value = int(x)
    if value not in ALPHABET:
        raise ValueError(f"invalid input {x!r}; expected one of {ALPHABET}")
    return value


def transition(
    previous: int | State,
    x: int,
    low: int = LOW_THRESHOLD,
    high: int = HIGH_THRESHOLD,
) -> State:
    previous_state = normalize_state(previous)
    value = validate_input_symbol(x)
    if value >= high:
        return State.ON
    if value <= low:
        return State.OFF
    return previous_state


def run_hysteresis(
    inputs: Sequence[int],
    initial_state: int | State = State.OFF,
    low: int = LOW_THRESHOLD,
    high: int = HIGH_THRESHOLD,
) -> list[int]:
    state = normalize_state(initial_state)
    outputs: list[int] = []
    for x in inputs:
        state = transition(state, x, low=low, high=high)
        outputs.append(int(state))
    return outputs


def state_labels(states: Iterable[int | State]) -> str:
    return "".join("N" if normalize_state(s) == State.ON else "O" for s in states)


def exhaustive_sequences(
    max_len: int,
    alphabet: Sequence[int] = ALPHABET,
    include_empty: bool = False,
) -> Iterator[tuple[int, ...]]:
    start = 0 if include_empty else 1
    for length in range(start, max_len + 1):
        yield from product(alphabet, repeat=length)


def transition_table() -> dict[tuple[int, int], int]:
    return {
        (int(state), x): int(transition(state, x))
        for state in State
        for x in ALPHABET
    }
