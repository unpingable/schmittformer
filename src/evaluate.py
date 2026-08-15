from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .reference import State, run_hysteresis, transition


@dataclass
class SequenceEvaluation:
    sequences: int
    tokens: int
    exact_sequences: int
    exact_tokens: int
    first_divergence: dict[str, Any] | None
    illegal_transitions: int

    @property
    def sequence_accuracy(self) -> float:
        return self.exact_sequences / self.sequences if self.sequences else 1.0

    @property
    def token_accuracy(self) -> float:
        return self.exact_tokens / self.tokens if self.tokens else 1.0

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sequence_accuracy"] = self.sequence_accuracy
        payload["token_accuracy"] = self.token_accuracy
        return payload


def first_divergence(expected: Sequence[int], actual: Sequence[int]) -> int | None:
    for index, (want, got) in enumerate(zip(expected, actual)):
        if int(want) != int(got):
            return index
    if len(expected) != len(actual):
        return min(len(expected), len(actual))
    return None


def count_illegal_transitions(
    inputs: Sequence[int],
    states: Sequence[int],
    initial_state: int | State = State.OFF,
) -> int:
    previous = State(int(initial_state))
    illegal = 0
    for x, predicted in zip(inputs, states):
        expected_next = transition(previous, int(x))
        predicted_state = State(int(predicted))
        if predicted_state != expected_next:
            illegal += 1
        previous = predicted_state
    return illegal


def evaluate_predictor(
    predictor: Callable[[Sequence[int], int | State], Sequence[int]],
    sequences: Iterable[Sequence[int]],
    initial_state: int | State = State.OFF,
) -> SequenceEvaluation:
    total_sequences = 0
    total_tokens = 0
    exact_sequences = 0
    exact_tokens = 0
    illegal = 0
    first_bad: dict[str, Any] | None = None

    for sequence in sequences:
        inputs = tuple(int(x) for x in sequence)
        expected = run_hysteresis(inputs, initial_state=initial_state)
        actual = [int(x) for x in predictor(inputs, initial_state)]
        div = first_divergence(expected, actual)

        total_sequences += 1
        total_tokens += len(inputs)
        exact_tokens += sum(int(a == b) for a, b in zip(expected, actual))
        if div is None:
            exact_sequences += 1
        elif first_bad is None:
            first_bad = {
                "inputs": list(inputs),
                "initial_state": int(initial_state),
                "expected": expected,
                "actual": actual,
                "index": div,
            }

        illegal += count_illegal_transitions(inputs, actual, initial_state)

    return SequenceEvaluation(
        sequences=total_sequences,
        tokens=total_tokens,
        exact_sequences=exact_sequences,
        exact_tokens=exact_tokens,
        first_divergence=first_bad,
        illegal_transitions=illegal,
    )
