from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .governance_reference import (
    EVENT_NAMES,
    OUTPUT_NAMES,
    Event,
    GovernanceState,
    Output,
    initial_state,
    normalize_event,
    transition,
)


@dataclass(frozen=True)
class TraceViolation:
    index: int
    event: int
    expected_output: int
    actual_output: int
    state_before: GovernanceState
    state_after_reference: GovernanceState

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "event": EVENT_NAMES[Event(self.event)],
            "event_id": self.event,
            "expected_output": OUTPUT_NAMES[Output(self.expected_output)],
            "expected_output_id": self.expected_output,
            "actual_output": OUTPUT_NAMES[Output(self.actual_output)],
            "actual_output_id": self.actual_output,
            "state_before": self.state_before.to_json(),
            "state_after_reference": self.state_after_reference.to_json(),
        }


def trace_violations(
    events: Sequence[int | str | Event],
    outputs: Sequence[int | Output],
    start: GovernanceState | None = None,
) -> list[TraceViolation]:
    if len(events) != len(outputs):
        raise ValueError("events and outputs must have the same length")
    state = start or initial_state()
    violations: list[TraceViolation] = []
    for index, (event_like, output_like) in enumerate(zip(events, outputs)):
        event = normalize_event(event_like)
        actual_output = int(Output(int(output_like)))
        result = transition(state, event)
        if result.output != actual_output:
            violations.append(
                TraceViolation(
                    index=index,
                    event=int(event),
                    expected_output=result.output,
                    actual_output=actual_output,
                    state_before=state,
                    state_after_reference=result.next_state,
                )
            )
        state = result.next_state
    return violations


def is_admissible_trace(
    events: Sequence[int | str | Event],
    outputs: Sequence[int | Output],
    start: GovernanceState | None = None,
) -> bool:
    return not trace_violations(events, outputs, start=start)


def admissibility_report(
    traces: dict[str, tuple[Sequence[int | str | Event], Sequence[int | Output]]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    total_violations = 0
    for name, (events, outputs) in traces.items():
        violations = trace_violations(events, outputs)
        total_violations += len(violations)
        out[name] = {
            "length": len(events),
            "admissible": not violations,
            "violations": [violation.to_json() for violation in violations[:20]],
            "violation_count": len(violations),
        }
    out["summary"] = {
        "traces": len(traces),
        "total_violations": total_violations,
        "passed": total_violations == 0,
    }
    return out

