from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
except Exception:  # pragma: no cover - torch is expected in this repo, but not required for the reference kernel.
    torch = None  # type: ignore[assignment]

from src.governance_admissibility import admissibility_report
from src.governance_reference import (
    EVENTS,
    adversarial_sequences,
    collect_equivalent_histories,
    enumerate_reachable_states,
    invariant_violations,
    output_sequence,
    reachable_graph,
    run_kernel,
    transition,
)


AG_NG_PATH = Path("/home/jbeck/ag_ng")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def command_output(args: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(args, cwd=cwd, text=True, timeout=10).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def environment_report() -> dict[str, Any]:
    torch_report: dict[str, Any] = {"torch_available": torch is not None}
    if torch is not None:
        torch_report.update(
            {
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    return {
        "python": platform.python_version(),
        "schmittformer_revision": command_output(["git", "rev-parse", "HEAD"]),
        "schmittformer_describe": command_output(["git", "describe", "--tags", "--always", "--dirty"]),
        "schmittformer_status_short": command_output(["git", "status", "--short"]),
        "ag_ng_path": str(AG_NG_PATH),
        "ag_ng_revision": command_output(["git", "rev-parse", "HEAD"], cwd=AG_NG_PATH),
        "ag_ng_status_short": command_output(["git", "status", "--short"], cwd=AG_NG_PATH),
        **torch_report,
    }


def invariant_report() -> dict[str, Any]:
    states, _ = enumerate_reachable_states()
    violations: list[dict[str, Any]] = []
    for state in states:
        for event in EVENTS:
            result = transition(state, event)
            for violation in invariant_violations(state, event, result):
                violations.append(
                    {
                        "state": state.to_json(),
                        "event": event.name,
                        "event_id": int(event),
                        "violation": violation,
                        "result": result.to_json(),
                    }
                )
    return {
        "reachable_states": len(states),
        "event_alphabet_size": len(EVENTS),
        "checked_transitions": len(states) * len(EVENTS),
        "violation_count": len(violations),
        "violations": violations[:50],
        "passed": not violations,
    }


def state_abstraction_report() -> dict[str, Any]:
    groups = collect_equivalent_histories(max_per_state=5, random_sequences=1200, random_length=80)
    suffixes = [
        ["NOOP", "CLAIM_AUTHORITY_RECORD", "MALFORMED"],
        ["PROPOSE_INITIAL_A_P0", "REQUIRE_STANDING", "RECORD_ADMISSIBLE_CURRENT"],
        ["CONSUME_AUTH_CURRENT", "ACCEPT_DOCKET_CUSTODY", "REQUIRE_RECONCILIATION"],
        ["RECORD_RECONCILED_FAILURE", "OPEN_CONTINUATION", "PROPOSE_RETRY_A_P0"],
        ["TICK", "TICK", "CONSUME_AUTH_CURRENT"],
        ["HALT", "HUMAN_RETURN", "COMPLETE"],
    ]
    comparisons = 0
    violations: list[dict[str, Any]] = []
    for state, histories in groups.items():
        if len(histories) < 2:
            continue
        for suffix in suffixes:
            expected = None
            expected_history = None
            for history in histories:
                reached = run_kernel(history, include_initial=True)[0][-1]
                if reached != state:
                    violations.append({
                        "kind": "history_does_not_reach_group_state",
                        "state": state.to_json(),
                        "history": list(history),
                        "reached": reached.to_json(),
                    })
                    continue
                future = tuple(output_sequence(suffix, start=reached))
                if expected is None:
                    expected = future
                    expected_history = history
                else:
                    comparisons += 1
                    if future != expected:
                        violations.append({
                            "kind": "history_equivalence_violation",
                            "state": state.to_json(),
                            "suffix": suffix,
                            "history_a": list(expected_history or ()),
                            "history_b": list(history),
                            "outputs_a": list(expected),
                            "outputs_b": list(future),
                        })
    return {
        "groups": len(groups),
        "groups_with_multiple_histories": sum(1 for histories in groups.values() if len(histories) > 1),
        "comparisons": comparisons,
        "violations": violations[:50],
        "violation_count": len(violations),
        "passed": not violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    start = time.time()
    graph = reachable_graph()
    states_payload = {
        "metadata": environment_report(),
        "source_model": graph["source_model"],
        "syntactic_states": graph["syntactic_states"],
        "reachable_states": graph["reachable_states"],
        "event_alphabet_size": graph["event_alphabet_size"],
        "state_records": "reachable states only",
        "states": graph["states"],
        "canonical_histories": graph["canonical_histories"],
    }
    transitions_payload = {
        "metadata": environment_report(),
        "source_model": graph["source_model"],
        "reachable_states": graph["reachable_states"],
        "event_alphabet_size": graph["event_alphabet_size"],
        "reachable_transitions": graph["reachable_transitions"],
        "admitted_transitions": graph["admitted_transitions"],
        "refusal_transitions": graph["refusal_transitions"],
        "transition_records": "reachable state/event transitions only",
        "transitions": graph["transitions"],
    }
    adversarial_traces = {
        name: (events, [result.output for result in run_kernel(events)[1]])
        for name, events in adversarial_sequences().items()
    }
    summary = {
        "metadata": environment_report(),
        "elapsed_seconds": time.time() - start,
        "graph_counts": {
            "syntactic_states": graph["syntactic_states"],
            "reachable_states": graph["reachable_states"],
            "event_alphabet_size": graph["event_alphabet_size"],
            "reachable_transitions": graph["reachable_transitions"],
            "admitted_transitions": graph["admitted_transitions"],
            "refusal_transitions": graph["refusal_transitions"],
        },
        "invariants": invariant_report(),
        "state_abstraction": state_abstraction_report(),
        "adversarial_admissibility": admissibility_report(adversarial_traces),
    }
    write_json(args.out_dir / "governance_states.json", states_payload)
    write_json(args.out_dir / "governance_transitions.json", transitions_payload)
    write_json(args.out_dir / "governance_summary.json", summary)


if __name__ == "__main__":
    main()
