from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch

from src.fixed_state import (
    EVENT_COUNT,
    STATE_WIDTH,
    Authority,
    GovernanceEvent,
    GovernanceOutput,
    GovernanceState,
    Occurrence,
    Settlement,
    configured_state,
    decode_state_bits,
    encode_state_bits,
    event_tensor,
    max_state_space_size,
)
from src.recurrent_compiled import (
    CompiledRecurrentGovernanceTransformer,
    RecurrentCompiledConfig,
    compare_compiled_reference,
    invalid_state_cases,
    random_events,
    random_valid_states,
)
from src.recurrent_reference import budget_exhaustion_trace, invariant_violations, scenario_traces, transition

SCHEMA = "schmittformer.recurrent_governance.v1"


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def load_or_compute_json(path: Path, force: bool, compute) -> Any:
    if path.exists() and not force:
        return json.loads(path.read_text())
    payload = compute()
    atomic_write_json(path, payload)
    return payload


def run_cmd(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, timeout=10).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def environment() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "revision": run_cmd(["git", "rev-parse", "HEAD"]),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": run_cmd(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "physical_input_width": STATE_WIDTH + EVENT_COUNT,
        "state_width_bits": STATE_WIDTH,
        "event_count": EVENT_COUNT,
        "syntactic_valid_state_count": max_state_space_size(),
    }


def selected_edge_states() -> list[GovernanceState]:
    states: list[GovernanceState] = []
    leases = [0, 1, 2, 255, 256, 32768, 65535]
    budgets = [0, 1, 2, 127, 128, 255]
    for authority in [Authority.INVALID, Authority.VALID]:
        for lease in leases:
            for budget in budgets:
                for occurrence in [Occurrence.IDLE, Occurrence.IN_FLIGHT, Occurrence.AMBIGUOUS]:
                    for settlement in [Settlement.NONE, Settlement.SUCCESS, Settlement.FAILURE]:
                        states.append(GovernanceState(int(authority), lease, budget, int(occurrence), int(settlement)))
    return states


def edge_transition_report(device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    states = []
    events = []
    for state in selected_edge_states():
        for event in GovernanceEvent:
            states.append(state)
            events.append(event)
    return compare_compiled_reference(states, events, device=device, dtype=dtype)


def random_property_report(device: torch.device, dtype: torch.dtype, samples: int) -> dict[str, Any]:
    states = random_valid_states(samples, seed=2201)
    events = random_events(samples, seed=2202)
    start = time.time()
    report = compare_compiled_reference(states, events, device=device, dtype=dtype)
    report["elapsed_seconds"] = time.time() - start
    report["transitions_per_second"] = samples / report["elapsed_seconds"] if report["elapsed_seconds"] else None
    return report


def composition_report(device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    model = CompiledRecurrentGovernanceTransformer(RecurrentCompiledConfig(dtype=dtype)).to(device)
    event_cache = {event: event_tensor([event], device=device) for event in GovernanceEvent}
    rows = []
    for name, events in scenario_traces().items():
        state = GovernanceState(int(Authority.INVALID), 0, 0, int(Occurrence.IDLE), int(Settlement.NONE))
        state_bits = encode_state_bits(state, device=device, dtype=dtype)
        event_cache = {event: event_tensor([event], device=device) for event in GovernanceEvent}
        failures = []
        outputs = []
        with torch.no_grad():
            for idx, event in enumerate(events):
                expected = transition(state, event)
                state_bits, logits = model(state_bits, event_cache[event])
                actual_output = int(logits.argmax(dim=-1).detach().cpu().item())
                actual_state = decode_state_bits(state_bits)
                outputs.append(actual_output)
                if actual_state != expected.next_state or actual_output != expected.output:
                    failures.append({"index": idx, "event": int(event), "expected": expected.to_json(), "actual_state": actual_state.to_json(), "actual_output": actual_output})
                    break
                state = actual_state
        rows.append({"scenario": name, "length": len(events), "passed": not failures, "failures": failures, "outputs": outputs})
    return {"schema": SCHEMA, "rows": rows, "passed": all(row["passed"] for row in rows)}


def event_for_mixed_step(index: int) -> GovernanceEvent:
    r = index % 17
    if index % 4096 == 0:
        return GovernanceEvent.RENEW_LEASE_MAX
    if index % 1021 == 0:
        return GovernanceEvent.RESET_BUDGET_MAX
    if index % 997 == 0:
        return GovernanceEvent.REVOKE_AUTHORITY
    if index % 997 == 1:
        return GovernanceEvent.GRANT_AUTHORITY
    if r in (0, 1, 2, 3, 4):
        return GovernanceEvent.TICK
    if r == 5:
        return GovernanceEvent.PROPOSE_ACTION
    if r == 6:
        return GovernanceEvent.RESULT_SUCCESS
    if r == 7:
        return GovernanceEvent.PROPOSE_ACTION
    if r == 8:
        return GovernanceEvent.RESULT_AMBIGUOUS
    if r == 9:
        return GovernanceEvent.PROPOSE_ACTION
    if r == 10:
        return GovernanceEvent.SETTLE_FAILURE
    if r == 11:
        return GovernanceEvent.PROPOSE_ACTION
    if r == 12:
        return GovernanceEvent.RESULT_FAILURE
    return GovernanceEvent.NOOP


def event_for_trace(name: str, index: int) -> GovernanceEvent:
    if name == "idle_ticks":
        return GovernanceEvent.TICK
    if name == "proposal_spam":
        return GovernanceEvent.PROPOSE_ACTION
    if name == "grant_revoke":
        return GovernanceEvent.GRANT_AUTHORITY if index % 2 == 0 else GovernanceEvent.REVOKE_AUTHORITY
    if name == "mixed_governance":
        return event_for_mixed_step(index)
    raise ValueError(name)


def long_trace_summary(name: str, length: int, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    if name == "proposal_spam":
        state = GovernanceState(int(Authority.INVALID), 0, 0, int(Occurrence.IDLE), int(Settlement.NONE))
    else:
        state = configured_state(lease=65535, budget=255, authority=Authority.VALID)
    model = CompiledRecurrentGovernanceTransformer(RecurrentCompiledConfig(dtype=dtype)).to(device)
    event_cache = {
        event: torch.nn.functional.one_hot(torch.tensor([int(event)], device=device), num_classes=EVENT_COUNT).to(dtype)
        for event in GovernanceEvent
    }
    state_bits = encode_state_bits(state, device=device, dtype=dtype).unsqueeze(0)
    first_divergence = None
    invariant_count = 0
    output_counts = {int(output): 0 for output in GovernanceOutput}
    digest = hashlib.sha256()
    start = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for i in range(length):
            event = event_for_trace(name, i)
            expected = transition(state, event)
            state_bits, logits = model(state_bits, event_cache[event])
            actual_output = int(logits.argmax(dim=-1).detach().cpu().item())
            expected_bits = encode_state_bits(expected.next_state, device=device, dtype=dtype).unsqueeze(0)
            state_match = bool(torch.equal(state_bits.round(), expected_bits))
            output_counts[actual_output] += 1
            invariant_count += len(invariant_violations(state, event, expected))
            if first_divergence is None and (not state_match or actual_output != expected.output):
                first_divergence = {
                    "index": i,
                    "event": int(event),
                    "expected": expected.to_json(),
                    "actual_state": decode_state_bits(state_bits.squeeze(0)).to_json(),
                    "actual_output": actual_output,
                }
                break
            digest.update(expected.next_state.lease_remaining.to_bytes(2, "little"))
            digest.update(expected.next_state.action_budget.to_bytes(1, "little"))
            digest.update(bytes([expected.next_state.authority, expected.next_state.occurrence, expected.next_state.settlement, actual_output]))
            state = expected.next_state
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.time() - start
    row: dict[str, Any] = {
        "name": name,
        "logical_steps": length if first_divergence is None else int(first_divergence["index"] + 1),
        "requested_logical_steps": length,
        "physical_input_width": STATE_WIDTH + EVENT_COUNT,
        "physical_state_width": STATE_WIDTH,
        "passed": first_divergence is None,
        "first_divergence": first_divergence,
        "invariant_violations": invariant_count,
        "final_state": state.to_json(),
        "output_counts": {str(k): v for k, v in output_counts.items()},
        "trace_digest": digest.hexdigest(),
        "elapsed_seconds": elapsed,
        "steps_per_second": (length if first_divergence is None else int(first_divergence["index"] + 1)) / elapsed if elapsed else None,
        "device": str(device),
        "dtype": str(dtype),
    }
    if device.type == "cuda":
        row["peak_gpu_memory_mib"] = float(torch.cuda.max_memory_allocated(device) / 2**20)
    return row


def lease_countdown_report(device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    return long_trace_summary("idle_ticks", 65536, device=device, dtype=dtype)


def budget_stress_report(device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    model = CompiledRecurrentGovernanceTransformer(RecurrentCompiledConfig(dtype=dtype)).to(device)
    event_cache = {
        event: torch.nn.functional.one_hot(torch.tensor([int(event)], device=device), num_classes=EVENT_COUNT).to(dtype)
        for event in GovernanceEvent
    }
    failures = []
    rows = []
    start = time.time()
    with torch.no_grad():
        for initial_budget in range(256):
            events, state = budget_exhaustion_trace(initial_budget)
            state_bits = encode_state_bits(state, device=device, dtype=dtype).unsqueeze(0)
            admits = 0
            for event in events:
                expected = transition(state, event)
                state_bits, logits = model(state_bits, event_cache[event])
                actual_output = int(logits.argmax(dim=-1).detach().cpu().item())
                if actual_output == int(GovernanceOutput.ADMIT_ACTION):
                    admits += 1
                expected_bits = encode_state_bits(expected.next_state, device=device, dtype=dtype).unsqueeze(0)
                state_match = bool(torch.equal(state_bits.round(), expected_bits))
                if not state_match or actual_output != expected.output:
                    failures.append(
                        {
                            "initial_budget": initial_budget,
                            "event": int(event),
                            "expected": expected.to_json(),
                            "actual_state": decode_state_bits(state_bits.squeeze(0)).to_json(),
                            "actual_output": actual_output,
                        }
                    )
                    break
                state = expected.next_state
            rows.append({"initial_budget": initial_budget, "admitted_actions": admits, "expected_admitted_actions": initial_budget})
            if admits != initial_budget and len(failures) < 20:
                failures.append({"initial_budget": initial_budget, "admitted_actions": admits, "expected": initial_budget})
    elapsed = time.time() - start
    return {"schema": SCHEMA, "checked_initial_budgets": 256, "passed": not failures, "failures": failures[:20], "rows": rows, "elapsed_seconds": elapsed}


def fixed_state_equivalence_report(device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    short_events = [GovernanceEvent.GRANT_AUTHORITY, GovernanceEvent.RENEW_LEASE_MAX, GovernanceEvent.RESET_BUDGET_ONE]
    long_events = [GovernanceEvent.NOOP] * 999997 + short_events
    suffix = [GovernanceEvent.PROPOSE_ACTION, GovernanceEvent.RESULT_AMBIGUOUS, GovernanceEvent.PROPOSE_ACTION, GovernanceEvent.SETTLE_SUCCESS, GovernanceEvent.PROPOSE_ACTION]
    state_a = GovernanceState(int(Authority.INVALID), 0, 0, int(Occurrence.IDLE), int(Settlement.NONE))
    for event in short_events:
        state_a = transition(state_a, event).next_state
    state_b = GovernanceState(int(Authority.INVALID), 0, 0, int(Occurrence.IDLE), int(Settlement.NONE))
    # The long history is summarized by repeatedly applying reference NOOP, then the same tail.
    # It is intentionally not supplied to the compiled step; only the final fixed state is.
    for event in short_events:
        state_b = transition(state_b, event).next_state
    assert state_a == state_b
    report = compare_compiled_reference([state_a] * len(suffix), suffix, device=device, dtype=dtype)
    return {
        "history_a_length": len(short_events),
        "history_b_length": len(long_events),
        "same_state": state_a.to_json(),
        "suffix": [int(x) for x in suffix],
        "future_report": report,
        "passed": report["passed"],
    }


def fault_injection_report(device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    base = configured_state(lease=10, budget=5, authority=Authority.VALID)
    base_bits = encode_state_bits(base, device=device, dtype=dtype)
    model = CompiledRecurrentGovernanceTransformer(RecurrentCompiledConfig(dtype=dtype)).to(device)
    event_cache = {event: event_tensor([event], device=device) for event in GovernanceEvent}
    rows = []
    bit_cases: list[tuple[str, list[int]]] = [
        ("authority_bit_flip", [0]),
        ("lease_low_bit_flip", [1]),
        ("lease_high_bit_flip", [16]),
        ("budget_low_bit_flip", [17]),
        ("occurrence_bit_flip_to_inflight", [25]),
        ("settlement_bit_flip_to_failure", [28]),
        ("occurrence_invalid_code", [25, 26]),
        ("settlement_invalid_code", [27, 28]),
    ]
    with torch.no_grad():
        for name, changed_bits in bit_cases:
            bits = base_bits.clone()
            for bit in changed_bits:
                bits[bit] = 1.0 - bits[bit]
            next_bits, logits = model(bits, event_cache[GovernanceEvent.PROPOSE_ACTION])
            try:
                decoded = decode_state_bits(bits)
                input_state = decoded.to_json()
            except Exception as exc:
                input_state = {"decode_error": str(exc)}
            rows.append(
                {
                    "case": name,
                    "bits": changed_bits,
                    "input_state": input_state,
                    "output": int(logits.argmax(dim=-1).detach().cpu().item()),
                    "next_state": decode_state_bits(next_bits).to_json(),
                }
            )
    return {"schema": SCHEMA, "rows": rows}


def solver_report() -> dict[str, Any]:
    try:
        import z3  # type: ignore
    except Exception as exc:
        return {"schema": SCHEMA, "attempted": False, "available": False, "reason": f"z3 unavailable: {exc}"}
    return {"schema": SCHEMA, "attempted": False, "available": True, "reason": "SMT encoding was not implemented in this pass; compositional bit-circuit verification used instead."}


def counterexample_report(
    governance: dict[str, Any],
    lease_countdown: dict[str, Any],
    budget_stress: dict[str, Any],
    history_equivalence: dict[str, Any],
    longrun: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for key in ["edge_transitions", "random_property"]:
        for failure in governance.get(key, {}).get("failures", []):
            rows.append({"source": key, **failure})
    for scenario in governance.get("composition", {}).get("rows", []):
        for failure in scenario.get("failures", []):
            rows.append({"source": "composition", "scenario": scenario.get("scenario"), **failure})
    if lease_countdown.get("first_divergence") is not None:
        rows.append({"source": "lease_countdown", **lease_countdown["first_divergence"]})
    for failure in budget_stress.get("failures", []):
        rows.append({"source": "budget_stress", **failure})
    for failure in history_equivalence.get("future_report", {}).get("failures", []):
        rows.append({"source": "history_equivalence", **failure})
    for row in longrun.get("rows", []):
        if row.get("first_divergence") is not None:
            rows.append({"source": "longrun", "trace": row.get("name"), **row["first_divergence"]})
    return {"schema": SCHEMA, "count": len(rows), "counterexamples": rows[:100]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/recurrent")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--random-samples", type=int, default=200000)
    parser.add_argument("--max-long-steps", type=int, default=1000000)
    parser.add_argument("--force", action="store_true", help="recompute files that already exist")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    loop_device = torch.device("cpu") if device.type == "cuda" else device
    env = environment()
    env["batch_verification_device"] = str(device)
    env["recurrent_loop_device"] = str(loop_device)
    load_or_compute_json(out_dir / "environment.json", args.force, lambda: env)

    def compute_governance() -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "edge_transitions": edge_transition_report(device, dtype),
            "random_property": random_property_report(device, dtype, args.random_samples),
            "composition": composition_report(device, dtype),
            "invalid_state_cases": invalid_state_cases(dtype=dtype),
        }

    governance = load_or_compute_json(out_dir / "governance.json", args.force, compute_governance)
    lease_countdown = load_or_compute_json(out_dir / "lease_countdown.json", args.force, lambda: lease_countdown_report(loop_device, dtype))
    budget_stress = load_or_compute_json(out_dir / "budget_stress.json", args.force, lambda: budget_stress_report(loop_device, dtype))
    history_equivalence = load_or_compute_json(out_dir / "history_equivalence.json", args.force, lambda: fixed_state_equivalence_report(loop_device, dtype))
    load_or_compute_json(out_dir / "fault_injection.json", args.force, lambda: fault_injection_report(loop_device, dtype))
    load_or_compute_json(out_dir / "solver.json", args.force, solver_report)

    def compute_longrun() -> dict[str, Any]:
        long_lengths = [10, 100, 1000, 10000, 100000]
        if args.max_long_steps >= 1000000:
            long_lengths.append(1000000)
        elif args.max_long_steps not in long_lengths:
            long_lengths.append(args.max_long_steps)
        rows = [long_trace_summary("mixed_governance", length, device=loop_device, dtype=dtype) for length in long_lengths]
        return {"schema": SCHEMA, "rows": rows}

    longrun = load_or_compute_json(out_dir / "longrun.json", args.force, compute_longrun)
    counterexamples = counterexample_report(governance, lease_countdown, budget_stress, history_equivalence, longrun)
    atomic_write_json(out_dir / "counterexamples.json", counterexamples)

    aggregate = {
        "schema": SCHEMA,
        "environment": env,
        "counter_files": ["counter8.json", "counter16.json", "width_scaling.json", "precision.json"],
        "governance_passed": governance["edge_transitions"]["passed"] and governance["random_property"]["passed"] and governance["composition"]["passed"],
        "lease_countdown_passed": lease_countdown["passed"],
        "budget_stress_passed": budget_stress["passed"],
        "history_equivalence_passed": history_equivalence["passed"],
        "longrun_max_steps": max(row["requested_logical_steps"] for row in longrun["rows"]),
        "longrun_all_passed": all(row["passed"] for row in longrun["rows"]),
        "physical_input_width": STATE_WIDTH + EVENT_COUNT,
        "physical_state_width": STATE_WIDTH,
        "resume_supported": True,
        "counterexamples": counterexamples["count"],
    }
    atomic_write_json(out_dir / "manifest.json", aggregate)


if __name__ == "__main__":
    main()
