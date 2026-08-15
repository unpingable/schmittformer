from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from src.compiled_bits import int_tensor_to_bits
from src.fixed_state import (
    EVENT_COUNT,
    STATE_WIDTH,
    GovernanceEvent,
    GovernanceOutput,
    GovernanceState,
    decode_state_bits,
    encode_state_bits,
    event_tensor,
)
from src.recurrent_compiled import random_events, random_valid_states
from src.recurrent_reference import invariant_violations, transition
from src.recurrent_softmax import (
    SlotSoftmaxConfig,
    SoftmaxRecurrentGovernanceTransformer,
    compare_softmax_reference,
    slot_leakage_bound,
    slot_softmax_weights,
    slot_attention_stats,
    verify_softmax_counter,
)
from src.stock_transformer_recurrent import (
    StockGatherConfig,
    StockGatherRecurrentModel,
    compare_stock_gather_to_reference,
    load_stock_model,
    save_stock_model,
)
from src.transition_smt import run_transition_equivalence
from experiments.run_recurrent_governance import event_for_trace, selected_edge_states

SCHEMA = "schmittformer.recurrent_softmax.v1"
BASE_REVISION = "cb6ce94 recurrent: add fixed-state counter governance"


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def load_or_compute(path: Path, force: bool, compute) -> Any:
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
        "base_checkpoint": BASE_REVISION,
        "revision": run_cmd(["git", "rev-parse", "HEAD"]),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": run_cmd(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "state_width": STATE_WIDTH,
        "event_count": EVENT_COUNT,
        "physical_input_width": STATE_WIDTH + EVENT_COUNT,
    }


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float64": torch.float64,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def primitive_margins(gaps: list[float]) -> dict[str, Any]:
    rows = []
    for width in [8, 16, STATE_WIDTH + EVENT_COUNT]:
        for gap in gaps:
            weights = slot_softmax_weights(width, gap, torch.float64, torch.device("cpu"))
            stats = slot_attention_stats(weights)
            rows.append({"width": width, "score_gap": gap, "leakage_bound": slot_leakage_bound(width, gap), **stats})
    return {"schema": SCHEMA, "rows": rows}


def counter_sweep(width: int, gaps: list[float], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    rows = []
    for gap in gaps:
        rows.append(verify_softmax_counter(width, gap, device=device, dtype=dtype, exhaustive=True, batch_size=65536))
    return {"schema": SCHEMA, "width": width, "rows": rows}


def precision_report(gap: float, device: torch.device) -> dict[str, Any]:
    plan: list[tuple[str, torch.device]] = [("float64", torch.device("cpu")), ("float32_cpu", torch.device("cpu")), ("float32", device)]
    if device.type == "cuda":
        plan.append(("float16", device))
        if torch.cuda.is_bf16_supported():
            plan.append(("bfloat16", device))
    rows = []
    for name, dev in plan:
        dtype_name = "float32" if name == "float32_cpu" else name
        try:
            rows.append(verify_softmax_counter(16, gap, device=dev, dtype=dtype_from_name(dtype_name), exhaustive=True, batch_size=65536) | {"precision_name": name})
        except Exception as exc:
            rows.append({"precision_name": name, "device": str(dev), "dtype": dtype_name, "passed": False, "classification": "NUMERIC_FAILURE", "error": str(exc)})
    return {"schema": SCHEMA, "score_gap": gap, "rows": rows}


def adversarial_states() -> list[GovernanceState]:
    rows: list[GovernanceState] = []
    leases = [0, 1, 2, 255, 256, 0x00FF, 0x0100, 0x0FFF, 0x1000, 0x7FFF, 0x8000, 0xFFFF]
    budgets = [0, 1, 2, 127, 128, 254, 255]
    for authority in [0, 1]:
        for lease in leases:
            for budget in budgets:
                for occurrence in [0, 1, 2]:
                    for settlement in [0, 1, 2]:
                        rows.append(GovernanceState(authority, lease, budget, occurrence, settlement))
    return rows


def governance_report(gap: float, device: torch.device, dtype: torch.dtype, random_samples: int) -> dict[str, Any]:
    edge_states = []
    edge_events = []
    for state in selected_edge_states():
        for event in GovernanceEvent:
            edge_states.append(state)
            edge_events.append(event)
    adv_states = []
    adv_events = []
    for state in adversarial_states():
        for event in GovernanceEvent:
            adv_states.append(state)
            adv_events.append(event)
    random_states = random_valid_states(random_samples, seed=3301)
    rand_events = random_events(random_samples, seed=3302)
    start = time.time()
    random_report = compare_softmax_reference(random_states, rand_events, gap, device=device, dtype=dtype)
    random_report["elapsed_seconds"] = time.time() - start
    return {
        "schema": SCHEMA,
        "score_gap": gap,
        "edge": compare_softmax_reference(edge_states, edge_events, gap, device=device, dtype=dtype),
        "adversarial": compare_softmax_reference(adv_states, adv_events, gap, device=device, dtype=dtype),
        "random": random_report,
    }


def long_trace(name: str, length: int, gap: float, dtype: torch.dtype, carry_mode: str) -> dict[str, Any]:
    device = torch.device("cpu")
    model = SoftmaxRecurrentGovernanceTransformer(SlotSoftmaxConfig(score_gap=gap, dtype=dtype)).to(device)
    state = GovernanceState(1, 0xFFFF, 0xFF, 0, 0)
    state_bits = encode_state_bits(state, device=device, dtype=dtype).unsqueeze(0)
    event_cache = {event: event_tensor([event], device=device) for event in GovernanceEvent}
    first_divergence = None
    invariant_count = 0
    min_next_margin = float("inf")
    min_output_margin = float("inf")
    max_next_error = 0.0
    output_counts = {int(output): 0 for output in GovernanceOutput}
    start = time.time()
    with torch.no_grad():
        for index in range(length):
            event = event_for_trace(name, index)
            expected = transition(state, event)
            next_bits, logits, debug = model(state_bits, event_cache[event], return_debug=True)
            actual_output = int(logits.argmax(dim=-1).detach().cpu().item())
            expected_bits = encode_state_bits(expected.next_state, device=device, dtype=dtype).unsqueeze(0)
            rounded_match = bool(torch.equal(next_bits.round(), expected_bits))
            output_counts[actual_output] += 1
            invariant_count += len(invariant_violations(state, event, expected))
            min_next_margin = min(min_next_margin, float(torch.abs(next_bits - 0.5).min().detach().cpu().item()))
            correct = logits.reshape(1, -1)[0, int(expected.output)]
            runner = torch.cat([logits.reshape(1, -1)[0, : int(expected.output)], logits.reshape(1, -1)[0, int(expected.output) + 1 :]]).max()
            min_output_margin = min(min_output_margin, float((correct - runner).detach().cpu().item()))
            max_next_error = max(max_next_error, float(torch.abs(next_bits - expected_bits).max().detach().cpu().item()))
            if not rounded_match or actual_output != expected.output:
                first_divergence = {
                    "index": index,
                    "event": int(event),
                    "expected": expected.to_json(),
                    "actual_state": decode_state_bits(next_bits.squeeze(0)).to_json(),
                    "actual_output": actual_output,
                }
                break
            state = expected.next_state
            state_bits = expected_bits if carry_mode == "discrete_reencode" else next_bits.reshape(1, -1)
    elapsed = time.time() - start
    return {
        "name": name,
        "carry_mode": carry_mode,
        "requested_logical_steps": length,
        "logical_steps": length if first_divergence is None else int(first_divergence["index"] + 1),
        "passed": first_divergence is None,
        "first_divergence": first_divergence,
        "physical_input_width": STATE_WIDTH + EVENT_COUNT,
        "invariant_violations": invariant_count,
        "min_next_bit_margin_to_half": min_next_margin,
        "min_output_margin": min_output_margin,
        "max_next_bit_error_before_decode": max_next_error,
        "output_counts": {str(k): v for k, v in output_counts.items()},
        "elapsed_seconds": elapsed,
        "steps_per_second": (length if first_divergence is None else int(first_divergence["index"] + 1)) / elapsed if elapsed else None,
        "final_state": state.to_json(),
    }


def longrun_report(gap: float, dtype: torch.dtype, max_steps: int) -> dict[str, Any]:
    base_lengths = [10, 100, 1000, 10000, 100000, 1000000]
    lengths = [length for length in base_lengths if length <= max_steps]
    if max_steps not in lengths:
        lengths.append(max_steps)
    lengths = sorted(set(lengths))
    rows = [long_trace("mixed_governance", length, gap, dtype, "discrete_reencode") for length in lengths]
    analog_lengths = [length for length in [10, 100, 1000, 10000] if length <= max_steps]
    analog_rows = [long_trace("mixed_governance", length, gap, dtype, "analog_carry") for length in analog_lengths]
    return {"schema": SCHEMA, "score_gap": gap, "rows": rows, "analog_carry_rows": analog_rows}


def stock_model_report(out_dir: Path, gap: float) -> dict[str, Any]:
    device = torch.device("cpu")
    dtype = torch.float32
    slots = torch.randint(0, 2, (64, STATE_WIDTH + EVENT_COUNT), generator=torch.Generator().manual_seed(4401)).to(dtype)
    gather = compare_stock_gather_to_reference(slots, gap, dtype)
    model = StockGatherRecurrentModel(StockGatherConfig(score_gap=gap, dtype_name="float32")).to(device)
    states = [GovernanceState(1, 1, 1, 0, 0), GovernanceState(0, 0, 0, 0, 0), GovernanceState(1, 10, 5, 2, 0)]
    events = [GovernanceEvent.PROPOSE_ACTION, GovernanceEvent.PROPOSE_ACTION, GovernanceEvent.SETTLE_FAILURE]
    failures = []
    with torch.no_grad():
        for idx, (state, event) in enumerate(zip(states, events)):
            bits = encode_state_bits(state, dtype=dtype)
            next_bits, logits = model(bits, torch.tensor([int(event)]))
            actual = decode_state_bits(next_bits)
            actual_output = int(logits.argmax().item())
            expected = transition(state, event)
            if actual != expected.next_state or actual_output != expected.output:
                failures.append({"index": idx, "expected": expected.to_json(), "actual_state": actual.to_json(), "actual_output": actual_output})
    ckpt = out_dir / "stock_gather_recurrent.pt"
    save_stock_model(ckpt, model)
    loaded = load_stock_model(ckpt, device)
    load_failures = []
    with torch.no_grad():
        for idx, (state, event) in enumerate(zip(states, events)):
            bits = encode_state_bits(state, dtype=dtype)
            a_bits, a_logits = model(bits, torch.tensor([int(event)]))
            b_bits, b_logits = loaded(bits, torch.tensor([int(event)]))
            if not torch.allclose(a_bits, b_bits) or not torch.allclose(a_logits, b_logits):
                load_failures.append(idx)
    return {
        "schema": SCHEMA,
        "score_gap": gap,
        "stock_gather": gather,
        "governance_smoke_checked": len(states),
        "governance_smoke_failures": failures,
        "save_load_failures": load_failures,
        "save_load_passed": not load_failures,
        "checkpoint_path": str(ckpt),
        "checkpoint_size_bytes": ckpt.stat().st_size,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "audit_classification": "stock nn.MultiheadAttention for retrieval; recurrent arithmetic remains custom tensor circuit",
    }


def perturbation_report(gap: float) -> dict[str, Any]:
    base_states = [state for state in selected_edge_states()[:32]]
    events = [GovernanceEvent.PROPOSE_ACTION, GovernanceEvent.TICK, GovernanceEvent.RESULT_AMBIGUOUS, GovernanceEvent.SETTLE_FAILURE]
    pairs = [(s, e) for s in base_states for e in events]
    rows = []
    for sigma in [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        gen = torch.Generator().manual_seed(5500 + int(sigma * 1e8))
        model = StockGatherRecurrentModel(StockGatherConfig(score_gap=gap, dtype_name="float32"))
        with torch.no_grad():
            for p in model.gather.parameters():
                if sigma > 0:
                    p.add_(torch.randn(p.shape, generator=gen, dtype=p.dtype) * sigma)
        failures = []
        with torch.no_grad():
            for idx, (state, event) in enumerate(pairs):
                bits = encode_state_bits(state, dtype=torch.float32)
                next_bits, logits = model(bits, torch.tensor([int(event)]))
                actual = decode_state_bits(next_bits)
                actual_output = int(logits.argmax().item())
                expected = transition(state, event)
                if actual != expected.next_state or actual_output != expected.output:
                    failures.append({"index": idx, "sigma": sigma, "state": state.to_json(), "event": int(event), "expected": expected.to_json(), "actual_state": actual.to_json(), "actual_output": actual_output})
                    break
        rows.append({"sigma": sigma, "checked": len(pairs) if not failures else failures[0]["index"] + 1, "passed": not failures, "first_failure": failures[0] if failures else None})
    return {"schema": SCHEMA, "score_gap": gap, "rows": rows}


def counterexamples(governance: dict[str, Any], longrun: dict[str, Any], stock: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for key in ["edge", "adversarial", "random"]:
        for failure in governance.get(key, {}).get("failures", []):
            rows.append({"source": key, "category": "main_semantic", **failure})
    for row in longrun.get("rows", []):
        if row.get("first_divergence") is not None:
            rows.append({"source": "longrun", "category": "main_semantic", "carry_mode": row.get("carry_mode"), **row["first_divergence"]})
    for row in longrun.get("analog_carry_rows", []):
        if row.get("first_divergence") is not None:
            rows.append({"source": "longrun", "category": "analog_carry_diagnostic", "carry_mode": row.get("carry_mode"), **row["first_divergence"]})
    for failure in stock.get("governance_smoke_failures", []):
        rows.append({"source": "stock_model", "category": "stock_smoke", **failure})
    return {
        "schema": SCHEMA,
        "count": len(rows),
        "main_semantic_count": sum(1 for row in rows if row["category"] == "main_semantic"),
        "analog_carry_diagnostic_count": sum(1 for row in rows if row["category"] == "analog_carry_diagnostic"),
        "counterexamples": rows[:100],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/recurrent_softmax")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--score-gap", type=float, default=8.0)
    parser.add_argument("--random-samples", type=int, default=200000)
    parser.add_argument("--max-long-steps", type=int, default=1000000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gaps = [2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0]
    env = load_or_compute(out_dir / "environment.json", args.force, environment)
    margins = load_or_compute(out_dir / "primitive_margins.json", args.force, lambda: primitive_margins(gaps))
    counter8 = load_or_compute(out_dir / "counter8.json", args.force, lambda: counter_sweep(8, gaps, device, torch.float32 if device.type == "cuda" else torch.float64))
    counter16 = load_or_compute(out_dir / "counter16.json", args.force, lambda: counter_sweep(16, gaps, device, torch.float32 if device.type == "cuda" else torch.float64))
    precision = load_or_compute(out_dir / "precision.json", args.force, lambda: precision_report(args.score_gap, device))
    governance = load_or_compute(out_dir / "governance.json", args.force, lambda: governance_report(args.score_gap, device, torch.float32 if device.type == "cuda" else torch.float64, args.random_samples))
    longrun = load_or_compute(out_dir / "longrun.json", args.force, lambda: longrun_report(args.score_gap, torch.float64, args.max_long_steps))
    stock = load_or_compute(out_dir / "stock_model.json", args.force, lambda: stock_model_report(out_dir, args.score_gap))
    solver = load_or_compute(out_dir / "solver.json", args.force, lambda: run_transition_equivalence())
    perturb = load_or_compute(out_dir / "perturbation.json", args.force, lambda: perturbation_report(args.score_gap))
    cex = counterexamples(governance, longrun, stock)
    atomic_write_json(out_dir / "counterexamples.json", cex)
    manifest = {
        "schema": SCHEMA,
        "environment": env,
        "score_gap": args.score_gap,
        "counter8_passed_at_gap": next((row["passed"] for row in counter8["rows"] if row["score_gap"] == args.score_gap), None),
        "counter16_passed_at_gap": next((row["passed"] for row in counter16["rows"] if row["score_gap"] == args.score_gap), None),
        "governance_passed": governance["edge"]["passed"] and governance["adversarial"]["passed"] and governance["random"]["passed"],
        "longrun_all_passed": all(row["passed"] for row in longrun["rows"]),
        "analog_carry_all_passed": all(row["passed"] for row in longrun["analog_carry_rows"]),
        "stock_save_load_passed": stock["save_load_passed"],
        "solver_result": solver.get("result"),
        "counterexamples": cex["count"],
        "main_semantic_counterexamples": cex["main_semantic_count"],
        "analog_carry_diagnostic_counterexamples": cex["analog_carry_diagnostic_count"],
        "perturbation_rows": perturb["rows"],
    }
    atomic_write_json(out_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
