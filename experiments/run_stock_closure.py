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

from experiments.run_recurrent_governance import event_for_trace, selected_edge_states
from experiments.run_recurrent_softmax import adversarial_states
from src.ffn_counter import verify_decrementer
from src.fixed_state import (
    EVENT_COUNT,
    STATE_WIDTH,
    GovernanceEvent,
    GovernanceOutput,
    GovernanceState,
    decode_state_bits,
    encode_state_bits,
)
from src.recurrent_compiled import random_events, random_valid_states
from src.recurrent_reference import transition
from src.stock_governance_transformer import (
    StockGovernanceConfig,
    StockGovernanceTransformer,
    architecture_summary,
    compare_stock_reference,
    encode_slots,
    load_stock_governance_model,
    output_margins,
    save_stock_governance_model,
)

SCHEMA = "schmittformer.stock_transformer_closure.v1"


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


def edge_dataset() -> tuple[list[GovernanceState], list[GovernanceEvent]]:
    states = []
    events = []
    for state in selected_edge_states():
        for event in GovernanceEvent:
            states.append(state)
            events.append(event)
    return states, events


def adversarial_dataset() -> tuple[list[GovernanceState], list[GovernanceEvent]]:
    states = []
    events = []
    for state in adversarial_states():
        for event in GovernanceEvent:
            states.append(state)
            events.append(event)
    return states, events


def primitive_report(device: torch.device) -> dict[str, Any]:
    rows = []
    plan: list[tuple[str, torch.device]] = [("float64", torch.device("cpu"))]
    if device.type == "cuda":
        plan.append(("float32", device))
        plan.append(("float16", device))
        if torch.cuda.is_bf16_supported():
            plan.append(("bfloat16", device))
    else:
        plan.append(("float32", torch.device("cpu")))
    for dtype_name, dev in plan:
        dtype = dtype_from_name(dtype_name)
        for width in [8, 16]:
            rows.append(verify_decrementer(width, dtype=dtype, device=dev))
    return {"schema": SCHEMA, "rows": rows}


def gap_sweep(device: torch.device, gaps: list[float]) -> dict[str, Any]:
    states, events = edge_dataset()
    random_states = random_valid_states(20000, seed=4401)
    random_evs = random_events(20000, seed=4402)
    rows = []
    for gap in gaps:
        model = StockGovernanceTransformer(StockGovernanceConfig(score_gap=gap, dtype_name="float32")).to(device)
        edge = compare_stock_reference(states, events, model, batch_size=4096 if device.type == "cuda" else 512, device=device)
        random = compare_stock_reference(random_states, random_evs, model, batch_size=4096 if device.type == "cuda" else 512, device=device)
        classification = "NON_SATURATED_SOFTMAX_SUCCESS" if edge["passed"] and random["passed"] else "SEMANTIC_FAILURE"
        rows.append({"score_gap": gap, "classification": classification, "edge": edge, "random": random})
    return {"schema": SCHEMA, "rows": rows}


def governance_report(device: torch.device, random_samples: int) -> dict[str, Any]:
    model = StockGovernanceTransformer(StockGovernanceConfig(score_gap=8.0, dtype_name="float32")).to(device)
    edge_states, edge_events = edge_dataset()
    adv_states, adv_events = adversarial_dataset()
    random_states = random_valid_states(random_samples, seed=5501)
    random_evs = random_events(random_samples, seed=5502)
    start = time.time()
    edge = compare_stock_reference(edge_states, edge_events, model, batch_size=4096 if device.type == "cuda" else 512, device=device)
    adversarial = compare_stock_reference(adv_states, adv_events, model, batch_size=4096 if device.type == "cuda" else 512, device=device)
    random = compare_stock_reference(random_states, random_evs, model, batch_size=4096 if device.type == "cuda" else 512, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.time() - start
    return {
        "schema": SCHEMA,
        "edge": edge,
        "adversarial": adversarial,
        "random": random,
        "elapsed_seconds": elapsed,
        "passed": edge["passed"] and adversarial["passed"] and random["passed"],
    }


def precision_report(device: torch.device) -> dict[str, Any]:
    edge_states, edge_events = edge_dataset()
    adv_states, adv_events = adversarial_dataset()
    rand_states = random_valid_states(50000, seed=6601)
    rand_events = random_events(50000, seed=6602)
    plan: list[tuple[str, torch.device, int]] = [("float64", torch.device("cpu"), 512)]
    if device.type == "cuda":
        plan.append(("float32", device, 4096))
        plan.append(("float16", device, 4096))
        if torch.cuda.is_bf16_supported():
            plan.append(("bfloat16", device, 4096))
    else:
        plan.append(("float32", torch.device("cpu"), 512))
    rows = []
    for dtype_name, dev, batch in plan:
        try:
            model = StockGovernanceTransformer(StockGovernanceConfig(score_gap=8.0, dtype_name=dtype_name)).to(dev)
            rows.append(
                {
                    "dtype": dtype_name,
                    "device": str(dev),
                    "edge": compare_stock_reference(edge_states, edge_events, model, batch_size=batch, device=dev),
                    "adversarial": compare_stock_reference(adv_states, adv_events, model, batch_size=batch, device=dev),
                    "random50k": compare_stock_reference(rand_states, rand_events, model, batch_size=batch, device=dev),
                }
            )
        except Exception as exc:
            rows.append({"dtype": dtype_name, "device": str(dev), "passed": False, "error": str(exc)})
    return {"schema": SCHEMA, "rows": rows}


@torch.no_grad()
def longrun_report(device: torch.device, lengths: list[int]) -> dict[str, Any]:
    model = StockGovernanceTransformer(StockGovernanceConfig(score_gap=8.0, dtype_name="float32")).to(device)
    rows = []
    for length in lengths:
        state = GovernanceState(1, 65535, 255, 0, 0)
        first_divergence = None
        min_state_margin = float("inf")
        min_output_margin = float("inf")
        max_state_error = 0.0
        output_counts = {int(output): 0 for output in GovernanceOutput}
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.time()
        for index in range(length):
            event = event_for_trace("mixed_governance", index)
            expected = transition(state, event)
            slots = encode_slots(state, event, device=device, dtype=torch.float32)
            next_bits, logits = model(slots)
            actual_state = decode_state_bits(next_bits)
            actual_output = int(logits.argmax().detach().cpu().item())
            expected_bits = encode_state_bits(expected.next_state, device=device, dtype=torch.float32)
            state_margin = torch.abs(next_bits.detach().float() - 0.5).min()
            out_margin = output_margins(logits.reshape(1, -1), torch.tensor([int(expected.output)], device=device))[0]
            min_state_margin = min(min_state_margin, float(state_margin.detach().cpu().item()))
            min_output_margin = min(min_output_margin, float(out_margin.detach().cpu().item()))
            max_state_error = max(max_state_error, float(torch.abs(next_bits - expected_bits).max().detach().cpu().item()))
            output_counts[actual_output] += 1
            if actual_state != expected.next_state or actual_output != expected.output:
                first_divergence = {
                    "index": index,
                    "event": int(event),
                    "expected": expected.to_json(),
                    "actual_state": actual_state.to_json(),
                    "actual_output": actual_output,
                }
                break
            state = actual_state
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.time() - start
        rows.append(
            {
                "requested_logical_steps": int(length),
                "logical_steps": int(length if first_divergence is None else first_divergence["index"] + 1),
                "passed": first_divergence is None,
                "first_divergence": first_divergence,
                "physical_input_width": STATE_WIDTH + EVENT_COUNT,
                "final_state": state.to_json(),
                "min_state_bit_margin": min_state_margin,
                "min_output_margin": min_output_margin,
                "max_state_error_before_decode": max_state_error,
                "output_counts": {str(k): v for k, v in output_counts.items()},
                "elapsed_seconds": elapsed,
                "steps_per_second": (length if first_divergence is None else first_divergence["index"] + 1) / elapsed if elapsed else None,
            }
        )
    return {"schema": SCHEMA, "rows": rows}


def checkpoint_report(out_dir: Path, device: torch.device) -> dict[str, Any]:
    model = StockGovernanceTransformer(StockGovernanceConfig(score_gap=8.0, dtype_name="float32")).to(device)
    ckpt = out_dir / "stock_governance.pt"
    save_stock_governance_model(ckpt, model)
    size = ckpt.stat().st_size
    del model
    loaded = load_stock_governance_model(ckpt, device=device)
    states = random_valid_states(2048, seed=7701)
    events = random_events(2048, seed=7702)
    validation = compare_stock_reference(states, events, loaded, batch_size=1024, device=device)
    return {
        "schema": SCHEMA,
        "checkpoint": str(ckpt),
        "checkpoint_size_bytes": int(size),
        "validation": validation,
        "passed": validation["passed"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/stock_transformer")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--random-samples", type=int, default=200000)
    parser.add_argument("--long-steps", type=int, default=10000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    env = load_or_compute(out_dir / "manifest.json", args.force, environment)
    architecture = load_or_compute(
        out_dir / "architecture.json",
        args.force,
        lambda: {"schema": SCHEMA, **architecture_summary(StockGovernanceTransformer(StockGovernanceConfig()).to("cpu"))},
    )
    primitives = load_or_compute(out_dir / "primitive_verification.json", args.force, lambda: primitive_report(device))
    gap = load_or_compute(out_dir / "gap_sweep.json", args.force, lambda: gap_sweep(device, [2.0, 4.0, 6.0, 8.0]))
    governance = load_or_compute(out_dir / "governance.json", args.force, lambda: governance_report(device, args.random_samples))
    precision = load_or_compute(out_dir / "precision.json", args.force, lambda: precision_report(device))
    lengths = [10, 100, 1000, 10000]
    if args.long_steps not in lengths:
        lengths.append(args.long_steps)
    lengths = sorted({length for length in lengths if length <= args.long_steps})
    longrun = load_or_compute(out_dir / "longrun.json", args.force, lambda: longrun_report(device, lengths))
    checkpoint = load_or_compute(out_dir / "checkpoint.json", args.force, lambda: checkpoint_report(out_dir, device))
    aggregate = {
        "schema": SCHEMA,
        "environment": env,
        "architecture": architecture,
        "primitive_passed": all(row["passed"] for row in primitives["rows"]),
        "gap8_passed": next(row for row in gap["rows"] if row["score_gap"] == 8.0)["classification"],
        "governance_passed": governance["passed"],
        "precision_passed": all(
            row.get("edge", {}).get("passed", False)
            and row.get("adversarial", {}).get("passed", False)
            and row.get("random50k", {}).get("passed", False)
            for row in precision["rows"]
        ),
        "longrun_passed": all(row["passed"] for row in longrun["rows"]),
        "longest_logical_trace": max(row["logical_steps"] for row in longrun["rows"]),
        "checkpoint_passed": checkpoint["passed"],
    }
    atomic_write_json(out_dir / "aggregate.json", aggregate)
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
