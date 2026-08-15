
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from src.circuit_reference import collect_equivalent_histories, initial_state as circuit_initial, run_controller
from src.circuit_softmax import (
    SoftmaxCircuitConfig,
    SoftmaxCircuitBreakerTransformer,
    evaluate_softmax_circuit_long_traces,
    verify_softmax_circuit_history_equivalence,
    verify_softmax_circuit_transition_graph,
)
from src.hysteresis_softmax import SoftmaxHysteresisConfig, SoftmaxHysteresisTransformer, verify_softmax_hysteresis_transitions
from src.reference import State, run_hysteresis
from src.softmax_attention import (
    EFFECTIVELY_HARD,
    NUMERIC_FAILURE,
    PASS_EXACT,
    SEMANTIC_FAILURE,
    SoftmaxAttentionConfig,
    classify_softmax_run,
    dtype_supported,
    finite_geometric_stale_bound,
    geometric_stale_state_unnormalized_bound,
    geometric_state_correct_probability_lower_bound,
    latest_state_attention,
    non_state_unnormalized_bound,
    simple_correct_probability_lower_bound,
    simple_leakage_upper_bound,
    state_leakage_sufficient_for_decoding,
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def dtype_from_name(name: str) -> torch.dtype:
    return getattr(torch, name)


def environment_report() -> dict[str, Any]:
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
            timeout=5,
        ).strip()
    except Exception as exc:
        gpu = f"unavailable: {exc}"
    try:
        hard_revision = subprocess.check_output(["git", "rev-parse", "hard-attention-success"], text=True).strip()
    except Exception as exc:
        hard_revision = f"unavailable: {exc}"
    try:
        current_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception as exc:
        current_revision = f"unavailable: {exc}"
    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": gpu,
        "hard_attention_checkpoint": hard_revision,
        "current_revision_at_run_start": current_revision,
    }


def numbers_to_digit_batch(start: int, count: int, length: int, device: torch.device) -> torch.Tensor:
    nums = torch.arange(start, start + count, device=device, dtype=torch.long)
    columns = []
    for power in reversed(range(length)):
        columns.append((nums // (10**power)) % 10)
    return torch.stack(columns, dim=1)


def hysteresis_expected_tensor(inputs: torch.Tensor, initial: State) -> torch.Tensor:
    prev = torch.full((inputs.shape[0],), int(initial), dtype=torch.long, device=inputs.device)
    outs = []
    for t in range(inputs.shape[1]):
        x = inputs[:, t]
        prev = torch.where(x >= 7, torch.ones_like(prev), torch.where(x <= 3, torch.zeros_like(prev), prev))
        outs.append(prev)
    return torch.stack(outs, dim=1)


def batched_hysteresis_decode(model: SoftmaxHysteresisTransformer, inputs: torch.Tensor, initial: State) -> torch.Tensor:
    batch = inputs.shape[0]
    tokens = torch.full((batch, 1), model.state_token(initial), dtype=torch.long, device=inputs.device)
    outputs = []
    for t in range(inputs.shape[1]):
        tokens = torch.cat([tokens, inputs[:, t : t + 1]], dim=1)
        with torch.no_grad():
            next_state = model.next_state_logits(tokens).argmax(dim=-1)
        outputs.append(next_state)
        tokens = torch.cat([tokens, next_state[:, None] + model.state_token_offset], dim=1)
    return torch.stack(outputs, dim=1)


def hysteresis_exhaustive(
    state_record_gap: float,
    non_state_penalty: float,
    dtype: torch.dtype,
    max_len: int,
    batch_size: int,
) -> dict[str, Any]:
    device = torch.device("cpu")
    model = SoftmaxHysteresisTransformer(SoftmaxHysteresisConfig(state_record_gap, non_state_penalty, dtype)).to(device)
    failures = []
    sequences = 0
    tokens_count = 0
    start_time = time.time()
    finite = True
    effectively_hard = True
    min_margin = float("inf")
    for initial in (State.OFF, State.ON):
        for length in range(1, max_len + 1):
            total = 10**length
            for start in range(0, total, batch_size):
                count = min(batch_size, total - start)
                batch = numbers_to_digit_batch(start, count, length, device)
                expected = hysteresis_expected_tensor(batch, initial)
                actual = batched_hysteresis_decode(model, batch, initial)
                bad = actual != expected
                if bad.any():
                    row = int(bad.any(dim=1).nonzero()[0].item())
                    col = int(bad[row].nonzero()[0].item())
                    failures.append({
                        "initial_state": int(initial),
                        "length": length,
                        "inputs": batch[row].tolist(),
                        "expected": expected[row].tolist(),
                        "actual": actual[row].tolist(),
                        "index": col,
                    })
                    break
                # Inspect the final step of the first row as a cheap numeric sentinel.
                probe_tokens = model.encode_history_from_reference(batch[0, :-1].tolist(), initial)
                _, debug = model.next_state_logits(torch.tensor([*probe_tokens, int(batch[0, -1].item())]), return_debug=True)
                finite = finite and bool(debug["finite"].item())
                effectively_hard = effectively_hard and bool(debug["effectively_hard"].item())
                min_margin = min(min_margin, float(debug["decision_margin"].item()))
                sequences += count
                tokens_count += count * length
            if failures:
                break
        if failures:
            break
    return {
        "passed": not failures,
        "max_len": max_len,
        "sequences": sequences,
        "tokens": tokens_count,
        "failures": failures[:20],
        "elapsed_seconds": time.time() - start_time,
        "finite": finite,
        "effectively_hard_sentinel": effectively_hard,
        "min_decision_margin_sentinel": min_margin,
    }


def hysteresis_equivalence(state_record_gap: float, non_state_penalty: float, dtype: torch.dtype) -> dict[str, Any]:
    model = SoftmaxHysteresisTransformer(SoftmaxHysteresisConfig(state_record_gap, non_state_penalty, dtype))
    groups = {
        State.OFF: [[], [4] * 64, [7, 3], [7, 5, 6, 3] + [4, 5, 6] * 16, [7, 3] * 50],
        State.ON: [[7], [7] + [4, 5, 6] * 64, [7, 3, 7], [7, 3] * 50 + [7]],
    }
    suffixes = [[4, 5, 6, 5], [7], [3], [7, 6, 5, 4, 3], [4, 5, 6] * 20]
    semantic_violations = []
    latent_diffs = []
    comparisons = 0
    for state, histories in groups.items():
        for suffix in suffixes:
            base_outputs = None
            base_mass = None
            for hist in histories:
                tokens = model.encode_history_from_reference(hist)
                outputs, _ = model.decode_from_tokens(tokens, suffix)
                probe = [*tokens, suffix[0]]
                _, debug = model.next_state_logits(torch.tensor(probe), return_debug=True)
                mass = debug["state_masses"].detach().cpu()
                if base_outputs is None:
                    base_outputs = tuple(outputs)
                    base_mass = mass
                else:
                    comparisons += 1
                    if tuple(outputs) != base_outputs:
                        semantic_violations.append({"state": int(state), "history": hist, "suffix": suffix, "outputs": outputs, "base_outputs": list(base_outputs)})
                    latent_diffs.append(float((mass - base_mass).abs().max().item()))
    return {
        "passed": not semantic_violations,
        "comparisons": comparisons,
        "history_equivalence_violations": len(semantic_violations),
        "examples": semantic_violations[:20],
        "max_latent_state_mass_diff": max(latent_diffs) if latent_diffs else 0.0,
        "mean_latent_state_mass_diff": sum(latent_diffs) / len(latent_diffs) if latent_diffs else 0.0,
    }


def hysteresis_long_traces(state_record_gap: float, non_state_penalty: float, dtype: torch.dtype) -> dict[str, Any]:
    model = SoftmaxHysteresisTransformer(SoftmaxHysteresisConfig(state_record_gap, non_state_penalty, dtype))
    scenarios = {
        "deadband_off_4096": ([4, 5, 6, 5] * 1024, State.OFF),
        "deadband_on_4096": ([4, 5, 6, 5] * 1024, State.ON),
        "threshold_crossings_2048": ([2, 7, 6, 5, 4, 3, 4, 5] * 256, State.OFF),
        "long_random_4096": (torch.randint(0, 10, (4096,)).tolist(), State.OFF),
    }
    out = {}
    for name, (inputs, initial) in scenarios.items():
        expected = run_hysteresis(inputs, initial)
        actual, _ = model.decode_inputs(inputs, initial)
        first_bad = next((i for i, (a, b) in enumerate(zip(expected, actual)) if a != b), None)
        out[name] = {"length": len(inputs), "exact": first_bad is None, "first_divergence": first_bad}
    return out


def synthetic_attention_probe(controller: str, context_updates: int, state_record_gap: float, non_state_penalty: float, dtype: torch.dtype) -> dict[str, Any]:
    if controller == "hysteresis":
        model = SoftmaxHysteresisTransformer(SoftmaxHysteresisConfig(state_record_gap, non_state_penalty, dtype))
        latest = 1
        stale = 0
        current_input = 5
    else:
        model = SoftmaxCircuitBreakerTransformer(SoftmaxCircuitConfig(state_record_gap, non_state_penalty, dtype))
        latest_state = circuit_initial()
        latest = model.state_to_id[latest_state]
        # Pick an older state whose FAILURE transition differs from CLOSED+FAILURE.
        stale = next(i for i, s in enumerate(model.states) if model.transition_table[i, 1].item() != model.transition_table[latest, 1].item())
        current_input = 1
    tokens = []
    for _ in range(max(0, context_updates - 1)):
        tokens.extend([model.state_token(stale), current_input])
    tokens.extend([model.state_token(latest), current_input])
    try:
        logits, debug = model.next_state_logits(torch.tensor(tokens, dtype=torch.long), return_debug=True)
        actual = int(logits.argmax(dim=-1).item())
        latest_next = int(model.transition_table[latest, current_input].item())
        exact = actual == latest_next
        finite = bool(debug["finite"].item())
        effectively_hard = bool(debug["effectively_hard"].item())
        classification = classify_softmax_run(exact, finite, effectively_hard)
        return {
            "classification": classification,
            "exact": exact,
            "actual": actual,
            "expected_from_latest": latest_next,
            "correct_mass": float(debug["correct_mass"].item()),
            "stale_state_mass": float(debug["stale_state_mass"].item()),
            "non_state_mass": float(debug["non_state_mass"].item()),
            "decision_margin": float(debug["decision_margin"].item()),
            "effectively_hard": effectively_hard,
            "finite": finite,
            "token_length": len(tokens),
        }
    except Exception as exc:
        return {"classification": NUMERIC_FAILURE, "error": repr(exc), "token_length": len(tokens)}


def run_sweep(out_dir: Path, dtypes: list[torch.dtype], deltas: list[float], contexts: list[int]) -> dict[str, Any]:
    rows = []
    for controller in ("hysteresis", "circuit"):
        for dtype in dtypes:
            if not dtype_supported(dtype):
                rows.append({"controller": controller, "dtype": dtype_name(dtype), "classification": NUMERIC_FAILURE, "error": "dtype unsupported"})
                continue
            for delta in deltas:
                penalty = 2.0 * delta
                for context in contexts:
                    probe = synthetic_attention_probe(controller, context, delta, penalty, dtype)
                    rows.append({
                        "controller": controller,
                        "dtype": dtype_name(dtype),
                        "state_record_gap": delta,
                        "non_state_penalty": penalty,
                        "context_updates": context,
                        "theory_stale_bound_infinite": geometric_stale_state_unnormalized_bound(delta),
                        "theory_stale_bound_finite": finite_geometric_stale_bound(delta, max(context - 1, 0)),
                        "theory_non_state_bound": non_state_unnormalized_bound(delta, penalty),
                        "theory_sufficient": state_leakage_sufficient_for_decoding(delta),
                        **probe,
                    })
    payload = {"rows": rows, "dtypes": [dtype_name(d) for d in dtypes], "deltas": deltas, "contexts": contexts}
    write_json(out_dir / "softmax_sweep.json", payload)
    return payload


def first_passing_margin(rows: list[dict[str, Any]], controller: str, dtype: str, max_context: int) -> float | None:
    candidates = sorted({row["state_record_gap"] for row in rows if row.get("controller") == controller and row.get("dtype") == dtype})
    for delta in candidates:
        selected = [row for row in rows if row.get("controller") == controller and row.get("dtype") == dtype and row.get("state_record_gap") == delta and row.get("context_updates", 0) <= max_context]
        if selected and all(row.get("classification") in {PASS_EXACT, EFFECTIVELY_HARD} for row in selected):
            return float(delta)
    return None


def run_precision(out_dir: Path, dtypes: list[torch.dtype], deltas: list[float]) -> dict[str, Any]:
    rows = []
    for dtype in dtypes:
        if not dtype_supported(dtype):
            rows.append({"dtype": dtype_name(dtype), "supported": False})
            continue
        for delta in deltas:
            penalty = 2.0 * delta
            h = verify_softmax_hysteresis_transitions(delta, penalty, dtype)
            try:
                c = verify_softmax_circuit_transition_graph(delta, penalty, dtype, max_histories_per_state=2)
            except Exception as exc:
                c = {"passed": False, "finite": False, "error": repr(exc), "effectively_hard": False, "min_decision_margin": None}
            rows.append({
                "dtype": dtype_name(dtype),
                "supported": True,
                "state_record_gap": delta,
                "non_state_penalty": penalty,
                "hysteresis_passed": h["passed"],
                "hysteresis_effectively_hard": h["effectively_hard"],
                "hysteresis_min_decision_margin": h["min_decision_margin"],
                "hysteresis_max_stale_state_mass": h["max_stale_state_mass"],
                "hysteresis_max_non_state_mass": h["max_non_state_mass"],
                "circuit_passed": c.get("passed", False),
                "circuit_effectively_hard": c.get("effectively_hard", False),
                "circuit_min_decision_margin": c.get("min_decision_margin"),
                "circuit_max_stale_state_mass": c.get("max_stale_state_mass"),
                "circuit_max_non_state_mass": c.get("max_non_state_mass"),
                "circuit_error": c.get("error"),
            })
    payload = {"rows": rows}
    write_json(out_dir / "softmax_precision.json", payload)
    return payload


def run_counterexample_search(out_dir: Path, deltas: list[float], contexts: list[int]) -> dict[str, Any]:
    examples = []
    for controller in ("hysteresis", "circuit"):
        for delta in deltas:
            for context in contexts:
                probe = synthetic_attention_probe(controller, context, delta, 2.0 * delta, torch.float64)
                if probe.get("classification") == SEMANTIC_FAILURE:
                    # The constructed case is already minimal-ish: shrink context while preserving failure.
                    lo = 1
                    hi = context
                    best = context
                    while lo <= hi:
                        mid = (lo + hi) // 2
                        mid_probe = synthetic_attention_probe(controller, mid, delta, 2.0 * delta, torch.float64)
                        if mid_probe.get("classification") == SEMANTIC_FAILURE:
                            best = mid
                            hi = mid - 1
                        else:
                            lo = mid + 1
                    best_probe = synthetic_attention_probe(controller, best, delta, 2.0 * delta, torch.float64)
                    examples.append({"controller": controller, "state_record_gap": delta, "first_failing_context_updates": best, **best_probe})
                    break
    payload = {"counterexamples": examples}
    write_json(out_dir / "softmax_counterexamples.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--hysteresis-max-len", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    env = environment_report()
    write_json(args.out_dir / "softmax_environment.json", env)

    dtypes = [torch.float64, torch.float32]
    for candidate in (torch.bfloat16, torch.float16):
        if dtype_supported(candidate):
            dtypes.append(candidate)
    deltas = [0.25, 0.5, 0.69, 0.7, 1.0, 1.5, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
    contexts = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    theory = {
        "simple_bound_examples": [
            {"delta": d, "competitors": n, "p_correct_lower": simple_correct_probability_lower_bound(d, n), "leakage_upper": simple_leakage_upper_bound(d, n)}
            for d in [1.0, 2.0, 4.0]
            for n in [1, 10, 1000]
        ],
        "geometric_bound_examples": [
            {"state_record_gap": d, "stale_unnormalized_bound": geometric_stale_state_unnormalized_bound(d), "p_correct_lower_ignoring_nonstate": geometric_state_correct_probability_lower_bound(d), "sufficient_for_decoding": state_leakage_sufficient_for_decoding(d), "non_state_bound_with_penalty_2delta": non_state_unnormalized_bound(d, 2.0 * d)}
            for d in deltas
        ],
        "critical_state_record_gap_ln2": math.log(2.0),
    }
    write_json(args.out_dir / "softmax_theory.json", theory)

    good_delta = 2.0
    good_penalty = 4.0
    h_transition = verify_softmax_hysteresis_transitions(good_delta, good_penalty, torch.float64)
    h_exhaustive = hysteresis_exhaustive(good_delta, good_penalty, torch.float64, args.hysteresis_max_len, args.batch_size)
    h_equivalence = hysteresis_equivalence(good_delta, good_penalty, torch.float64)
    h_long = hysteresis_long_traces(good_delta, good_penalty, torch.float64)
    hysteresis_payload = {
        "config": {"state_record_gap": good_delta, "non_state_penalty": good_penalty, "dtype": "float64"},
        "transition_check": h_transition,
        "exhaustive": h_exhaustive,
        "history_equivalence": h_equivalence,
        "long_traces": h_long,
    }
    write_json(args.out_dir / "softmax_hysteresis.json", hysteresis_payload)

    c_transition = verify_softmax_circuit_transition_graph(good_delta, good_penalty, torch.float64, max_histories_per_state=8)
    c_equivalence = verify_softmax_circuit_history_equivalence(good_delta, good_penalty, torch.float64, max_histories_per_state=8)
    c_long = evaluate_softmax_circuit_long_traces(good_delta, good_penalty, torch.float64)
    circuit_payload = {
        "config": {"state_record_gap": good_delta, "non_state_penalty": good_penalty, "dtype": "float64"},
        "transition_check": c_transition,
        "history_equivalence": c_equivalence,
        "long_traces": c_long,
    }
    write_json(args.out_dir / "softmax_circuit.json", circuit_payload)

    equivalence_payload = {"hysteresis": h_equivalence, "circuit": c_equivalence}
    write_json(args.out_dir / "softmax_equivalence.json", equivalence_payload)

    sweep = run_sweep(args.out_dir, dtypes, deltas, contexts)
    precision = run_precision(args.out_dir, dtypes, deltas)
    counterexamples = run_counterexample_search(args.out_dir, [0.25, 0.5, 0.69, 0.7, 1.0], contexts)
    summary = {
        "hard_attention_checkpoint": env["hard_attention_checkpoint"],
        "hysteresis_passed": h_transition["passed"] and h_exhaustive["passed"] and h_equivalence["passed"] and all(v["exact"] for v in h_long.values()),
        "circuit_passed": c_transition["passed"] and c_equivalence["passed"] and all(v["exact_state"] for v in c_long.values()),
        "good_config": {"state_record_gap": good_delta, "non_state_penalty": good_penalty, "dtype": "float64"},
        "first_passing_margin_by_controller_dtype_16384": {
            f"{controller}/{dtype_name(dtype)}": first_passing_margin(sweep["rows"], controller, dtype_name(dtype), 16384)
            for controller in ("hysteresis", "circuit")
            for dtype in dtypes
        },
        "counterexamples_found": len(counterexamples["counterexamples"]),
    }
    write_json(args.out_dir / "softmax_summary.json", summary)


if __name__ == "__main__":
    main()
