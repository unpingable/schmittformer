from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from src.circuit_compiled import (
    CompiledCircuitBreakerTransformer,
    evaluate_compiled_long_traces,
    verify_compiled_history_equivalence,
    verify_compiled_transition_graph,
)
from src.circuit_hybrid import CircuitHybridConfig, evaluate_hybrid, train_hybrid
from src.circuit_learned import (
    CircuitLearnedConfig,
    evaluate_sequences,
    learned_history_equivalence,
    minimize_sequence_for_predicate,
    predict_state_ids,
    train_circuit_model,
)
from src.circuit_reference import (
    Event,
    collect_equivalent_histories,
    initial_state,
    invariant_violations,
    reachable_graph,
    run_controller,
    state_id_maps,
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def environment_report() -> dict[str, Any]:
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
            timeout=5,
        ).strip()
    except Exception as exc:
        gpu = f"unavailable: {exc}"
    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": gpu,
    }


def evaluation_sequences() -> dict[str, list[int]]:
    trip_recover = [1, 1, 1, 2, 2, 2, 2, 2, 0, 0]
    threshold_edge = [0, 1, 0, 1, 2, 0, 1, 2]
    return {
        "natural_32": [0, 0, 2, 0, 1, 0, 0, 2] * 4,
        "natural_64": [0, 0, 2, 0, 1, 0, 0, 2] * 8,
        "healthy_128": [0, 2, 0, 0] * 32,
        "sustained_unknown_256": [2] * 256,
        "sustained_failure_128": [1] * 128,
        "repeated_trip_recover_256": trip_recover * 26,
        "threshold_edge_256": threshold_edge * 32,
        "adversarial_mix_1024": (threshold_edge + trip_recover + [2] * 11 + [0, 2, 0, 1, 1]) * 31,
    }


def first_illegal_predicate(model, sequence: list[int]) -> bool:
    states, _ = state_id_maps()
    pred_ids = predict_state_ids(model, sequence)
    prev = initial_state()
    for event, pred_id in zip(sequence, pred_ids):
        pred_state = states[pred_id]
        if invariant_violations(prev, event, pred_state):
            return True
        prev = pred_state
    return False


def run_learned_variant(out_dir: Path, config: CircuitLearnedConfig, prefix: str) -> dict[str, Any]:
    model, train_metrics = train_circuit_model(config)
    scenarios = evaluate_sequences(model, evaluation_sequences())
    groups = collect_equivalent_histories(max_per_state=6)
    suffixes = [[0, 0], [1], [2] * 8, [1, 1, 1, 2, 2, 2, 2, 2, 0, 0], [0, 2, 0, 1]]
    equivalence = learned_history_equivalence(model, groups, suffixes)
    counterexamples = {}
    for name, seq in evaluation_sequences().items():
        if first_illegal_predicate(model, seq):
            minimized = minimize_sequence_for_predicate(seq, lambda s: first_illegal_predicate(model, s))
            counterexamples[name] = {"original_length": len(seq), "minimized_length": len(minimized), "sequence": minimized}
    payload = {
        "train": train_metrics,
        "scenarios": scenarios,
        "history_equivalence": equivalence,
        "counterexamples": counterexamples,
    }
    write_json(out_dir / f"circuit_learned_{prefix}.json", payload)
    return payload


def run_compiled(out_dir: Path) -> dict[str, Any]:
    start = time.time()
    graph = reachable_graph()
    write_json(out_dir / "circuit_graph.json", graph)
    transition_check = verify_compiled_transition_graph(max_histories_per_state=8)
    equivalence = verify_compiled_history_equivalence(max_histories_per_state=8)
    long_traces = evaluate_compiled_long_traces()
    payload = {
        "graph_counts": {
            "syntactic_normalized_states": graph["syntactic_normalized_states"],
            "reachable_states": graph["reachable_states"],
            "reachable_transitions": graph["reachable_transitions"],
        },
        "transition_check": transition_check,
        "history_equivalence": equivalence,
        "long_traces": long_traces,
        "elapsed_seconds": time.time() - start,
        "representation": {
            "input_tokens": 3,
            "state_tokens": graph["reachable_states"],
            "token_overhead": "one generated complete-state token per input token plus initial state token",
            "attention": "hard argmax over prior state-token positions selects the latest complete logical state",
            "lookup": "deterministic transition table over reachable_state_id x event_id",
            "decoding": "greedy argmax emits the next complete state token",
        },
    }
    write_json(out_dir / "circuit_compiled.json", payload)
    return payload


def run_hybrid_experiment(out_dir: Path, classifier_steps: int, e2e_steps: int, train_len: int) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = CircuitHybridConfig(classifier_steps=classifier_steps, e2e_steps=e2e_steps, train_len=train_len)
    classifier, e2e, train_metrics = train_hybrid(config, device)
    scenarios = {
        "in_distribution": evaluate_hybrid(classifier, e2e, config, device, sigma=config.sigma, seq_len=128, batches=4, distribution="natural"),
        "shifted_noise": evaluate_hybrid(classifier, e2e, config, device, sigma=0.75, bias=0.15, seq_len=256, batches=4, distribution="natural"),
        "adversarial_events": evaluate_hybrid(classifier, e2e, config, device, sigma=0.55, bias=0.0, seq_len=256, batches=4, distribution="adversarial"),
        "long_duration": evaluate_hybrid(classifier, e2e, config, device, sigma=0.50, bias=0.05, seq_len=1024, batches=1, distribution="balanced"),
    }
    payload = {"train": train_metrics, "scenarios": scenarios}
    write_json(out_dir / "circuit_hybrid.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--natural-steps", type=int, default=1500)
    parser.add_argument("--balanced-steps", type=int, default=1200)
    parser.add_argument("--classifier-steps", type=int, default=1000)
    parser.add_argument("--e2e-steps", type=int, default=1500)
    parser.add_argument("--train-len", type=int, default=64)
    parser.add_argument("--skip-learned", action="store_true")
    parser.add_argument("--skip-hybrid", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "circuit_environment.json", environment_report())
    compiled = run_compiled(args.out_dir)
    learned = {}
    if not args.skip_learned:
        learned["natural"] = run_learned_variant(
            args.out_dir,
            CircuitLearnedConfig(steps=args.natural_steps, distribution="natural", train_len=args.train_len),
            "natural",
        )
        learned["adversarial"] = run_learned_variant(
            args.out_dir,
            CircuitLearnedConfig(seed=22, steps=args.balanced_steps, distribution="adversarial", train_len=args.train_len),
            "adversarial",
        )
    hybrid = None if args.skip_hybrid else run_hybrid_experiment(args.out_dir, args.classifier_steps, args.e2e_steps, args.train_len)
    write_json(
        args.out_dir / "circuit_summary.json",
        {
            "compiled_transition_check_passed": compiled["transition_check"]["passed"],
            "compiled_history_equivalence_passed": compiled["history_equivalence"]["passed"],
            "learned_variants": list(learned.keys()),
            "hybrid_ran": hybrid is not None,
        },
    )


if __name__ == "__main__":
    main()
