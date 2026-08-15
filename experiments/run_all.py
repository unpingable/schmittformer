from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from src.compiled import (
    CompiledConfig,
    CompiledHysteresisTransformer,
    predict_compiled,
    verify_reachable_transitions,
)
from src.evaluate import count_illegal_transitions, evaluate_predictor
from src.hybrid import (
    HybridConfig,
    evaluate_hybrid_models,
    train_hybrid_system,
)
from src.learned import (
    LearnedConfig,
    predict_learned,
    states_from_inputs_tensor,
    train_model,
)
from src.reference import State, exhaustive_sequences, run_hysteresis


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def environment_report() -> dict[str, Any]:
    gpu = None
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=5,
        ).strip()
        gpu = raw
    except Exception as exc:  # pragma: no cover - diagnostic only.
        gpu = f"unavailable: {exc}"
    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": gpu,
    }


def numbers_to_digit_batch(start: int, count: int, length: int, device: torch.device) -> torch.Tensor:
    nums = torch.arange(start, start + count, device=device, dtype=torch.long)
    columns = []
    for power in reversed(range(length)):
        columns.append((nums // (10**power)) % 10)
    return torch.stack(columns, dim=1)


def compiled_exhaustive(max_len: int, batch_size: int, device: torch.device) -> dict[str, Any]:
    model = CompiledHysteresisTransformer().to(device)
    start_time = time.time()
    failures = []
    sequences = 0
    tokens = 0

    with torch.no_grad():
        for initial in (State.OFF, State.ON):
            for length in range(1, max_len + 1):
                total = 10**length
                for start in range(0, total, batch_size):
                    count = min(batch_size, total - start)
                    batch = numbers_to_digit_batch(start, count, length, device)
                    initial_states = torch.full(
                        (count,),
                        int(initial),
                        dtype=torch.long,
                        device=device,
                    )
                    expected = states_from_inputs_tensor(batch, initial_states)
                    actual = model(batch, initial_state=initial_states).argmax(dim=-1)
                    bad = actual != expected
                    if bad.any():
                        row = int(bad.any(dim=1).nonzero()[0].item())
                        col = int(bad[row].nonzero()[0].item())
                        failures.append(
                            {
                                "initial_state": int(initial),
                                "length": length,
                                "inputs": batch[row].cpu().tolist(),
                                "expected": expected[row].cpu().tolist(),
                                "actual": actual[row].cpu().tolist(),
                                "index": col,
                            }
                        )
                        break
                    sequences += count
                    tokens += count * length
                if failures:
                    break
            if failures:
                break

    return {
        "attention": "hard",
        "max_len": max_len,
        "initial_states": [0, 1],
        "sequences": sequences,
        "tokens": tokens,
        "passed": not failures,
        "failures": failures,
        "elapsed_seconds": time.time() - start_time,
    }


def run_compiled(out_dir: Path, max_len: int, batch_size: int) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_inputs = [2, 5, 8, 6, 4, 3, 5, 7]
    repeated_thresholds = [2, 7, 6, 5, 4, 3, 4, 5, 8, 6] * 100
    near_thresholds = [3, 4, 6, 7, 6, 4, 3, 7] * 128
    long_random = torch.randint(0, 10, (10000,)).tolist()

    def scenario(inputs: list[int], initial: State = State.OFF) -> dict[str, Any]:
        expected = run_hysteresis(inputs, initial)
        actual = predict_compiled(inputs, initial, device=device)
        first_bad = next((i for i, (a, b) in enumerate(zip(expected, actual)) if a != b), None)
        return {
            "length": len(inputs),
            "exact": expected == actual,
            "first_divergence": first_bad,
            "illegal_transitions": count_illegal_transitions(inputs, actual, initial),
        }

    dtype_results = {}
    for dtype in (torch.float32, torch.float64):
        dtype_results[str(dtype)] = verify_reachable_transitions(dtype=dtype, device=device)
    if torch.cuda.is_available():
        dtype_results[str(torch.float16)] = verify_reachable_transitions(
            dtype=torch.float16,
            device=device,
        )

    payload = {
        "sample": {
            "inputs": sample_inputs,
            "expected": run_hysteresis(sample_inputs),
            "actual": predict_compiled(sample_inputs, device=device),
        },
        "reachable_transition_check": dtype_results,
        "exhaustive": compiled_exhaustive(max_len, batch_size, device),
        "deadband_off": {
            "inputs": [4, 5, 6, 5, 4, 6] * 8,
            "actual": predict_compiled([4, 5, 6, 5, 4, 6] * 8, State.OFF, device=device),
        },
        "deadband_on": {
            "inputs": [4, 5, 6, 5, 4, 6] * 8,
            "actual": predict_compiled([4, 5, 6, 5, 4, 6] * 8, State.ON, device=device),
        },
        "scenarios": {
            "repeated_threshold_crossings": scenario(repeated_thresholds),
            "near_thresholds": scenario(near_thresholds),
            "long_random": scenario(long_random),
            "deadband_off": scenario([4, 5, 6, 5, 4, 6] * 256, State.OFF),
            "deadband_on": scenario([4, 5, 6, 5, 4, 6] * 256, State.ON),
        },
    }
    write_json(out_dir / "compiled.json", payload)
    return payload


def learned_eval_cases(model, out_dir: Path) -> dict[str, Any]:
    device = next(model.parameters()).device
    predictor = lambda seq, init: predict_learned(model, seq, init, device=device)
    exhaustive = {
        "initial_off_len4": evaluate_predictor(
            predictor,
            exhaustive_sequences(4),
            State.OFF,
        ).to_json(),
        "initial_on_len4": evaluate_predictor(
            predictor,
            exhaustive_sequences(4),
            State.ON,
        ).to_json(),
    }
    adversarial_inputs = {
        "example": [2, 5, 8, 6, 4, 3, 5, 7],
        "deadband_off": [4, 5, 6, 5, 4, 6] * 16,
        "threshold_crossings": [2, 7, 6, 5, 4, 3, 4, 5, 8, 6] * 12,
        "near_thresholds": [3, 4, 6, 7, 6, 4, 3, 7] * 16,
        "long_random": torch.randint(0, 10, (512,)).tolist(),
    }
    adversarial = {}
    for name, inputs in adversarial_inputs.items():
        expected = run_hysteresis(inputs)
        actual = predictor(inputs, State.OFF)
        first_bad = next((i for i, (a, b) in enumerate(zip(expected, actual)) if a != b), None)
        adversarial[name] = {
            "length": len(inputs),
            "exact": expected == actual,
            "token_accuracy": sum(a == b for a, b in zip(expected, actual)) / len(inputs),
            "first_divergence": first_bad,
        }
    payload = {"exhaustive": exhaustive, "adversarial": adversarial}
    write_json(out_dir / "learned_eval.json", payload)
    return payload


def run_learned(out_dir: Path, steps: int, train_len: int) -> dict[str, Any]:
    config = LearnedConfig(steps=steps, train_len=train_len)
    model, train_metrics = train_model(config)
    eval_metrics = learned_eval_cases(model, out_dir)
    torch.save(
        {"config": train_metrics["config"], "state_dict": model.state_dict()},
        out_dir / "learned_model.pt",
    )
    payload = {"train": train_metrics, "eval": eval_metrics}
    write_json(out_dir / "learned.json", payload)
    return payload


def run_hybrid(out_dir: Path, classifier_steps: int, e2e_steps: int, train_len: int) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = HybridConfig(
        classifier_steps=classifier_steps,
        e2e_steps=e2e_steps,
        train_len=train_len,
    )
    classifier, e2e, train_metrics = train_hybrid_system(config, device)
    scenarios = {
        "in_distribution": evaluate_hybrid_models(
            classifier,
            e2e,
            config,
            device,
            sigma=config.sigma,
            seq_len=128,
            batches=8,
        ),
        "shifted_noise": evaluate_hybrid_models(
            classifier,
            e2e,
            config,
            device,
            sigma=0.80,
            bias=0.25,
            seq_len=256,
            batches=8,
        ),
        "near_thresholds": evaluate_hybrid_models(
            classifier,
            e2e,
            config,
            device,
            sigma=0.60,
            bias=0.10,
            seq_len=256,
            batches=8,
            near_threshold_prob=1.0,
        ),
        "long_duration": evaluate_hybrid_models(
            classifier,
            e2e,
            config,
            device,
            sigma=0.50,
            seq_len=1024,
            batches=2,
            near_threshold_prob=0.5,
        ),
    }
    payload = {"train": train_metrics, "scenarios": scenarios}
    write_json(out_dir / "hybrid.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--compiled-max-len", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learned-steps", type=int, default=800)
    parser.add_argument("--hybrid-steps", type=int, default=800)
    parser.add_argument("--classifier-steps", type=int, default=600)
    parser.add_argument("--train-len", type=int, default=16)
    parser.add_argument("--skip-learned", action="store_true")
    parser.add_argument("--skip-hybrid", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "environment.json", environment_report())
    compiled = run_compiled(args.out_dir, args.compiled_max_len, args.batch_size)
    learned = None if args.skip_learned else run_learned(
        args.out_dir,
        args.learned_steps,
        args.train_len,
    )
    hybrid = None if args.skip_hybrid else run_hybrid(
        args.out_dir,
        args.classifier_steps,
        args.hybrid_steps,
        args.train_len,
    )
    write_json(
        args.out_dir / "summary.json",
        {
            "compiled_passed": compiled["exhaustive"]["passed"],
            "learned_ran": learned is not None,
            "hybrid_ran": hybrid is not None,
        },
    )


if __name__ == "__main__":
    main()
