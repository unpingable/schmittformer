from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from src.compiled_counter import verify_counter

SCHEMA = "schmittformer.recurrent_counter_verification.v1"


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
    }


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float64": torch.float64,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/recurrent")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
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
    load_or_compute_json(out_dir / "counter_environment.json", args.force, environment)

    load_or_compute_json(out_dir / "counter8.json", args.force, lambda: verify_counter(8, device=device, dtype=torch.float32, exhaustive=True))
    load_or_compute_json(out_dir / "counter16.json", args.force, lambda: verify_counter(16, device=device, dtype=torch.float32, exhaustive=True))

    def compute_width_scaling() -> dict[str, Any]:
        width_rows = []
        for width in [4, 8, 16, 32]:
            exhaustive = width <= 16
            width_rows.append(
                verify_counter(
                    width,
                    device=device,
                    dtype=torch.float32,
                    exhaustive=exhaustive,
                    random_samples=200000,
                    batch_size=65536,
                    seed=9000 + width,
                )
            )
        return {"schema": SCHEMA, "rows": width_rows}

    load_or_compute_json(out_dir / "width_scaling.json", args.force, compute_width_scaling)

    def compute_precision() -> dict[str, Any]:
        precision_rows = []
        precision_plan: list[tuple[str, torch.device]] = [("float64", torch.device("cpu")), ("float32", device)]
        if device.type == "cuda":
            precision_plan.append(("float16", device))
            if torch.cuda.is_bf16_supported():
                precision_plan.append(("bfloat16", device))
        for dtype_name, dtype_device in precision_plan:
            precision_rows.append(
                verify_counter(
                    16,
                    device=dtype_device,
                    dtype=dtype_from_name(dtype_name),
                    exhaustive=True,
                    batch_size=65536,
                )
            )
        return {"schema": SCHEMA, "rows": precision_rows}

    load_or_compute_json(out_dir / "precision.json", args.force, compute_precision)


if __name__ == "__main__":
    main()
