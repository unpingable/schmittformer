from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from src.transition_smt import run_transition_equivalence


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/recurrent_softmax/solver.json")
    parser.add_argument("--timeout-ms", type=int, default=120000)
    args = parser.parse_args()
    atomic_write_json(Path(args.out), run_transition_equivalence(timeout_ms=args.timeout_ms))


if __name__ == "__main__":
    main()
