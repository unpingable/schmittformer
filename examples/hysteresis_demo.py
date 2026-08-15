from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compiled import predict_compiled
from src.reference import State, run_hysteresis, state_labels


def main() -> None:
    inputs = [2, 5, 8, 6, 4, 3, 5, 7]
    reference = run_hysteresis(inputs, initial_state=State.OFF)
    compiled = predict_compiled(inputs, initial_state=State.OFF, attention="hard")
    soft = predict_compiled(inputs, initial_state=State.OFF, attention="soft")

    print("inputs:       ", " ".join(str(x) for x in inputs))
    print("reference:    ", " ".join(state_labels([s]) for s in reference))
    print("compiled hard:", " ".join(state_labels([s]) for s in compiled))
    print("compiled soft:", " ".join(state_labels([s]) for s in soft))
    print("match:", reference == compiled == soft)


if __name__ == "__main__":
    main()
