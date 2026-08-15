import torch

from src.compiled import CompiledHysteresisTransformer
from src.evaluate import count_illegal_transitions
from src.hybrid import hybrid_predict_from_levels
from src.reference import State, run_hysteresis


def test_hybrid_controller_preserves_invariants_for_wrong_beliefs() -> None:
    true_levels = torch.tensor([[4, 5, 6, 5, 4, 6]], dtype=torch.long)
    wrong_beliefs = torch.tensor([[4, 7, 6, 5, 3, 6]], dtype=torch.long)
    initial = torch.tensor([int(State.OFF)], dtype=torch.long)
    states = hybrid_predict_from_levels(
        wrong_beliefs,
        initial,
        CompiledHysteresisTransformer(),
    )
    predicted = states[0].tolist()

    assert count_illegal_transitions(wrong_beliefs[0].tolist(), predicted, State.OFF) == 0
    assert predicted != run_hysteresis(true_levels[0].tolist(), State.OFF)


def test_hybrid_controller_batches_initial_states() -> None:
    levels = torch.tensor(
        [
            [4, 5, 6, 7],
            [4, 5, 6, 3],
        ],
        dtype=torch.long,
    )
    initial = torch.tensor([int(State.OFF), int(State.ON)], dtype=torch.long)
    states = hybrid_predict_from_levels(levels, initial)
    assert states.tolist() == [
        run_hysteresis(levels[0].tolist(), State.OFF),
        run_hysteresis(levels[1].tolist(), State.ON),
    ]
