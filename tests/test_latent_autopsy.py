import torch

from src.latent_autopsy import (
    alignment_quality,
    apply_affine_alignment,
    classification_margin,
    controlled_position_batch,
    fit_affine_alignment,
    fit_linear_readout,
    representative_states,
)
from src.projection_task import Scope, Witness


def test_closed_form_linear_readout_separates_simple_labels() -> None:
    x = torch.tensor([[-1.0, 0.0], [-2.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    labels = torch.tensor([0, 0, 1, 1])

    readout = fit_linear_readout(x, labels, classes=2)

    assert readout.predict(x).tolist() == labels.tolist()


def test_affine_alignment_restores_shifted_coordinates() -> None:
    source = torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]])
    target = source + torch.tensor([10.0, -3.0])

    weights = fit_affine_alignment(source, target)
    mapped = apply_affine_alignment(source, weights)

    assert alignment_quality(mapped, target)["rmse"] < 1.0e-4


def test_classification_margin_positive_for_correct_winner() -> None:
    logits = torch.tensor([[3.0, 1.0, 2.0], [0.0, 5.0, 1.0]])
    labels = torch.tensor([0, 1])

    assert classification_margin(logits, labels).tolist() == [1.0, 4.0]


def test_controlled_position_batch_sets_requested_semantics() -> None:
    states = representative_states(repeats=1)[:4]
    batch = controlled_position_batch(states, seq_len=64, mode="fixed_distance", device=torch.device("cpu"))

    assert batch.tokens.shape == (4, 64)
    assert set(batch.witness.tolist()).issubset({int(Witness.INVALID), int(Witness.VALID)})
    assert set(batch.scope.tolist()).issubset({int(Scope.A), int(Scope.B)})
