import torch

from src.causal_interventions import directional_swap


def test_directional_swap_replaces_only_requested_component() -> None:
    source = torch.tensor([[2.0, 10.0], [4.0, 20.0]])
    target = torch.tensor([[1.0, -1.0], [3.0, -3.0]])
    direction = torch.tensor([1.0, 0.0])

    swapped = directional_swap(source, target, direction)

    assert torch.allclose(swapped[:, 0], source[:, 0])
    assert torch.allclose(swapped[:, 1], target[:, 1])


def test_directional_swap_rejects_zero_direction() -> None:
    source = torch.zeros(1, 2)
    target = torch.zeros(1, 2)
    direction = torch.zeros(2)

    try:
        directional_swap(source, target, direction)
    except ValueError as exc:
        assert "zero direction" in str(exc)
    else:
        raise AssertionError("zero direction should fail")
