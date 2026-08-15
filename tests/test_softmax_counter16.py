import torch

from src.recurrent_softmax import verify_softmax_counter


def test_softmax_counter16_exhaustive_gap8() -> None:
    result = verify_softmax_counter(16, 8.0, device="cpu", dtype=torch.float64, exhaustive=True, batch_size=65536)
    assert result["passed"]
    assert result["classification"] == "PASS_EXACT"
    assert not result["attention"]["effectively_hard"]


def test_softmax_counter16_gap4_finds_boundary() -> None:
    result = verify_softmax_counter(16, 4.0, device="cpu", dtype=torch.float64, exhaustive=True, batch_size=65536)
    assert not result["passed"]
    assert result["classification"] == "SEMANTIC_FAILURE"
