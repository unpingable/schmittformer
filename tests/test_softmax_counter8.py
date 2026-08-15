import torch

from src.recurrent_softmax import verify_softmax_counter


def test_softmax_counter8_non_saturated_gap8() -> None:
    result = verify_softmax_counter(8, 8.0, device="cpu", dtype=torch.float64, exhaustive=True, batch_size=256)
    assert result["passed"]
    assert result["classification"] == "PASS_EXACT"
    assert not result["attention"]["effectively_hard"]
