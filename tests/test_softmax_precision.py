import torch

from src.hysteresis_softmax import verify_softmax_hysteresis_transitions
from src.softmax_attention import dtype_supported


def test_float32_softmax_hysteresis_not_saturated_at_moderate_gap() -> None:
    result = verify_softmax_hysteresis_transitions(2.0, 4.0, torch.float32)
    assert result["passed"], result["failures"]
    assert result["finite"]
    assert not result["effectively_hard"]


def test_dtype_support_probe_runs() -> None:
    assert dtype_supported(torch.float64)
    assert dtype_supported(torch.float32)
