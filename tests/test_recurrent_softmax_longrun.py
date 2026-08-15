import torch

from experiments.run_recurrent_softmax import long_trace


def test_recurrent_softmax_discrete_reencode_longrun_small() -> None:
    result = long_trace("mixed_governance", 100, 8.0, torch.float64, "discrete_reencode")
    assert result["passed"]
    assert result["physical_input_width"] == 43


def test_recurrent_softmax_analog_carry_drifts() -> None:
    result = long_trace("mixed_governance", 100, 8.0, torch.float64, "analog_carry")
    assert not result["passed"]
    assert result["first_divergence"]["index"] == 4
