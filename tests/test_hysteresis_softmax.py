import torch

from src.hysteresis_softmax import SoftmaxHysteresisConfig, SoftmaxHysteresisTransformer, verify_softmax_hysteresis_transitions
from src.reference import State, exhaustive_sequences, run_hysteresis


def test_softmax_hysteresis_transition_table_float64() -> None:
    result = verify_softmax_hysteresis_transitions(2.0, 4.0, torch.float64)
    assert result["passed"], result["failures"]
    assert result["min_decision_margin"] > 0.0
    assert not result["effectively_hard"]


def test_softmax_hysteresis_exhaustive_small_bound() -> None:
    model = SoftmaxHysteresisTransformer(SoftmaxHysteresisConfig(2.0, 4.0, torch.float64))
    for initial in (State.OFF, State.ON):
        for seq in exhaustive_sequences(3):
            actual, _ = model.decode_inputs(seq, initial_state=initial)
            assert actual == run_hysteresis(seq, initial)


def test_softmax_hysteresis_deadband_initial_on() -> None:
    model = SoftmaxHysteresisTransformer(SoftmaxHysteresisConfig(2.0, 4.0, torch.float64))
    inputs = [4, 5, 6, 5, 4, 6] * 3
    actual, _ = model.decode_inputs(inputs, initial_state=State.ON)
    assert actual == [1] * len(inputs)
