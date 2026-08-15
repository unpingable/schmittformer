import torch

from src.circuit_softmax import SoftmaxCircuitConfig, SoftmaxCircuitBreakerTransformer, verify_softmax_circuit_transition_graph
from src.circuit_reference import Event, run_controller


def test_softmax_circuit_matches_reference_trace() -> None:
    model = SoftmaxCircuitBreakerTransformer(SoftmaxCircuitConfig(2.0, 4.0, torch.float64))
    inputs = [Event.FAILURE, Event.FAILURE, Event.FAILURE, Event.UNKNOWN, Event.UNKNOWN, Event.UNKNOWN, Event.UNKNOWN, Event.UNKNOWN, Event.SUCCESS, Event.UNKNOWN, Event.SUCCESS]
    actual, _ = model.decode_inputs(inputs)
    assert actual == run_controller(inputs)


def test_softmax_circuit_transition_graph_float64() -> None:
    result = verify_softmax_circuit_transition_graph(2.0, 4.0, torch.float64, max_histories_per_state=4)
    assert result["passed"], result["failures"]
    assert result["transitions_checked"] == 48 * 4 * 3
    assert result["min_decision_margin"] > 0.0
    assert not result["effectively_hard"]
