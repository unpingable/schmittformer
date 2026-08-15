from src.circuit_compiled import CompiledCircuitBreakerTransformer, verify_compiled_transition_graph
from src.circuit_reference import Event, run_controller


def test_compiled_matches_reference_trace() -> None:
    inputs = [Event.FAILURE, Event.SUCCESS, Event.FAILURE, Event.UNKNOWN, Event.SUCCESS, Event.FAILURE, Event.UNKNOWN, Event.UNKNOWN, Event.UNKNOWN, Event.UNKNOWN, Event.UNKNOWN, Event.SUCCESS, Event.UNKNOWN, Event.SUCCESS]
    model = CompiledCircuitBreakerTransformer()
    actual, _ = model.decode_inputs(inputs)
    assert actual == run_controller(inputs)


def test_compiled_transition_graph() -> None:
    result = verify_compiled_transition_graph(max_histories_per_state=4)
    assert result["passed"], result["failures"]
    assert result["reachable_states"] == 48
