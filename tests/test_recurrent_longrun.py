from src.fixed_state import Authority, GovernanceEvent, configured_state
from src.recurrent_compiled import run_compiled_trace
from src.recurrent_reference import run_reference


def test_recurrent_fixed_state_trace_longer_than_physical_width() -> None:
    events = [GovernanceEvent.TICK] * 200
    start = configured_state(lease=100, budget=1, authority=Authority.VALID)
    compiled_states, compiled_outputs = run_compiled_trace(events, start)
    reference_states, reference_outputs = run_reference(events, start)
    assert compiled_states == reference_states
    assert compiled_outputs == reference_outputs
    assert compiled_states[-1].lease_remaining == 0
