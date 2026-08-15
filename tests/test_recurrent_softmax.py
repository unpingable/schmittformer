import torch

from experiments.run_recurrent_governance import selected_edge_states
from src.fixed_state import GovernanceEvent
from src.recurrent_softmax import compare_softmax_reference


def test_recurrent_softmax_edge_transitions_gap8() -> None:
    states = []
    events = []
    for state in selected_edge_states():
        for event in GovernanceEvent:
            states.append(state)
            events.append(event)
    result = compare_softmax_reference(states, events, 8.0, device="cpu", dtype=torch.float64)
    assert result["passed"]
    assert result["classification"] == "PASS_EXACT"
    assert not result["attention"]["effectively_hard"]
