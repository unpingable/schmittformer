import torch

from src.circuit_softmax import verify_softmax_circuit_history_equivalence


def test_softmax_circuit_semantic_history_equivalence() -> None:
    result = verify_softmax_circuit_history_equivalence(2.0, 4.0, torch.float64, max_histories_per_state=4)
    assert result["passed"], result["examples"]
    assert result["history_equivalence_violations"] == 0
    assert result["max_latent_state_mass_diff"] > 0.0
