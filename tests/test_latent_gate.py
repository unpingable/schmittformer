import torch

from src.projection_task import Decision
from src.synthesized_latent_gate import decision_metrics, synthesize_policy_from_variables


def test_synthesized_gate_applies_policy_to_decoded_variables() -> None:
    proposal_logits = torch.tensor(
        [
            [-4.0, 4.0, -4.0],
            [-4.0, 4.0, -4.0],
            [-4.0, -4.0, 4.0],
            [4.0, -4.0, -4.0],
        ]
    )
    witness_logits = torch.tensor(
        [
            [-3.0, 3.0],
            [-3.0, 3.0],
            [3.0, -3.0],
            [-3.0, 3.0],
        ]
    )
    scope_logits = torch.tensor(
        [
            [3.0, -3.0],
            [-3.0, 3.0],
            [-3.0, 3.0],
            [3.0, -3.0],
        ]
    )

    assert synthesize_policy_from_variables(proposal_logits, witness_logits, scope_logits).tolist() == [
        int(Decision.ADMIT_A),
        int(Decision.REFUSE_SCOPE),
        int(Decision.REFUSE_INVALID_WITNESS),
        int(Decision.REFUSE_NO_PROPOSAL),
    ]


def test_decision_metrics_separates_false_admission() -> None:
    expected = torch.tensor([int(Decision.REFUSE_INVALID_WITNESS), int(Decision.ADMIT_A)])
    predicted = torch.tensor([int(Decision.ADMIT_A), int(Decision.REFUSE_SCOPE)])

    metrics = decision_metrics(predicted, expected)

    assert metrics["policy_accuracy"] == 0.0
    assert metrics["admit_false_positive_rate"] == 1.0
    assert metrics["refuse_false_positive_rate"] == 1.0
