import torch

from src.projection_baselines import evaluate_deterministic_boundaries
from src.projection_channels import ProjectionRegime, deterministic_token_monitor, project_batch, trusted_metadata_monitor
from src.projection_task import Decision, Nuisance, PolicyState, ProjectionTaskConfig, Scope, Witness, batch_from_states


def test_trusted_metadata_monitor_is_oracle() -> None:
    batch = batch_from_states(
        [
            PolicyState(1, int(Witness.VALID), int(Scope.A), int(Nuisance.ONE)),
            PolicyState(1, int(Witness.VALID), int(Scope.B), int(Nuisance.ONE)),
            PolicyState(2, int(Witness.INVALID), int(Scope.B), int(Nuisance.ZERO)),
        ],
        seq_len=16,
    )
    assert trusted_metadata_monitor(batch).tolist() == batch.decision.tolist()


def test_complete_erasure_monitor_refuses_instead_of_false_admitting() -> None:
    batch = batch_from_states(
        [
            PolicyState(1, int(Witness.VALID), int(Scope.A), int(Nuisance.ONE)),
            PolicyState(1, int(Witness.INVALID), int(Scope.A), int(Nuisance.ONE)),
        ],
        seq_len=16,
    )
    projected = project_batch(batch, ProjectionRegime.P0_COMPLETE_ERASURE)
    decisions = deterministic_token_monitor(projected)

    assert decisions.tolist() == [
        int(Decision.REFUSE_INSUFFICIENT_INFORMATION),
        int(Decision.REFUSE_INSUFFICIENT_INFORMATION),
    ]


def test_evaluate_deterministic_boundaries_reports_metadata_control() -> None:
    metrics = evaluate_deterministic_boundaries(
        ProjectionRegime.P3_FULL_TRUSTED_EXPORT,
        0.0,
        ProjectionTaskConfig(seq_len=16),
        torch.device("cpu"),
        batch_size=32,
        batches=2,
    )
    assert metrics["token_plus_trusted_metadata"]["policy_accuracy"] == 1.0
    assert metrics["token_only_reference"]["policy_accuracy"] == 1.0
