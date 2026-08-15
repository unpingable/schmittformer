import torch

from src.projection_channels import (
    MISSING,
    ProjectionRegime,
    bayes_bounds,
    decoded_scope,
    decoded_witness,
    deterministic_token_monitor,
    project_batch,
)
from src.projection_task import (
    Decision,
    Nuisance,
    PolicyState,
    ProjectionTaskConfig,
    Scope,
    Witness,
    batch_from_states,
)


def test_complete_erasure_removes_policy_bits() -> None:
    states = [
        PolicyState(1, int(Witness.VALID), int(Scope.A), int(Nuisance.ONE)),
        PolicyState(1, int(Witness.INVALID), int(Scope.A), int(Nuisance.ONE)),
    ]
    batch = batch_from_states(states, seq_len=16)
    projected = project_batch(batch, ProjectionRegime.P0_COMPLETE_ERASURE)

    assert projected.proposal.tolist() == [1, 1]
    assert projected.witness.tolist() == [MISSING, MISSING]
    assert projected.scope.tolist() == [MISSING, MISSING]
    assert batch.decision.tolist() == [int(Decision.ADMIT_A), int(Decision.REFUSE_INVALID_WITNESS)]


def test_full_trusted_export_restores_token_monitor() -> None:
    states = [
        PolicyState(1, int(Witness.VALID), int(Scope.A), int(Nuisance.ONE)),
        PolicyState(2, int(Witness.VALID), int(Scope.A), int(Nuisance.ONE)),
        PolicyState(2, int(Witness.INVALID), int(Scope.B), int(Nuisance.ZERO)),
    ]
    batch = batch_from_states(states, seq_len=16)
    projected = project_batch(batch, ProjectionRegime.P3_FULL_TRUSTED_EXPORT)

    assert deterministic_token_monitor(projected).tolist() == batch.decision.tolist()


def test_redundant_export_majority_decodes_carriers() -> None:
    states = [PolicyState(1, int(Witness.VALID), int(Scope.A), int(Nuisance.ONE))]
    batch = batch_from_states(states, seq_len=16)
    projected = project_batch(batch, ProjectionRegime.P4_REDUNDANT_EXPORT, noise=0.0)

    assert decoded_witness(projected).tolist() == [int(Witness.VALID)]
    assert decoded_scope(projected).tolist() == [int(Scope.A)]
    assert deterministic_token_monitor(projected).tolist() == [int(Decision.ADMIT_A)]


def test_bayes_bounds_detect_ambiguous_erasure() -> None:
    bounds = bayes_bounds(ProjectionTaskConfig(seq_len=16), ProjectionRegime.P0_COMPLETE_ERASURE)

    assert bounds["bayes_optimal_error"] > 0.0
    assert bounds["ambiguous_projection_keys"] > 0
    assert bounds["collision_example"] is not None
