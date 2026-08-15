from src.projection_channels import ProjectionRegime, project_state_distribution
from src.projection_task import Nuisance, PolicyState, Scope, Witness, policy_decision


def test_downstream_projection_function_cannot_separate_collided_states() -> None:
    admit_state = PolicyState(1, int(Witness.VALID), int(Scope.A), int(Nuisance.ONE))
    refuse_state = PolicyState(1, int(Witness.INVALID), int(Scope.A), int(Nuisance.ONE))

    admit_projection = project_state_distribution(admit_state, ProjectionRegime.P0_COMPLETE_ERASURE)
    refuse_projection = project_state_distribution(refuse_state, ProjectionRegime.P0_COMPLETE_ERASURE)

    assert admit_projection == refuse_projection
    assert policy_decision(admit_state.proposal, admit_state.witness, admit_state.scope) != policy_decision(
        refuse_state.proposal,
        refuse_state.witness,
        refuse_state.scope,
    )
