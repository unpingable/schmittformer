from src.projection_task import PolicyState, batch_from_states
from src.register_interventions import evaluate_register_intervention, semantic_swap_register
from src.semantic_register import RegisterEncoding, decode_register, register_code


def test_semantic_swap_register_changes_only_requested_variable() -> None:
    target = register_code(RegisterEncoding.GROUPED_ONE_HOT, batch_from_states([PolicyState(1, 0, 0, 0)], 64).witness, batch_from_states([PolicyState(1, 0, 0, 0)], 64).scope)
    source = register_code(RegisterEncoding.GROUPED_ONE_HOT, batch_from_states([PolicyState(1, 1, 1, 0)], 64).witness, batch_from_states([PolicyState(1, 1, 1, 0)], 64).scope)
    swapped = semantic_swap_register(target, source, RegisterEncoding.GROUPED_ONE_HOT, "witness")
    decoded = decode_register(swapped, RegisterEncoding.GROUPED_ONE_HOT)
    assert decoded.witness.item() == 1
    assert decoded.scope.item() == 0


def test_register_intervention_is_semantically_consistent() -> None:
    target = batch_from_states([PolicyState(1, 0, 0, 0)], 64)
    source = batch_from_states([PolicyState(1, 1, 0, 0)], 64)
    target_register = register_code(RegisterEncoding.BINARY_PAIR, target.witness, target.scope)
    source_register = register_code(RegisterEncoding.BINARY_PAIR, source.witness, source.scope)
    result = evaluate_register_intervention(target, source, target_register, source_register, RegisterEncoding.BINARY_PAIR, "witness")
    assert result.semantic_consistency == 1.0
    assert result.world_target_consistency == 1.0
    assert result.changed_decision_rate == 1.0
