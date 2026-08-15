from src.circuit_compiled import verify_compiled_history_equivalence
from src.circuit_reference import collect_equivalent_histories


def test_equivalent_history_generator_has_multiple_histories() -> None:
    groups = collect_equivalent_histories(max_per_state=4)
    multi = sum(1 for histories in groups.values() if len(histories) >= 2)
    assert multi == 48


def test_compiled_history_equivalence() -> None:
    result = verify_compiled_history_equivalence(max_histories_per_state=4)
    assert result["passed"], result["examples"]
    assert result["history_equivalence_violations"] == 0
