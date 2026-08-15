from src.compiled import verify_reachable_transitions


def test_every_reachable_one_step_transition_matches_reference() -> None:
    result = verify_reachable_transitions()
    assert result["passed"], result["failures"]
    assert result["checked"] == 40
