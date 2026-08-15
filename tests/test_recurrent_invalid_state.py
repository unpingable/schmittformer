from src.fixed_state import GovernanceOutput
from src.recurrent_compiled import invalid_state_cases


def test_invalid_enum_state_refuses_and_sanitizes() -> None:
    cases = invalid_state_cases()
    assert cases
    for case in cases:
        assert case["output"] == int(GovernanceOutput.REFUSE_INVALID_STATE)
        assert case["next_state"]["authority"] == 0
        assert case["next_state"]["lease_remaining"] == 0
        assert case["next_state"]["action_budget"] == 0
