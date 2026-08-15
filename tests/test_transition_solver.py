from src.transition_smt import run_transition_equivalence


def test_transition_solver_reports_result_or_unavailable() -> None:
    result = run_transition_equivalence(timeout_ms=1000)
    assert result["result"] in {"UNSAT", "SAT", "UNKNOWN", "UNAVAILABLE"}
