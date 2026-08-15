import math

from src.softmax_attention import (
    finite_geometric_stale_bound,
    geometric_stale_state_unnormalized_bound,
    non_state_unnormalized_bound,
    state_leakage_sufficient_for_decoding,
)


def test_finite_geometric_bound_approaches_infinite_bound() -> None:
    infinite = geometric_stale_state_unnormalized_bound(2.0)
    finite = finite_geometric_stale_bound(2.0, 10)
    assert finite < infinite
    assert math.isclose(finite, infinite, rel_tol=1e-8)


def test_decoding_sufficient_condition() -> None:
    assert state_leakage_sufficient_for_decoding(2.0)
    assert not state_leakage_sufficient_for_decoding(0.5)


def test_non_state_bound_decreases_with_penalty() -> None:
    assert non_state_unnormalized_bound(2.0, 6.0) < non_state_unnormalized_bound(2.0, 4.0)
