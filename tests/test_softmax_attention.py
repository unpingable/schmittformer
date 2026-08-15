import math

import torch

from src.softmax_attention import (
    SoftmaxAttentionConfig,
    geometric_stale_state_unnormalized_bound,
    latest_state_attention,
    non_state_unnormalized_bound,
    simple_correct_probability_lower_bound,
    simple_leakage_upper_bound,
)


def test_simple_softmax_bound_matches_constructed_case() -> None:
    delta = 2.0
    competitors = 7
    scores = torch.tensor([0.0] + [-delta] * competitors, dtype=torch.float64)
    weights = torch.softmax(scores, dim=0)
    assert math.isclose(float(weights[0]), simple_correct_probability_lower_bound(delta, competitors), rel_tol=1e-12)
    assert math.isclose(float(weights[1:].sum()), simple_leakage_upper_bound(delta, competitors), rel_tol=1e-12)


def test_geometric_bound_is_below_one_above_log_two() -> None:
    assert geometric_stale_state_unnormalized_bound(math.log(2) + 1e-3) < 1.0
    assert geometric_stale_state_unnormalized_bound(math.log(2) - 1e-3) > 1.0


def test_recency_scoring_orders_latest_state_record() -> None:
    tokens = torch.tensor([10, 0, 11, 5, 10, 6], dtype=torch.long)
    _, debug = latest_state_attention(tokens, state_token_offset=10, num_states=2, config=SoftmaxAttentionConfig(state_record_gap=2.0, non_state_penalty=4.0, dtype=torch.float64))
    assert int(debug["latest_index"].item()) == 4
    assert int(debug["latest_state_id"].item()) == 0
    assert float(debug["correct_mass"].item()) > float(debug["stale_state_mass"].item())


def test_non_state_contamination_is_measured_not_masked() -> None:
    cfg = SoftmaxAttentionConfig(state_record_gap=2.0, non_state_penalty=4.0, dtype=torch.float64)
    tokens = torch.tensor([10, 0, 11, 5, 10, 6], dtype=torch.long)
    _, debug = latest_state_attention(tokens, 10, 2, cfg)
    assert float(debug["non_state_mass"].item()) > 0.0
    assert float(debug["non_state_mass"].item()) < non_state_unnormalized_bound(2.0, 4.0)
