from __future__ import annotations

import torch

from experiments.run_recurrent_governance import selected_edge_states
from src.fixed_state import GovernanceEvent, GovernanceState
from src.recurrent_compiled import random_events, random_valid_states
from src.stock_governance_transformer import (
    StockGovernanceConfig,
    StockGovernanceTransformer,
    compare_stock_reference,
    load_stock_governance_model,
    save_stock_governance_model,
)


def test_stock_governance_edge_subset_cpu():
    model = StockGovernanceTransformer(StockGovernanceConfig(score_gap=8.0, dtype_name="float32"))
    edge_states = selected_edge_states()[:32]
    states = [state for state in edge_states for _ in GovernanceEvent]
    events = [event for _ in edge_states for event in GovernanceEvent]
    result = compare_stock_reference(states, events, model, batch_size=128)
    assert result["passed"], result["first_failures"]
    assert result["min_state_bit_margin"] > 0.25


def test_stock_governance_random_subset_cpu():
    model = StockGovernanceTransformer(StockGovernanceConfig(score_gap=8.0, dtype_name="float32"))
    result = compare_stock_reference(
        random_valid_states(512, seed=901),
        random_events(512, seed=902),
        model,
        batch_size=128,
    )
    assert result["passed"], result["first_failures"]


def test_stock_governance_save_load_checkpoint(tmp_path):
    model = StockGovernanceTransformer(StockGovernanceConfig(score_gap=8.0, dtype_name="float32"))
    path = tmp_path / "stock_governance.pt"
    save_stock_governance_model(path, model)
    loaded = load_stock_governance_model(path)
    states = [
        GovernanceState(1, 1, 1, 0, 0),
        GovernanceState(0, 0, 0, 0, 0),
        GovernanceState(1, 65535, 255, 1, 0),
        GovernanceState(1, 9, 3, 2, 0),
    ]
    events = [
        GovernanceEvent.PROPOSE_ACTION,
        GovernanceEvent.PROPOSE_ACTION,
        GovernanceEvent.RESULT_AMBIGUOUS,
        GovernanceEvent.SETTLE_FAILURE,
    ]
    result = compare_stock_reference(states, events, loaded, batch_size=4)
    assert result["passed"], result["first_failures"]


def test_gap4_is_not_enough_for_stock_governance():
    model = StockGovernanceTransformer(StockGovernanceConfig(score_gap=4.0, dtype_name="float32"))
    states = [GovernanceState(0, 0, 0, 0, 0)]
    events = [GovernanceEvent.PROPOSE_ACTION]
    result = compare_stock_reference(states, events, model, batch_size=1)
    assert not result["passed"]
    assert result["first_failures"][0]["actual_output"] != result["first_failures"][0]["expected"]["output"]
