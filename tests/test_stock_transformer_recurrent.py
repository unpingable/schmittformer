import torch

from src.stock_transformer_recurrent import compare_stock_gather_to_reference


def test_stock_mha_gather_matches_direct_softmax() -> None:
    gen = torch.Generator().manual_seed(7101)
    slots = torch.randint(0, 2, (16, 43), generator=gen).float()
    result = compare_stock_gather_to_reference(slots, 8.0, torch.float32)
    assert result["passed"]
    assert result["max_abs_diff"] < 1e-5
