from __future__ import annotations

import torch

from src.ffn_counter import verify_decrementer


def test_ffn_counter8_exhaustive_cpu():
    result = verify_decrementer(8, dtype=torch.float64, device="cpu")
    assert result["checked"] == 256
    assert result["passed"]
    assert result["semantic_failures"] == 0


def test_ffn_counter16_exhaustive_cpu():
    result = verify_decrementer(16, dtype=torch.float32, device="cpu")
    assert result["checked"] == 65536
    assert result["passed"]
    assert result["semantic_failures"] == 0
