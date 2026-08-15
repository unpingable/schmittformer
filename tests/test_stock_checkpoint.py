import tempfile

import torch

from src.fixed_state import GovernanceEvent, configured_state, encode_state_bits
from src.stock_transformer_recurrent import StockGatherRecurrentModel, load_stock_model, save_stock_model


def test_stock_gather_checkpoint_roundtrip() -> None:
    model = StockGatherRecurrentModel()
    state = configured_state(lease=1, budget=1)
    bits = encode_state_bits(state)
    event = torch.tensor([int(GovernanceEvent.PROPOSE_ACTION)])
    before_bits, before_logits = model(bits, event)
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        save_stock_model(f.name, model)
        loaded = load_stock_model(f.name)
    after_bits, after_logits = loaded(bits, event)
    assert torch.allclose(before_bits, after_bits)
    assert torch.allclose(before_logits, after_logits)
