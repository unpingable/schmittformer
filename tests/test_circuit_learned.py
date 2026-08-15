import torch

from src.circuit_learned import CircuitBatcher, CircuitLearnedConfig, TinyCircuitTransformer, make_tokens


def test_circuit_learned_forward_shapes() -> None:
    device = torch.device("cpu")
    batcher = CircuitBatcher(device)
    config = CircuitLearnedConfig(steps=1, batch_size=2, train_len=8, d_model=24, n_heads=4, d_ff=48, max_len=64)
    model = TinyCircuitTransformer(config, len(batcher.states))
    inputs = batcher.sample_inputs(2, 8, "natural")
    logits = model(make_tokens(inputs))
    assert logits.shape == (2, 9, len(batcher.states))
