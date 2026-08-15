import torch

from src.compiled import CompiledConfig, CompiledHysteresisTransformer, predict_compiled
from src.reference import State, exhaustive_sequences, run_hysteresis


def test_compiled_matches_example() -> None:
    inputs = [2, 5, 8, 6, 4, 3, 5, 7]
    assert predict_compiled(inputs) == run_hysteresis(inputs)


def test_compiled_supports_initial_on() -> None:
    inputs = [4, 5, 6, 3, 4, 7]
    assert predict_compiled(inputs, State.ON) == run_hysteresis(inputs, State.ON)


def test_hard_attention_selects_latest_threshold_event() -> None:
    model = CompiledHysteresisTransformer()
    inputs = torch.tensor([7, 6, 5, 4, 3, 6, 7], dtype=torch.long)
    logits, debug = model(inputs, return_debug=True)
    assert logits.argmax(dim=-1).tolist() == run_hysteresis(inputs.tolist())
    assert debug["attention_indices"].tolist() == [1, 1, 1, 1, 5, 5, 7]


def test_soft_attention_is_margin_stable_on_small_case() -> None:
    model = CompiledHysteresisTransformer(CompiledConfig(attention="soft"))
    inputs = torch.tensor([7, 6, 5, 4, 3, 6, 7], dtype=torch.long)
    logits = model(inputs)
    assert logits.argmax(dim=-1).tolist() == run_hysteresis(inputs.tolist())


def test_compiled_exhaustive_small_bound() -> None:
    for initial in (State.OFF, State.ON):
        for sequence in exhaustive_sequences(3):
            assert predict_compiled(sequence, initial) == run_hysteresis(sequence, initial)
