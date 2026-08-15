import torch

from src.compiled_bits import hard_and, hard_eq_bits, hard_not, hard_or, hard_xor, int_tensor_to_bits, bits_to_int_tensor, saturating_decrement_bits


def test_boolean_gates_exact_on_bits() -> None:
    a = torch.tensor([0.0, 0.0, 1.0, 1.0])
    b = torch.tensor([0.0, 1.0, 0.0, 1.0])
    assert hard_not(a).tolist() == [1.0, 1.0, 0.0, 0.0]
    assert hard_and(a, b).tolist() == [0.0, 0.0, 0.0, 1.0]
    assert hard_or(a, b).tolist() == [0.0, 1.0, 1.0, 1.0]
    assert hard_xor(a, b).tolist() == [0.0, 1.0, 1.0, 0.0]


def test_bit_encoding_little_endian_round_trip() -> None:
    values = torch.tensor([0, 1, 2, 7, 255], dtype=torch.long)
    bits = int_tensor_to_bits(values, 8)
    assert bits[1, :4].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert bits_to_int_tensor(bits).tolist() == values.tolist()


def test_eq_bits() -> None:
    bits = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert hard_eq_bits(bits, [0, 1]).tolist() == [0.0, 0.0, 1.0, 0.0]


def test_saturating_decrement_small_values() -> None:
    values = torch.arange(8, dtype=torch.long)
    bits = int_tensor_to_bits(values, 4)
    out, is_zero, is_nonzero = saturating_decrement_bits(bits)
    assert bits_to_int_tensor(out).tolist() == [0, 0, 1, 2, 3, 4, 5, 6]
    assert is_zero.tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert is_nonzero.tolist() == [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
