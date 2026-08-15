from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor


def as_float_bits(bits: Tensor, dtype: torch.dtype | None = None) -> Tensor:
    dtype = dtype or (bits.dtype if bits.dtype.is_floating_point else torch.float32)
    return bits.to(dtype=dtype)


def hard_not(a: Tensor) -> Tensor:
    return 1.0 - a


def hard_and(a: Tensor, b: Tensor) -> Tensor:
    # Exact for a,b in {0,1}; implemented with ReLU as a two-input threshold gate.
    return torch.relu(a + b - 1.0)


def hard_or(a: Tensor, b: Tensor) -> Tensor:
    # Exact min(1, a + b) for binary inputs.
    s = a + b
    return s - torch.relu(s - 1.0)


def hard_xor(a: Tensor, b: Tensor) -> Tensor:
    # Exact abs(a-b) for binary inputs.
    return torch.relu(a - b) + torch.relu(b - a)


def hard_and_many(bits: Tensor) -> Tensor:
    if bits.shape[-1] == 0:
        return torch.ones(bits.shape[:-1], dtype=bits.dtype, device=bits.device)
    return torch.relu(bits.sum(dim=-1) - float(bits.shape[-1] - 1))


def hard_or_many(bits: Tensor) -> Tensor:
    if bits.shape[-1] == 0:
        return torch.zeros(bits.shape[:-1], dtype=bits.dtype, device=bits.device)
    s = bits.sum(dim=-1)
    return s - torch.relu(s - 1.0)


def hard_eq_bits(bits: Tensor, pattern: Sequence[int]) -> Tensor:
    if bits.shape[-1] != len(pattern):
        raise ValueError("pattern width mismatch")
    pieces = []
    for index, bit in enumerate(pattern):
        pieces.append(bits[..., index] if int(bit) else hard_not(bits[..., index]))
    return hard_and_many(torch.stack(pieces, dim=-1))


def hard_mux(cond: Tensor, if_true: Tensor, if_false: Tensor) -> Tensor:
    cond = cond.to(dtype=if_true.dtype, device=if_true.device)
    while cond.ndim < if_true.ndim:
        cond = cond.unsqueeze(-1)
    return cond * if_true + (1.0 - cond) * if_false


def hard_one_hot(index: int, size: int, batch_shape: Sequence[int], device: torch.device, dtype: torch.dtype) -> Tensor:
    out = torch.zeros((*batch_shape, size), dtype=dtype, device=device)
    out[..., int(index)] = 1.0
    return out


def int_tensor_to_bits(values: Tensor, width: int, dtype: torch.dtype | None = None) -> Tensor:
    values = values.to(torch.long)
    dtype = dtype or torch.float32
    shifts = torch.arange(width, device=values.device, dtype=torch.long)
    return ((values[..., None] >> shifts) & 1).to(dtype)


def bits_to_int_tensor(bits: Tensor) -> Tensor:
    bits_long = bits.round().to(torch.long)
    shifts = torch.arange(bits.shape[-1], device=bits.device, dtype=torch.long)
    weights = (1 << shifts).to(torch.long)
    return (bits_long * weights).sum(dim=-1)


def zero_test(bits: Tensor) -> Tensor:
    return hard_not(hard_or_many(bits))


def nonzero_test(bits: Tensor) -> Tensor:
    return hard_or_many(bits)


def saturating_decrement_bits(bits: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return max(x-1,0), is_zero(x), is_nonzero(x) for little-endian bits.

    This is a ripple-borrow subtract-one circuit, followed by a saturation mask
    that clears the all-ones wraparound result when x == 0.
    """
    bits = as_float_bits(bits)
    borrow = torch.ones(bits.shape[:-1], dtype=bits.dtype, device=bits.device)
    raw = []
    for index in range(bits.shape[-1]):
        bit = bits[..., index]
        raw_bit = hard_xor(bit, borrow)
        raw.append(raw_bit)
        borrow = hard_and(borrow, hard_not(bit))
    raw_bits = torch.stack(raw, dim=-1)
    nonzero = nonzero_test(bits)
    dec = hard_mux(nonzero, raw_bits, torch.zeros_like(raw_bits))
    return dec, hard_not(nonzero), nonzero


def bit_margin(bits: Tensor) -> Tensor:
    return torch.abs(bits - 0.5)
