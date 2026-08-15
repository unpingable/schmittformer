from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .relu_boolean import ReLUCircuitBuilder, WireSpec, AffineExpr, SynthesizedReLUCircuit


@dataclass(frozen=True)
class FFNCounterConfig:
    width: int
    dtype: torch.dtype = torch.float32


def add_saturating_decrement(builder: ReLUCircuitBuilder, bits: list[str], prefix: str) -> tuple[list[str], str, str]:
    """Append a scalable ripple-borrow saturating decrement circuit.

    Inputs and outputs are little-endian binary wires. The construction is
    exact on Boolean inputs and uses O(width) generated ReLU blocks rather than
    a lookup table over counter values.
    """

    borrow = builder.affine(f"{prefix}_borrow_start", {}, 1.0)
    raw_bits: list[str] = []
    for i, bit in enumerate(bits):
        raw = f"{prefix}_raw_{i}"
        next_borrow = f"{prefix}_borrow_{i + 1}"
        builder.append_block(
            [
                WireSpec(
                    name=raw,
                    coeffs={},
                    relus=[
                        (1.0, AffineExpr({bit: 1.0, borrow: -1.0})),
                        (1.0, AffineExpr({bit: -1.0, borrow: 1.0})),
                    ],
                ),
                WireSpec(
                    name=next_borrow,
                    coeffs={},
                    relus=[(1.0, AffineExpr({borrow: 1.0, bit: -1.0}))],
                ),
            ]
        )
        raw_bits.append(raw)
        borrow = next_borrow

    nonzero = builder.or_many(f"{prefix}_nonzero", bits)
    zero = builder.not_wire(f"{prefix}_zero", nonzero)
    dec_bits = builder.mux_many(f"{prefix}_dec", nonzero, raw_bits, [builder_const0(builder)] * len(bits))
    return dec_bits, zero, nonzero


def builder_const0(builder: ReLUCircuitBuilder) -> str:
    if "__const0" not in builder.wires:
        builder.affine("__const0", {}, 0.0)
    return "__const0"


def builder_const1(builder: ReLUCircuitBuilder) -> str:
    if "__const1" not in builder.wires:
        builder.affine("__const1", {}, 1.0)
    return "__const1"


class FFNSaturatingDecrementer(nn.Module):
    """Stock Linear/ReLU realization of max(x-1, 0), is_zero, is_nonzero."""

    def __init__(self, config: FFNCounterConfig):
        super().__init__()
        self.config = config
        input_names = [f"x_{i}" for i in range(config.width)]
        builder = ReLUCircuitBuilder(input_names, dtype=config.dtype)
        builder_const0(builder)
        dec_bits, zero, nonzero = add_saturating_decrement(builder, input_names, "dec")
        self.network = builder.build([(wire, 1.0, 0.0) for wire in dec_bits] + [(zero, 1.0, 0.0), (nonzero, 1.0, 0.0)])

    def forward(self, bits: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        squeeze = False
        if bits.ndim == 1:
            bits = bits.unsqueeze(0)
            squeeze = True
        out = self.network(bits.to(dtype=self.config.dtype))
        dec = out[:, : self.config.width]
        zero = out[:, self.config.width]
        nonzero = out[:, self.config.width + 1]
        if squeeze:
            return dec.squeeze(0), zero.squeeze(0), nonzero.squeeze(0)
        return dec, zero, nonzero


def bit_tensor_from_ints(values: Tensor, width: int, dtype: torch.dtype) -> Tensor:
    values = values.to(torch.long)
    shifts = torch.arange(width, device=values.device, dtype=torch.long)
    return ((values[:, None] >> shifts) & 1).to(dtype)


def ints_from_bit_tensor(bits: Tensor) -> Tensor:
    rounded = bits.round().to(torch.long)
    shifts = torch.arange(bits.shape[-1], device=bits.device, dtype=torch.long)
    return (rounded * (1 << shifts)).sum(dim=-1)


@torch.no_grad()
def verify_decrementer(width: int, dtype: torch.dtype = torch.float32, device: torch.device | str = "cpu") -> dict[str, float | int | bool | str]:
    model = FFNSaturatingDecrementer(FFNCounterConfig(width=width, dtype=dtype)).to(device)
    values = torch.arange(1 << width, device=device)
    bits = bit_tensor_from_ints(values, width, dtype)
    dec_bits, zero, nonzero = model(bits)
    decoded = ints_from_bit_tensor(dec_bits)
    expected = torch.clamp(values - 1, min=0)
    expected_zero = (values == 0).to(dtype)
    expected_nonzero = (values != 0).to(dtype)
    failures = (decoded != expected).sum().item()
    zero_failures = (zero.round().to(torch.long) != expected_zero.to(torch.long)).sum().item()
    nonzero_failures = (nonzero.round().to(torch.long) != expected_nonzero.to(torch.long)).sum().item()
    outputs = torch.cat([dec_bits, zero[:, None], nonzero[:, None]], dim=-1)
    margin = torch.abs(outputs - 0.5).min().item()
    max_error = torch.abs(outputs - outputs.round()).max().item()
    return {
        "width": int(width),
        "checked": int(1 << width),
        "semantic_failures": int(failures),
        "zero_failures": int(zero_failures),
        "nonzero_failures": int(nonzero_failures),
        "min_bit_margin": float(margin),
        "max_abs_error_from_binary": float(max_error),
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "block_count": int(len(model.network.blocks)),
        "passed": bool(failures == 0 and zero_failures == 0 and nonzero_failures == 0),
    }
