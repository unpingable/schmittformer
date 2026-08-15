from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class AffineExpr:
    coeffs: dict[str, float]
    bias: float = 0.0


@dataclass(frozen=True)
class WireSpec:
    name: str
    coeffs: dict[str, float]
    bias: float = 0.0
    relus: Sequence[tuple[float, AffineExpr]] = ()


class SynthesizedReLUCircuit(nn.Module):
    """A frozen network made only of standard Linear/ReLU operations."""

    def __init__(self, blocks: Sequence[nn.Sequential], readout: nn.Linear, wire_names: Sequence[str]):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.readout = readout
        self.wire_names = tuple(wire_names)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return self.readout(x)


class ReLUCircuitBuilder:
    """Builds exact Boolean circuits over binary inputs as Linear/ReLU layers.

    Each block appends new wires while preserving existing nonnegative wires.
    The generated inference path contains no semantic custom operation: it is a
    list of ordinary Linear/ReLU modules plus a final Linear readout.
    """

    def __init__(self, input_names: Sequence[str], dtype: torch.dtype = torch.float32):
        self.wires = list(input_names)
        self.dtype = dtype
        self.blocks: list[nn.Sequential] = []

    def append_block(self, specs: Sequence[WireSpec]) -> list[str]:
        if not specs:
            return []
        old = list(self.wires)
        old_index = {name: i for i, name in enumerate(old)}
        relu_terms: list[AffineExpr] = []
        for spec in specs:
            relu_terms.extend(expr for _, expr in spec.relus)

        in_dim = len(old)
        const_idx = in_dim
        hidden_dim = in_dim + 1 + len(relu_terms)
        out_dim = in_dim + len(specs)
        lin1 = nn.Linear(in_dim, hidden_dim, bias=True, dtype=self.dtype)
        lin2 = nn.Linear(hidden_dim, out_dim, bias=True, dtype=self.dtype)
        with torch.no_grad():
            lin1.weight.zero_()
            lin1.bias.zero_()
            lin2.weight.zero_()
            lin2.bias.zero_()
            for i in range(in_dim):
                lin1.weight[i, i] = 1.0
                lin2.weight[i, i] = 1.0
            lin1.bias[const_idx] = 1.0

            hidden_offset = in_dim + 1
            for j, expr in enumerate(relu_terms):
                row = hidden_offset + j
                lin1.bias[row] = float(expr.bias)
                for name, coeff in expr.coeffs.items():
                    lin1.weight[row, old_index[name]] = float(coeff)

            relu_cursor = 0
            for out_offset, spec in enumerate(specs):
                row = in_dim + out_offset
                lin2.bias[row] = float(spec.bias)
                for name, coeff in spec.coeffs.items():
                    lin2.weight[row, old_index[name]] = float(coeff)
                for coeff, _ in spec.relus:
                    lin2.weight[row, hidden_offset + relu_cursor] = float(coeff)
                    relu_cursor += 1

        self.blocks.append(nn.Sequential(lin1, nn.ReLU(), lin2))
        for spec in specs:
            self.wires.append(spec.name)
        return [spec.name for spec in specs]

    def affine(self, name: str, coeffs: dict[str, float] | None = None, bias: float = 0.0) -> str:
        self.append_block([WireSpec(name=name, coeffs=coeffs or {}, bias=bias)])
        return name

    def relu_affine(self, name: str, coeffs: dict[str, float], bias: float = 0.0) -> str:
        self.append_block([WireSpec(name=name, coeffs={}, relus=[(1.0, AffineExpr(coeffs, bias))])])
        return name

    def not_wire(self, name: str, a: str) -> str:
        return self.affine(name, {a: -1.0}, 1.0)

    def and_many(self, name: str, wires: Sequence[str]) -> str:
        if not wires:
            return self.affine(name, {}, 1.0)
        return self.relu_affine(name, {wire: 1.0 for wire in wires}, -float(len(wires) - 1))

    def or_many(self, name: str, wires: Sequence[str]) -> str:
        if not wires:
            return self.affine(name, {}, 0.0)
        coeffs = {wire: 1.0 for wire in wires}
        self.append_block(
            [
                WireSpec(
                    name=name,
                    coeffs=coeffs,
                    relus=[(-1.0, AffineExpr(coeffs, -1.0))],
                )
            ]
        )
        return name

    def xor(self, name: str, a: str, b: str) -> str:
        self.append_block(
            [
                WireSpec(
                    name=name,
                    coeffs={},
                    relus=[
                        (1.0, AffineExpr({a: 1.0, b: -1.0})),
                        (1.0, AffineExpr({a: -1.0, b: 1.0})),
                    ],
                )
            ]
        )
        return name

    def eq_bits(self, name: str, bits: Sequence[str], pattern: Sequence[int]) -> str:
        if len(bits) != len(pattern):
            raise ValueError("pattern width mismatch")
        coeffs: dict[str, float] = {}
        zeros = 0
        for bit, expected in zip(bits, pattern):
            if int(expected):
                coeffs[bit] = coeffs.get(bit, 0.0) + 1.0
            else:
                coeffs[bit] = coeffs.get(bit, 0.0) - 1.0
                zeros += 1
        return self.relu_affine(name, coeffs, float(zeros - (len(bits) - 1)))

    def mux_many(self, prefix: str, cond: str, if_true: Sequence[str], if_false: Sequence[str]) -> list[str]:
        if len(if_true) != len(if_false):
            raise ValueError("mux width mismatch")
        first: list[WireSpec] = []
        true_terms: list[str] = []
        false_terms: list[str] = []
        for i, (t, f) in enumerate(zip(if_true, if_false)):
            at = f"{prefix}_and_true_{i}"
            af = f"{prefix}_and_false_{i}"
            first.append(WireSpec(at, {}, relus=[(1.0, AffineExpr({cond: 1.0, t: 1.0}, -1.0))]))
            first.append(WireSpec(af, {}, relus=[(1.0, AffineExpr({f: 1.0, cond: -1.0}))]))
            true_terms.append(at)
            false_terms.append(af)
        self.append_block(first)
        out_specs: list[WireSpec] = []
        out_names: list[str] = []
        for i, (at, af) in enumerate(zip(true_terms, false_terms)):
            out = f"{prefix}_{i}"
            out_names.append(out)
            coeffs = {at: 1.0, af: 1.0}
            out_specs.append(WireSpec(out, coeffs=coeffs, relus=[(-1.0, AffineExpr(coeffs, -1.0))]))
        self.append_block(out_specs)
        return out_names

    def readout(self, outputs: Sequence[tuple[str, float, float]]) -> nn.Linear:
        layer = nn.Linear(len(self.wires), len(outputs), bias=True, dtype=self.dtype)
        index = {name: i for i, name in enumerate(self.wires)}
        with torch.no_grad():
            layer.weight.zero_()
            layer.bias.zero_()
            for row, (wire, scale, bias) in enumerate(outputs):
                layer.weight[row, index[wire]] = float(scale)
                layer.bias[row] = float(bias)
        return layer

    def build(self, outputs: Sequence[tuple[str, float, float]]) -> SynthesizedReLUCircuit:
        return SynthesizedReLUCircuit(self.blocks, self.readout(outputs), self.wires)
