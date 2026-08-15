from __future__ import annotations

import torch

from src.relu_boolean import ReLUCircuitBuilder


def test_relu_boolean_primitives_exact_on_binary_domain():
    builder = ReLUCircuitBuilder(["a", "b"], dtype=torch.float64)
    not_a = builder.not_wire("not_a", "a")
    and_ab = builder.and_many("and_ab", ["a", "b"])
    or_ab = builder.or_many("or_ab", ["a", "b"])
    xor_ab = builder.xor("xor_ab", "a", "b")
    mux = builder.mux_many("mux", "a", ["b"], [not_a])[0]
    net = builder.build([(wire, 1.0, 0.0) for wire in [not_a, and_ab, or_ab, xor_ab, mux]])

    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    out = net(x).round().to(torch.long)
    expected = torch.tensor(
        [
            [1, 0, 0, 0, 1],
            [1, 0, 1, 1, 1],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 0, 1],
        ],
        dtype=torch.long,
    )
    assert torch.equal(out, expected)
