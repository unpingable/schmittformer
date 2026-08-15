from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .compiled_bits import bits_to_int_tensor, int_tensor_to_bits, nonzero_test, saturating_decrement_bits, zero_test


@dataclass(frozen=True)
class CounterVerificationConfig:
    width: int
    exhaustive: bool = True
    random_samples: int = 100000
    batch_size: int = 65536
    dtype: str = "torch.float32"
    device: str = "cpu"


class CompiledSaturatingDecrementer(nn.Module):
    """Deterministically synthesized fixed-width decrement circuit."""

    def __init__(self, width: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.width = int(width)
        self.dtype = dtype

    def forward(self, bits: Tensor) -> dict[str, Tensor]:
        bits = bits.to(self.dtype)
        dec, is_zero, is_nonzero = saturating_decrement_bits(bits)
        return {"decremented_bits": dec, "is_zero": is_zero, "is_nonzero": is_nonzero}


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def verify_counter(
    width: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    exhaustive: bool = True,
    random_samples: int = 100000,
    batch_size: int = 65536,
    seed: int = 1234,
) -> dict[str, Any]:
    device = torch.device(device)
    model = CompiledSaturatingDecrementer(width, dtype=dtype).to(device)
    max_value = 1 << width
    if exhaustive:
        values = torch.arange(max_value, dtype=torch.long, device=device)
    else:
        generator = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
        generator.manual_seed(seed)
        random_values = torch.randint(0, max_value, (random_samples,), dtype=torch.long, device=device, generator=generator)
        boundaries = torch.tensor(
            [0, 1, 2, 3, 15, 16, 31, 32, 127, 128, 255, 256, 32767, 32768, max_value - 2, max_value - 1],
            dtype=torch.long,
            device=device,
        )
        boundaries = boundaries[(boundaries >= 0) & (boundaries < max_value)]
        values = torch.cat([boundaries, random_values], dim=0).unique()
    checked = 0
    failures: list[dict[str, int]] = []
    min_margin = float("inf")
    start = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for start_idx in range(0, values.numel(), batch_size):
            chunk = values[start_idx : start_idx + batch_size]
            bits = int_tensor_to_bits(chunk, width, dtype=dtype)
            out = model(bits)
            actual = bits_to_int_tensor(out["decremented_bits"])
            expected = torch.clamp(chunk - 1, min=0)
            expected_zero = (chunk == 0).to(dtype)
            expected_nonzero = (chunk != 0).to(dtype)
            bad = (actual != expected) | (out["is_zero"].round().to(torch.long) != expected_zero.to(torch.long)) | (out["is_nonzero"].round().to(torch.long) != expected_nonzero.to(torch.long))
            if bad.any():
                idxs = torch.nonzero(bad, as_tuple=False).flatten()[:20]
                for idx in idxs.detach().cpu().tolist():
                    failures.append(
                        {
                            "input": int(chunk[idx].detach().cpu().item()),
                            "expected": int(expected[idx].detach().cpu().item()),
                            "actual": int(actual[idx].detach().cpu().item()),
                        }
                    )
            margins = torch.minimum(torch.abs(out["decremented_bits"] - 0.5).min(), torch.minimum(torch.abs(out["is_zero"] - 0.5).min(), torch.abs(out["is_nonzero"] - 0.5).min()))
            min_margin = min(min_margin, float(margins.detach().cpu().item()))
            checked += int(chunk.numel())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.time() - start
    result: dict[str, Any] = {
        "width": width,
        "device": str(device),
        "dtype": str(dtype),
        "exhaustive": bool(exhaustive),
        "checked": checked,
        "passed": not failures,
        "failures": failures[:20],
        "elapsed_seconds": elapsed,
        "values_per_second": checked / elapsed if elapsed else None,
        "min_bit_margin_to_half": min_margin,
        "construction": "ripple_borrow_saturating_decrement",
        "complexity": {"gates": {"xor": width, "and_borrow": width, "or_tree_linear_threshold": 1, "mux_bits": width}, "depth_ripple": width},
    }
    if device.type == "cuda":
        result["peak_gpu_memory_mib"] = float(torch.cuda.max_memory_allocated(device) / 2**20)
    return result
