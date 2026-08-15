from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from torch import Tensor

from .projection_task import Decision, Scope, Witness, decisions_from_tensors


class RegisterEncoding(str, Enum):
    BINARY_PAIR = "binary_pair"
    GROUPED_ONE_HOT = "grouped_one_hot"
    JOINT_ONE_HOT = "joint_one_hot"


@dataclass(frozen=True)
class RegisterDecode:
    witness: Tensor
    scope: Tensor
    valid: Tensor
    distance: Tensor
    margin: Tensor


def register_dim(encoding: RegisterEncoding | str) -> int:
    encoding = RegisterEncoding(encoding)
    if encoding == RegisterEncoding.BINARY_PAIR:
        return 2
    if encoding == RegisterEncoding.GROUPED_ONE_HOT:
        return 4
    if encoding == RegisterEncoding.JOINT_ONE_HOT:
        return 4
    raise AssertionError(encoding)


def _labels_to_sign(values: Tensor) -> Tensor:
    return values.to(torch.float32) * 2.0 - 1.0


def register_code(encoding: RegisterEncoding | str, witness: Tensor, scope: Tensor) -> Tensor:
    encoding = RegisterEncoding(encoding)
    witness = witness.to(torch.long)
    scope = scope.to(torch.long)
    if encoding == RegisterEncoding.BINARY_PAIR:
        return torch.stack([_labels_to_sign(witness), _labels_to_sign(scope)], dim=-1)
    if encoding == RegisterEncoding.GROUPED_ONE_HOT:
        return torch.cat(
            [
                torch.nn.functional.one_hot(witness, num_classes=2).to(torch.float32),
                torch.nn.functional.one_hot(scope, num_classes=2).to(torch.float32),
            ],
            dim=-1,
        )
    if encoding == RegisterEncoding.JOINT_ONE_HOT:
        joint = witness * 2 + scope
        return torch.nn.functional.one_hot(joint, num_classes=4).to(torch.float32)
    raise AssertionError(encoding)


def all_codebook(encoding: RegisterEncoding | str, device: torch.device | None = None) -> tuple[Tensor, Tensor, Tensor]:
    device = device or torch.device("cpu")
    witness = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=device)
    scope = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=device)
    return register_code(encoding, witness, scope), witness, scope


def register_variable_logits(register: Tensor, encoding: RegisterEncoding | str) -> tuple[Tensor, Tensor]:
    encoding = RegisterEncoding(encoding)
    if encoding == RegisterEncoding.BINARY_PAIR:
        return torch.stack([-register[:, 0], register[:, 0]], dim=-1), torch.stack([-register[:, 1], register[:, 1]], dim=-1)
    if encoding == RegisterEncoding.GROUPED_ONE_HOT:
        return register[:, 0:2], register[:, 2:4]
    if encoding == RegisterEncoding.JOINT_ONE_HOT:
        invalid = torch.logsumexp(register[:, 0:2], dim=-1)
        valid = torch.logsumexp(register[:, 2:4], dim=-1)
        scope_a = torch.logsumexp(register[:, [0, 2]], dim=-1)
        scope_b = torch.logsumexp(register[:, [1, 3]], dim=-1)
        return torch.stack([invalid, valid], dim=-1), torch.stack([scope_a, scope_b], dim=-1)
    raise AssertionError(encoding)


def register_state_logits(register: Tensor, encoding: RegisterEncoding | str) -> Tensor:
    codebook, _, _ = all_codebook(encoding, register.device)
    distances = ((register[:, None, :] - codebook[None, :, :]) ** 2).sum(dim=-1)
    return -distances


def decode_register(register: Tensor, encoding: RegisterEncoding | str, tolerance: float = 0.75) -> RegisterDecode:
    encoding = RegisterEncoding(encoding)
    finite = torch.isfinite(register).all(dim=-1)
    codebook, witness_values, scope_values = all_codebook(encoding, register.device)
    distances = ((register[:, None, :] - codebook[None, :, :]) ** 2).sum(dim=-1)
    ordered = distances.argsort(dim=-1)
    best_index = ordered[:, 0]
    second_index = ordered[:, 1]
    best_distance = distances.gather(1, best_index[:, None]).squeeze(1)
    second_distance = distances.gather(1, second_index[:, None]).squeeze(1)
    valid = finite & (best_distance <= float(tolerance))
    return RegisterDecode(
        witness=witness_values[best_index].to(torch.long),
        scope=scope_values[best_index].to(torch.long),
        valid=valid,
        distance=best_distance,
        margin=second_distance - best_distance,
    )


def register_policy_decision(
    proposal: Tensor,
    register: Tensor,
    encoding: RegisterEncoding | str,
    tolerance: float = 0.75,
) -> tuple[Tensor, RegisterDecode]:
    proposal = proposal.to(torch.long)
    decoded = decode_register(register, encoding, tolerance)
    oracle = decisions_from_tensors(proposal, decoded.witness, decoded.scope)
    invalid_refusal = torch.full_like(proposal, int(Decision.REFUSE_INSUFFICIENT_INFORMATION))
    decision = torch.where(decoded.valid, oracle, invalid_refusal)
    decision = torch.where(
        proposal == 0,
        torch.full_like(decision, int(Decision.REFUSE_NO_PROPOSAL)),
        decision,
    )
    return decision.to(torch.long), decoded


def register_relative_expected(proposal: Tensor, decoded: RegisterDecode) -> Tensor:
    proposal = proposal.to(torch.long)
    oracle = decisions_from_tensors(proposal, decoded.witness, decoded.scope)
    invalid_refusal = torch.full_like(proposal, int(Decision.REFUSE_INSUFFICIENT_INFORMATION))
    decision = torch.where(decoded.valid, oracle, invalid_refusal)
    return torch.where(
        proposal == 0,
        torch.full_like(decision, int(Decision.REFUSE_NO_PROPOSAL)),
        decision,
    ).to(torch.long)


def metadata_monitor_from_register(
    proposal: Tensor,
    register: Tensor,
    encoding: RegisterEncoding | str,
    tolerance: float = 0.75,
) -> Tensor:
    decision, _ = register_policy_decision(proposal, register, encoding, tolerance)
    return decision


def register_accuracy(decoded: RegisterDecode, witness: Tensor, scope: Tensor) -> dict[str, float]:
    witness = witness.to(torch.long)
    scope = scope.to(torch.long)
    valid = decoded.valid
    return {
        "register_valid_rate": float(valid.to(torch.float32).mean().item()),
        "witness_accuracy": float(((decoded.witness == witness) & valid).to(torch.float32).mean().item()),
        "scope_accuracy": float(((decoded.scope == scope) & valid).to(torch.float32).mean().item()),
        "joint_accuracy": float(((decoded.witness == witness) & (decoded.scope == scope) & valid).to(torch.float32).mean().item()),
        "decode_margin_mean": float(decoded.margin.detach().cpu().mean().item()),
        "decode_margin_min": float(decoded.margin.detach().cpu().min().item()),
        "decode_distance_mean": float(decoded.distance.detach().cpu().mean().item()),
        "decode_distance_max": float(decoded.distance.detach().cpu().max().item()),
    }


def corrupt_register(register: Tensor, encoding: RegisterEncoding | str, mode: str) -> Tensor:
    encoding = RegisterEncoding(encoding)
    out = register.clone()
    if mode == "clean":
        return out
    if mode == "zero_vector":
        return torch.zeros_like(out)
    if mode == "nan":
        out[:, 0] = float("nan")
        return out
    if mode == "large_out_of_domain":
        return torch.full_like(out, 3.0)
    decoded = decode_register(register, encoding, tolerance=10_000.0)
    witness = decoded.witness.clone()
    scope = decoded.scope.clone()
    if mode == "bit_flip_witness":
        witness = 1 - witness
        return register_code(encoding, witness, scope).to(register.device)
    if mode == "bit_flip_scope":
        scope = 1 - scope
        return register_code(encoding, witness, scope).to(register.device)
    if mode == "partial_stale_witness_valid":
        witness = torch.full_like(witness, int(Witness.VALID))
        return register_code(encoding, witness, scope).to(register.device)
    if mode == "partial_stale_scope_a":
        scope = torch.full_like(scope, int(Scope.A))
        return register_code(encoding, witness, scope).to(register.device)
    raise ValueError(f"unknown corruption mode: {mode}")


def corruption_modes() -> list[str]:
    return [
        "clean",
        "zero_vector",
        "nan",
        "large_out_of_domain",
        "bit_flip_witness",
        "bit_flip_scope",
        "partial_stale_witness_valid",
        "partial_stale_scope_a",
    ]


def tensor_stats(values: Tensor) -> dict[str, float]:
    values = values.detach().to(torch.float32).cpu()
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def decode_summary(register: Tensor, encoding: RegisterEncoding | str, witness: Tensor, scope: Tensor) -> dict[str, Any]:
    decoded = decode_register(register, encoding)
    return {**register_accuracy(decoded, witness, scope), "margin_stats": tensor_stats(decoded.margin)}
