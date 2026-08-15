from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .projection_task import ProjectionBatch, decisions_from_tensors
from .semantic_register import RegisterEncoding, decode_register, register_code, register_policy_decision
from .synthesized_latent_gate import decision_metrics


@dataclass(frozen=True)
class RegisterInterventionResult:
    intervention: str
    samples: int
    semantic_consistency: float
    world_target_consistency: float
    changed_decision_rate: float
    invalid_rate: float

    def to_json(self) -> dict[str, Any]:
        return {
            "intervention": self.intervention,
            "samples": self.samples,
            "semantic_consistency": self.semantic_consistency,
            "world_target_consistency": self.world_target_consistency,
            "changed_decision_rate": self.changed_decision_rate,
            "invalid_rate": self.invalid_rate,
        }


def semantic_swap_register(
    target_register: Tensor,
    source_register: Tensor,
    encoding: RegisterEncoding | str,
    variable: str,
) -> Tensor:
    target = decode_register(target_register, encoding, tolerance=10_000.0)
    source = decode_register(source_register, encoding, tolerance=10_000.0)
    witness = target.witness.clone()
    scope = target.scope.clone()
    if variable == "witness":
        witness = source.witness.clone()
    elif variable == "scope":
        scope = source.scope.clone()
    elif variable == "both":
        witness = source.witness.clone()
        scope = source.scope.clone()
    elif variable == "nuisance" or variable == "none":
        pass
    else:
        raise ValueError(f"unknown intervention variable: {variable}")
    return register_code(encoding, witness, scope).to(target_register.device)


def random_register_control(target_register: Tensor, scale: float = 0.05, generator: torch.Generator | None = None) -> Tensor:
    noise = torch.randn(target_register.shape, device=target_register.device, generator=generator) * scale
    return target_register + noise


def evaluate_register_intervention(
    target_batch: ProjectionBatch,
    source_batch: ProjectionBatch,
    target_register: Tensor,
    source_register: Tensor,
    encoding: RegisterEncoding | str,
    variable: str,
) -> RegisterInterventionResult:
    before, _ = register_policy_decision(target_batch.proposal, target_register, encoding)
    if variable == "random":
        intervened = random_register_control(target_register)
    else:
        intervened = semantic_swap_register(target_register, source_register, encoding, variable)
    after, decoded = register_policy_decision(target_batch.proposal, intervened, encoding)
    semantic_expected = decisions_from_tensors(target_batch.proposal, decoded.witness, decoded.scope)
    semantic_expected = torch.where(decoded.valid, semantic_expected, torch.full_like(semantic_expected, 5))
    if variable == "witness":
        world_expected = decisions_from_tensors(target_batch.proposal, source_batch.witness, target_batch.scope)
    elif variable == "scope":
        world_expected = decisions_from_tensors(target_batch.proposal, target_batch.witness, source_batch.scope)
    elif variable == "both":
        world_expected = decisions_from_tensors(target_batch.proposal, source_batch.witness, source_batch.scope)
    else:
        world_expected = target_batch.decision
    return RegisterInterventionResult(
        intervention=variable,
        samples=int(after.numel()),
        semantic_consistency=float((after == semantic_expected).to(torch.float32).mean().item()),
        world_target_consistency=float((after == world_expected).to(torch.float32).mean().item()),
        changed_decision_rate=float((after != before).to(torch.float32).mean().item()),
        invalid_rate=float((~decoded.valid).to(torch.float32).mean().item()),
    )
