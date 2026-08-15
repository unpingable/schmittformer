from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .projection_task import Decision, decisions_from_tensors
from .semantic_register import RegisterEncoding, decode_register, metadata_monitor_from_register, register_policy_decision
from .synthesized_latent_gate import decision_metrics


def explicit_register_synthesized_gate(
    proposal: Tensor,
    register: Tensor,
    encoding: RegisterEncoding | str,
    tolerance: float = 0.75,
) -> Tensor:
    decision, _ = register_policy_decision(proposal, register, encoding, tolerance)
    return decision


def external_metadata_monitor_from_register(
    proposal: Tensor,
    register: Tensor,
    encoding: RegisterEncoding | str,
    tolerance: float = 0.75,
) -> Tensor:
    return metadata_monitor_from_register(proposal, register, encoding, tolerance)


def metadata_equivalence_report(
    proposal: Tensor,
    register: Tensor,
    encoding: RegisterEncoding | str,
    tolerance: float = 0.75,
) -> dict[str, Any]:
    internal = explicit_register_synthesized_gate(proposal, register, encoding, tolerance)
    external = external_metadata_monitor_from_register(proposal, register, encoding, tolerance)
    return {
        "samples": int(internal.numel()),
        "exact_match_rate": float((internal == external).to(torch.float32).mean().item()),
        "mismatches": int((internal != external).sum().item()),
    }


def register_relative_report(
    proposal: Tensor,
    register: Tensor,
    encoding: RegisterEncoding | str,
    tolerance: float = 0.75,
) -> dict[str, Any]:
    decision, decoded = register_policy_decision(proposal, register, encoding, tolerance)
    expected = decisions_from_tensors(proposal.to(torch.long), decoded.witness, decoded.scope)
    expected = torch.where(
        decoded.valid,
        expected,
        torch.full_like(expected, int(Decision.REFUSE_INSUFFICIENT_INFORMATION)),
    )
    expected = torch.where(
        proposal.to(torch.long) == 0,
        torch.full_like(expected, int(Decision.REFUSE_NO_PROPOSAL)),
        expected,
    )
    return decision_metrics(decision.detach().cpu(), expected.detach().cpu())
