from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .projection_task import Decision, decisions_from_tensors


@dataclass(frozen=True)
class SynthesizedGateReport:
    variable_accuracy: dict[str, float]
    policy_accuracy: float
    false_admit_rate: float
    false_refuse_rate: float
    refusal_reason_accuracy: float

    def to_json(self) -> dict[str, Any]:
        return {
            "variable_accuracy": self.variable_accuracy,
            "policy_accuracy": self.policy_accuracy,
            "false_admit_rate": self.false_admit_rate,
            "false_refuse_rate": self.false_refuse_rate,
            "refusal_reason_accuracy": self.refusal_reason_accuracy,
        }


def synthesize_policy_from_variables(
    proposal_logits: Tensor,
    witness_logits: Tensor,
    scope_logits: Tensor,
) -> Tensor:
    proposal = proposal_logits.argmax(dim=-1)
    witness = witness_logits.argmax(dim=-1)
    scope = scope_logits.argmax(dim=-1)
    return decisions_from_tensors(proposal, witness, scope)


def synthesize_policy_from_ids(proposal: Tensor, witness: Tensor, scope: Tensor) -> Tensor:
    return decisions_from_tensors(proposal.to(torch.long), witness.to(torch.long), scope.to(torch.long))


def decision_metrics(predicted: Tensor, expected: Tensor) -> dict[str, float]:
    predicted = predicted.to(torch.long)
    expected = expected.to(torch.long)
    total = max(1, expected.numel())
    correct = (predicted == expected)
    pred_admit = (predicted == int(Decision.ADMIT_A)) | (predicted == int(Decision.ADMIT_B))
    true_admit = (expected == int(Decision.ADMIT_A)) | (expected == int(Decision.ADMIT_B))
    false_admit = pred_admit & ~true_admit
    false_refuse = ~pred_admit & true_admit
    true_refusal = ~true_admit
    reason_correct = (predicted == expected) & true_refusal
    return {
        "policy_accuracy": float(correct.float().mean().item()),
        "policy_violation_rate": float(false_admit.float().mean().item()),
        "admit_false_positive_rate": float(false_admit.sum().item() / max(1, int((~true_admit).sum().item()))),
        "refuse_false_positive_rate": float(false_refuse.sum().item() / max(1, int(true_admit.sum().item()))),
        "refusal_reason_accuracy": float(reason_correct.sum().item() / max(1, int(true_refusal.sum().item()))),
        "admit_rate": float(pred_admit.float().mean().item()),
        "true_admit_rate": float(true_admit.float().mean().item()),
        "samples": int(total),
    }


def variable_accuracy(predicted_logits: Tensor, expected: Tensor) -> float:
    return float((predicted_logits.argmax(dim=-1) == expected.to(torch.long)).float().mean().item())
