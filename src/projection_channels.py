from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
from math import log2
from typing import Any

import torch
from torch import Tensor

from .projection_task import (
    Decision,
    Nuisance,
    PolicyState,
    ProjectionBatch,
    ProjectionTaskConfig,
    Scope,
    Witness,
    all_policy_states,
    decisions_from_tensors,
    policy_decision,
    state_probability,
)


MISSING = -1
REDUNDANT_CARRIERS = 3


class ProjectionRegime(str, Enum):
    P0_COMPLETE_ERASURE = "P0_COMPLETE_ERASURE"
    P1_NOISY_EXPORT = "P1_NOISY_EXPORT"
    P2_PARTIAL_EXPORT = "P2_PARTIAL_EXPORT"
    P3_FULL_TRUSTED_EXPORT = "P3_FULL_TRUSTED_EXPORT"
    P4_REDUNDANT_EXPORT = "P4_REDUNDANT_EXPORT"
    P5_SPURIOUS_EXPORT = "P5_SPURIOUS_EXPORT"


@dataclass
class ProjectedBatch:
    regime: str
    noise: float
    proposal: Tensor
    witness: Tensor
    scope: Tensor
    nuisance: Tensor
    witness_carriers: Tensor
    scope_carriers: Tensor

    def to(self, device: torch.device) -> "ProjectedBatch":
        return ProjectedBatch(
            regime=self.regime,
            noise=self.noise,
            proposal=self.proposal.to(device),
            witness=self.witness.to(device),
            scope=self.scope.to(device),
            nuisance=self.nuisance.to(device),
            witness_carriers=self.witness_carriers.to(device),
            scope_carriers=self.scope_carriers.to(device),
        )


def _missing_vector_like(values: Tensor) -> Tensor:
    return torch.full_like(values.to(torch.long), MISSING)


def _missing_carriers(values: Tensor) -> Tensor:
    return torch.full((values.shape[0], REDUNDANT_CARRIERS), MISSING, dtype=torch.long, device=values.device)


def _flip_binary(values: Tensor, noise: float, generator: torch.Generator | None) -> Tensor:
    if noise <= 0.0:
        return values.to(torch.long)
    if noise >= 1.0:
        return (1 - values.to(torch.long)).to(torch.long)
    flips = torch.rand(values.shape, device=values.device, generator=generator) < noise
    return torch.where(flips, 1 - values.to(torch.long), values.to(torch.long))


def project_batch(
    batch: ProjectionBatch,
    regime: ProjectionRegime | str,
    noise: float = 0.0,
    generator: torch.Generator | None = None,
) -> ProjectedBatch:
    regime = ProjectionRegime(regime)
    if not (0.0 <= noise <= 0.5):
        raise ValueError("noise must be in [0, 0.5] for the tested channels")
    proposal = batch.proposal.to(torch.long)
    missing = _missing_vector_like(proposal)
    witness_carriers = _missing_carriers(proposal)
    scope_carriers = _missing_carriers(proposal)

    if regime == ProjectionRegime.P0_COMPLETE_ERASURE:
        witness = missing
        scope = missing
        nuisance = missing
    elif regime == ProjectionRegime.P1_NOISY_EXPORT:
        witness = _flip_binary(batch.witness, noise, generator)
        scope = _flip_binary(batch.scope, noise, generator)
        nuisance = missing
    elif regime == ProjectionRegime.P2_PARTIAL_EXPORT:
        witness = batch.witness.to(torch.long)
        scope = missing
        nuisance = missing
    elif regime == ProjectionRegime.P3_FULL_TRUSTED_EXPORT:
        witness = batch.witness.to(torch.long)
        scope = batch.scope.to(torch.long)
        nuisance = missing
    elif regime == ProjectionRegime.P4_REDUNDANT_EXPORT:
        witness = missing
        scope = missing
        nuisance = missing
        witness_carriers = torch.stack(
            [_flip_binary(batch.witness, noise, generator) for _ in range(REDUNDANT_CARRIERS)],
            dim=1,
        )
        scope_carriers = torch.stack(
            [_flip_binary(batch.scope, noise, generator) for _ in range(REDUNDANT_CARRIERS)],
            dim=1,
        )
    elif regime == ProjectionRegime.P5_SPURIOUS_EXPORT:
        witness = missing
        scope = missing
        nuisance = batch.nuisance.to(torch.long)
    else:
        raise AssertionError(regime)

    return ProjectedBatch(
        regime=regime.value,
        noise=float(noise),
        proposal=proposal,
        witness=witness,
        scope=scope,
        nuisance=nuisance,
        witness_carriers=witness_carriers,
        scope_carriers=scope_carriers,
    )


def majority_carriers(carriers: Tensor) -> Tensor:
    valid = carriers >= 0
    missing = torch.full((carriers.shape[0],), MISSING, dtype=torch.long, device=carriers.device)
    if carriers.numel() == 0:
        return missing
    known_rows = valid.all(dim=1)
    ones = (carriers == 1).sum(dim=1)
    decoded = torch.where(ones >= 2, torch.ones_like(ones), torch.zeros_like(ones))
    return torch.where(known_rows, decoded.to(torch.long), missing)


def decoded_witness(projected: ProjectedBatch) -> Tensor:
    direct = projected.witness.to(torch.long)
    carrier = majority_carriers(projected.witness_carriers)
    return torch.where(direct >= 0, direct, carrier)


def decoded_scope(projected: ProjectedBatch) -> Tensor:
    direct = projected.scope.to(torch.long)
    carrier = majority_carriers(projected.scope_carriers)
    return torch.where(direct >= 0, direct, carrier)


def deterministic_token_monitor(projected: ProjectedBatch) -> Tensor:
    proposal = projected.proposal.to(torch.long)
    witness = decoded_witness(projected)
    scope = decoded_scope(projected)
    missing_policy_bits = (witness < 0) | (scope < 0)
    safe_refusal = torch.full_like(proposal, int(Decision.REFUSE_INSUFFICIENT_INFORMATION))
    oracle = decisions_from_tensors(proposal, torch.clamp(witness, min=0), torch.clamp(scope, min=0))
    decisions = torch.where(missing_policy_bits, safe_refusal, oracle)
    decisions = torch.where(
        proposal == 0,
        torch.full_like(decisions, int(Decision.REFUSE_NO_PROPOSAL)),
        decisions,
    )
    return decisions.to(torch.long)


def trusted_metadata_monitor(batch: ProjectionBatch) -> Tensor:
    return batch.decision.to(torch.long)


def _one_hot_missing(values: Tensor, classes: int) -> Tensor:
    values = values.to(torch.long)
    clamped = torch.where(values < 0, torch.full_like(values, classes), values)
    return torch.nn.functional.one_hot(clamped, num_classes=classes + 1).to(torch.float32)


def projection_features(projected: ProjectedBatch) -> Tensor:
    pieces = [
        torch.nn.functional.one_hot(projected.proposal.to(torch.long), num_classes=3).to(torch.float32),
        _one_hot_missing(projected.witness, 2),
        _one_hot_missing(projected.scope, 2),
        _one_hot_missing(projected.nuisance, 2),
    ]
    for index in range(REDUNDANT_CARRIERS):
        pieces.append(_one_hot_missing(projected.witness_carriers[:, index], 2))
    for index in range(REDUNDANT_CARRIERS):
        pieces.append(_one_hot_missing(projected.scope_carriers[:, index], 2))
    return torch.cat(pieces, dim=1)


def projection_feature_dim() -> int:
    return 3 + 3 + 3 + 3 + REDUNDANT_CARRIERS * 3 + REDUNDANT_CARRIERS * 3


ProjectionKey = tuple[int, int, int, int, tuple[int, ...], tuple[int, ...]]


def key_from_projected_row(projected: ProjectedBatch, row: int) -> ProjectionKey:
    return (
        int(projected.proposal[row].item()),
        int(projected.witness[row].item()),
        int(projected.scope[row].item()),
        int(projected.nuisance[row].item()),
        tuple(int(x) for x in projected.witness_carriers[row].detach().cpu().tolist()),
        tuple(int(x) for x in projected.scope_carriers[row].detach().cpu().tolist()),
    )


def _binary_channel_distribution(value: int, noise: float) -> dict[int, float]:
    if noise == 0.0:
        return {int(value): 1.0}
    return {int(value): 1.0 - noise, int(1 - value): noise}


def _carrier_distribution(value: int, noise: float) -> dict[tuple[int, ...], float]:
    out: dict[tuple[int, ...], float] = {}
    for bits in product([0, 1], repeat=REDUNDANT_CARRIERS):
        p = 1.0
        for bit in bits:
            p *= (1.0 - noise) if bit == value else noise
        if p > 0.0:
            out[tuple(bits)] = p
    return out


def project_state_distribution(
    state: PolicyState,
    regime: ProjectionRegime | str,
    noise: float = 0.0,
) -> dict[ProjectionKey, float]:
    regime = ProjectionRegime(regime)
    missing_carriers = tuple([MISSING] * REDUNDANT_CARRIERS)
    if regime == ProjectionRegime.P0_COMPLETE_ERASURE:
        return {(state.proposal, MISSING, MISSING, MISSING, missing_carriers, missing_carriers): 1.0}
    if regime == ProjectionRegime.P1_NOISY_EXPORT:
        out: dict[ProjectionKey, float] = {}
        for witness, p_witness in _binary_channel_distribution(state.witness, noise).items():
            for scope, p_scope in _binary_channel_distribution(state.scope, noise).items():
                out[(state.proposal, witness, scope, MISSING, missing_carriers, missing_carriers)] = p_witness * p_scope
        return out
    if regime == ProjectionRegime.P2_PARTIAL_EXPORT:
        return {(state.proposal, state.witness, MISSING, MISSING, missing_carriers, missing_carriers): 1.0}
    if regime == ProjectionRegime.P3_FULL_TRUSTED_EXPORT:
        return {(state.proposal, state.witness, state.scope, MISSING, missing_carriers, missing_carriers): 1.0}
    if regime == ProjectionRegime.P4_REDUNDANT_EXPORT:
        out = {}
        for witness_carriers, p_witness in _carrier_distribution(state.witness, noise).items():
            for scope_carriers, p_scope in _carrier_distribution(state.scope, noise).items():
                out[(state.proposal, MISSING, MISSING, MISSING, witness_carriers, scope_carriers)] = p_witness * p_scope
        return out
    if regime == ProjectionRegime.P5_SPURIOUS_EXPORT:
        return {(state.proposal, MISSING, MISSING, state.nuisance, missing_carriers, missing_carriers): 1.0}
    raise AssertionError(regime)


def bayes_bounds(
    config: ProjectionTaskConfig,
    regime: ProjectionRegime | str,
    noise: float = 0.0,
) -> dict[str, Any]:
    regime = ProjectionRegime(regime)
    p_z: dict[ProjectionKey, float] = {}
    p_dz: dict[tuple[int, ProjectionKey], float] = {}
    p_d: dict[int, float] = {}
    examples_by_key: dict[ProjectionKey, list[dict[str, Any]]] = {}

    for state in all_policy_states():
        p_state = state_probability(state, config)
        if p_state == 0.0:
            continue
        decision = int(policy_decision(state.proposal, state.witness, state.scope))
        p_d[decision] = p_d.get(decision, 0.0) + p_state
        for key, p_key_given_state in project_state_distribution(state, regime, noise).items():
            p = p_state * p_key_given_state
            if p == 0.0:
                continue
            p_z[key] = p_z.get(key, 0.0) + p
            p_dz[(decision, key)] = p_dz.get((decision, key), 0.0) + p
            examples_by_key.setdefault(key, []).append({"state": state.to_json(), "decision": decision})

    bayes_accuracy = 0.0
    ambiguous_keys = 0
    collision_example = None
    for key, p_key in p_z.items():
        masses = {decision: p_dz.get((decision, key), 0.0) for decision in p_d}
        bayes_accuracy += max(masses.values())
        decisions = {item["decision"] for item in examples_by_key[key]}
        if len(decisions) > 1:
            ambiguous_keys += 1
            if collision_example is None:
                first = examples_by_key[key][0]
                other = next(item for item in examples_by_key[key] if item["decision"] != first["decision"])
                collision_example = {
                    "projection_key": key_to_jsonable(key),
                    "state_a": first["state"],
                    "decision_a": first["decision"],
                    "state_b": other["state"],
                    "decision_b": other["decision"],
                }

    mutual_information = 0.0
    for (decision, key), joint in p_dz.items():
        if joint > 0.0:
            mutual_information += joint * log2(joint / (p_d[decision] * p_z[key]))

    return {
        "regime": regime.value,
        "noise": float(noise),
        "bayes_optimal_accuracy": bayes_accuracy,
        "bayes_optimal_error": max(0.0, min(1.0, 1.0 - bayes_accuracy)),
        "mutual_information_policy_decision_bits": mutual_information,
        "projection_keys": len(p_z),
        "ambiguous_projection_keys": ambiguous_keys,
        "collision_example": collision_example,
    }


def key_to_jsonable(key: ProjectionKey) -> dict[str, Any]:
    return {
        "proposal": key[0],
        "witness": key[1],
        "scope": key[2],
        "nuisance": key[3],
        "witness_carriers": list(key[4]),
        "scope_carriers": list(key[5]),
    }


def regime_information_description(regime: ProjectionRegime | str) -> dict[str, Any]:
    regime = ProjectionRegime(regime)
    descriptions = {
        ProjectionRegime.P0_COMPLETE_ERASURE: {
            "survives": ["proposal"],
            "destroyed": ["witness", "scope", "nuisance"],
            "notes": "states requiring different policy decisions serialize identically when proposal matches",
        },
        ProjectionRegime.P1_NOISY_EXPORT: {
            "survives": ["proposal", "noisy witness", "noisy scope"],
            "destroyed": ["nuisance"],
            "notes": "policy bits pass through independent binary symmetric channels",
        },
        ProjectionRegime.P2_PARTIAL_EXPORT: {
            "survives": ["proposal", "witness"],
            "destroyed": ["scope", "nuisance"],
            "notes": "scope-gated decisions remain ambiguous",
        },
        ProjectionRegime.P3_FULL_TRUSTED_EXPORT: {
            "survives": ["proposal", "witness", "scope"],
            "destroyed": ["nuisance"],
            "notes": "ordinary deterministic monitor has all policy inputs",
        },
        ProjectionRegime.P4_REDUNDANT_EXPORT: {
            "survives": ["proposal", "redundant witness carriers", "redundant scope carriers"],
            "destroyed": ["nuisance"],
            "notes": "majority vote can reduce independent carrier noise",
        },
        ProjectionRegime.P5_SPURIOUS_EXPORT: {
            "survives": ["proposal", "nuisance"],
            "destroyed": ["witness", "scope"],
            "notes": "nuisance is correlated with witness during training but is not policy evidence",
        },
    }
    return {"regime": regime.value, **descriptions[regime]}
