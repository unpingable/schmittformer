from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import torch
from torch import Tensor

from .projection_model import ProjectionCausalTransformer, make_generator
from .projection_task import (
    Decision,
    Nuisance,
    PolicyState,
    ProjectionBatch,
    ProjectionTaskConfig,
    Proposal,
    Scope,
    Token,
    Witness,
    decisions_from_tensors,
    sample_policy_batch,
    token_for_proposal,
)
from .synthesized_latent_gate import decision_metrics


@dataclass(frozen=True)
class LinearReadout:
    weights: Tensor
    classes: int

    def logits(self, representation: Tensor) -> Tensor:
        ones = torch.ones((representation.shape[0], 1), dtype=representation.dtype, device=representation.device)
        return torch.cat([representation, ones], dim=1) @ self.weights.to(representation.device, representation.dtype)

    def predict(self, representation: Tensor) -> Tensor:
        return self.logits(representation).argmax(dim=-1)


@dataclass(frozen=True)
class VariableReadouts:
    proposal: LinearReadout
    witness: LinearReadout
    scope: LinearReadout
    nuisance: LinearReadout
    decision: LinearReadout


@dataclass
class RepresentationSet:
    representations: list[Tensor]
    proposal: Tensor
    witness: Tensor
    scope: Tensor
    nuisance: Tensor
    decision: Tensor


def _with_bias(x: Tensor) -> Tensor:
    ones = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
    return torch.cat([x, ones], dim=1)


def fit_linear_readout(x: Tensor, labels: Tensor, classes: int, ridge: float = 1.0e-3) -> LinearReadout:
    x = x.detach().to(torch.float64)
    labels = labels.detach().to(torch.long)
    xb = _with_bias(x)
    y = torch.nn.functional.one_hot(labels, num_classes=classes).to(torch.float64)
    eye = torch.eye(xb.shape[1], dtype=torch.float64, device=xb.device)
    eye[-1, -1] = 0.0
    weights = torch.linalg.solve(xb.T @ xb + ridge * eye, xb.T @ y)
    return LinearReadout(weights=weights.to(torch.float32).cpu(), classes=classes)


def fit_variable_readouts(data: RepresentationSet, layer_index: int, ridge: float = 1.0e-3) -> VariableReadouts:
    x = data.representations[layer_index]
    return VariableReadouts(
        proposal=fit_linear_readout(x, data.proposal, 3, ridge),
        witness=fit_linear_readout(x, data.witness, 2, ridge),
        scope=fit_linear_readout(x, data.scope, 2, ridge),
        nuisance=fit_linear_readout(x, data.nuisance, 2, ridge),
        decision=fit_linear_readout(x, data.decision, len(Decision), ridge),
    )


def collect_representations(
    model: ProjectionCausalTransformer,
    batch: ProjectionBatch,
    device: torch.device,
) -> RepresentationSet:
    model.eval()
    batch = batch.to(device)
    with torch.no_grad():
        hidden = model.hidden_states(batch.tokens)
    reps = [h[:, -1, :].detach().cpu().to(torch.float32) for h in hidden]
    return RepresentationSet(
        representations=reps,
        proposal=batch.proposal.detach().cpu(),
        witness=batch.witness.detach().cpu(),
        scope=batch.scope.detach().cpu(),
        nuisance=batch.nuisance.detach().cpu(),
        decision=batch.decision.detach().cpu(),
    )


def sample_representation_set(
    model: ProjectionCausalTransformer,
    seq_len: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    nuisance_corr: float = 0.95,
) -> RepresentationSet:
    generator = make_generator(seed, device)
    batch = sample_policy_batch(
        batch_size,
        config=ProjectionTaskConfig(
            seq_len=seq_len,
            nuisance_corr=nuisance_corr,
        ),
        device=device,
        generator=generator,
    )
    return collect_representations(model, batch, device)


def readout_accuracy(readout: LinearReadout, x: Tensor, labels: Tensor) -> float:
    return float((readout.predict(x) == labels.to(torch.long)).float().mean().item())


def classification_margin(logits: Tensor, labels: Tensor) -> Tensor:
    labels = labels.to(torch.long)
    correct = logits.gather(1, labels[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels[:, None], float("-inf"))
    runner_up = masked.max(dim=1).values
    return correct - runner_up


def synthesized_decision_scores(
    proposal_logits: Tensor,
    witness_logits: Tensor,
    scope_logits: Tensor,
) -> Tensor:
    scores = torch.full(
        (proposal_logits.shape[0], len(Decision)),
        float("-inf"),
        dtype=proposal_logits.dtype,
        device=proposal_logits.device,
    )
    for proposal in Proposal:
        for witness in Witness:
            for scope in Scope:
                decision = int(decisions_from_tensors(
                    torch.tensor([int(proposal)], device=proposal_logits.device),
                    torch.tensor([int(witness)], device=proposal_logits.device),
                    torch.tensor([int(scope)], device=proposal_logits.device),
                )[0].item())
                triple_score = proposal_logits[:, int(proposal)] + witness_logits[:, int(witness)] + scope_logits[:, int(scope)]
                scores[:, decision] = torch.maximum(scores[:, decision], triple_score)
    return scores


def evaluate_readouts(
    readouts: VariableReadouts,
    x: Tensor,
    labels: RepresentationSet,
) -> dict[str, Any]:
    proposal_logits = readouts.proposal.logits(x)
    witness_logits = readouts.witness.logits(x)
    scope_logits = readouts.scope.logits(x)
    nuisance_logits = readouts.nuisance.logits(x)
    decision_logits = readouts.decision.logits(x)
    synthesized_scores = synthesized_decision_scores(proposal_logits, witness_logits, scope_logits)
    synthesized_pred = synthesized_scores.argmax(dim=-1)
    learned_pred = decision_logits.argmax(dim=-1)
    margins = classification_margin(synthesized_scores, labels.decision)
    variable_margins = {
        "proposal": margin_summary(classification_margin(proposal_logits, labels.proposal)),
        "witness": margin_summary(classification_margin(witness_logits, labels.witness)),
        "scope": margin_summary(classification_margin(scope_logits, labels.scope)),
        "nuisance": margin_summary(classification_margin(nuisance_logits, labels.nuisance)),
        "decision_probe": margin_summary(classification_margin(decision_logits, labels.decision)),
    }
    out = {
        "probe_accuracy": {
            "proposal": readout_accuracy(readouts.proposal, x, labels.proposal),
            "witness": readout_accuracy(readouts.witness, x, labels.witness),
            "scope": readout_accuracy(readouts.scope, x, labels.scope),
            "nuisance": readout_accuracy(readouts.nuisance, x, labels.nuisance),
            "decision": readout_accuracy(readouts.decision, x, labels.decision),
        },
        "latent_learned": decision_metrics(learned_pred, labels.decision),
        "latent_synthesized": decision_metrics(synthesized_pred, labels.decision),
        "synthesized_decision_margin": margin_summary(margins),
        "variable_margins": variable_margins,
        "samples": int(labels.decision.numel()),
    }
    return out


def margin_summary(values: Tensor) -> dict[str, float]:
    values = values.detach().cpu().to(torch.float32)
    if values.numel() == 0:
        return {"mean": float("nan"), "min": float("nan"), "p05": float("nan"), "p50": float("nan")}
    return {
        "mean": float(values.mean().item()),
        "min": float(values.min().item()),
        "p05": float(torch.quantile(values, 0.05).item()),
        "p50": float(torch.quantile(values, 0.50).item()),
    }


def centroid_geometry(x: Tensor, labels: Tensor, classes: int) -> dict[str, Any]:
    x = x.detach().cpu().to(torch.float32)
    labels = labels.detach().cpu().to(torch.long)
    centroids = []
    within = []
    counts = []
    for label in range(classes):
        rows = x[labels == label]
        counts.append(int(rows.shape[0]))
        if rows.numel() == 0:
            centroid = torch.zeros(x.shape[1], dtype=torch.float32)
            dist = torch.tensor(float("nan"))
        else:
            centroid = rows.mean(dim=0)
            dist = torch.linalg.norm(rows - centroid, dim=1).mean()
        centroids.append(centroid)
        within.append(float(dist.item()))
    centroid_tensor = torch.stack(centroids)
    pairwise = torch.cdist(centroid_tensor, centroid_tensor)
    nonzero = pairwise[pairwise > 0]
    between_min = float(nonzero.min().item()) if nonzero.numel() else 0.0
    between_mean = float(nonzero.mean().item()) if nonzero.numel() else 0.0
    normalized = torch.nn.functional.normalize(centroid_tensor, dim=1)
    cosine = normalized @ normalized.T
    off_diag = cosine[~torch.eye(classes, dtype=torch.bool)]
    return {
        "counts": counts,
        "within_class_distance_mean": float(torch.tensor(within).nanmean().item()),
        "within_class_distance_by_class": within,
        "between_centroid_distance_min": between_min,
        "between_centroid_distance_mean": between_mean,
        "centroid_cosine_offdiag_mean": float(off_diag.mean().item()) if off_diag.numel() else 1.0,
        "centroid_cosine_offdiag_min": float(off_diag.min().item()) if off_diag.numel() else 1.0,
        "centroid_norms": [float(torch.linalg.norm(c).item()) for c in centroids],
    }


def fit_affine_alignment(source: Tensor, target: Tensor, ridge: float = 1.0e-3) -> Tensor:
    source = source.detach().to(torch.float64)
    target = target.detach().to(torch.float64)
    xb = _with_bias(source)
    eye = torch.eye(xb.shape[1], dtype=torch.float64, device=xb.device)
    eye[-1, -1] = 0.0
    weights = torch.linalg.solve(xb.T @ xb + ridge * eye, xb.T @ target)
    return weights.to(torch.float32).cpu()


def apply_affine_alignment(source: Tensor, weights: Tensor) -> Tensor:
    return (_with_bias(source.to(torch.float32)) @ weights.to(source.device, source.dtype)).detach().cpu()


@dataclass(frozen=True)
class ProcrustesAlignment:
    rotation: Tensor
    source_mean: Tensor
    target_mean: Tensor


def fit_orthogonal_procrustes(source: Tensor, target: Tensor) -> ProcrustesAlignment:
    source = source.detach().to(torch.float64)
    target = target.detach().to(torch.float64)
    source_mean = source.mean(dim=0, keepdim=True)
    target_mean = target.mean(dim=0, keepdim=True)
    source_centered = source - source_mean
    target_centered = target - target_mean
    u, _, vh = torch.linalg.svd(source_centered.T @ target_centered, full_matrices=False)
    rotation = u @ vh
    return ProcrustesAlignment(
        rotation=rotation.to(torch.float32).cpu(),
        source_mean=source_mean.squeeze(0).to(torch.float32).cpu(),
        target_mean=target_mean.squeeze(0).to(torch.float32).cpu(),
    )


def apply_orthogonal_procrustes(source: Tensor, alignment: ProcrustesAlignment) -> Tensor:
    return (
        (source.to(torch.float32) - alignment.source_mean.to(source.device))
        @ alignment.rotation.to(source.device)
        + alignment.target_mean.to(source.device)
    ).detach().cpu()


def alignment_quality(mapped: Tensor, target: Tensor) -> dict[str, float]:
    mapped = mapped.detach().cpu().to(torch.float32)
    target = target.detach().cpu().to(torch.float32)
    residual = mapped - target
    cosine = torch.nn.functional.cosine_similarity(mapped, target, dim=1)
    return {
        "rmse": float(torch.sqrt((residual * residual).mean()).item()),
        "mean_l2": float(torch.linalg.norm(residual, dim=1).mean().item()),
        "mean_cosine": float(cosine.mean().item()),
    }


def tokens_from_latents_with_positions(
    proposal: Tensor,
    witness: Tensor,
    scope: Tensor,
    nuisance: Tensor,
    seq_len: int,
    witness_pos: int,
    scope_pos: int,
    nuisance_positions: Sequence[int] | None = None,
) -> Tensor:
    if not (0 < witness_pos < seq_len - 1 and 0 < scope_pos < seq_len - 1):
        raise ValueError("witness and scope positions must be inside the sequence before the proposal")
    batch_size = proposal.shape[0]
    device = proposal.device
    tokens = torch.full((batch_size, seq_len), int(Token.FILLER), dtype=torch.long, device=device)
    tokens[:, 0] = int(Token.BOS)
    tokens[:, witness_pos] = torch.where(witness.to(torch.long) == int(Witness.VALID), int(Token.WITNESS_VALID), int(Token.WITNESS_INVALID))
    tokens[:, scope_pos] = torch.where(scope.to(torch.long) == int(Scope.A), int(Token.SCOPE_A), int(Token.SCOPE_B))
    nuisance_token = torch.where(nuisance.to(torch.long) == int(Nuisance.ONE), int(Token.NUISANCE_ONE), int(Token.NUISANCE_ZERO))
    positions = list(nuisance_positions or [])
    if not positions:
        start = min(max(witness_pos, scope_pos) + 1, seq_len - 2)
        positions = torch.linspace(start, seq_len - 2, steps=max(1, min(8, seq_len - start - 1))).round().to(torch.long).tolist()
    for pos in positions:
        if 0 < int(pos) < seq_len - 1 and int(pos) not in (witness_pos, scope_pos):
            tokens[:, int(pos)] = nuisance_token
    tokens[:, -1] = token_for_proposal(proposal)
    return tokens


def controlled_position_batch(
    states: Sequence[PolicyState],
    seq_len: int,
    mode: Literal["scaled", "fixed_absolute", "fixed_distance", "early", "middle", "late"],
    device: torch.device,
) -> ProjectionBatch:
    proposal = torch.tensor([state.proposal for state in states], dtype=torch.long, device=device)
    witness = torch.tensor([state.witness for state in states], dtype=torch.long, device=device)
    scope = torch.tensor([state.scope for state in states], dtype=torch.long, device=device)
    nuisance = torch.tensor([state.nuisance for state in states], dtype=torch.long, device=device)
    if mode == "scaled":
        witness_pos = max(1, seq_len // 8)
        scope_pos = max(witness_pos + 1, seq_len // 3)
    elif mode == "fixed_absolute":
        witness_pos = min(8, seq_len - 3)
        scope_pos = min(21, seq_len - 2)
        if scope_pos <= witness_pos:
            scope_pos = witness_pos + 1
    elif mode == "fixed_distance":
        scope_pos = seq_len - 8
        witness_pos = max(1, scope_pos - 16)
    elif mode == "early":
        witness_pos = 2
        scope_pos = min(4, seq_len - 2)
    elif mode == "middle":
        witness_pos = max(1, seq_len // 2 - 4)
        scope_pos = min(seq_len - 2, witness_pos + 8)
    elif mode == "late":
        scope_pos = seq_len - 3
        witness_pos = max(1, scope_pos - 2)
    else:
        raise ValueError(f"unknown position mode {mode}")
    decision = decisions_from_tensors(proposal, witness, scope)
    tokens = tokens_from_latents_with_positions(proposal, witness, scope, nuisance, seq_len, witness_pos, scope_pos)
    return ProjectionBatch(tokens, proposal, witness, scope, nuisance, decision)


def representative_states(repeats: int = 32) -> list[PolicyState]:
    states = [
        PolicyState(int(proposal), int(witness), int(scope), int(nuisance))
        for _ in range(repeats)
        for proposal in Proposal
        for witness in Witness
        for scope in Scope
        for nuisance in Nuisance
    ]
    return states


def minimize_padding_failure(
    predicate,
    low: int,
    high: int,
) -> int | None:
    if not predicate(high):
        return None
    best = high
    lo = low
    hi = high
    while lo <= hi:
        mid = (lo + hi) // 2
        if predicate(mid):
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return best
