from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .projection_channels import (
    ProjectionRegime,
    deterministic_token_monitor,
    project_batch,
    projection_feature_dim,
    projection_features,
    trusted_metadata_monitor,
)
from .projection_model import make_generator
from .projection_task import ProjectionTaskConfig, sample_policy_batch
from .synthesized_latent_gate import decision_metrics


@dataclass(frozen=True)
class TokenGuardConfig:
    seed: int = 1301
    steps: int = 300
    batch_size: int = 512
    learning_rate: float = 1.0e-3
    train_nuisance_corr: float = 0.95
    hidden_dim: int = 48
    log_every: int = 100


class TokenDecisionGuard(nn.Module):
    def __init__(self, hidden_dim: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(projection_feature_dim(), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 6),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)


def evaluate_deterministic_boundaries(
    regime: ProjectionRegime | str,
    noise: float,
    task_config: ProjectionTaskConfig,
    device: torch.device,
    batch_size: int = 1024,
    batches: int = 8,
    seed: int = 4401,
) -> dict[str, Any]:
    regime = ProjectionRegime(regime)
    generator = make_generator(seed, device)
    deterministic_predictions: list[Tensor] = []
    trusted_predictions: list[Tensor] = []
    expected: list[Tensor] = []
    for _ in range(batches):
        batch = sample_policy_batch(batch_size, task_config, device, generator)
        projected = project_batch(batch, regime, noise, generator)
        deterministic_predictions.append(deterministic_token_monitor(projected).detach().cpu())
        trusted_predictions.append(trusted_metadata_monitor(batch).detach().cpu())
        expected.append(batch.decision.detach().cpu())
    expected_cat = torch.cat(expected, dim=0)
    return {
        "token_only_reference": decision_metrics(torch.cat(deterministic_predictions, dim=0), expected_cat),
        "token_plus_trusted_metadata": decision_metrics(torch.cat(trusted_predictions, dim=0), expected_cat),
    }


def train_token_guard(
    regime: ProjectionRegime | str,
    noise: float,
    task_config: ProjectionTaskConfig,
    config: TokenGuardConfig,
    device: torch.device,
) -> tuple[TokenDecisionGuard, dict[str, Any]]:
    torch.manual_seed(config.seed)
    regime = ProjectionRegime(regime)
    train_task = ProjectionTaskConfig(
        seq_len=task_config.seq_len,
        p_valid=task_config.p_valid,
        p_scope_a=task_config.p_scope_a,
        p_noop=task_config.p_noop,
        p_remediate_a=task_config.p_remediate_a,
        nuisance_corr=config.train_nuisance_corr,
        nuisance_events=task_config.nuisance_events,
    )
    guard = TokenDecisionGuard(config.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(guard.parameters(), lr=config.learning_rate)
    generator = make_generator(config.seed + 53, device)
    history: list[dict[str, Any]] = []
    start = time.time()
    for step in range(1, config.steps + 1):
        batch = sample_policy_batch(config.batch_size, train_task, device, generator)
        projected = project_batch(batch, regime, noise, generator)
        logits = guard(projection_features(projected))
        loss = torch.nn.functional.cross_entropy(logits, batch.decision)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step == config.steps or step % config.log_every == 0:
            with torch.no_grad():
                metrics = decision_metrics(logits.argmax(dim=-1), batch.decision)
            history.append({"step": step, "loss": float(loss.item()), **metrics})
    return guard, {"config": asdict(config), "training_time_seconds": time.time() - start, "history": history}


@torch.no_grad()
def evaluate_token_guard(
    guard: TokenDecisionGuard,
    regime: ProjectionRegime | str,
    noise: float,
    task_config: ProjectionTaskConfig,
    device: torch.device,
    batch_size: int = 1024,
    batches: int = 8,
    seed: int = 5501,
) -> dict[str, Any]:
    regime = ProjectionRegime(regime)
    guard.eval()
    generator = make_generator(seed, device)
    predictions: list[Tensor] = []
    expected: list[Tensor] = []
    for _ in range(batches):
        batch = sample_policy_batch(batch_size, task_config, device, generator)
        projected = project_batch(batch, regime, noise, generator)
        predictions.append(guard(projection_features(projected)).argmax(dim=-1).detach().cpu())
        expected.append(batch.decision.detach().cpu())
    return decision_metrics(torch.cat(predictions, dim=0), torch.cat(expected, dim=0))
