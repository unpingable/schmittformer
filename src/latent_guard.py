from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .projection_model import ProjectionCausalTransformer, final_representations, layer_names, make_generator
from .projection_task import ProjectionBatch, ProjectionTaskConfig, sample_policy_batch
from .synthesized_latent_gate import decision_metrics, synthesize_policy_from_variables


@dataclass(frozen=True)
class GuardTrainingConfig:
    seed: int = 701
    steps: int = 350
    batch_size: int = 256
    learning_rate: float = 7.0e-4
    train_nuisance_corr: float = 0.95
    log_every: int = 100


class VariableProbe(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proposal = nn.Linear(d_model, 3)
        self.witness = nn.Linear(d_model, 2)
        self.scope = nn.Linear(d_model, 2)
        self.nuisance = nn.Linear(d_model, 2)

    def forward(self, representation: Tensor) -> dict[str, Tensor]:
        return {
            "proposal": self.proposal(representation),
            "witness": self.witness(representation),
            "scope": self.scope(representation),
            "nuisance": self.nuisance(representation),
        }


class DecisionProbe(nn.Module):
    def __init__(self, d_model: int, num_decisions: int):
        super().__init__()
        self.linear = nn.Linear(d_model, num_decisions)

    def forward(self, representation: Tensor) -> Tensor:
        return self.linear(representation)


@dataclass
class LayerGuardSet:
    layer_labels: list[str]
    variable_probes: nn.ModuleList
    decision_probes: nn.ModuleList

    def modules_for_optim(self) -> list[nn.Module]:
        return [*self.variable_probes, *self.decision_probes]

    def to(self, device: torch.device) -> "LayerGuardSet":
        self.variable_probes.to(device)
        self.decision_probes.to(device)
        return self


def _variable_loss(outputs: dict[str, Tensor], batch: ProjectionBatch) -> Tensor:
    ce = torch.nn.functional.cross_entropy
    return (
        ce(outputs["proposal"], batch.proposal)
        + ce(outputs["witness"], batch.witness)
        + ce(outputs["scope"], batch.scope)
        + ce(outputs["nuisance"], batch.nuisance)
    )


def make_layer_guard_set(model: ProjectionCausalTransformer) -> LayerGuardSet:
    labels = layer_names(model.config)
    variable_probes = nn.ModuleList([VariableProbe(model.config.d_model) for _ in labels])
    decision_probes = nn.ModuleList([DecisionProbe(model.config.d_model, 6) for _ in labels])
    return LayerGuardSet(labels, variable_probes, decision_probes)


def train_layer_guards(
    model: ProjectionCausalTransformer,
    config: GuardTrainingConfig,
    device: torch.device,
) -> tuple[LayerGuardSet, dict[str, Any]]:
    torch.manual_seed(config.seed)
    guards = make_layer_guard_set(model).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for module in guards.modules_for_optim() for parameter in module.parameters()],
        lr=config.learning_rate,
    )
    model.eval()
    task_config = ProjectionTaskConfig(seq_len=model.config.seq_len, nuisance_corr=config.train_nuisance_corr)
    generator = make_generator(config.seed + 31, device)
    history: list[dict[str, Any]] = []
    start = time.time()

    for step in range(1, config.steps + 1):
        batch = sample_policy_batch(config.batch_size, task_config, device, generator)
        with torch.no_grad():
            reps = final_representations(model, batch)
        loss = torch.tensor(0.0, device=device)
        for index, rep in enumerate(reps):
            variable_outputs = guards.variable_probes[index](rep)
            decision_logits = guards.decision_probes[index](rep)
            loss = loss + _variable_loss(variable_outputs, batch)
            loss = loss + torch.nn.functional.cross_entropy(decision_logits, batch.decision)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step == config.steps or step % config.log_every == 0:
            history.append({"step": step, "loss": float(loss.item())})

    return guards, {
        "config": asdict(config),
        "training_time_seconds": time.time() - start,
        "history": history,
    }


def _concat_batches(values: list[Tensor]) -> Tensor:
    return torch.cat(values, dim=0) if values else torch.empty(0, dtype=torch.long)


@torch.no_grad()
def evaluate_layer_guards(
    model: ProjectionCausalTransformer,
    guards: LayerGuardSet,
    task_config: ProjectionTaskConfig,
    device: torch.device,
    batch_size: int = 512,
    batches: int = 8,
    seed: int = 9001,
) -> list[dict[str, Any]]:
    model.eval()
    for module in guards.modules_for_optim():
        module.eval()
    generator = make_generator(seed, device)
    by_layer: list[dict[str, list[Tensor]]] = [
        {
            "expected_decision": [],
            "learned_decision": [],
            "synth_decision": [],
            "proposal_pred": [],
            "witness_pred": [],
            "scope_pred": [],
            "nuisance_pred": [],
            "proposal_true": [],
            "witness_true": [],
            "scope_true": [],
            "nuisance_true": [],
        }
        for _ in guards.layer_labels
    ]

    for _ in range(batches):
        batch = sample_policy_batch(batch_size, task_config, device, generator)
        reps = final_representations(model, batch)
        for index, rep in enumerate(reps):
            variable_outputs = guards.variable_probes[index](rep)
            learned_logits = guards.decision_probes[index](rep)
            synth_decision = synthesize_policy_from_variables(
                variable_outputs["proposal"],
                variable_outputs["witness"],
                variable_outputs["scope"],
            )
            bucket = by_layer[index]
            bucket["expected_decision"].append(batch.decision.detach().cpu())
            bucket["learned_decision"].append(learned_logits.argmax(dim=-1).detach().cpu())
            bucket["synth_decision"].append(synth_decision.detach().cpu())
            bucket["proposal_pred"].append(variable_outputs["proposal"].argmax(dim=-1).detach().cpu())
            bucket["witness_pred"].append(variable_outputs["witness"].argmax(dim=-1).detach().cpu())
            bucket["scope_pred"].append(variable_outputs["scope"].argmax(dim=-1).detach().cpu())
            bucket["nuisance_pred"].append(variable_outputs["nuisance"].argmax(dim=-1).detach().cpu())
            bucket["proposal_true"].append(batch.proposal.detach().cpu())
            bucket["witness_true"].append(batch.witness.detach().cpu())
            bucket["scope_true"].append(batch.scope.detach().cpu())
            bucket["nuisance_true"].append(batch.nuisance.detach().cpu())

    rows = []
    for index, label in enumerate(guards.layer_labels):
        bucket = by_layer[index]
        expected = _concat_batches(bucket["expected_decision"])
        learned = _concat_batches(bucket["learned_decision"])
        synth = _concat_batches(bucket["synth_decision"])
        row = {
            "layer": label,
            "layer_index": index,
            "latent_learned": decision_metrics(learned, expected),
            "latent_synthesized": decision_metrics(synth, expected),
            "probe_accuracy": {
                "proposal": float((_concat_batches(bucket["proposal_pred"]) == _concat_batches(bucket["proposal_true"])).float().mean().item()),
                "witness": float((_concat_batches(bucket["witness_pred"]) == _concat_batches(bucket["witness_true"])).float().mean().item()),
                "scope": float((_concat_batches(bucket["scope_pred"]) == _concat_batches(bucket["scope_true"])).float().mean().item()),
                "nuisance": float((_concat_batches(bucket["nuisance_pred"]) == _concat_batches(bucket["nuisance_true"])).float().mean().item()),
            },
            "samples": int(expected.numel()),
        }
        rows.append(row)
    return rows


@torch.no_grad()
def evaluate_upstream_policy_head(
    model: ProjectionCausalTransformer,
    task_config: ProjectionTaskConfig,
    device: torch.device,
    batch_size: int = 512,
    batches: int = 8,
    seed: int = 9101,
) -> dict[str, Any]:
    model.eval()
    generator = make_generator(seed, device)
    predictions: list[Tensor] = []
    expected: list[Tensor] = []
    for _ in range(batches):
        batch = sample_policy_batch(batch_size, task_config, device, generator)
        outputs = model(batch.tokens)
        predictions.append(outputs["decision_logits"].argmax(dim=-1).detach().cpu())
        expected.append(batch.decision.detach().cpu())
    return decision_metrics(_concat_batches(predictions), _concat_batches(expected))
