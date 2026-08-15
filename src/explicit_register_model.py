from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .projection_model import accuracy, make_generator, sinusoidal_positions
from .projection_task import (
    VOCAB_SIZE,
    Decision,
    ProjectionBatch,
    ProjectionTaskConfig,
    decisions_from_tensors,
    sample_policy_batch,
)
from .semantic_register import (
    RegisterEncoding,
    decode_register,
    register_accuracy,
    register_code,
    register_dim,
    register_policy_decision,
    register_state_logits,
    register_variable_logits,
)
from .synthesized_latent_gate import decision_metrics


@dataclass(frozen=True)
class ExplicitRegisterModelConfig:
    seed: int = 101
    seq_len: int = 64
    max_len: int = 4096
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 192
    steps: int = 1200
    batch_size: int = 256
    learning_rate: float = 3.0e-4
    train_nuisance_corr: float = 0.95
    encoding: str = RegisterEncoding.BINARY_PAIR.value
    register_mse_weight: float = 2.0
    register_ce_weight: float = 1.0
    register_gate_weight: float = 0.75
    e2e_decision_weight: float = 0.5
    log_every: int = 200


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def labels_from_projection_batch(batch: ProjectionBatch) -> dict[str, Tensor]:
    return {
        "proposal": batch.proposal.to(torch.long),
        "witness": batch.witness.to(torch.long),
        "scope": batch.scope.to(torch.long),
        "nuisance": batch.nuisance.to(torch.long),
        "decision": batch.decision.to(torch.long),
    }


class ExplicitRegisterTransformer(nn.Module):
    def __init__(self, config: ExplicitRegisterModelConfig):
        super().__init__()
        self.config = config
        self.encoding = RegisterEncoding(config.encoding)
        self.token_embedding = nn.Embedding(VOCAB_SIZE, config.d_model)
        self.register_buffer("position_encoding", sinusoidal_positions(config.max_len, config.d_model), persistent=False)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.d_model,
                    nhead=config.n_heads,
                    dim_feedforward=config.d_ff,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config.n_layers)
            ]
        )
        dim = register_dim(self.encoding)
        self.proposal_head = nn.Linear(config.d_model, 3)
        self.witness_head = nn.Linear(config.d_model, 2)
        self.scope_head = nn.Linear(config.d_model, 2)
        self.nuisance_head = nn.Linear(config.d_model, 2)
        self.e2e_decision_head = nn.Linear(config.d_model, 6)
        self.register_writer = nn.Linear(config.d_model, dim)
        self.register_decision_head = nn.Sequential(nn.Linear(dim, 32), nn.GELU(), nn.Linear(32, 6))

    def _causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def hidden_states(self, tokens: Tensor) -> list[Tensor]:
        batch_size, seq_len = tokens.shape
        if seq_len > self.config.max_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_len {self.config.max_len}")
        del batch_size
        hidden = self.token_embedding(tokens) + self.position_encoding[:seq_len].to(tokens.device)[None, :, :]
        states = [hidden]
        mask = self._causal_mask(seq_len, tokens.device)
        for layer in self.layers:
            hidden = layer(hidden, src_mask=mask)
            states.append(hidden)
        return states

    def forward(self, tokens: Tensor, return_hidden: bool = False) -> dict[str, Tensor | list[Tensor]]:
        states = self.hidden_states(tokens)
        final_hidden = states[-1][:, -1, :]
        register = self.register_writer(final_hidden)
        out: dict[str, Tensor | list[Tensor]] = {
            "proposal_logits": self.proposal_head(final_hidden),
            "witness_logits": self.witness_head(final_hidden),
            "scope_logits": self.scope_head(final_hidden),
            "nuisance_logits": self.nuisance_head(final_hidden),
            "e2e_decision_logits": self.e2e_decision_head(final_hidden),
            "register": register,
            "register_decision_logits": self.register_decision_head(register),
        }
        if return_hidden:
            out["hidden_states"] = states
        return out


def register_loss(outputs: dict[str, Tensor | list[Tensor]], batch: ProjectionBatch, encoding: RegisterEncoding | str) -> Tensor:
    register = outputs["register"]
    assert isinstance(register, Tensor)
    target = register_code(encoding, batch.witness, batch.scope).to(register.device)
    mse = torch.nn.functional.mse_loss(register, target)
    witness_logits, scope_logits = register_variable_logits(register, encoding)
    ce = torch.nn.functional.cross_entropy
    loss = mse + ce(witness_logits, batch.witness.to(torch.long)) + ce(scope_logits, batch.scope.to(torch.long))
    if RegisterEncoding(encoding) == RegisterEncoding.JOINT_ONE_HOT:
        joint = batch.witness.to(torch.long) * 2 + batch.scope.to(torch.long)
        loss = loss + ce(register_state_logits(register, encoding), joint)
    return loss


def multitask_register_loss(outputs: dict[str, Tensor | list[Tensor]], batch: ProjectionBatch, config: ExplicitRegisterModelConfig) -> Tensor:
    labels = labels_from_projection_batch(batch)
    ce = torch.nn.functional.cross_entropy
    return (
        ce(outputs["proposal_logits"], labels["proposal"])
        + ce(outputs["witness_logits"], labels["witness"])
        + ce(outputs["scope_logits"], labels["scope"])
        + ce(outputs["nuisance_logits"], labels["nuisance"])
        + config.e2e_decision_weight * ce(outputs["e2e_decision_logits"], labels["decision"])
        + config.register_gate_weight * ce(outputs["register_decision_logits"], labels["decision"])
        + config.register_mse_weight * torch.nn.functional.mse_loss(
            outputs["register"], register_code(config.encoding, batch.witness, batch.scope).to(outputs["register"].device)
        )
        + config.register_ce_weight * register_loss(outputs, batch, config.encoding)
    )


def _batch_metrics(model: ExplicitRegisterTransformer, outputs: dict[str, Tensor | list[Tensor]], batch: ProjectionBatch) -> dict[str, Any]:
    labels = labels_from_projection_batch(batch)
    register = outputs["register"]
    assert isinstance(register, Tensor)
    synthesized, decoded = register_policy_decision(batch.proposal, register, model.encoding)
    reg_metrics = register_accuracy(decoded, batch.witness, batch.scope)
    return {
        "proposal_accuracy": accuracy(outputs["proposal_logits"], labels["proposal"]),
        "witness_head_accuracy": accuracy(outputs["witness_logits"], labels["witness"]),
        "scope_head_accuracy": accuracy(outputs["scope_logits"], labels["scope"]),
        "nuisance_head_accuracy": accuracy(outputs["nuisance_logits"], labels["nuisance"]),
        "e2e_decision_accuracy": accuracy(outputs["e2e_decision_logits"], labels["decision"]),
        "register_learned_gate_accuracy": accuracy(outputs["register_decision_logits"], labels["decision"]),
        "register_synthesized_accuracy": float((synthesized == labels["decision"]).to(torch.float32).mean().item()),
        **reg_metrics,
    }


def train_explicit_register_model(
    config: ExplicitRegisterModelConfig,
    device: torch.device | None = None,
) -> tuple[ExplicitRegisterTransformer, dict[str, Any]]:
    set_seed(config.seed)
    device = device or choose_device()
    task_config = ProjectionTaskConfig(seq_len=config.seq_len, nuisance_corr=config.train_nuisance_corr)
    generator = make_generator(config.seed + 31, device)
    model = ExplicitRegisterTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, Any]] = []
    start = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, config.steps + 1):
        model.train()
        batch = sample_policy_batch(config.batch_size, task_config, device, generator)
        outputs = model(batch.tokens)
        loss = multitask_register_loss(outputs, batch, config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step == config.steps or step % config.log_every == 0:
            with torch.no_grad():
                metrics = _batch_metrics(model, outputs, batch)
            history.append({"step": step, "loss": float(loss.detach().item()), **metrics})
    metrics: dict[str, Any] = {
        "config": asdict(config),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "training_time_seconds": time.time() - start,
        "history": history,
    }
    if device.type == "cuda":
        metrics["peak_gpu_memory_mib"] = float(torch.cuda.max_memory_allocated(device) / 2**20)
    return model, metrics


@torch.no_grad()
def evaluate_explicit_register_model(
    model: ExplicitRegisterTransformer,
    task_config: ProjectionTaskConfig,
    device: torch.device,
    batch_size: int = 512,
    batches: int = 4,
    seed: int = 9001,
) -> dict[str, Any]:
    generator = make_generator(seed, device)
    predicted_synth: list[Tensor] = []
    predicted_learned: list[Tensor] = []
    predicted_e2e: list[Tensor] = []
    expected: list[Tensor] = []
    decoded_witness: list[Tensor] = []
    decoded_scope: list[Tensor] = []
    valid: list[Tensor] = []
    true_witness: list[Tensor] = []
    true_scope: list[Tensor] = []
    margins: list[Tensor] = []
    distances: list[Tensor] = []
    proposal_head: list[Tensor] = []
    true_proposal: list[Tensor] = []
    model.eval()
    for _ in range(batches):
        batch = sample_policy_batch(batch_size, task_config, device, generator)
        outputs = model(batch.tokens)
        register = outputs["register"]
        assert isinstance(register, Tensor)
        synth, decoded = register_policy_decision(batch.proposal, register, model.encoding)
        predicted_synth.append(synth.detach().cpu())
        predicted_learned.append(outputs["register_decision_logits"].argmax(dim=-1).detach().cpu())
        predicted_e2e.append(outputs["e2e_decision_logits"].argmax(dim=-1).detach().cpu())
        expected.append(batch.decision.detach().cpu())
        decoded_witness.append(decoded.witness.detach().cpu())
        decoded_scope.append(decoded.scope.detach().cpu())
        valid.append(decoded.valid.detach().cpu())
        true_witness.append(batch.witness.detach().cpu())
        true_scope.append(batch.scope.detach().cpu())
        margins.append(decoded.margin.detach().cpu())
        distances.append(decoded.distance.detach().cpu())
        proposal_head.append(outputs["proposal_logits"].argmax(dim=-1).detach().cpu())
        true_proposal.append(batch.proposal.detach().cpu())
    expected_cat = torch.cat(expected)
    synth_cat = torch.cat(predicted_synth)
    learned_cat = torch.cat(predicted_learned)
    e2e_cat = torch.cat(predicted_e2e)
    witness_cat = torch.cat(true_witness)
    scope_cat = torch.cat(true_scope)
    decoded_witness_cat = torch.cat(decoded_witness)
    decoded_scope_cat = torch.cat(decoded_scope)
    valid_cat = torch.cat(valid)
    margins_cat = torch.cat(margins)
    distances_cat = torch.cat(distances)
    proposal_head_cat = torch.cat(proposal_head)
    true_proposal_cat = torch.cat(true_proposal)
    register_relative_expected = decisions_from_tensors(true_proposal_cat, decoded_witness_cat, decoded_scope_cat)
    register_relative_expected = torch.where(
        valid_cat,
        register_relative_expected,
        torch.full_like(register_relative_expected, int(Decision.REFUSE_INSUFFICIENT_INFORMATION)),
    )
    register_relative_expected = torch.where(
        true_proposal_cat == 0,
        torch.full_like(register_relative_expected, int(Decision.REFUSE_NO_PROPOSAL)),
        register_relative_expected,
    )
    return {
        "samples": int(expected_cat.numel()),
        "register": {
            "register_valid_rate": float(valid_cat.to(torch.float32).mean().item()),
            "witness_accuracy": float(((decoded_witness_cat == witness_cat) & valid_cat).to(torch.float32).mean().item()),
            "scope_accuracy": float(((decoded_scope_cat == scope_cat) & valid_cat).to(torch.float32).mean().item()),
            "joint_accuracy": float(((decoded_witness_cat == witness_cat) & (decoded_scope_cat == scope_cat) & valid_cat).to(torch.float32).mean().item()),
            "decode_margin_mean": float(margins_cat.to(torch.float32).mean().item()),
            "decode_margin_min": float(margins_cat.to(torch.float32).min().item()),
            "decode_distance_mean": float(distances_cat.to(torch.float32).mean().item()),
            "decode_distance_max": float(distances_cat.to(torch.float32).max().item()),
        },
        "explicit_register_synthesized_gate": decision_metrics(synth_cat, expected_cat),
        "explicit_register_learned_gate": decision_metrics(learned_cat, expected_cat),
        "end_to_end_learned": decision_metrics(e2e_cat, expected_cat),
        "register_relative_synthesized": decision_metrics(synth_cat, register_relative_expected),
        "proposal_head_accuracy": float((proposal_head_cat == true_proposal_cat).to(torch.float32).mean().item()),
    }


def save_explicit_register_model(model: ExplicitRegisterTransformer, path: str) -> None:
    torch.save({"config": asdict(model.config), "state_dict": model.state_dict()}, path)


def load_explicit_register_model(path: str, device: torch.device | None = None) -> ExplicitRegisterTransformer:
    device = device or choose_device()
    payload = torch.load(path, map_location=device)
    model = ExplicitRegisterTransformer(ExplicitRegisterModelConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
