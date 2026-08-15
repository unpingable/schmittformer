from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .projection_task import (
    VOCAB_SIZE,
    Decision,
    ProjectionBatch,
    ProjectionTaskConfig,
    sample_policy_batch,
)


@dataclass(frozen=True)
class ProjectionModelConfig:
    seed: int = 101
    seq_len: int = 64
    max_len: int = 1024
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 192
    steps: int = 900
    batch_size: int = 192
    learning_rate: float = 3.0e-4
    train_nuisance_corr: float = 0.95
    log_every: int = 100


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_generator(seed: int, device: torch.device) -> torch.Generator:
    # CUDA generators are useful when CUDA is visible; the current checked-in
    # environment uses a CPU PyTorch wheel.
    generator = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
    generator.manual_seed(seed)
    return generator


def sinusoidal_positions(max_len: int, d_model: int) -> Tensor:
    positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
    encoding = torch.zeros(max_len, d_model, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * div)
    encoding[:, 1::2] = torch.cos(positions * div[: encoding[:, 1::2].shape[1]])
    return encoding


class ProjectionCausalTransformer(nn.Module):
    def __init__(self, config: ProjectionModelConfig):
        super().__init__()
        self.config = config
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
        self.proposal_head = nn.Linear(config.d_model, 3)
        self.witness_head = nn.Linear(config.d_model, 2)
        self.scope_head = nn.Linear(config.d_model, 2)
        self.nuisance_head = nn.Linear(config.d_model, 2)
        self.decision_head = nn.Linear(config.d_model, len(Decision))

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
        out: dict[str, Tensor | list[Tensor]] = {
            "proposal_logits": self.proposal_head(final_hidden),
            "witness_logits": self.witness_head(final_hidden),
            "scope_logits": self.scope_head(final_hidden),
            "nuisance_logits": self.nuisance_head(final_hidden),
            "decision_logits": self.decision_head(final_hidden),
        }
        if return_hidden:
            out["hidden_states"] = states
        return out


def labels_from_batch(batch: ProjectionBatch) -> dict[str, Tensor]:
    return {
        "proposal": batch.proposal.to(torch.long),
        "witness": batch.witness.to(torch.long),
        "scope": batch.scope.to(torch.long),
        "nuisance": batch.nuisance.to(torch.long),
        "decision": batch.decision.to(torch.long),
    }


def multitask_loss(outputs: dict[str, Tensor | list[Tensor]], batch: ProjectionBatch) -> Tensor:
    labels = labels_from_batch(batch)
    ce = torch.nn.functional.cross_entropy
    return (
        ce(outputs["proposal_logits"], labels["proposal"])
        + ce(outputs["witness_logits"], labels["witness"])
        + ce(outputs["scope_logits"], labels["scope"])
        + ce(outputs["nuisance_logits"], labels["nuisance"])
        + 0.75 * ce(outputs["decision_logits"], labels["decision"])
    )


def accuracy(logits: Tensor, labels: Tensor) -> float:
    return float((logits.argmax(dim=-1) == labels.to(torch.long)).float().mean().item())


def train_projection_model(
    config: ProjectionModelConfig,
    device: torch.device | None = None,
) -> tuple[ProjectionCausalTransformer, dict[str, Any]]:
    set_seed(config.seed)
    device = device or choose_device()
    task_config = ProjectionTaskConfig(seq_len=config.seq_len, nuisance_corr=config.train_nuisance_corr)
    generator = make_generator(config.seed + 17, device)
    model = ProjectionCausalTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, Any]] = []
    start = time.time()

    for step in range(1, config.steps + 1):
        model.train()
        batch = sample_policy_batch(config.batch_size, task_config, device, generator)
        outputs = model(batch.tokens)
        loss = multitask_loss(outputs, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step == config.steps or step % config.log_every == 0:
            with torch.no_grad():
                labels = labels_from_batch(batch)
                history.append(
                    {
                        "step": step,
                        "loss": float(loss.item()),
                        "proposal_accuracy": accuracy(outputs["proposal_logits"], labels["proposal"]),
                        "witness_accuracy": accuracy(outputs["witness_logits"], labels["witness"]),
                        "scope_accuracy": accuracy(outputs["scope_logits"], labels["scope"]),
                        "nuisance_accuracy": accuracy(outputs["nuisance_logits"], labels["nuisance"]),
                        "decision_accuracy": accuracy(outputs["decision_logits"], labels["decision"]),
                    }
                )

    metrics = {
        "config": asdict(config),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "training_time_seconds": time.time() - start,
        "history": history,
    }
    return model, metrics


def layer_names(config: ProjectionModelConfig) -> list[str]:
    return ["embedding"] + [f"layer_{index}" for index in range(1, config.n_layers + 1)]


@torch.no_grad()
def final_representations(model: ProjectionCausalTransformer, batch: ProjectionBatch) -> list[Tensor]:
    model.eval()
    states = model.hidden_states(batch.tokens.to(next(model.parameters()).device))
    return [state[:, -1, :].detach() for state in states]


def save_projection_model(model: ProjectionCausalTransformer, path: str) -> None:
    torch.save({"config": asdict(model.config), "state_dict": model.state_dict()}, path)


def load_projection_model(path: str, device: torch.device | None = None) -> ProjectionCausalTransformer:
    payload = torch.load(path, map_location=device or choose_device())
    model = ProjectionCausalTransformer(ProjectionModelConfig(**payload["config"])).to(device or choose_device())
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
