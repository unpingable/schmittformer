from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from .compiled import CompiledHysteresisTransformer
from .evaluate import count_illegal_transitions
from .learned import LearnedConfig, choose_device, set_seed, states_from_inputs_tensor
from .reference import State


@dataclass
class HybridConfig:
    seed: int = 11
    classifier_steps: int = 600
    e2e_steps: int = 800
    batch_size: int = 128
    train_len: int = 16
    sigma: float = 0.35
    learning_rate: float = 5.0e-4
    d_model: int = 32
    n_heads: int = 2
    n_layers: int = 2
    d_ff: int = 64
    max_len: int = 1024


class LevelClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, 10),
        )

    def forward(self, observations: Tensor) -> Tensor:
        return self.net(observations.unsqueeze(-1))


class ContinuousCausalTransformer(nn.Module):
    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        self.initial_embedding = nn.Embedding(2, config.d_model)
        self.observation_projection = nn.Linear(1, config.d_model)
        self.position_embedding = nn.Embedding(config.max_len + 1, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.output = nn.Linear(config.d_model, 2)

    def forward(self, observations: Tensor, initial_states: Tensor) -> Tensor:
        batch_size, seq_len = observations.shape
        total_len = seq_len + 1
        if total_len > self.config.max_len + 1:
            raise ValueError("sequence too long")
        initial_hidden = self.initial_embedding(initial_states.to(torch.long))[:, None, :]
        obs_hidden = self.observation_projection(observations.unsqueeze(-1))
        hidden = torch.cat([initial_hidden, obs_hidden], dim=1)
        positions = torch.arange(total_len, device=observations.device)
        hidden = hidden + self.position_embedding(positions)[None, :, :]
        mask = torch.full((total_len, total_len), float("-inf"), device=observations.device)
        mask = torch.triu(mask, diagonal=1)
        hidden = self.blocks(hidden, mask=mask)
        return self.output(hidden)


def generate_noisy_batch(
    batch_size: int,
    seq_len: int,
    sigma: float,
    device: torch.device,
    bias: float = 0.0,
    near_threshold_prob: float = 0.0,
    initial: str = "random",
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if near_threshold_prob > 0:
        uniform = torch.randint(0, 10, (batch_size, seq_len), device=device)
        near = torch.tensor([2, 3, 4, 6, 7, 8], device=device)
        near_samples = near[torch.randint(0, len(near), (batch_size, seq_len), device=device)]
        choose_near = torch.rand((batch_size, seq_len), device=device) < near_threshold_prob
        latent = torch.where(choose_near, near_samples, uniform)
    else:
        latent = torch.randint(0, 10, (batch_size, seq_len), device=device)

    if initial == "off":
        initial_states = torch.zeros(batch_size, dtype=torch.long, device=device)
    elif initial == "on":
        initial_states = torch.ones(batch_size, dtype=torch.long, device=device)
    else:
        initial_states = torch.randint(0, 2, (batch_size,), dtype=torch.long, device=device)

    observations = latent.to(torch.float32) + bias + sigma * torch.randn(
        batch_size,
        seq_len,
        device=device,
    )
    labels = states_from_inputs_tensor(latent, initial_states)
    return observations, latent, labels, initial_states


def train_classifier(
    config: HybridConfig,
    device: torch.device,
) -> tuple[LevelClassifier, dict]:
    classifier = LevelClassifier().to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=config.learning_rate)
    history = []
    start = time.time()
    for step in range(1, config.classifier_steps + 1):
        latent = torch.randint(0, 10, (config.batch_size,), device=device)
        observations = latent.to(torch.float32) + config.sigma * torch.randn(
            config.batch_size,
            device=device,
        )
        logits = classifier(observations)
        loss = torch.nn.functional.cross_entropy(logits, latent)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step == config.classifier_steps or step % 100 == 0:
            acc = (logits.argmax(dim=-1) == latent).float().mean().item()
            history.append({"step": step, "loss": float(loss.item()), "accuracy": acc})
    return classifier, {"history": history, "training_time_seconds": time.time() - start}


def train_e2e(
    config: HybridConfig,
    device: torch.device,
) -> tuple[ContinuousCausalTransformer, dict]:
    model = ContinuousCausalTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    history = []
    start = time.time()
    for step in range(1, config.e2e_steps + 1):
        observations, _, labels, initial_states = generate_noisy_batch(
            config.batch_size,
            config.train_len,
            config.sigma,
            device,
        )
        logits = model(observations, initial_states)[:, 1:, :]
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step == config.e2e_steps or step % 100 == 0:
            acc = (logits.argmax(dim=-1) == labels).float().mean().item()
            history.append({"step": step, "loss": float(loss.item()), "accuracy": acc})
    return model, {"history": history, "training_time_seconds": time.time() - start}


def train_hybrid_system(
    config: HybridConfig | None = None,
    device: torch.device | None = None,
) -> tuple[LevelClassifier, ContinuousCausalTransformer, dict]:
    config = config or HybridConfig()
    set_seed(config.seed)
    device = device or choose_device()
    classifier, classifier_metrics = train_classifier(config, device)
    e2e, e2e_metrics = train_e2e(config, device)
    return classifier, e2e, {
        "config": asdict(config),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "classifier": classifier_metrics,
        "end_to_end": e2e_metrics,
    }


def hybrid_predict_from_levels(
    predicted_levels: Tensor,
    initial_states: Tensor,
    controller: CompiledHysteresisTransformer | None = None,
) -> Tensor:
    controller = controller or CompiledHysteresisTransformer()
    controller = controller.to(predicted_levels.device)
    with torch.no_grad():
        logits = controller(predicted_levels.to(torch.long), initial_state=initial_states)
        return logits.argmax(dim=-1)


def evaluate_hybrid_models(
    classifier: LevelClassifier,
    e2e: ContinuousCausalTransformer,
    config: HybridConfig,
    device: torch.device,
    sigma: float,
    bias: float = 0.0,
    seq_len: int = 128,
    batches: int = 8,
    near_threshold_prob: float = 0.0,
) -> dict:
    classifier.eval()
    e2e.eval()
    controller = CompiledHysteresisTransformer().to(device)

    total = 0
    classifier_correct = 0
    hybrid_correct = 0
    e2e_correct = 0
    hybrid_illegal_true = 0
    hybrid_illegal_belief = 0
    e2e_illegal_true = 0

    for _ in range(batches):
        observations, latent, labels, initial_states = generate_noisy_batch(
            config.batch_size,
            seq_len,
            sigma,
            device,
            bias=bias,
            near_threshold_prob=near_threshold_prob,
        )
        with torch.no_grad():
            predicted_levels = classifier(observations).argmax(dim=-1)
            hybrid_states = hybrid_predict_from_levels(
                predicted_levels,
                initial_states,
                controller,
            )
            e2e_states = e2e(observations, initial_states)[:, 1:, :].argmax(dim=-1)

        total += labels.numel()
        classifier_correct += (predicted_levels == latent).sum().item()
        hybrid_correct += (hybrid_states == labels).sum().item()
        e2e_correct += (e2e_states == labels).sum().item()

        for row in range(latent.shape[0]):
            init = int(initial_states[row].item())
            hybrid_illegal_true += count_illegal_transitions(
                latent[row].tolist(),
                hybrid_states[row].tolist(),
                init,
            )
            hybrid_illegal_belief += count_illegal_transitions(
                predicted_levels[row].tolist(),
                hybrid_states[row].tolist(),
                init,
            )
            e2e_illegal_true += count_illegal_transitions(
                latent[row].tolist(),
                e2e_states[row].tolist(),
                init,
            )

    return {
        "sigma": sigma,
        "bias": bias,
        "seq_len": seq_len,
        "batches": batches,
        "near_threshold_prob": near_threshold_prob,
        "tokens": total,
        "classifier_accuracy": classifier_correct / total,
        "hybrid_state_accuracy": hybrid_correct / total,
        "end_to_end_state_accuracy": e2e_correct / total,
        "hybrid_illegal_transitions_vs_true_latent": hybrid_illegal_true,
        "hybrid_illegal_transitions_vs_predicted_belief": hybrid_illegal_belief,
        "end_to_end_illegal_transitions_vs_true_latent": e2e_illegal_true,
    }


def predict_hybrid_sequence(
    classifier: LevelClassifier,
    observations: Sequence[float],
    initial_state: int | State = State.OFF,
    device: torch.device | None = None,
) -> tuple[list[int], list[int]]:
    classifier.eval()
    device = device or next(classifier.parameters()).device
    obs = torch.tensor([list(observations)], dtype=torch.float32, device=device)
    initial = torch.tensor([int(initial_state)], dtype=torch.long, device=device)
    with torch.no_grad():
        levels = classifier(obs).argmax(dim=-1)
        states = hybrid_predict_from_levels(levels, initial)
    return levels[0].cpu().tolist(), states[0].cpu().tolist()
