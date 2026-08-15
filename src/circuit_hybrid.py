from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from .circuit_compiled import CompiledCircuitBreakerTransformer
from .circuit_learned import CircuitBatcher, CircuitLearnedConfig, TinyCircuitTransformer, choose_device, make_tokens, set_seed, sinusoidal_positions
from .circuit_reference import Event, initial_state, invariant_violations, run_controller, state_id_maps, transition


@dataclass
class CircuitHybridConfig:
    seed: int = 31
    classifier_steps: int = 1000
    e2e_steps: int = 1500
    batch_size: int = 128
    train_len: int = 64
    sigma: float = 0.35
    learning_rate: float = 5.0e-4
    d_model: int = 48
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 96
    max_len: int = 4096


EVENT_CENTERS = torch.tensor([1.0, -1.0, 0.0], dtype=torch.float32)


class EventClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, 3),
        )

    def forward(self, observations: Tensor) -> Tensor:
        return self.net(observations.unsqueeze(-1))


class ContinuousCircuitTransformer(nn.Module):
    def __init__(self, config: CircuitHybridConfig, num_states: int):
        super().__init__()
        self.config = config
        self.bos = nn.Parameter(torch.zeros(config.d_model))
        self.observation_projection = nn.Linear(1, config.d_model)
        self.register_buffer("position_encoding", sinusoidal_positions(config.max_len + 1, config.d_model), persistent=False)
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
        self.output = nn.Linear(config.d_model, num_states)

    def forward(self, observations: Tensor) -> Tensor:
        batch_size, seq_len = observations.shape
        total_len = seq_len + 1
        if total_len > self.position_encoding.shape[0]:
            raise ValueError("sequence too long")
        bos = self.bos[None, None, :].expand(batch_size, 1, -1)
        obs = self.observation_projection(observations.unsqueeze(-1))
        hidden = torch.cat([bos, obs], dim=1)
        hidden = hidden + self.position_encoding[:total_len].to(observations.device)[None, :, :]
        mask = torch.full((total_len, total_len), float("-inf"), device=observations.device)
        mask = torch.triu(mask, diagonal=1)
        hidden = self.blocks(hidden, mask=mask)
        return self.output(hidden)


def observations_from_events(events: Tensor, sigma: float, bias: float = 0.0) -> Tensor:
    centers = EVENT_CENTERS.to(events.device)
    return centers[events] + bias + sigma * torch.randn(events.shape, device=events.device)


def train_event_classifier(config: CircuitHybridConfig, device: torch.device) -> tuple[EventClassifier, dict[str, Any]]:
    classifier = EventClassifier().to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=config.learning_rate)
    history = []
    start = time.time()
    # Balanced perception training keeps failures from disappearing into the majority class.
    for step in range(1, config.classifier_steps + 1):
        labels = torch.randint(0, 3, (config.batch_size,), device=device)
        obs = observations_from_events(labels, config.sigma)
        logits = classifier(obs)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step == config.classifier_steps or step % 150 == 0:
            acc = (logits.argmax(dim=-1) == labels).float().mean().item()
            history.append({"step": step, "loss": float(loss.item()), "accuracy": acc})
    return classifier, {"history": history, "training_time_seconds": time.time() - start}


def train_e2e_controller(config: CircuitHybridConfig, device: torch.device) -> tuple[ContinuousCircuitTransformer, dict[str, Any]]:
    batcher = CircuitBatcher(device)
    model = ContinuousCircuitTransformer(config, len(batcher.states)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    history = []
    start = time.time()
    for step in range(1, config.e2e_steps + 1):
        latent = batcher.sample_inputs(config.batch_size, config.train_len, "natural")
        observations = observations_from_events(latent, config.sigma)
        labels = batcher.labels_for_inputs(latent)
        logits = model(observations)[:, 1:, :]
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step == config.e2e_steps or step % 150 == 0:
            pred = logits.argmax(dim=-1)
            state_acc = (pred == labels).float().mean().item()
            mode_acc = (batcher.mode_ids_for_state_ids(pred) == batcher.mode_ids_for_state_ids(labels)).float().mean().item()
            history.append({"step": step, "loss": float(loss.item()), "state_accuracy": state_acc, "mode_accuracy": mode_acc})
    return model, {"history": history, "training_time_seconds": time.time() - start}


def train_hybrid(config: CircuitHybridConfig | None = None, device: torch.device | None = None):
    config = config or CircuitHybridConfig()
    set_seed(config.seed)
    device = device or choose_device()
    classifier, classifier_metrics = train_event_classifier(config, device)
    e2e, e2e_metrics = train_e2e_controller(config, device)
    return classifier, e2e, {
        "config": asdict(config),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "classifier": classifier_metrics,
        "end_to_end": e2e_metrics,
    }


def compiled_states_for_event_batch(events: Tensor) -> Tensor:
    device = events.device
    model = CompiledCircuitBreakerTransformer().to(device)
    state_id = model.state_to_id[initial_state()]
    state_ids = []
    tokens_batch = None
    # This helper is only used for moderate evaluation batches; a Python loop is
    # decoding, not state-transition logic.
    outputs = []
    for row in events.cpu().tolist():
        states, _ = model.decode_inputs(row, device=device)
        outputs.append([model.state_to_id[s] for s in states])
    return torch.tensor(outputs, dtype=torch.long, device=device)


def evaluate_hybrid(
    classifier: EventClassifier,
    e2e: ContinuousCircuitTransformer,
    config: CircuitHybridConfig,
    device: torch.device,
    sigma: float,
    bias: float = 0.0,
    seq_len: int = 256,
    batches: int = 4,
    distribution: str = "natural",
) -> dict[str, Any]:
    classifier.eval()
    e2e.eval()
    batcher = CircuitBatcher(device)
    states, _ = state_id_maps()
    total = 0
    classifier_correct = 0
    hybrid_world_correct = 0
    e2e_world_correct = 0
    hybrid_belief_violations = 0
    hybrid_world_violations = 0
    e2e_world_violations = 0

    for _ in range(batches):
        latent = batcher.sample_inputs(config.batch_size, seq_len, distribution)
        observations = observations_from_events(latent, sigma, bias=bias)
        true_state_ids = batcher.labels_for_inputs(latent)
        with torch.no_grad():
            predicted_events = classifier(observations).argmax(dim=-1)
            hybrid_state_ids = compiled_states_for_event_batch(predicted_events)
            e2e_state_ids = e2e(observations)[:, 1:, :].argmax(dim=-1)

        total += latent.numel()
        classifier_correct += (predicted_events == latent).sum().item()
        hybrid_world_correct += (hybrid_state_ids == true_state_ids).sum().item()
        e2e_world_correct += (e2e_state_ids == true_state_ids).sum().item()

        for row in range(latent.shape[0]):
            prev_hybrid_belief = initial_state()
            prev_hybrid_world = initial_state()
            prev_e2e_world = initial_state()
            for t in range(seq_len):
                belief_event = int(predicted_events[row, t].item())
                true_event = int(latent[row, t].item())
                hybrid_state = states[int(hybrid_state_ids[row, t].item())]
                e2e_state = states[int(e2e_state_ids[row, t].item())]
                hybrid_belief_violations += int(bool(invariant_violations(prev_hybrid_belief, belief_event, hybrid_state)))
                hybrid_world_violations += int(bool(invariant_violations(prev_hybrid_world, true_event, hybrid_state)))
                e2e_world_violations += int(bool(invariant_violations(prev_e2e_world, true_event, e2e_state)))
                prev_hybrid_belief = hybrid_state
                prev_hybrid_world = hybrid_state
                prev_e2e_world = e2e_state

    return {
        "sigma": sigma,
        "bias": bias,
        "seq_len": seq_len,
        "batches": batches,
        "distribution": distribution,
        "tokens": total,
        "classifier_accuracy": classifier_correct / total,
        "hybrid_world_state_accuracy": hybrid_world_correct / total,
        "end_to_end_world_state_accuracy": e2e_world_correct / total,
        "hybrid_belief_relative_semantic_violations": hybrid_belief_violations,
        "hybrid_world_relative_semantic_violations": hybrid_world_violations,
        "end_to_end_world_relative_semantic_violations": e2e_world_violations,
    }
