from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn

from .reference import State


OFF_INIT_TOKEN = 10
ON_INIT_TOKEN = 11
VOCAB_SIZE = 12


@dataclass
class LearnedConfig:
    seed: int = 7
    steps: int = 800
    batch_size: int = 128
    train_len: int = 16
    d_model: int = 32
    n_heads: int = 2
    n_layers: int = 2
    d_ff: int = 64
    learning_rate: float = 3.0e-4
    max_len: int = 1024
    log_every: int = 100


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def states_from_inputs_tensor(inputs: Tensor, initial_states: Tensor) -> Tensor:
    previous = initial_states.to(dtype=torch.long, device=inputs.device)
    outputs = []
    for t in range(inputs.shape[1]):
        x = inputs[:, t]
        previous = torch.where(
            x >= 7,
            torch.ones_like(previous),
            torch.where(x <= 3, torch.zeros_like(previous), previous),
        )
        outputs.append(previous)
    return torch.stack(outputs, dim=1) if outputs else inputs.new_empty(inputs.shape)


def make_tokens(inputs: Tensor, initial_states: Tensor) -> Tensor:
    init_tokens = torch.where(
        initial_states.to(torch.long) == int(State.ON),
        torch.full_like(initial_states.to(torch.long), ON_INIT_TOKEN),
        torch.full_like(initial_states.to(torch.long), OFF_INIT_TOKEN),
    )
    return torch.cat([init_tokens[:, None], inputs.to(torch.long)], dim=1)


def generate_batch(
    batch_size: int,
    seq_len: int,
    device: torch.device,
    initial: str = "random",
    near_threshold_prob: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor]:
    if near_threshold_prob > 0:
        uniform = torch.randint(0, 10, (batch_size, seq_len), device=device)
        near = torch.tensor([2, 3, 4, 6, 7, 8], device=device)
        near_samples = near[torch.randint(0, len(near), (batch_size, seq_len), device=device)]
        choose_near = torch.rand((batch_size, seq_len), device=device) < near_threshold_prob
        inputs = torch.where(choose_near, near_samples, uniform)
    else:
        inputs = torch.randint(0, 10, (batch_size, seq_len), device=device)

    if initial == "off":
        initial_states = torch.zeros(batch_size, dtype=torch.long, device=device)
    elif initial == "on":
        initial_states = torch.ones(batch_size, dtype=torch.long, device=device)
    else:
        initial_states = torch.randint(0, 2, (batch_size,), device=device)

    labels = states_from_inputs_tensor(inputs, initial_states)
    tokens = make_tokens(inputs, initial_states)
    return tokens, labels, initial_states


class TinyCausalTransformer(nn.Module):
    def __init__(self, config: LearnedConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(VOCAB_SIZE, config.d_model)
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

    def forward(self, tokens: Tensor) -> Tensor:
        batch_size, seq_len = tokens.shape
        if seq_len > self.config.max_len + 1:
            raise ValueError(f"sequence length {seq_len} exceeds max_len")
        positions = torch.arange(seq_len, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)[None, :, :]
        mask = torch.full((seq_len, seq_len), float("-inf"), device=tokens.device)
        mask = torch.triu(mask, diagonal=1)
        hidden = self.blocks(hidden, mask=mask)
        return self.output(hidden)


def train_model(
    config: LearnedConfig,
    device: torch.device | None = None,
) -> tuple[TinyCausalTransformer, dict]:
    set_seed(config.seed)
    device = device or choose_device()
    model = TinyCausalTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    history: list[dict] = []
    start = time.time()

    for step in range(1, config.steps + 1):
        model.train()
        tokens, labels, _ = generate_batch(config.batch_size, config.train_len, device)
        logits = model(tokens)[:, 1:, :]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 2),
            labels.reshape(-1),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % config.log_every == 0 or step == config.steps:
            with torch.no_grad():
                predictions = logits.argmax(dim=-1)
                accuracy = (predictions == labels).float().mean().item()
            history.append({"step": step, "loss": float(loss.item()), "accuracy": accuracy})

    elapsed = time.time() - start
    metrics = {
        "config": asdict(config),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "training_time_seconds": elapsed,
        "history": history,
    }
    return model, metrics


def predict_learned(
    model: TinyCausalTransformer,
    inputs: Sequence[int],
    initial_state: int | State = State.OFF,
    device: torch.device | None = None,
) -> list[int]:
    model.eval()
    device = device or next(model.parameters()).device
    input_tensor = torch.tensor([list(inputs)], dtype=torch.long, device=device)
    initial = torch.tensor([int(initial_state)], dtype=torch.long, device=device)
    tokens = make_tokens(input_tensor, initial)
    with torch.no_grad():
        logits = model(tokens)[0, 1:, :]
    return [int(x) for x in logits.argmax(dim=-1).cpu().tolist()]


def save_model(model: TinyCausalTransformer, config: LearnedConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(config), "state_dict": model.state_dict()}, path)


def load_model(path: Path, device: torch.device | None = None) -> TinyCausalTransformer:
    device = device or choose_device()
    payload = torch.load(path, map_location=device)
    model = TinyCausalTransformer(LearnedConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--train-len", type=int, default=16)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    config = LearnedConfig(steps=args.steps, train_len=args.train_len)
    model, metrics = train_model(config)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_model(model, config, args.out_dir / "learned_model.pt")
    (args.out_dir / "learned_train.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
