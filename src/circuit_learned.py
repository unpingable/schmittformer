from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from .circuit_reference import (
    EVENTS,
    Event,
    Mode,
    initial_state,
    invariant_violations,
    run_controller,
    state_id_maps,
    transition,
    transition_label,
)

BOS_TOKEN = 3
VOCAB_SIZE = 4


@dataclass
class CircuitLearnedConfig:
    seed: int = 21
    distribution: str = "natural"
    steps: int = 1500
    batch_size: int = 128
    train_len: int = 64
    d_model: int = 48
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 96
    learning_rate: float = 3.0e-4
    max_len: int = 4096
    log_every: int = 150


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sinusoidal_positions(max_len: int, d_model: int) -> Tensor:
    positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
    pe = torch.zeros(max_len, d_model, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(positions * div)
    pe[:, 1::2] = torch.cos(positions * div[: pe[:, 1::2].shape[1]])
    return pe


class TinyCircuitTransformer(nn.Module):
    def __init__(self, config: CircuitLearnedConfig, num_states: int):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(VOCAB_SIZE, config.d_model)
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

    def forward(self, tokens: Tensor) -> Tensor:
        batch_size, seq_len = tokens.shape
        if seq_len > self.position_encoding.shape[0]:
            raise ValueError("sequence longer than configured max_len")
        hidden = self.token_embedding(tokens) + self.position_encoding[:seq_len].to(tokens.device)[None, :, :]
        mask = torch.full((seq_len, seq_len), float("-inf"), device=tokens.device)
        mask = torch.triu(mask, diagonal=1)
        hidden = self.blocks(hidden, mask=mask)
        return self.output(hidden)


class CircuitBatcher:
    def __init__(self, device: torch.device):
        states, state_to_id = state_id_maps()
        self.states = states
        self.state_to_id = state_to_id
        table = torch.empty((len(states), len(EVENTS)), dtype=torch.long, device=device)
        mode_table = torch.empty((len(states),), dtype=torch.long, device=device)
        for state, state_id in state_to_id.items():
            mode_table[state_id] = state.mode
            for event in EVENTS:
                table[state_id, int(event)] = state_to_id[transition(state, event)]
        self.transition_table = table
        self.mode_table = mode_table
        self.initial_id = state_to_id[initial_state()]
        self.device = device

    def sample_inputs(self, batch_size: int, seq_len: int, distribution: str) -> Tensor:
        if distribution == "natural":
            probs = torch.tensor([0.865, 0.035, 0.100], device=self.device)
            return torch.multinomial(probs, batch_size * seq_len, replacement=True).reshape(batch_size, seq_len)
        if distribution == "balanced":
            probs = torch.tensor([0.45, 0.35, 0.20], device=self.device)
            return torch.multinomial(probs, batch_size * seq_len, replacement=True).reshape(batch_size, seq_len)
        if distribution == "adversarial":
            patterns = torch.tensor(
                [
                    [1, 1, 1, 2, 2, 2, 2, 2, 0, 0],
                    [0, 1, 0, 1, 2, 0, 1, 2, 1, 2],
                    [2, 2, 2, 2, 2, 0, 0, 1, 1, 1],
                    [0, 2, 0, 1, 2, 1, 0, 1, 2, 0],
                ],
                dtype=torch.long,
                device=self.device,
            )
            rows = patterns[torch.randint(0, patterns.shape[0], (batch_size,), device=self.device)]
            reps = (seq_len + rows.shape[1] - 1) // rows.shape[1]
            tiled = rows.repeat_interleave(reps, dim=1)[:, :seq_len]
            noise = torch.randint(0, 3, (batch_size, seq_len), device=self.device)
            choose_noise = torch.rand(batch_size, seq_len, device=self.device) < 0.08
            return torch.where(choose_noise, noise, tiled)
        raise ValueError(f"unknown distribution {distribution}")

    def labels_for_inputs(self, inputs: Tensor) -> Tensor:
        state_ids = torch.full((inputs.shape[0],), self.initial_id, dtype=torch.long, device=inputs.device)
        labels = []
        for t in range(inputs.shape[1]):
            state_ids = self.transition_table[state_ids, inputs[:, t]]
            labels.append(state_ids)
        return torch.stack(labels, dim=1)

    def mode_ids_for_state_ids(self, state_ids: Tensor) -> Tensor:
        return self.mode_table[state_ids]


def make_tokens(inputs: Tensor) -> Tensor:
    bos = torch.full((inputs.shape[0], 1), BOS_TOKEN, dtype=torch.long, device=inputs.device)
    return torch.cat([bos, inputs.to(torch.long)], dim=1)


def train_circuit_model(
    config: CircuitLearnedConfig,
    device: torch.device | None = None,
) -> tuple[TinyCircuitTransformer, dict[str, Any]]:
    set_seed(config.seed)
    device = device or choose_device()
    batcher = CircuitBatcher(device)
    model = TinyCircuitTransformer(config, len(batcher.states)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, Any]] = []
    start = time.time()
    for step in range(1, config.steps + 1):
        model.train()
        inputs = batcher.sample_inputs(config.batch_size, config.train_len, config.distribution)
        labels = batcher.labels_for_inputs(inputs)
        logits = model(make_tokens(inputs))[:, 1:, :]
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step == config.steps or step % config.log_every == 0:
            with torch.no_grad():
                predictions = logits.argmax(dim=-1)
                state_acc = (predictions == labels).float().mean().item()
                mode_acc = (batcher.mode_ids_for_state_ids(predictions) == batcher.mode_ids_for_state_ids(labels)).float().mean().item()
            history.append({"step": step, "loss": float(loss.item()), "state_accuracy": state_acc, "mode_accuracy": mode_acc})
    return model, {
        "config": asdict(config),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "training_time_seconds": time.time() - start,
        "history": history,
    }


def predict_state_ids(model: TinyCircuitTransformer, inputs: Sequence[int | Event], device: torch.device | None = None) -> list[int]:
    model.eval()
    device = device or next(model.parameters()).device
    tensor = torch.tensor([[int(Event(x)) for x in inputs]], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(make_tokens(tensor))[0, 1:, :]
    return [int(x) for x in logits.argmax(dim=-1).cpu().tolist()]


def evaluate_sequences(
    model: TinyCircuitTransformer,
    sequences: dict[str, Sequence[int | Event]],
) -> dict[str, Any]:
    device = next(model.parameters()).device
    states, state_to_id = state_id_maps()
    mode_table = torch.tensor([s.mode for s in states], dtype=torch.long)
    out: dict[str, Any] = {}
    for name, seq in sequences.items():
        inputs = [int(Event(x)) for x in seq]
        expected_states = run_controller(inputs)
        expected_ids = [state_to_id[s] for s in expected_states]
        expected_modes = [s.mode for s in expected_states]
        pred_ids = predict_state_ids(model, inputs, device)
        pred_modes = [states[i].mode for i in pred_ids]
        first_state_bad = next((i for i, (a, b) in enumerate(zip(expected_ids, pred_ids)) if a != b), None)
        first_mode_bad = next((i for i, (a, b) in enumerate(zip(expected_modes, pred_modes)) if a != b), None)
        true_prev = initial_state()
        pred_prev = initial_state()
        violation_counts: dict[str, int] = {}
        illegal = 0
        transition_conditioned: dict[str, dict[str, int]] = {}
        for i, (event, pred_id, expected_state) in enumerate(zip(inputs, pred_ids, expected_states)):
            pred_state = states[pred_id]
            label = transition_label(true_prev, expected_state)
            bucket = transition_conditioned.setdefault(label, {"total": 0, "state_correct": 0, "mode_correct": 0})
            bucket["total"] += 1
            bucket["state_correct"] += int(pred_state == expected_state)
            bucket["mode_correct"] += int(pred_state.mode == expected_state.mode)
            violations = invariant_violations(pred_prev, event, pred_state)
            if violations:
                illegal += 1
                for violation in violations:
                    violation_counts[violation] = violation_counts.get(violation, 0) + 1
            pred_prev = pred_state
            true_prev = expected_state
        for bucket in transition_conditioned.values():
            total = bucket["total"]
            bucket["state_accuracy"] = bucket["state_correct"] / total
            bucket["mode_accuracy"] = bucket["mode_correct"] / total
        out[name] = {
            "length": len(inputs),
            "state_accuracy": sum(a == b for a, b in zip(expected_ids, pred_ids)) / len(inputs) if inputs else 1.0,
            "mode_accuracy": sum(a == b for a, b in zip(expected_modes, pred_modes)) / len(inputs) if inputs else 1.0,
            "first_state_divergence": first_state_bad,
            "first_mode_divergence": first_mode_bad,
            "illegal_transition_steps": illegal,
            "violation_counts": violation_counts,
            "transition_conditioned": transition_conditioned,
        }
    return out


def learned_history_equivalence(
    model: TinyCircuitTransformer,
    history_groups: dict[Any, list[tuple[int, ...]]],
    suffixes: Sequence[Sequence[int | Event]],
    max_histories_per_state: int = 5,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    states, state_to_id = state_id_maps()
    violations = []
    comparisons = 0
    groups_checked = 0
    for state, histories in history_groups.items():
        histories = histories[:max_histories_per_state]
        if len(histories) < 2:
            continue
        groups_checked += 1
        for suffix in suffixes:
            base_outputs = None
            base_history = None
            for history in histories:
                full = [*history, *[int(Event(x)) for x in suffix]]
                pred = predict_state_ids(model, full, device)
                suffix_pred = tuple(pred[-len(suffix):]) if suffix else ()
                if base_outputs is None:
                    base_outputs = suffix_pred
                    base_history = history
                else:
                    comparisons += 1
                    if suffix_pred != base_outputs:
                        violations.append(
                            {
                                "state": state.to_json(),
                                "suffix": [int(Event(x)) for x in suffix],
                                "history_a_len": len(base_history or ()),
                                "history_b_len": len(history),
                                "history_a": list(base_history or ()),
                                "history_b": list(history),
                                "outputs_a": list(base_outputs),
                                "outputs_b": list(suffix_pred),
                            }
                        )
                        if len(violations) >= 50:
                            return {
                                "groups_checked": groups_checked,
                                "comparisons": comparisons,
                                "history_equivalence_violations": len(violations),
                                "examples": violations,
                            }
    return {
        "groups_checked": groups_checked,
        "comparisons": comparisons,
        "history_equivalence_violations": len(violations),
        "examples": violations,
    }


def minimize_sequence_for_predicate(
    sequence: Sequence[int],
    predicate,
    max_passes: int = 4,
) -> list[int]:
    candidate = list(sequence)
    if not predicate(candidate):
        return candidate
    for _ in range(max_passes):
        changed = False
        i = 0
        while i < len(candidate):
            trial = candidate[:i] + candidate[i + 1 :]
            if trial and predicate(trial):
                candidate = trial
                changed = True
            else:
                i += 1
        if not changed:
            break
    return candidate
