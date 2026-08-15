from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor, nn

from .ffn_counter import add_saturating_decrement, builder_const0, builder_const1
from .fixed_state import (
    BUDGET_BITS,
    EVENT_COUNT,
    LEASE_BITS,
    OUTPUT_COUNT,
    STATE_WIDTH,
    GovernanceEvent,
    GovernanceOutput,
    GovernanceState,
    decode_state_bits,
    decode_states_bits,
    encode_state_bits,
    encode_states_bits,
)
from .relu_boolean import ReLUCircuitBuilder, SynthesizedReLUCircuit
from .stock_transformer_recurrent import StockSoftmaxGather


@dataclass(frozen=True)
class StockGovernanceConfig:
    score_gap: float = 8.0
    dtype_name: str = "float32"
    logit_margin: float = 16.0

    @property
    def dtype(self) -> torch.dtype:
        return {
            "float64": torch.float64,
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[self.dtype_name]


def _bits_for_int(value: int, width: int) -> list[int]:
    return [(int(value) >> i) & 1 for i in range(width)]


def _code_wires(builder: ReLUCircuitBuilder, value: int, width: int) -> list[str]:
    zero = builder_const0(builder)
    one = builder_const1(builder)
    return [one if bit else zero for bit in _bits_for_int(value, width)]


def _onehot_wires(builder: ReLUCircuitBuilder, index: int, width: int) -> list[str]:
    zero = builder_const0(builder)
    one = builder_const1(builder)
    return [one if i == int(index) else zero for i in range(width)]


def _select_code(builder: ReLUCircuitBuilder, prefix: str, current: list[str], cond: str, value: int, width: int) -> list[str]:
    return builder.mux_many(prefix, cond, _code_wires(builder, value, width), current)


def _select_onehot(builder: ReLUCircuitBuilder, prefix: str, current: list[str], cond: str, index: int, width: int) -> list[str]:
    return builder.mux_many(prefix, cond, _onehot_wires(builder, index, width), current)


def _event_name(event: GovernanceEvent) -> str:
    return f"event_{int(event)}"


def build_governance_ffn(dtype: torch.dtype = torch.float32, logit_margin: float = 16.0) -> nn.Module:
    input_names = [f"slot_{i}" for i in range(STATE_WIDTH + EVENT_COUNT)]
    builder = ReLUCircuitBuilder(input_names, dtype=dtype)
    zero = builder_const0(builder)
    one = builder_const1(builder)

    pos = 0
    authority = input_names[pos]
    pos += 1
    lease = input_names[pos : pos + LEASE_BITS]
    pos += LEASE_BITS
    budget = input_names[pos : pos + BUDGET_BITS]
    pos += BUDGET_BITS
    occurrence = input_names[pos : pos + 2]
    pos += 2
    settlement = input_names[pos : pos + 2]
    events = {event: input_names[STATE_WIDTH + int(event)] for event in GovernanceEvent}

    is_idle = builder.eq_bits("is_idle", occurrence, [0, 0])
    is_in_flight = builder.eq_bits("is_in_flight", occurrence, [1, 0])
    is_ambiguous = builder.eq_bits("is_ambiguous", occurrence, [0, 1])
    valid_occurrence = builder.or_many("valid_occurrence", [is_idle, is_in_flight, is_ambiguous])
    is_settlement_none = builder.eq_bits("is_settlement_none", settlement, [0, 0])
    is_settlement_success = builder.eq_bits("is_settlement_success", settlement, [1, 0])
    is_settlement_failure = builder.eq_bits("is_settlement_failure", settlement, [0, 1])
    valid_settlement = builder.or_many("valid_settlement", [is_settlement_none, is_settlement_success, is_settlement_failure])
    valid_state = builder.and_many("valid_state", [valid_occurrence, valid_settlement])
    invalid_state = builder.not_wire("invalid_state", valid_state)

    lease_dec, lease_zero, lease_nonzero = add_saturating_decrement(builder, lease, "lease")
    budget_dec, budget_zero, budget_nonzero = add_saturating_decrement(builder, budget, "budget")
    del lease_zero, budget_zero

    authority_invalid = builder.not_wire("authority_invalid", authority)
    proposal = events[GovernanceEvent.PROPOSE_ACTION]
    result_success = events[GovernanceEvent.RESULT_SUCCESS]
    result_failure = events[GovernanceEvent.RESULT_FAILURE]
    result_ambiguous = events[GovernanceEvent.RESULT_AMBIGUOUS]
    settle_success = events[GovernanceEvent.SETTLE_SUCCESS]
    settle_failure = events[GovernanceEvent.SETTLE_FAILURE]
    any_settle = builder.or_many("any_settle", [settle_success, settle_failure])

    propose_and_valid = builder.and_many("propose_and_valid", [proposal, valid_state])
    propose_idle = builder.and_many("propose_idle", [propose_and_valid, is_idle])
    admit = builder.and_many("admit", [propose_idle, authority, lease_nonzero, budget_nonzero])
    refuse_ambiguous = builder.and_many("refuse_ambiguous", [propose_and_valid, is_ambiguous])
    refuse_in_flight = builder.and_many("refuse_in_flight", [propose_and_valid, is_in_flight])
    refuse_no_authority = builder.and_many("refuse_no_authority", [propose_idle, authority_invalid])
    lease_empty = builder.not_wire("lease_empty", lease_nonzero)
    budget_empty = builder.not_wire("budget_empty", budget_nonzero)
    refuse_expired = builder.and_many("refuse_expired", [propose_idle, authority, lease_empty])
    refuse_budget = builder.and_many("refuse_budget", [propose_idle, authority, lease_nonzero, budget_empty])

    authority_next = [authority]
    authority_next = builder.mux_many("authority_grant", events[GovernanceEvent.GRANT_AUTHORITY], [one], authority_next)
    authority_next = builder.mux_many("authority_revoke", events[GovernanceEvent.REVOKE_AUTHORITY], [zero], authority_next)

    lease_next = lease
    lease_next = builder.mux_many("lease_tick", events[GovernanceEvent.TICK], lease_dec, lease_next)
    lease_next = builder.mux_many("lease_renew_max", events[GovernanceEvent.RENEW_LEASE_MAX], [one] * LEASE_BITS, lease_next)
    lease_next = builder.mux_many("lease_renew_one", events[GovernanceEvent.RENEW_LEASE_ONE], _code_wires(builder, 1, LEASE_BITS), lease_next)

    budget_next = budget
    budget_next = builder.mux_many("budget_admit", admit, budget_dec, budget_next)
    budget_next = builder.mux_many("budget_reset_max", events[GovernanceEvent.RESET_BUDGET_MAX], [one] * BUDGET_BITS, budget_next)
    budget_next = builder.mux_many("budget_reset_one", events[GovernanceEvent.RESET_BUDGET_ONE], _code_wires(builder, 1, BUDGET_BITS), budget_next)

    success_from_inflight = builder.and_many("success_from_inflight", [is_in_flight, result_success])
    failure_from_inflight = builder.and_many("failure_from_inflight", [is_in_flight, result_failure])
    ambiguous_from_inflight = builder.and_many("ambiguous_from_inflight", [is_in_flight, result_ambiguous])
    settle_from_ambiguous = builder.and_many("settle_from_ambiguous", [is_ambiguous, any_settle])
    settle_success_from_ambiguous = builder.and_many("settle_success_from_ambiguous", [is_ambiguous, settle_success])
    settle_failure_from_ambiguous = builder.and_many("settle_failure_from_ambiguous", [is_ambiguous, settle_failure])

    occurrence_next = occurrence
    occurrence_next = _select_code(builder, "occ_admit", occurrence_next, admit, 1, 2)
    occurrence_next = _select_code(builder, "occ_success", occurrence_next, success_from_inflight, 0, 2)
    occurrence_next = _select_code(builder, "occ_failure", occurrence_next, failure_from_inflight, 0, 2)
    occurrence_next = _select_code(builder, "occ_ambiguous", occurrence_next, ambiguous_from_inflight, 2, 2)
    occurrence_next = _select_code(builder, "occ_settle", occurrence_next, settle_from_ambiguous, 0, 2)

    settlement_next = settlement
    settlement_next = _select_code(builder, "settle_admit", settlement_next, admit, 0, 2)
    settlement_next = _select_code(builder, "settle_success", settlement_next, success_from_inflight, 1, 2)
    settlement_next = _select_code(builder, "settle_failure", settlement_next, failure_from_inflight, 2, 2)
    settlement_next = _select_code(builder, "settle_ambiguous", settlement_next, ambiguous_from_inflight, 0, 2)
    settlement_next = _select_code(builder, "settle_explicit_success", settlement_next, settle_success_from_ambiguous, 1, 2)
    settlement_next = _select_code(builder, "settle_explicit_failure", settlement_next, settle_failure_from_ambiguous, 2, 2)

    safe_state = [zero] + [zero] * LEASE_BITS + [zero] * BUDGET_BITS + _code_wires(builder, 0, 2) + _code_wires(builder, 0, 2)
    next_state = authority_next + lease_next + budget_next + occurrence_next + settlement_next
    next_state = builder.mux_many("invalid_safe_state", invalid_state, safe_state, next_state)

    output = _onehot_wires(builder, int(GovernanceOutput.NO_ACTION), OUTPUT_COUNT)
    output = _select_onehot(builder, "out_admit", output, admit, int(GovernanceOutput.ADMIT_ACTION), OUTPUT_COUNT)
    output = _select_onehot(builder, "out_budget", output, refuse_budget, int(GovernanceOutput.REFUSE_BUDGET), OUTPUT_COUNT)
    output = _select_onehot(builder, "out_expired", output, refuse_expired, int(GovernanceOutput.REFUSE_EXPIRED), OUTPUT_COUNT)
    output = _select_onehot(builder, "out_no_auth", output, refuse_no_authority, int(GovernanceOutput.REFUSE_NO_AUTHORITY), OUTPUT_COUNT)
    output = _select_onehot(builder, "out_in_flight", output, refuse_in_flight, int(GovernanceOutput.REFUSE_IN_FLIGHT), OUTPUT_COUNT)
    output = _select_onehot(builder, "out_ambiguous", output, refuse_ambiguous, int(GovernanceOutput.REFUSE_AMBIGUOUS), OUTPUT_COUNT)
    output = _select_onehot(builder, "out_invalid", output, invalid_state, int(GovernanceOutput.REFUSE_INVALID_STATE), OUTPUT_COUNT)

    readout = [(wire, 1.0, 0.0) for wire in next_state]
    readout.extend((wire, 2.0 * logit_margin, -logit_margin) for wire in output)
    return builder.build(readout)


class StockGovernanceTransformer(nn.Module):
    """Ordinary finite-softmax attention plus synthesized stock ReLU FFNs."""

    def __init__(self, config: StockGovernanceConfig | None = None, ffn: nn.Module | None = None):
        super().__init__()
        self.config = config or StockGovernanceConfig()
        self.input_width = STATE_WIDTH + EVENT_COUNT
        self.gather = StockSoftmaxGather(self.input_width, self.config.score_gap, dtype=self.config.dtype)
        self.ffn = ffn if ffn is not None else build_governance_ffn(dtype=self.config.dtype, logit_margin=self.config.logit_margin)

    def forward(self, slots: Tensor) -> tuple[Tensor, Tensor]:
        squeeze = False
        if slots.ndim == 1:
            slots = slots.unsqueeze(0)
            squeeze = True
        retrieved = self.gather(slots.to(dtype=self.config.dtype))
        out = self.ffn(retrieved)
        next_bits = out[:, :STATE_WIDTH]
        logits = out[:, STATE_WIDTH:]
        if squeeze:
            return next_bits.squeeze(0), logits.squeeze(0)
        return next_bits, logits


def encode_slots(state: GovernanceState, event: int | GovernanceEvent, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> Tensor:
    state_bits = encode_state_bits(state, device=torch.device(device), dtype=dtype)
    event_bits = torch.zeros(EVENT_COUNT, dtype=dtype, device=device)
    event_bits[int(GovernanceEvent(int(event)))] = 1.0
    return torch.cat([state_bits, event_bits], dim=0)


def encode_slot_batch(
    states: Sequence[GovernanceState],
    events: Sequence[int | GovernanceEvent],
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    if len(states) != len(events):
        raise ValueError("state/event count mismatch")
    device = torch.device(device)
    state_bits = encode_states_bits(states, device=device, dtype=dtype)
    event_ids = torch.tensor([int(GovernanceEvent(int(event))) for event in events], dtype=torch.long, device=device)
    event_bits = torch.nn.functional.one_hot(event_ids, num_classes=EVENT_COUNT).to(dtype)
    return torch.cat([state_bits, event_bits], dim=-1)


def decode_step_output(next_bits: Tensor, logits: Tensor) -> tuple[GovernanceState, int]:
    return decode_state_bits(next_bits), int(torch.argmax(logits.detach()).cpu().item())


def output_margins(logits: Tensor, expected_outputs: Tensor | None = None) -> Tensor:
    logits = logits.detach().float()
    if expected_outputs is None:
        sorted_logits = torch.sort(logits, dim=-1, descending=True).values
        return sorted_logits[:, 0] - sorted_logits[:, 1]
    expected_outputs = expected_outputs.to(logits.device).reshape(-1, 1)
    correct = logits.gather(1, expected_outputs).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, expected_outputs, float("-inf"))
    runner = masked.max(dim=-1).values
    return correct - runner


@torch.no_grad()
def stock_step(
    model: StockGovernanceTransformer,
    state: GovernanceState,
    event: int | GovernanceEvent,
    device: torch.device | str = "cpu",
) -> tuple[GovernanceState, int, float, float]:
    dtype = model.config.dtype
    slots = encode_slots(state, event, device=device, dtype=dtype)
    next_bits, logits = model(slots)
    decoded = decode_step_output(next_bits, logits)
    state_margin = float(torch.abs(next_bits.detach().float() - 0.5).min().cpu().item())
    sorted_logits = torch.sort(logits.detach().float(), descending=True).values
    logit_margin = float((sorted_logits[0] - sorted_logits[1]).cpu().item())
    return decoded[0], decoded[1], state_margin, logit_margin


def save_stock_governance_model(path: str | Path, model: StockGovernanceTransformer) -> None:
    payload = {"config": asdict(model.config), "state_dict": model.state_dict()}
    torch.save(payload, path)


def _empty_ffn_from_state_dict(state_dict: dict[str, Tensor], dtype: torch.dtype) -> SynthesizedReLUCircuit:
    blocks: list[nn.Sequential] = []
    index = 0
    while f"ffn.blocks.{index}.0.weight" in state_dict:
        w0 = state_dict[f"ffn.blocks.{index}.0.weight"]
        w2 = state_dict[f"ffn.blocks.{index}.2.weight"]
        lin1 = nn.Linear(w0.shape[1], w0.shape[0], bias=True, dtype=dtype)
        lin2 = nn.Linear(w2.shape[1], w2.shape[0], bias=True, dtype=dtype)
        blocks.append(nn.Sequential(lin1, nn.ReLU(), lin2))
        index += 1
    readout_w = state_dict["ffn.readout.weight"]
    readout = nn.Linear(readout_w.shape[1], readout_w.shape[0], bias=True, dtype=dtype)
    return SynthesizedReLUCircuit(blocks, readout, ())


def load_stock_governance_model(path: str | Path, device: torch.device | str = "cpu") -> StockGovernanceTransformer:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = StockGovernanceConfig(**payload["config"])
    ffn = _empty_ffn_from_state_dict(payload["state_dict"], config.dtype)
    model = StockGovernanceTransformer(config, ffn=ffn)
    model.load_state_dict(payload["state_dict"])
    return model.to(device)


def architecture_summary(model: StockGovernanceTransformer) -> dict[str, Any]:
    blocks = len(model.ffn.blocks) if hasattr(model.ffn, "blocks") else None
    max_width = 0
    for block in getattr(model.ffn, "blocks", []):
        for module in block:
            if isinstance(module, nn.Linear):
                max_width = max(max_width, module.in_features, module.out_features)
    return {
        "input_width": int(model.input_width),
        "state_width": int(STATE_WIDTH),
        "event_count": int(EVENT_COUNT),
        "output_count": int(OUTPUT_COUNT),
        "score_gap": float(model.config.score_gap),
        "dtype": model.config.dtype_name,
        "logit_margin": float(model.config.logit_margin),
        "attention": "torch.nn.MultiheadAttention",
        "ffn_blocks": int(blocks or 0),
        "max_ffn_width": int(max_width),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
    }

@torch.no_grad()
def compare_stock_reference(
    states: Sequence[GovernanceState],
    events: Sequence[int | GovernanceEvent],
    model: StockGovernanceTransformer,
    batch_size: int = 1024,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    from .recurrent_reference import transition

    if len(states) != len(events):
        raise ValueError("state/event count mismatch")
    device = torch.device(device)
    dtype = model.config.dtype
    failures: list[dict[str, Any]] = []
    checked = 0
    min_state_margin = float("inf")
    min_output_margin = float("inf")
    max_state_error = 0.0
    for start in range(0, len(states), batch_size):
        chunk_states = states[start : start + batch_size]
        chunk_events = events[start : start + batch_size]
        slots = encode_slot_batch(chunk_states, chunk_events, device=device, dtype=dtype)
        next_bits, logits = model(slots)
        expected_results = [transition(state, event) for state, event in zip(chunk_states, chunk_events)]
        expected_states = [result.next_state for result in expected_results]
        expected_outputs = torch.tensor([int(result.output) for result in expected_results], dtype=torch.long, device=device)
        expected_bits = encode_states_bits(expected_states, device=device, dtype=dtype)
        rounded_match = torch.all(next_bits.round() == expected_bits, dim=-1)
        output_ids = logits.argmax(dim=-1)
        output_match = output_ids == expected_outputs
        state_margins = torch.abs(next_bits.detach().float() - 0.5).min(dim=-1).values
        margins = output_margins(logits, expected_outputs)
        min_state_margin = min(min_state_margin, float(state_margins.min().detach().cpu().item()))
        min_output_margin = min(min_output_margin, float(margins.min().detach().cpu().item()))
        max_state_error = max(max_state_error, float(torch.abs(next_bits - expected_bits).max().detach().cpu().item()))
        bad = (~(rounded_match & output_match)).nonzero(as_tuple=False).reshape(-1)
        if bad.numel() and len(failures) < 5:
            decode_count = min(5 - len(failures), int(bad.numel()))
            decoded_states = decode_states_bits(next_bits[bad[:decode_count]])
            for local, actual_state in zip(bad[:decode_count].tolist(), decoded_states):
                idx = start + int(local)
                failures.append(
                    {
                        "index": idx,
                        "state": states[idx].to_json(),
                        "event": int(GovernanceEvent(int(events[idx]))),
                        "expected": expected_results[int(local)].to_json(),
                        "actual_state": actual_state.to_json(),
                        "actual_output": int(output_ids[int(local)].detach().cpu().item()),
                    }
                )
        checked += len(chunk_states)
    return {
        "checked": int(checked),
        "failures": int(len(failures)),
        "first_failures": failures,
        "passed": not failures,
        "min_state_bit_margin": float(min_state_margin),
        "min_output_margin": float(min_output_margin),
        "max_state_error_before_decode": float(max_state_error),
        "batch_size": int(batch_size),
        "device": str(device),
        "dtype": model.config.dtype_name,
        "score_gap": float(model.config.score_gap),
    }

