from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class SoftmaxAttentionConfig:
    state_record_gap: float = 2.0
    non_state_penalty: float = 4.0
    dtype: torch.dtype = torch.float32

    @property
    def position_scale(self) -> float:
        # State records and input records alternate, so adjacent state records are
        # two token positions apart.
        return self.state_record_gap / 2.0


PASS_EXACT = "PASS_EXACT"
SEMANTIC_FAILURE = "SEMANTIC_FAILURE"
NUMERIC_FAILURE = "NUMERIC_FAILURE"
EFFECTIVELY_HARD = "EFFECTIVELY_HARD"


def simple_correct_probability_lower_bound(delta: float, competitors: int) -> float:
    if competitors < 0:
        raise ValueError("competitors must be non-negative")
    if competitors == 0:
        return 1.0
    z = competitors * exp(-float(delta))
    return 1.0 / (1.0 + z)


def simple_leakage_upper_bound(delta: float, competitors: int) -> float:
    return 1.0 - simple_correct_probability_lower_bound(delta, competitors)


def geometric_stale_state_unnormalized_bound(state_record_gap: float) -> float:
    r = exp(-float(state_record_gap))
    if r >= 1.0:
        return float("inf")
    return r / (1.0 - r)


def geometric_state_correct_probability_lower_bound(state_record_gap: float) -> float:
    stale = geometric_stale_state_unnormalized_bound(state_record_gap)
    return 1.0 / (1.0 + stale)


def state_leakage_sufficient_for_decoding(state_record_gap: float) -> bool:
    # With one-hot state values and non-state zero values, the correct next-state
    # logit is proportional to unnormalized mass 1 plus any helpful stale mass.
    # The worst wrong next-state logit is at most the total stale-state mass S.
    # Therefore S < 1 is sufficient for decoded argmax correctness.
    return geometric_stale_state_unnormalized_bound(state_record_gap) < 1.0


def non_state_unnormalized_bound(state_record_gap: float, non_state_penalty: float) -> float:
    # Correct state record is one token before the current input. Current input
    # gap is beta - state_record_gap/2. Older inputs then decay by state_record_gap.
    alpha = state_record_gap / 2.0
    beta = non_state_penalty
    r = exp(-state_record_gap)
    if r >= 1.0:
        return float("inf")
    current = exp(-(beta - alpha))
    older = exp(-(beta + alpha)) / (1.0 - r)
    return current + older


def finite_geometric_stale_bound(state_record_gap: float, stale_state_records: int) -> float:
    if stale_state_records <= 0:
        return 0.0
    r = exp(-float(state_record_gap))
    return r * (1.0 - r ** stale_state_records) / (1.0 - r)


def score_history(
    token_history: Tensor,
    state_token_offset: int,
    config: SoftmaxAttentionConfig,
) -> tuple[Tensor, Tensor]:
    tokens = token_history.to(torch.long)
    positions = torch.arange(tokens.shape[-1], device=tokens.device, dtype=config.dtype)
    is_state = tokens >= state_token_offset
    # Equivalent scalar QK score: q=[alpha,beta], k=[position,-1(non-state)].
    scores = positions.to(tokens.device) * config.position_scale
    if tokens.ndim == 2:
        scores = scores[None, :].expand(tokens.shape[0], -1).clone()
    else:
        scores = scores.clone()
    scores = scores - (~is_state).to(config.dtype) * config.non_state_penalty
    return scores, is_state


def latest_state_attention(
    token_history: Tensor,
    state_token_offset: int,
    num_states: int,
    config: SoftmaxAttentionConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    squeeze = False
    if token_history.ndim == 1:
        token_history = token_history.unsqueeze(0)
        squeeze = True
    if token_history.ndim != 2:
        raise ValueError("token_history must have shape [time] or [batch, time]")
    tokens = token_history.to(torch.long)
    scores, is_state = score_history(tokens, state_token_offset, config)
    if not is_state.any(dim=-1).all():
        raise ValueError("each history must contain at least one state token")
    weights = torch.softmax(scores, dim=-1)
    state_ids = (tokens - state_token_offset).clamp(min=0, max=max(num_states - 1, 0))
    state_masses = torch.zeros(
        (tokens.shape[0], num_states),
        dtype=config.dtype,
        device=tokens.device,
    )
    state_masses.scatter_add_(1, state_ids.masked_fill(~is_state, 0), weights * is_state.to(config.dtype))

    state_positions_for_argmax = torch.arange(tokens.shape[-1], device=tokens.device)[None, :].expand_as(tokens)
    latest_indices = state_positions_for_argmax.masked_fill(~is_state, -1).argmax(dim=-1)
    latest_tokens = tokens.gather(1, latest_indices[:, None]).squeeze(1)
    latest_state_ids = latest_tokens - state_token_offset
    correct_mass = state_masses.gather(1, latest_state_ids[:, None]).squeeze(1)
    total_state_mass = (weights * is_state.to(config.dtype)).sum(dim=-1)
    stale_state_mass = total_state_mass - correct_mass
    non_state_mass = (weights * (~is_state).to(config.dtype)).sum(dim=-1)
    losing_mass = 1.0 - correct_mass
    effectively_hard = losing_mass == 0
    finite = torch.isfinite(weights).all(dim=-1) & torch.isfinite(state_masses).all(dim=-1)
    debug = {
        "scores": scores,
        "weights": weights,
        "is_state": is_state,
        "latest_index": latest_indices,
        "latest_state_id": latest_state_ids,
        "correct_mass": correct_mass,
        "stale_state_mass": stale_state_mass,
        "non_state_mass": non_state_mass,
        "losing_mass": losing_mass,
        "effectively_hard": effectively_hard,
        "finite": finite,
        "state_masses": state_masses,
    }
    if squeeze:
        state_masses = state_masses.squeeze(0)
        debug = {k: v.squeeze(0) for k, v in debug.items()}
    return state_masses, debug


def classify_softmax_run(
    exact: bool,
    finite: bool,
    effectively_hard: bool,
) -> str:
    if not finite:
        return NUMERIC_FAILURE
    if exact and effectively_hard:
        return EFFECTIVELY_HARD
    if exact:
        return PASS_EXACT
    return SEMANTIC_FAILURE


def attention_stats_to_json(debug: dict[str, Tensor]) -> dict[str, Any]:
    return {
        "min_correct_mass": float(debug["correct_mass"].min().item()),
        "max_stale_state_mass": float(debug["stale_state_mass"].max().item()),
        "max_non_state_mass": float(debug["non_state_mass"].max().item()),
        "max_losing_mass": float(debug["losing_mass"].max().item()),
        "effectively_hard": bool(debug["effectively_hard"].all().item()),
        "finite": bool(debug["finite"].all().item()),
    }


def dtype_supported(dtype: torch.dtype) -> bool:
    try:
        x = torch.tensor([0.0, -1.0], dtype=dtype)
        torch.softmax(x, dim=0)
        return True
    except Exception:
        return False
