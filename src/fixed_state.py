from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor


LEASE_BITS = 16
BUDGET_BITS = 8
OCCURRENCE_BITS = 2
SETTLEMENT_BITS = 2
STATE_WIDTH = 1 + LEASE_BITS + BUDGET_BITS + OCCURRENCE_BITS + SETTLEMENT_BITS
EVENT_COUNT = 14
OUTPUT_COUNT = 8


class Authority(IntEnum):
    INVALID = 0
    VALID = 1


class Occurrence(IntEnum):
    IDLE = 0
    IN_FLIGHT = 1
    AMBIGUOUS = 2


class Settlement(IntEnum):
    NONE = 0
    SUCCESS = 1
    FAILURE = 2


class GovernanceEvent(IntEnum):
    NOOP = 0
    TICK = 1
    PROPOSE_ACTION = 2
    RESULT_SUCCESS = 3
    RESULT_FAILURE = 4
    RESULT_AMBIGUOUS = 5
    SETTLE_SUCCESS = 6
    SETTLE_FAILURE = 7
    GRANT_AUTHORITY = 8
    REVOKE_AUTHORITY = 9
    RENEW_LEASE_MAX = 10
    RENEW_LEASE_ONE = 11
    RESET_BUDGET_MAX = 12
    RESET_BUDGET_ONE = 13


class GovernanceOutput(IntEnum):
    ADMIT_ACTION = 0
    REFUSE_NO_AUTHORITY = 1
    REFUSE_EXPIRED = 2
    REFUSE_BUDGET = 3
    REFUSE_IN_FLIGHT = 4
    REFUSE_AMBIGUOUS = 5
    REFUSE_INVALID_STATE = 6
    NO_ACTION = 7


AUTHORITY_NAMES = {int(x): x.name for x in Authority}
OCCURRENCE_NAMES = {int(x): x.name for x in Occurrence}
SETTLEMENT_NAMES = {int(x): x.name for x in Settlement}
EVENT_NAMES = {int(x): x.name for x in GovernanceEvent}
OUTPUT_NAMES = {int(x): x.name for x in GovernanceOutput}
EVENTS = tuple(GovernanceEvent)
OUTPUTS = tuple(GovernanceOutput)


@dataclass(frozen=True, order=True)
class GovernanceState:
    authority: int
    lease_remaining: int
    action_budget: int
    occurrence: int
    settlement: int

    def __post_init__(self) -> None:
        Authority(int(self.authority))
        if not (0 <= int(self.lease_remaining) <= 0xFFFF):
            raise ValueError("lease_remaining must be uint16")
        if not (0 <= int(self.action_budget) <= 0xFF):
            raise ValueError("action_budget must be uint8")
        Occurrence(int(self.occurrence))
        Settlement(int(self.settlement))

    def to_json(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "GovernanceState":
        return cls(
            int(data["authority"]),
            int(data["lease_remaining"]),
            int(data["action_budget"]),
            int(data["occurrence"]),
            int(data["settlement"]),
        )


@dataclass(frozen=True)
class TransitionResult:
    next_state: GovernanceState
    output: int

    def to_json(self) -> dict[str, Any]:
        return {"next_state": self.next_state.to_json(), "output": int(self.output)}


def initial_state() -> GovernanceState:
    return GovernanceState(
        int(Authority.INVALID),
        0,
        0,
        int(Occurrence.IDLE),
        int(Settlement.NONE),
    )


def configured_state(lease: int = 65535, budget: int = 255, authority: Authority = Authority.VALID) -> GovernanceState:
    return GovernanceState(int(authority), int(lease), int(budget), int(Occurrence.IDLE), int(Settlement.NONE))


def int_to_bits(value: int, width: int) -> list[int]:
    if not (0 <= int(value) < (1 << width)):
        raise ValueError(f"value {value} does not fit in {width} bits")
    return [(int(value) >> i) & 1 for i in range(width)]


def bits_to_int(bits: Sequence[int]) -> int:
    out = 0
    for i, bit in enumerate(bits):
        out |= (int(bit) & 1) << i
    return out


def encode_state_bits(state: GovernanceState, device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> Tensor:
    bits: list[int] = []
    bits.append(int(state.authority))
    bits.extend(int_to_bits(state.lease_remaining, LEASE_BITS))
    bits.extend(int_to_bits(state.action_budget, BUDGET_BITS))
    bits.extend(int_to_bits(state.occurrence, OCCURRENCE_BITS))
    bits.extend(int_to_bits(state.settlement, SETTLEMENT_BITS))
    return torch.tensor(bits, dtype=dtype, device=device)


def encode_states_bits(states: Sequence[GovernanceState], device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> Tensor:
    return torch.stack([encode_state_bits(state, device=device, dtype=dtype) for state in states], dim=0)


def decode_state_bits(bits: Tensor) -> GovernanceState:
    values = [int(round(float(x))) for x in bits.detach().cpu().reshape(-1).tolist()]
    if len(values) != STATE_WIDTH:
        raise ValueError(f"expected {STATE_WIDTH} bits, got {len(values)}")
    pos = 0
    authority = values[pos]
    pos += 1
    lease = bits_to_int(values[pos : pos + LEASE_BITS])
    pos += LEASE_BITS
    budget = bits_to_int(values[pos : pos + BUDGET_BITS])
    pos += BUDGET_BITS
    occurrence = bits_to_int(values[pos : pos + OCCURRENCE_BITS])
    pos += OCCURRENCE_BITS
    settlement = bits_to_int(values[pos : pos + SETTLEMENT_BITS])
    return GovernanceState(authority, lease, budget, occurrence, settlement)


def decode_states_bits(bits: Tensor) -> list[GovernanceState]:
    if bits.ndim == 1:
        return [decode_state_bits(bits)]
    return [decode_state_bits(row) for row in bits]


def state_to_int_tuple(state: GovernanceState) -> tuple[int, int, int, int, int]:
    return (state.authority, state.lease_remaining, state.action_budget, state.occurrence, state.settlement)


def event_tensor(events: Iterable[int | GovernanceEvent], device: torch.device | None = None) -> Tensor:
    return torch.tensor([int(GovernanceEvent(int(e))) for e in events], dtype=torch.long, device=device)


def state_field_names() -> list[str]:
    return ["authority", "lease_remaining", "action_budget", "occurrence", "settlement"]


def max_state_space_size() -> int:
    return 2 * (1 << LEASE_BITS) * (1 << BUDGET_BITS) * len(Occurrence) * len(Settlement)
