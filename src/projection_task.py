from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any, Sequence

import torch
from torch import Tensor


class Proposal(IntEnum):
    NOOP = 0
    REMEDIATE_A = 1
    REMEDIATE_B = 2


class Witness(IntEnum):
    INVALID = 0
    VALID = 1


class Scope(IntEnum):
    A = 0
    B = 1


class Nuisance(IntEnum):
    ZERO = 0
    ONE = 1


class Decision(IntEnum):
    ADMIT_A = 0
    ADMIT_B = 1
    REFUSE_NO_PROPOSAL = 2
    REFUSE_INVALID_WITNESS = 3
    REFUSE_SCOPE = 4
    REFUSE_INSUFFICIENT_INFORMATION = 5


class Token(IntEnum):
    BOS = 0
    FILLER = 1
    WITNESS_INVALID = 2
    WITNESS_VALID = 3
    SCOPE_A = 4
    SCOPE_B = 5
    NUISANCE_ZERO = 6
    NUISANCE_ONE = 7
    PROPOSE_NOOP = 8
    PROPOSE_REMEDIATE_A = 9
    PROPOSE_REMEDIATE_B = 10


VOCAB_SIZE = len(Token)

PROPOSAL_NAMES = {value: value.name for value in Proposal}
WITNESS_NAMES = {value: value.name for value in Witness}
SCOPE_NAMES = {value: value.name for value in Scope}
NUISANCE_NAMES = {value: value.name for value in Nuisance}
DECISION_NAMES = {value: value.name for value in Decision}
TOKEN_NAMES = {value: value.name for value in Token}


@dataclass(frozen=True, order=True)
class PolicyState:
    proposal: int
    witness: int
    scope: int
    nuisance: int

    def __post_init__(self) -> None:
        Proposal(int(self.proposal))
        Witness(int(self.witness))
        Scope(int(self.scope))
        Nuisance(int(self.nuisance))

    def to_json(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class ProjectionTaskConfig:
    seq_len: int = 64
    p_valid: float = 0.5
    p_scope_a: float = 0.5
    p_noop: float = 0.10
    p_remediate_a: float = 0.45
    nuisance_corr: float = 0.95
    nuisance_events: int = 8

    def __post_init__(self) -> None:
        if self.seq_len < 8:
            raise ValueError("seq_len must be at least 8")
        if not (0.0 <= self.p_valid <= 1.0 and 0.0 <= self.p_scope_a <= 1.0):
            raise ValueError("probabilities must be in [0, 1]")
        if not (0.0 <= self.p_noop <= 1.0 and 0.0 <= self.p_remediate_a <= 1.0):
            raise ValueError("proposal probabilities must be in [0, 1]")
        if self.p_noop + self.p_remediate_a > 1.0:
            raise ValueError("proposal probabilities exceed 1")
        if not (0.0 <= self.nuisance_corr <= 1.0):
            raise ValueError("nuisance_corr must be in [0, 1]")
        if self.nuisance_events < 1:
            raise ValueError("nuisance_events must be positive")

    @property
    def p_remediate_b(self) -> float:
        return 1.0 - self.p_noop - self.p_remediate_a


@dataclass
class ProjectionBatch:
    tokens: Tensor
    proposal: Tensor
    witness: Tensor
    scope: Tensor
    nuisance: Tensor
    decision: Tensor

    def to(self, device: torch.device) -> "ProjectionBatch":
        return ProjectionBatch(
            tokens=self.tokens.to(device),
            proposal=self.proposal.to(device),
            witness=self.witness.to(device),
            scope=self.scope.to(device),
            nuisance=self.nuisance.to(device),
            decision=self.decision.to(device),
        )


def is_admit(decision: int | Decision) -> bool:
    return Decision(int(decision)) in (Decision.ADMIT_A, Decision.ADMIT_B)


def policy_decision(
    proposal: int | Proposal,
    witness: int | Witness,
    scope: int | Scope,
) -> Decision:
    proposal = Proposal(int(proposal))
    witness = Witness(int(witness))
    scope = Scope(int(scope))
    if proposal == Proposal.NOOP:
        return Decision.REFUSE_NO_PROPOSAL
    if witness != Witness.VALID:
        return Decision.REFUSE_INVALID_WITNESS
    if proposal == Proposal.REMEDIATE_A:
        return Decision.ADMIT_A if scope == Scope.A else Decision.REFUSE_SCOPE
    return Decision.ADMIT_B if scope == Scope.B else Decision.REFUSE_SCOPE


def decisions_from_tensors(proposal: Tensor, witness: Tensor, scope: Tensor) -> Tensor:
    out = torch.full_like(proposal.to(torch.long), int(Decision.REFUSE_SCOPE))
    proposal = proposal.to(torch.long)
    witness = witness.to(torch.long)
    scope = scope.to(torch.long)
    out = torch.where(proposal == int(Proposal.NOOP), int(Decision.REFUSE_NO_PROPOSAL), out)
    out = torch.where(
        (proposal != int(Proposal.NOOP)) & (witness != int(Witness.VALID)),
        int(Decision.REFUSE_INVALID_WITNESS),
        out,
    )
    out = torch.where(
        (proposal == int(Proposal.REMEDIATE_A)) & (witness == int(Witness.VALID)) & (scope == int(Scope.A)),
        int(Decision.ADMIT_A),
        out,
    )
    out = torch.where(
        (proposal == int(Proposal.REMEDIATE_B)) & (witness == int(Witness.VALID)) & (scope == int(Scope.B)),
        int(Decision.ADMIT_B),
        out,
    )
    return out.to(torch.long)


def all_policy_states() -> list[PolicyState]:
    return [
        PolicyState(int(proposal), int(witness), int(scope), int(nuisance))
        for proposal in Proposal
        for witness in Witness
        for scope in Scope
        for nuisance in Nuisance
    ]


def state_probability(state: PolicyState, config: ProjectionTaskConfig) -> float:
    if Proposal(state.proposal) == Proposal.NOOP:
        p_proposal = config.p_noop
    elif Proposal(state.proposal) == Proposal.REMEDIATE_A:
        p_proposal = config.p_remediate_a
    else:
        p_proposal = config.p_remediate_b

    p_witness = config.p_valid if Witness(state.witness) == Witness.VALID else 1.0 - config.p_valid
    p_scope = config.p_scope_a if Scope(state.scope) == Scope.A else 1.0 - config.p_scope_a
    expected_nuisance = int(Nuisance.ONE if Witness(state.witness) == Witness.VALID else Nuisance.ZERO)
    p_nuisance = config.nuisance_corr if state.nuisance == expected_nuisance else 1.0 - config.nuisance_corr
    return p_proposal * p_witness * p_scope * p_nuisance


def token_for_proposal(proposal: Tensor) -> Tensor:
    proposal = proposal.to(torch.long)
    return torch.where(
        proposal == int(Proposal.NOOP),
        torch.full_like(proposal, int(Token.PROPOSE_NOOP)),
        torch.where(
            proposal == int(Proposal.REMEDIATE_A),
            torch.full_like(proposal, int(Token.PROPOSE_REMEDIATE_A)),
            torch.full_like(proposal, int(Token.PROPOSE_REMEDIATE_B)),
        ),
    )


def tokens_from_latents(
    proposal: Tensor,
    witness: Tensor,
    scope: Tensor,
    nuisance: Tensor,
    seq_len: int,
) -> Tensor:
    batch_size = proposal.shape[0]
    device = proposal.device
    tokens = torch.full((batch_size, seq_len), int(Token.FILLER), dtype=torch.long, device=device)
    tokens[:, 0] = int(Token.BOS)
    witness_pos = max(1, seq_len // 8)
    scope_pos = max(witness_pos + 1, seq_len // 3)
    tokens[:, witness_pos] = torch.where(
        witness.to(torch.long) == int(Witness.VALID),
        int(Token.WITNESS_VALID),
        int(Token.WITNESS_INVALID),
    )
    tokens[:, scope_pos] = torch.where(
        scope.to(torch.long) == int(Scope.A),
        int(Token.SCOPE_A),
        int(Token.SCOPE_B),
    )
    start = min(scope_pos + 1, seq_len - 2)
    nuisance_positions = torch.linspace(start, seq_len - 2, steps=max(1, min(8, seq_len - start - 1))).round().to(torch.long)
    nuisance_token = torch.where(
        nuisance.to(torch.long) == int(Nuisance.ONE),
        int(Token.NUISANCE_ONE),
        int(Token.NUISANCE_ZERO),
    )
    for pos in nuisance_positions.tolist():
        if 0 < pos < seq_len - 1:
            tokens[:, pos] = nuisance_token
    tokens[:, -1] = token_for_proposal(proposal)
    return tokens


def sample_policy_batch(
    batch_size: int,
    config: ProjectionTaskConfig,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> ProjectionBatch:
    proposal_draw = torch.rand(batch_size, device=device, generator=generator)
    proposal = torch.where(
        proposal_draw < config.p_noop,
        torch.full((batch_size,), int(Proposal.NOOP), dtype=torch.long, device=device),
        torch.where(
            proposal_draw < config.p_noop + config.p_remediate_a,
            torch.full((batch_size,), int(Proposal.REMEDIATE_A), dtype=torch.long, device=device),
            torch.full((batch_size,), int(Proposal.REMEDIATE_B), dtype=torch.long, device=device),
        ),
    )
    witness = (torch.rand(batch_size, device=device, generator=generator) < config.p_valid).to(torch.long)
    scope = torch.where(
        torch.rand(batch_size, device=device, generator=generator) < config.p_scope_a,
        torch.full((batch_size,), int(Scope.A), dtype=torch.long, device=device),
        torch.full((batch_size,), int(Scope.B), dtype=torch.long, device=device),
    )
    expected_nuisance = torch.where(witness == int(Witness.VALID), int(Nuisance.ONE), int(Nuisance.ZERO))
    flip = torch.rand(batch_size, device=device, generator=generator) >= config.nuisance_corr
    nuisance = torch.where(flip, 1 - expected_nuisance, expected_nuisance).to(torch.long)
    decision = decisions_from_tensors(proposal, witness, scope)
    tokens = tokens_from_latents(proposal, witness, scope, nuisance, config.seq_len)
    return ProjectionBatch(tokens, proposal, witness, scope, nuisance, decision)


def batch_from_states(
    states: Sequence[PolicyState],
    seq_len: int,
    device: torch.device | None = None,
) -> ProjectionBatch:
    device = device or torch.device("cpu")
    proposal = torch.tensor([state.proposal for state in states], dtype=torch.long, device=device)
    witness = torch.tensor([state.witness for state in states], dtype=torch.long, device=device)
    scope = torch.tensor([state.scope for state in states], dtype=torch.long, device=device)
    nuisance = torch.tensor([state.nuisance for state in states], dtype=torch.long, device=device)
    decision = decisions_from_tensors(proposal, witness, scope)
    tokens = tokens_from_latents(proposal, witness, scope, nuisance, seq_len)
    return ProjectionBatch(tokens, proposal, witness, scope, nuisance, decision)


def tensor_to_list(tensor: Tensor) -> list[int]:
    return [int(x) for x in tensor.detach().cpu().reshape(-1).tolist()]


def batch_summary(batch: ProjectionBatch) -> dict[str, Any]:
    return {
        "tokens_shape": list(batch.tokens.shape),
        "proposal": tensor_to_list(batch.proposal),
        "witness": tensor_to_list(batch.witness),
        "scope": tensor_to_list(batch.scope),
        "nuisance": tensor_to_list(batch.nuisance),
        "decision": tensor_to_list(batch.decision),
    }
