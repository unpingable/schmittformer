import torch

from src.projection_task import Decision, Scope, Witness
from src.register_governance import metadata_equivalence_report
from src.semantic_register import (
    RegisterEncoding,
    corrupt_register,
    decode_register,
    register_code,
    register_policy_decision,
)


def test_register_encodings_round_trip() -> None:
    witness = torch.tensor([int(Witness.INVALID), int(Witness.VALID), int(Witness.VALID)])
    scope = torch.tensor([int(Scope.A), int(Scope.A), int(Scope.B)])
    for encoding in RegisterEncoding:
        register = register_code(encoding, witness, scope)
        decoded = decode_register(register, encoding)
        assert decoded.valid.all()
        assert decoded.witness.tolist() == witness.tolist()
        assert decoded.scope.tolist() == scope.tolist()


def test_register_governance_applies_policy() -> None:
    proposal = torch.tensor([1, 1, 2, 0])
    witness = torch.tensor([1, 1, 0, 1])
    scope = torch.tensor([0, 1, 1, 0])
    register = register_code(RegisterEncoding.GROUPED_ONE_HOT, witness, scope)
    decision, decoded = register_policy_decision(proposal, register, RegisterEncoding.GROUPED_ONE_HOT)
    assert decoded.valid.all()
    assert decision.tolist() == [
        int(Decision.ADMIT_A),
        int(Decision.REFUSE_SCOPE),
        int(Decision.REFUSE_INVALID_WITNESS),
        int(Decision.REFUSE_NO_PROPOSAL),
    ]


def test_invalid_register_refuses_when_detectable() -> None:
    proposal = torch.tensor([1, 2])
    register = torch.zeros(2, 4)
    decision, decoded = register_policy_decision(proposal, register, RegisterEncoding.GROUPED_ONE_HOT)
    assert not decoded.valid.any()
    assert decision.tolist() == [int(Decision.REFUSE_INSUFFICIENT_INFORMATION)] * 2


def test_metadata_equivalence_from_same_register() -> None:
    proposal = torch.tensor([1, 2])
    witness = torch.tensor([1, 1])
    scope = torch.tensor([0, 1])
    register = register_code(RegisterEncoding.JOINT_ONE_HOT, witness, scope)
    report = metadata_equivalence_report(proposal, register, RegisterEncoding.JOINT_ONE_HOT)
    assert report["exact_match_rate"] == 1.0
    assert report["mismatches"] == 0


def test_fault_modes_are_classified() -> None:
    witness = torch.tensor([1])
    scope = torch.tensor([0])
    register = register_code(RegisterEncoding.BINARY_PAIR, witness, scope)
    assert decode_register(corrupt_register(register, RegisterEncoding.BINARY_PAIR, "nan"), RegisterEncoding.BINARY_PAIR).valid.item() is False
    flipped = decode_register(corrupt_register(register, RegisterEncoding.BINARY_PAIR, "bit_flip_witness"), RegisterEncoding.BINARY_PAIR)
    assert flipped.valid.item() is True
    assert flipped.witness.item() == 0
