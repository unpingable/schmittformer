from __future__ import annotations

import time
from enum import IntEnum
from typing import Any


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


class Occurrence(IntEnum):
    IDLE = 0
    IN_FLIGHT = 1
    AMBIGUOUS = 2


class Settlement(IntEnum):
    NONE = 0
    SUCCESS = 1
    FAILURE = 2


def run_transition_equivalence(timeout_ms: int = 30000) -> dict[str, Any]:
    try:
        import z3  # type: ignore
    except Exception as exc:
        return {
            "attempted": False,
            "available": False,
            "result": "UNAVAILABLE",
            "reason": f"z3 unavailable: {exc}",
        }

    start = time.time()
    one1 = z3.BitVecVal(1, 1)
    zero1 = z3.BitVecVal(0, 1)

    authority = z3.Bool("authority")
    lease = z3.BitVec("lease", 16)
    budget = z3.BitVec("budget", 8)
    occurrence = z3.BitVec("occurrence", 2)
    settlement = z3.BitVec("settlement", 2)
    event = z3.BitVec("event", 4)

    def ev(e: GovernanceEvent):
        return event == z3.BitVecVal(int(e), 4)

    def bv_dec_sat(x, width: int):
        return z3.If(x == z3.BitVecVal(0, width), z3.BitVecVal(0, width), x - z3.BitVecVal(1, width))

    def bv_to_bits(x, width: int):
        return [z3.Extract(i, i, x) == one1 for i in range(width)]

    def bits_to_bv(bits):
        pieces = [z3.If(bit, one1, zero1) for bit in reversed(bits)]
        return z3.Concat(*pieces)

    def eq_bits(bits, pattern: list[int]):
        terms = [bit if value else z3.Not(bit) for bit, value in zip(bits, pattern)]
        return z3.And(*terms)

    def ripple_dec(bits):
        borrow = z3.BoolVal(True)
        raw = []
        for bit in bits:
            raw_bit = z3.Xor(bit, borrow)
            raw.append(raw_bit)
            borrow = z3.And(borrow, z3.Not(bit))
        nonzero = z3.Or(*bits)
        dec = [z3.And(nonzero, bit) for bit in raw]
        return dec, z3.Not(nonzero), nonzero

    def mux(cond, t, f):
        return z3.If(cond, t, f)

    # Reference semantics: bounded bit-vector arithmetic plus source-level event cases.
    is_idle_ref = occurrence == z3.BitVecVal(int(Occurrence.IDLE), 2)
    is_inflight_ref = occurrence == z3.BitVecVal(int(Occurrence.IN_FLIGHT), 2)
    is_ambiguous_ref = occurrence == z3.BitVecVal(int(Occurrence.AMBIGUOUS), 2)
    admit_ref = z3.And(ev(GovernanceEvent.PROPOSE_ACTION), is_idle_ref, authority, lease != z3.BitVecVal(0, 16), budget != z3.BitVecVal(0, 8))

    out_ref = z3.BitVecVal(int(GovernanceOutput.NO_ACTION), 4)
    out_ref = z3.If(
        ev(GovernanceEvent.PROPOSE_ACTION),
        z3.If(
            is_ambiguous_ref,
            z3.BitVecVal(int(GovernanceOutput.REFUSE_AMBIGUOUS), 4),
            z3.If(
                is_inflight_ref,
                z3.BitVecVal(int(GovernanceOutput.REFUSE_IN_FLIGHT), 4),
                z3.If(
                    z3.Not(authority),
                    z3.BitVecVal(int(GovernanceOutput.REFUSE_NO_AUTHORITY), 4),
                    z3.If(
                        lease == z3.BitVecVal(0, 16),
                        z3.BitVecVal(int(GovernanceOutput.REFUSE_EXPIRED), 4),
                        z3.If(
                            budget == z3.BitVecVal(0, 8),
                            z3.BitVecVal(int(GovernanceOutput.REFUSE_BUDGET), 4),
                            z3.BitVecVal(int(GovernanceOutput.ADMIT_ACTION), 4),
                        ),
                    ),
                ),
            ),
        ),
        out_ref,
    )

    authority_ref = z3.If(ev(GovernanceEvent.GRANT_AUTHORITY), z3.BoolVal(True), z3.If(ev(GovernanceEvent.REVOKE_AUTHORITY), z3.BoolVal(False), authority))
    lease_ref = z3.If(
        ev(GovernanceEvent.TICK),
        bv_dec_sat(lease, 16),
        z3.If(ev(GovernanceEvent.RENEW_LEASE_MAX), z3.BitVecVal(0xFFFF, 16), z3.If(ev(GovernanceEvent.RENEW_LEASE_ONE), z3.BitVecVal(1, 16), lease)),
    )
    budget_ref = z3.If(
        ev(GovernanceEvent.RESET_BUDGET_MAX),
        z3.BitVecVal(0xFF, 8),
        z3.If(ev(GovernanceEvent.RESET_BUDGET_ONE), z3.BitVecVal(1, 8), z3.If(admit_ref, bv_dec_sat(budget, 8), budget)),
    )
    occurrence_ref = occurrence
    occurrence_ref = z3.If(admit_ref, z3.BitVecVal(int(Occurrence.IN_FLIGHT), 2), occurrence_ref)
    occurrence_ref = z3.If(z3.And(is_inflight_ref, ev(GovernanceEvent.RESULT_SUCCESS)), z3.BitVecVal(int(Occurrence.IDLE), 2), occurrence_ref)
    occurrence_ref = z3.If(z3.And(is_inflight_ref, ev(GovernanceEvent.RESULT_FAILURE)), z3.BitVecVal(int(Occurrence.IDLE), 2), occurrence_ref)
    occurrence_ref = z3.If(z3.And(is_inflight_ref, ev(GovernanceEvent.RESULT_AMBIGUOUS)), z3.BitVecVal(int(Occurrence.AMBIGUOUS), 2), occurrence_ref)
    occurrence_ref = z3.If(z3.And(is_ambiguous_ref, z3.Or(ev(GovernanceEvent.SETTLE_SUCCESS), ev(GovernanceEvent.SETTLE_FAILURE))), z3.BitVecVal(int(Occurrence.IDLE), 2), occurrence_ref)

    settlement_ref = settlement
    settlement_ref = z3.If(admit_ref, z3.BitVecVal(int(Settlement.NONE), 2), settlement_ref)
    settlement_ref = z3.If(z3.And(is_inflight_ref, ev(GovernanceEvent.RESULT_SUCCESS)), z3.BitVecVal(int(Settlement.SUCCESS), 2), settlement_ref)
    settlement_ref = z3.If(z3.And(is_inflight_ref, ev(GovernanceEvent.RESULT_FAILURE)), z3.BitVecVal(int(Settlement.FAILURE), 2), settlement_ref)
    settlement_ref = z3.If(z3.And(is_inflight_ref, ev(GovernanceEvent.RESULT_AMBIGUOUS)), z3.BitVecVal(int(Settlement.NONE), 2), settlement_ref)
    settlement_ref = z3.If(z3.And(is_ambiguous_ref, ev(GovernanceEvent.SETTLE_SUCCESS)), z3.BitVecVal(int(Settlement.SUCCESS), 2), settlement_ref)
    settlement_ref = z3.If(z3.And(is_ambiguous_ref, ev(GovernanceEvent.SETTLE_FAILURE)), z3.BitVecVal(int(Settlement.FAILURE), 2), settlement_ref)

    # Compiled logical circuit: bit-level ripple decrement and hard gate/mux schedule.
    lease_bits = bv_to_bits(lease, 16)
    budget_bits = bv_to_bits(budget, 8)
    occ_bits = bv_to_bits(occurrence, 2)
    set_bits = bv_to_bits(settlement, 2)
    lease_dec_bits, _lease_zero, lease_nonzero = ripple_dec(lease_bits)
    budget_dec_bits, _budget_zero, budget_nonzero = ripple_dec(budget_bits)

    is_idle = eq_bits(occ_bits, [0, 0])
    is_inflight = eq_bits(occ_bits, [1, 0])
    is_ambiguous = eq_bits(occ_bits, [0, 1])
    is_settle_none = eq_bits(set_bits, [0, 0])
    is_settle_success = eq_bits(set_bits, [1, 0])
    is_settle_failure = eq_bits(set_bits, [0, 1])
    valid_state = z3.And(z3.Or(is_idle, is_inflight, is_ambiguous), z3.Or(is_settle_none, is_settle_success, is_settle_failure))

    proposal = ev(GovernanceEvent.PROPOSE_ACTION)
    propose_and_valid = z3.And(proposal, valid_state)
    propose_idle = z3.And(propose_and_valid, is_idle)
    admit = z3.And(propose_idle, authority, lease_nonzero, budget_nonzero)
    refuse_ambiguous = z3.And(propose_and_valid, is_ambiguous)
    refuse_inflight = z3.And(propose_and_valid, is_inflight)
    refuse_no_authority = z3.And(propose_idle, z3.Not(authority))
    refuse_expired = z3.And(propose_idle, authority, z3.Not(lease_nonzero))
    refuse_budget = z3.And(propose_idle, authority, lease_nonzero, z3.Not(budget_nonzero))

    authority_comp = mux(ev(GovernanceEvent.GRANT_AUTHORITY), z3.BoolVal(True), authority)
    authority_comp = mux(ev(GovernanceEvent.REVOKE_AUTHORITY), z3.BoolVal(False), authority_comp)

    lease_comp_bits = lease_bits[:]
    lease_comp_bits = [mux(ev(GovernanceEvent.TICK), d, old) for d, old in zip(lease_dec_bits, lease_comp_bits)]
    lease_comp_bits = [mux(ev(GovernanceEvent.RENEW_LEASE_MAX), z3.BoolVal(True), old) for old in lease_comp_bits]
    lease_one_bits = [z3.BoolVal(i == 0) for i in range(16)]
    lease_comp_bits = [mux(ev(GovernanceEvent.RENEW_LEASE_ONE), one, old) for one, old in zip(lease_one_bits, lease_comp_bits)]

    budget_comp_bits = budget_bits[:]
    budget_comp_bits = [mux(admit, d, old) for d, old in zip(budget_dec_bits, budget_comp_bits)]
    budget_comp_bits = [mux(ev(GovernanceEvent.RESET_BUDGET_MAX), z3.BoolVal(True), old) for old in budget_comp_bits]
    budget_one_bits = [z3.BoolVal(i == 0) for i in range(8)]
    budget_comp_bits = [mux(ev(GovernanceEvent.RESET_BUDGET_ONE), one, old) for one, old in zip(budget_one_bits, budget_comp_bits)]

    in_flight_bits = [z3.BoolVal(True), z3.BoolVal(False)]
    idle_bits = [z3.BoolVal(False), z3.BoolVal(False)]
    ambiguous_bits = [z3.BoolVal(False), z3.BoolVal(True)]
    none_bits = [z3.BoolVal(False), z3.BoolVal(False)]
    success_bits = [z3.BoolVal(True), z3.BoolVal(False)]
    failure_bits = [z3.BoolVal(False), z3.BoolVal(True)]

    result_success = ev(GovernanceEvent.RESULT_SUCCESS)
    result_failure = ev(GovernanceEvent.RESULT_FAILURE)
    result_ambiguous = ev(GovernanceEvent.RESULT_AMBIGUOUS)
    settle_success = ev(GovernanceEvent.SETTLE_SUCCESS)
    settle_failure = ev(GovernanceEvent.SETTLE_FAILURE)
    any_settle = z3.Or(settle_success, settle_failure)
    success_from_inflight = z3.And(is_inflight, result_success)
    failure_from_inflight = z3.And(is_inflight, result_failure)
    ambiguous_from_inflight = z3.And(is_inflight, result_ambiguous)
    settle_from_ambiguous = z3.And(is_ambiguous, any_settle)
    settle_success_from_ambiguous = z3.And(is_ambiguous, settle_success)
    settle_failure_from_ambiguous = z3.And(is_ambiguous, settle_failure)

    occ_comp_bits = occ_bits[:]
    occ_comp_bits = [mux(admit, t, f) for t, f in zip(in_flight_bits, occ_comp_bits)]
    occ_comp_bits = [mux(success_from_inflight, t, f) for t, f in zip(idle_bits, occ_comp_bits)]
    occ_comp_bits = [mux(failure_from_inflight, t, f) for t, f in zip(idle_bits, occ_comp_bits)]
    occ_comp_bits = [mux(ambiguous_from_inflight, t, f) for t, f in zip(ambiguous_bits, occ_comp_bits)]
    occ_comp_bits = [mux(settle_from_ambiguous, t, f) for t, f in zip(idle_bits, occ_comp_bits)]

    set_comp_bits = set_bits[:]
    set_comp_bits = [mux(admit, t, f) for t, f in zip(none_bits, set_comp_bits)]
    set_comp_bits = [mux(success_from_inflight, t, f) for t, f in zip(success_bits, set_comp_bits)]
    set_comp_bits = [mux(failure_from_inflight, t, f) for t, f in zip(failure_bits, set_comp_bits)]
    set_comp_bits = [mux(ambiguous_from_inflight, t, f) for t, f in zip(none_bits, set_comp_bits)]
    set_comp_bits = [mux(settle_success_from_ambiguous, t, f) for t, f in zip(success_bits, set_comp_bits)]
    set_comp_bits = [mux(settle_failure_from_ambiguous, t, f) for t, f in zip(failure_bits, set_comp_bits)]

    output_comp = z3.BitVecVal(int(GovernanceOutput.NO_ACTION), 4)
    output_comp = z3.If(admit, z3.BitVecVal(int(GovernanceOutput.ADMIT_ACTION), 4), output_comp)
    output_comp = z3.If(refuse_budget, z3.BitVecVal(int(GovernanceOutput.REFUSE_BUDGET), 4), output_comp)
    output_comp = z3.If(refuse_expired, z3.BitVecVal(int(GovernanceOutput.REFUSE_EXPIRED), 4), output_comp)
    output_comp = z3.If(refuse_no_authority, z3.BitVecVal(int(GovernanceOutput.REFUSE_NO_AUTHORITY), 4), output_comp)
    output_comp = z3.If(refuse_inflight, z3.BitVecVal(int(GovernanceOutput.REFUSE_IN_FLIGHT), 4), output_comp)
    output_comp = z3.If(refuse_ambiguous, z3.BitVecVal(int(GovernanceOutput.REFUSE_AMBIGUOUS), 4), output_comp)

    lease_comp = bits_to_bv(lease_comp_bits)
    budget_comp = bits_to_bv(budget_comp_bits)
    occurrence_comp = bits_to_bv(occ_comp_bits)
    settlement_comp = bits_to_bv(set_comp_bits)

    valid_constraints = z3.And(
        z3.ULE(event, z3.BitVecVal(13, 4)),
        occurrence != z3.BitVecVal(3, 2),
        settlement != z3.BitVecVal(3, 2),
    )
    differs = z3.Or(
        authority_ref != authority_comp,
        lease_ref != lease_comp,
        budget_ref != budget_comp,
        occurrence_ref != occurrence_comp,
        settlement_ref != settlement_comp,
        out_ref != output_comp,
    )
    solver = z3.Solver()
    solver.set(timeout=timeout_ms)
    solver.add(valid_constraints)
    solver.add(differs)
    check = solver.check()
    elapsed = time.time() - start
    payload: dict[str, Any] = {
        "attempted": True,
        "available": True,
        "z3_version": z3.get_version_string(),
        "timeout_ms": timeout_ms,
        "runtime_seconds": elapsed,
        "query": "exists valid bounded state/event where reference transition differs from compiled ripple/Boolean transition",
        "valid_domain": "authority bool, occurrence in {0,1,2}, settlement in {0,1,2}, event in 0..13; invalid enum states excluded",
        "result": str(check).upper(),
    }
    if check == z3.sat:
        model = solver.model()
        payload["counterexample"] = {
            "authority": bool(z3.is_true(model.eval(authority, model_completion=True))),
            "lease": int(model.eval(lease, model_completion=True).as_long()),
            "budget": int(model.eval(budget, model_completion=True).as_long()),
            "occurrence": int(model.eval(occurrence, model_completion=True).as_long()),
            "settlement": int(model.eval(settlement, model_completion=True).as_long()),
            "event": int(model.eval(event, model_completion=True).as_long()),
        }
    return payload
