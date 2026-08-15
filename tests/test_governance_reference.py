from src.governance_reference import (
    Event,
    Output,
    ProgramCounter,
    RefusalReason,
    initial_state,
    run_kernel,
    transition,
)


def outputs(events):
    return [result.output_enum for result in run_kernel(events)[1]]


def test_normal_path_consumes_authority_once_and_settles():
    events = [
        Event.PROPOSE_INITIAL_A_P0,
        Event.REQUIRE_STANDING,
        Event.RECORD_ADMISSIBLE_CURRENT,
        Event.CONSUME_AUTH_CURRENT,
        Event.ACCEPT_DOCKET_CUSTODY,
        Event.RECORD_SETTLEMENT_SUCCESS,
    ]
    states, results = run_kernel(events)
    assert states[-1].pc_enum == ProgramCounter.SETTLED_OBSERVATION_REQUIRED
    assert states[-1].settlement_outcome != 0
    assert results[3].admitted_action is not None
    assert outputs(events) == [
        Output.PROPOSAL_RECORDED,
        Output.STANDING_REQUIRED,
        Output.ADMISSIBLE_RECORDED,
        Output.AUTHORIZATION_CONSUMED_A,
        Output.DOCKET_CUSTODY_ACCEPTED,
        Output.SETTLED_SUCCESS,
    ]

    second_burn = transition(states[3], Event.CONSUME_AUTH_CURRENT)
    assert second_burn.refusal_reason == RefusalReason.ILLEGAL_TRANSITION


def test_claim_authority_record_is_not_runtime_authority():
    states, results = run_kernel([
        Event.CLAIM_AUTHORITY_RECORD,
        Event.CONSUME_AUTH_CURRENT,
        Event.ACCEPT_DOCKET_CUSTODY,
    ])
    assert states[-1] == initial_state()
    assert [r.output_enum for r in results] == [
        Output.CLAIM_IGNORED,
        Output.REFUSE_ILLEGAL_TRANSITION,
        Output.REFUSE_ILLEGAL_TRANSITION,
    ]


def test_standing_expiry_prevents_authority_burn():
    states, results = run_kernel([
        Event.PROPOSE_INITIAL_A_P0,
        Event.REQUIRE_STANDING,
        Event.RECORD_ADMISSIBLE_CURRENT,
        Event.TICK,
        Event.TICK,
        Event.CONSUME_AUTH_CURRENT,
    ])
    assert states[4].pc_enum == ProgramCounter.STANDING_REQUIRED
    assert results[-1].refusal_reason == RefusalReason.ILLEGAL_TRANSITION
    assert all(result.admitted_action is None for result in results[4:])


def test_indeterminate_attempt_requires_reconciliation_before_continuation():
    states, results = run_kernel([
        Event.PROPOSE_INITIAL_A_P0,
        Event.REQUIRE_STANDING,
        Event.RECORD_ADMISSIBLE_CURRENT,
        Event.CONSUME_AUTH_CURRENT,
        Event.ACCEPT_DOCKET_CUSTODY,
        Event.REQUIRE_RECONCILIATION,
        Event.OPEN_CONTINUATION,
        Event.ACCEPT_DOCKET_CUSTODY,
        Event.RECORD_RECONCILED_FAILURE,
        Event.OPEN_CONTINUATION,
    ])
    assert states[5].pc_enum == ProgramCounter.RECONCILIATION_REQUIRED
    assert results[6].refusal_reason == RefusalReason.ILLEGAL_TRANSITION
    assert results[7].refusal_reason == RefusalReason.ILLEGAL_TRANSITION
    assert states[-1].pc_enum == ProgramCounter.OBSERVATION_REQUIRED
    assert states[-1].has_prior == 1


def test_retry_requires_same_proposal_and_same_preconditions():
    events = [
        Event.PROPOSE_INITIAL_A_P0,
        Event.REQUIRE_STANDING,
        Event.RECORD_ADMISSIBLE_CURRENT,
        Event.CONSUME_AUTH_CURRENT,
        Event.ACCEPT_DOCKET_CUSTODY,
        Event.RECORD_SETTLEMENT_FAILURE,
        Event.OPEN_CONTINUATION,
    ]
    states, _ = run_kernel(events)
    retry_state = states[-1]
    changed = transition(retry_state, Event.PROPOSE_RETRY_A_P1)
    assert changed.refusal_reason == RefusalReason.RETRY_PRECONDITIONS_CHANGED
    wrong_proposal = transition(retry_state, Event.PROPOSE_RETRY_B_P0)
    assert wrong_proposal.refusal_reason == RefusalReason.RETRY_PROPOSAL_CHANGED
    ok = transition(retry_state, Event.PROPOSE_RETRY_A_P0)
    assert ok.next_state.pc_enum == ProgramCounter.PROPOSAL_RECORDED
    assert ok.next_state.retries_used == 1


def test_successor_cannot_reuse_prior_proposal():
    states, _ = run_kernel([
        Event.PROPOSE_INITIAL_A_P0,
        Event.REQUIRE_STANDING,
        Event.RECORD_ADMISSIBLE_CURRENT,
        Event.CONSUME_AUTH_CURRENT,
        Event.ACCEPT_DOCKET_CUSTODY,
        Event.RECORD_SETTLEMENT_SUCCESS,
        Event.OPEN_CONTINUATION,
    ])
    successor_state = states[-1]
    reused = transition(successor_state, Event.PROPOSE_SUCCESSOR_A_P0)
    assert reused.refusal_reason == RefusalReason.SUCCESSOR_PROPOSAL_REUSED
    fresh = transition(successor_state, Event.PROPOSE_SUCCESSOR_B_P0)
    assert fresh.next_state.pc_enum == ProgramCounter.PROPOSAL_RECORDED


def test_completed_is_terminal_and_halted_is_effect_free():
    completed = transition(initial_state(), Event.COMPLETE).next_state
    assert completed.pc_enum == ProgramCounter.COMPLETED
    assert transition(completed, Event.OPEN_CONTINUATION).next_state == completed
    assert transition(completed, Event.OPEN_CONTINUATION).refusal_reason == RefusalReason.ILLEGAL_TRANSITION

    halted = transition(initial_state(), Event.HALT).next_state
    assert halted.pc_enum == ProgramCounter.HALTED
    assert transition(halted, Event.REQUIRE_STANDING).refusal_reason == RefusalReason.ILLEGAL_TRANSITION
    assert transition(halted, Event.CONSUME_AUTH_CURRENT).refusal_reason == RefusalReason.ILLEGAL_TRANSITION
