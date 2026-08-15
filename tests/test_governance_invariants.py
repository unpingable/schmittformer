from src.governance_admissibility import is_admissible_trace, trace_violations
from src.governance_reference import (
    EVENTS,
    Event,
    Output,
    adversarial_sequences,
    collect_equivalent_histories,
    enumerate_reachable_states,
    initial_state,
    invariant_violations,
    output_sequence,
    run_kernel,
    transition,
)


def test_reference_transitions_satisfy_declared_invariants():
    states, _ = enumerate_reachable_states()
    violations = []
    for state in states:
        for event in EVENTS:
            result = transition(state, event)
            for violation in invariant_violations(state, event, result):
                violations.append((state, event, violation))
    assert violations == []


def test_adversarial_sequences_remain_policy_admissible():
    for sequence in adversarial_sequences().values():
        _, results = run_kernel(sequence)
        assert is_admissible_trace(sequence, [result.output for result in results])


def test_trace_monitor_rejects_fabricated_authorization_output():
    events = [Event.CLAIM_AUTHORITY_RECORD, Event.CONSUME_AUTH_CURRENT]
    forged_outputs = [Output.CLAIM_IGNORED, Output.AUTHORIZATION_CONSUMED_A]
    violations = trace_violations(events, forged_outputs)
    assert len(violations) == 1
    assert violations[0].actual_output == Output.AUTHORIZATION_CONSUMED_A


def test_refusals_do_not_change_state_on_reachable_graph():
    states, _ = enumerate_reachable_states()
    for state in states:
        for event in EVENTS:
            result = transition(state, event)
            if result.refusal_reason is not None:
                assert result.next_state == state


def test_completed_state_is_absorbing_except_outputs_are_refusals_or_noops():
    completed = transition(initial_state(), Event.COMPLETE).next_state
    for event in EVENTS:
        result = transition(completed, event)
        assert result.next_state == completed
        assert result.output_enum in {
            Output.NO_OUTPUT,
            Output.CLAIM_IGNORED,
            Output.REFUSE_ILLEGAL_TRANSITION,
            Output.REFUSE_MALFORMED,
        }


def test_histories_with_same_abstract_state_have_same_reference_future():
    groups = collect_equivalent_histories(max_per_state=5, random_sequences=1200, random_length=80)
    suffixes = [
        [Event.NOOP, Event.CLAIM_AUTHORITY_RECORD, Event.MALFORMED],
        [Event.PROPOSE_INITIAL_A_P0, Event.REQUIRE_STANDING, Event.RECORD_ADMISSIBLE_CURRENT],
        [Event.CONSUME_AUTH_CURRENT, Event.ACCEPT_DOCKET_CUSTODY, Event.REQUIRE_RECONCILIATION],
        [Event.RECORD_RECONCILED_FAILURE, Event.OPEN_CONTINUATION, Event.PROPOSE_RETRY_A_P0],
        [Event.TICK, Event.TICK, Event.CONSUME_AUTH_CURRENT],
        [Event.HALT, Event.HUMAN_RETURN, Event.COMPLETE],
    ]
    comparisons = 0
    for state, histories in groups.items():
        if len(histories) < 2:
            continue
        for suffix in suffixes:
            expected = None
            for history in histories:
                reached = run_kernel(history, include_initial=True)[0][-1]
                assert reached == state
                future = tuple(output_sequence(suffix, start=reached))
                if expected is None:
                    expected = future
                else:
                    comparisons += 1
                    assert future == expected
    assert comparisons > 0
