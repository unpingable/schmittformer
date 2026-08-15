from src.governance_conformance import conformance_corpus, compact_transition_relation, semantic_digest
from src.governance_reference import Event, state_id_maps, transition

EXPECTED_SEMANTIC_DIGEST = "1926060d9a20ff1f19d4e67d31c0c0cf4725a11b1a3f401419fd6c033333cf8c"


def test_governance_semantic_digest_is_stable_for_current_kernel():
    assert semantic_digest() == EXPECTED_SEMANTIC_DIGEST


def test_compact_transition_relation_counts():
    relation = compact_transition_relation()
    assert relation["counts"] == {
        "reachable_states": 912,
        "event_alphabet_size": 39,
        "reachable_transitions": 35568,
        "admitted_transitions": 144,
        "refusal_transitions": 30724,
    }


def test_conformance_corpus_replays_reference_transitions():
    corpus = conformance_corpus()
    states, _ = state_id_maps()
    assert corpus["semantic_digest_sha256"] == EXPECTED_SEMANTIC_DIGEST
    assert len(corpus["cases"]) == 35568
    for case in corpus["cases"]:
        state = states[case["initial_state_id"]]
        result = transition(state, Event(case["event_id"]))
        assert states[case["expected_next_state_id"]] == result.next_state
        assert case["expected_output_id"] == result.output
        assert case["expected_refusal_id"] == result.refusal_reason
        assert case["expected_admitted_action_id"] == result.admitted_action
