from __future__ import annotations

import hashlib
import json
from typing import Any

from .governance_reference import (
    EVENT_NAMES,
    EVENTS,
    OUTPUT_NAMES,
    PROPOSAL_NAMES,
    REFUSAL_NAMES,
    Event,
    ProposalKind,
    RefusalReason,
    state_id_maps,
    transition,
)

SEMANTIC_GRAPH_SCHEMA = "schmittformer.governance.transition_graph.v1"
CONFORMANCE_SCHEMA = "schmittformer.governance.conformance.v1"
SEMANTIC_NAME = "schmittformer.governance.ag_ng_occurrence_core"
SEMANTIC_VERSION = "0.1.0"
AG_NG_REVISION = "aab771b636d0e7f09b5e281fa2104d94dde7a595"


def compact_transition_relation() -> dict[str, Any]:
    states, state_to_id = state_id_maps()
    transitions = []
    admitted = 0
    refusals = 0
    for state in states:
        from_id = state_to_id[state]
        for event in EVENTS:
            result = transition(state, event)
            if result.admitted_action is not None:
                admitted += 1
            if result.refusal_reason is not None:
                refusals += 1
            transitions.append(
                [
                    from_id,
                    int(event),
                    state_to_id[result.next_state],
                    result.output,
                    -1 if result.refusal_reason is None else int(result.refusal_reason),
                    -1 if result.admitted_action is None else int(result.admitted_action),
                ]
            )
    return {
        "schema": SEMANTIC_GRAPH_SCHEMA,
        "semantic_name": SEMANTIC_NAME,
        "semantic_version": SEMANTIC_VERSION,
        "source_model": "finite AG-ng governed-loop occurrence abstraction",
        "ag_ng_revision": AG_NG_REVISION,
        "state_records": "reachable states only",
        "transition_records": "reachable state/event transitions only",
        "state_fields": [
            "pc",
            "proposal",
            "preconditions",
            "has_prior",
            "prior_proposal",
            "prior_preconditions",
            "retries_used",
            "probes_used",
            "escalations_used",
            "standing_lease",
            "settlement_outcome",
            "halted_unresolved_attempt",
        ],
        "events": [[int(event), EVENT_NAMES[Event(event)]] for event in EVENTS],
        "states": [state.to_json() for state in states],
        "transitions": transitions,
        "counts": {
            "reachable_states": len(states),
            "event_alphabet_size": len(EVENTS),
            "reachable_transitions": len(transitions),
            "admitted_transitions": admitted,
            "refusal_transitions": refusals,
        },
    }


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def semantic_digest(payload: dict[str, Any] | None = None) -> str:
    relation = compact_transition_relation() if payload is None else payload
    return hashlib.sha256(canonical_json_bytes(relation)).hexdigest()


def conformance_corpus() -> dict[str, Any]:
    relation = compact_transition_relation()
    digest = semantic_digest(relation)
    states = relation["states"]
    cases = []
    for index, (from_id, event_id, to_id, output_id, refusal_id, admitted_id) in enumerate(relation["transitions"]):
        cases.append(
            {
                "case_id": index,
                "initial_state_id": from_id,
                "event": EVENT_NAMES[Event(event_id)],
                "event_id": event_id,
                "expected_next_state_id": to_id,
                "expected_output": OUTPUT_NAMES[output_id],
                "expected_output_id": output_id,
                "expected_refusal": None if refusal_id < 0 else REFUSAL_NAMES[RefusalReason(refusal_id)],
                "expected_refusal_id": None if refusal_id < 0 else refusal_id,
                "expected_admitted_action": None if admitted_id < 0 else PROPOSAL_NAMES[ProposalKind(admitted_id)],
                "expected_admitted_action_id": None if admitted_id < 0 else admitted_id,
            }
        )
    return {
        "schema": CONFORMANCE_SCHEMA,
        "semantic_name": SEMANTIC_NAME,
        "semantic_version": SEMANTIC_VERSION,
        "semantic_digest_sha256": digest,
        "source_graph_schema": SEMANTIC_GRAPH_SCHEMA,
        "ag_ng_revision": AG_NG_REVISION,
        "state_records": "reachable states only; cases reference state IDs",
        "transition_records": "one case per reachable state/event transition",
        "states": states,
        "events": relation["events"],
        "cases": cases,
        "counts": relation["counts"],
    }
