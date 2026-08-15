# Governance Semantic-Core Checkpoint

This checkpoint stops before substantial transformer-backend work. It extracts an implementation-neutral governance semantics from AG-ng and verifies its finite transition graph.

## Reproduce

```bash
.venv/bin/python experiments/run_governance.py --out-dir results
.venv/bin/python -m pytest tests/test_governance_reference.py tests/test_governance_invariants.py tests/test_governance_reachable.py -q
.venv/bin/python -m pytest -q
```

The checkpoint run used Python 3.12.3 and `torch==2.13.0+cpu`. CUDA was not visible to the installed PyTorch wheel. No training or GPU compute is involved in this semantic-core pass.

Source provenance:

- AG-ng source: `/home/jbeck/ag_ng` at `aab771b636d0e7f09b5e281fa2104d94dde7a595`, clean.
- schmittformer base: `9710f2b4e92fc97624e9829a7e3388c06bad7a5a` (`softmax-bounded-margin-success`) plus this uncommitted governance checkpoint.
- Historical hysteresis/circuit/softmax results are unchanged.

## Files Added

- `GOVERNANCE_SOURCE.md`
- `GOVERNANCE_RESULTS.md`
- `src/governance_reference.py`
- `src/governance_admissibility.py`
- `experiments/run_governance.py`
- `tests/test_governance_reference.py`
- `tests/test_governance_invariants.py`
- `tests/test_governance_reachable.py`
- `results/governance_states.json`
- `results/governance_transitions.json`
- `results/governance_summary.json`

A pre-addendum `src/governance_compiled.py` exists but is not part of this checkpoint. It contains stale event names in helper scenarios and should be treated as paused transformer-backend work, not as a validated implementation.

## Extracted Kernel

The reference kernel is a deterministic finite state machine:

```python
transition(state, event) -> TransitionResult
```

`TransitionResult` records:

```text
next_state
output
refusal_reason | None
admitted_action | None
```

The complete state is hashable/serializable and includes the program counter, current proposal/precondition basis, prior occurrence basis, retry/probe/escalation counters, a tiny standing lease, settlement outcome, and unresolved-attempt halt flag.

This is deliberately independent of attention, tokens, positional encodings, dtype, logits, or neural representation choices.

## Results

Machine-readable outputs:

- `results/governance_states.json`
- `results/governance_transitions.json`
- `results/governance_summary.json`

Finite graph:

| measure | value |
| --- | ---: |
| normalized syntactic states | 3,840 |
| reachable states | 912 |
| event alphabet size | 39 |
| reachable transitions | 35,568 |
| admitted action transitions | 144 |
| refusal transitions | 30,724 |

Checks:

| check | value |
| --- | ---: |
| invariant transition checks | 35,568 |
| invariant violations | 0 |
| adversarial trace sets | 11 |
| adversarial admissibility violations | 0 |
| history-equivalence groups | 912 |
| sampled history-equivalence comparisons | 21,888 |
| history-equivalence violations | 0 |
| full pytest suite | 55 passed, 1 warning |

## DERIVED / PROVED UNDER STATED ASSUMPTIONS

Given the Python reference transition function as the oracle, the admissible trace checker is exact for this finite model: it replays the same transition function and flags any observable output that differs from the policy language.

Given a complete `GovernanceState`, future reference behavior is a pure function of that state and the suffix events. Therefore histories that reach the same complete state are definitionally equivalent for the reference backend. The tests exercise that quotient with many non-canonical histories, but the stronger statement follows from the deterministic API shape.

## EXHAUSTIVELY VERIFIED

Every reachable state/event pair in the finite model was enumerated and checked against the declared invariants:

- no action without admitted pending authorization and live standing lease
- one-use burn only
- claim/proposal records cannot create authority
- malformed/invalid events produce explicit refusal
- refusal leaves state unchanged
- ambiguous dispatched attempts require reconciliation before continuation
- no blind retry while reconciliation is unresolved
- unresolved halted attempts block human return/termination
- budgets never exceed their finite limits
- completed state is absorbing

The normalized syntactic state count is also saved in JSON; the transition table covers reachable states only.

## EMPIRICALLY OBSERVED

The adversarial trace set includes authority-claim spam, action attempts without authority, standing expiry before burn, one-use burn attempts, blind retry after ambiguity, retry-precondition changes, successor proposal reuse, probe budget exhaustion, halted unresolved human return, and long malformed/no-op/claim spam. All produced policy-admissible output traces under the reference checker.

The sampled history-equivalence test generated multiple histories for every reachable abstract state and compared common suffix behavior. It found zero violations.

## SUPPORTED BY SOURCE FORMALISM

AG-ng's formal-calculus crosswalk supports the high-level separation between formal evidence, adapter evidence, runtime evidence, and authority, and it emphasizes typed refusals and operational indeterminacy as distinct outcomes. It is pinned to Lean revision `ff491b808ebeab2a132d9ade46d234cf85dcfbe9`.

This checkpoint does not claim a Lean theorem for the Python model. The correspondence is approximate and documented in `GOVERNANCE_SOURCE.md`.

## NOT ESTABLISHED

- No transformer governance backend has been validated in this checkpoint.
- No finite-softmax margins or neural representation-preservation argument have been derived for governance composition.
- No Rust/AG-ng conformance theorem has been proven.
- The model omits AG-ng production concerns: cryptographic digests, JCS canonicalization, principals, signatures, store atomicity, concurrency, crash recovery, exact Docket envelopes, residual sets, and real resolver calls.
- The finite reduction is useful for semantics testing, but it is not a production authority boundary.

## Constellation Usefulness Review

If transformer research stopped here, the semantic kernel could still be useful as shared infrastructure.

Concrete potential consumers visible from AG-ng:

- `ag-campaign` and `ag-store`: conformance vectors for the governed-loop transition law and refusal vocabulary.
- `ag-app` / `CampaignEngineV1`: a small independent oracle for admission/settlement lifecycle tests.
- Docket-facing issuance/custody work: a trace-language checker that distinguishes admitted issuance, refusal, and reconciliation-required behavior.
- Effectd/worker/campaign-driver tests: a governance-relative monitor that can reject impossible action traces without executing effects.
- Audit/receipt tooling: compact finite examples for typed refusal, no-blind-retry, one-use burn, and terminal-state behavior.

The useful shared artifact is the semantics, not the transformer. A production backend should probably be a boring deterministic monitor or Rust library unless a specific deployment boundary requires something else.

## Backend Separation

REFERENCE: implemented here as `src/governance_reference.py` plus `src/governance_admissibility.py`.

FORMAL: not implemented. AG-ng has a formal-calculus crosswalk, but no direct local Lean source in this checkout and no proof connecting this Python finite model to Lean.

TRANSFORMER: paused. Existing historical transformer successes remain valid for hysteresis/circuit breaker/softmax. Governance transformer compilation should wait until the semantic kernel is reviewed.

## Next Steps

CONSTELLATION TRACK:

1. Compare this finite model against AG-ng Rust tests and decide whether it should become a conformance-vector generator rather than an independent doctrine source.
2. Factor the stable parts into a tiny semantic IR only if another backend needs it.
3. Consider a deterministic production/reference monitor first. Rust is the obvious eventual runtime target; Python is sufficient for this research checkpoint.

RESEARCH TRACK:

1. Projection-loss experiment: test whether policy-relevant latent state is erased at the proposal-token boundary. This most directly addresses whether a downstream compiled kernel merely duplicates a reference monitor.
2. Fixed-width recurrent governance state: avoid long history retrieval and test whether the structural state abstraction survives without explicit full-history state tokens.
3. Synthesized latent/pre-serialization gate: more interesting than a downstream token monitor, but harder to make falsifiable.
4. Counter/lease representation without enumerating every concrete value: useful if the finite kernel becomes too table-shaped.
5. Full token-level governance transformer compilation: lower priority for the next research pass, because it risks demonstrating only that the already-built reference monitor can be serialized into transformer-shaped lookup machinery.

## Checkpoint Ruling

Goal A is viable at this checkpoint: an implementation-neutral, finite, deterministic governance semantic kernel has been extracted and exhaustively checked over its reachable transition graph.

Goal B is not yet adjudicated for governance. The next research question should be chosen deliberately rather than assuming full downstream transformer compilation is the best experiment.
