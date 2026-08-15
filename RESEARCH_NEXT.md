# Governance Research Next Steps

Goal A and Goal B remain separate.

Goal A: useful constellation semantics, likely as a deterministic reference/conformance model.

Goal B: research on whether neural/transformer machinery can structurally enforce governance semantics in ways a conventional reference monitor cannot.

## Ranking

### 1. Projection-loss / latent-boundary experiment

Question answered: Does a downstream token-level governance kernel arrive too late because policy-relevant latent state is erased or laundered during proposal serialization?

What would falsify it: If all policy-relevant distinctions survive serialization in the tested setting and the downstream compiled/ref monitor catches every violation with no loss of utility.

Relation to prior art: Connects compiled automata/transformer-program work to representation bottlenecks and causal abstraction rather than another finite-state implementation demo.

What a conventional Rust/reference monitor cannot answer: A Rust monitor can reject serialized proposals, but it cannot reveal whether the proposal interface erased the state needed for correct governance before the monitor saw it.

Estimated cost: Medium.

### 2. Fixed-width recurrent governance state

Question answered: Can synthesized neural computation maintain governance state without full history retrieval/state-token growth?

What would falsify it: Counter/lease/settlement state drifts, aliases, or loses history equivalence under long adversarial traces.

Relation to prior art: Moves beyond RASP/Tracr-style history selection toward recurrent state abstraction and finite controller preservation.

What a conventional Rust/reference monitor cannot answer: Whether neural state can be structurally constrained to a governance quotient rather than memorizing histories.

Estimated cost: Medium/high.

### 3. Causal-interchange experiment

Question answered: Do internal activations in a learned or hybrid model causally correspond to governance state variables such as authority, evidence, unsettled attempt, or budget?

What would falsify it: Intervening on the proposed latent state fails to produce the expected policy behavior, or multiple incompatible latent encodings drive the same output only accidentally.

Relation to prior art: Connects transformer-program verification and causal abstraction/interchange methods.

What a conventional Rust/reference monitor cannot answer: Whether the learned part has aligned latent policy variables before the final discrete proposal boundary.

Estimated cost: Medium.

### 4. Parameterized counters / leases

Question answered: Can synthesized transformer machinery represent counters and expiry parametrically rather than enumerating every concrete value in a table?

What would falsify it: The construction only works by enumerating all counter states or loses exact decoded behavior under scale changes.

Relation to prior art: Tests the boundary between finite automata compilation and arithmetic/control representations in transformer programs.

What a conventional Rust/reference monitor cannot answer: Whether the neural substrate can express reusable temporal machinery instead of a finite lookup table.

Estimated cost: Medium.

### 5. Full finite-softmax governance compilation

Question answered: Can the 912-state governance occurrence lifecycle be compiled into the existing finite-softmax state-token substrate with the same bounded-margin guarantees as circuit breaker?

What would falsify it: Softmax margin collapse, state aliasing, history-equivalence failure, or the result becoming merely a bulky transition-table serializer.

Relation to prior art: Direct continuation of schmittformer hard/soft attention experiments and Tracr/RASP-style compiled transformer programs.

What a conventional Rust/reference monitor cannot answer: Very little unless paired with a stronger neural-boundary hypothesis. Alone, this risks confirming that transformers can implement finite tables.

Estimated cost: High.

### 6. Comparison against conventional reference monitor

Question answered: In the same hybrid setup, does a compiled transformer backend offer any behavioral or engineering advantage over a deterministic monitor fed the same tokens?

What would falsify it: The reference monitor is simpler, faster, more auditable, and provides the same guarantees at the same boundary.

Relation to prior art: Mostly an engineering baseline, not a novel transformer result.

What a conventional Rust/reference monitor cannot answer: Nothing by itself. It is the control arm that keeps the research honest.

Estimated cost: Low.

## Recommended Next Research Move

Run the projection-loss / latent-boundary experiment first. It directly addresses the strongest criticism of a downstream compiled token-level kernel: if the only thing it can enforce is the already-serialized proposal language, it may duplicate a conventional monitor. A projection-loss experiment can show whether there is research value upstream of serialization.

Full finite-softmax governance compilation should wait. It is technically feasible in outline, but currently the least likely to answer a question that the deterministic conformance corpus does not already answer.
