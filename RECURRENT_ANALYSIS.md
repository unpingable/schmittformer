# Recurrent Fixed-State Analysis

This research pass starts from the pushed semantic-register checkpoint:

```text
f58b429a96d397f122f1ba477ff9d012591e1b93
projection: add explicit semantic register experiment
```

It preserves the historical hard-attention, finite-softmax, circuit-breaker, projection-loss, latent-autopsy, and semantic-register results. This pass does not retrain the semantic-register writer and does not modify the implementation-neutral governance semantic core.

## Question

Earlier compiled controllers represented persistent state by retrieving the latest relevant state record from autoregressive history. That made logical horizon depend on context length and represented counters as small enumerated finite-state fields.

This experiment asks whether synthesized transformer-style tensor computation can instead implement one recurrent transition:

```text
fixed-width state_t + event_t -> fixed-width state_(t+1) + output_t
```

The execution harness feeds only the emitted fixed-width state into the next invocation. Old history is not provided to the compiled step.

## State Definition

The recurrent state is 29 bits:

```text
authority:       1 bit   INVALID=0, VALID=1
lease_remaining: 16 bits unsigned, little-endian
action_budget:   8 bits unsigned, little-endian
occurrence:      2 bits  IDLE=0, IN_FLIGHT=1, AMBIGUOUS=2, 3 invalid
settlement:      2 bits  NONE=0, SUCCESS=1, FAILURE=2, 3 invalid
```

Counter fields use every binary value. Enum-like fields reserve one invalid code each. Invalid occurrence or settlement encodings are defined to emit `REFUSE_INVALID_STATE` and transition to a safe state:

```text
authority=INVALID, lease=0, budget=0, occurrence=IDLE, settlement=NONE
```

The physical input width is constant:

```text
state bits: 29
event one-hot: 14
total input slots: 43
```

The physical output width is also constant:

```text
next-state bits: 29
output one-hot/logits: 8
total output slots: 37
```

## Event Alphabet

Events are finite and compact:

```text
NOOP
TICK
PROPOSE_ACTION
RESULT_SUCCESS
RESULT_FAILURE
RESULT_AMBIGUOUS
SETTLE_SUCCESS
SETTLE_FAILURE
GRANT_AUTHORITY
REVOKE_AUTHORITY
RENEW_LEASE_MAX
RENEW_LEASE_ONE
RESET_BUDGET_MAX
RESET_BUDGET_ONE
```

Payload-setting events are intentionally limited to constants in this pass. The arithmetic under test is decrement and zero/nonzero testing during ordinary governance transitions, not arbitrary integer parsing.

## Reference Semantics

The exact oracle is `src/recurrent_reference.py`:

```python
transition(state, event) -> TransitionResult(next_state, output)
```

Key rules:

```text
TICK:
    lease_remaining = max(lease_remaining - 1, 0)

PROPOSE_ACTION from IDLE:
    admit iff authority is VALID, lease_remaining > 0, and action_budget > 0
    admission consumes exactly one budget unit and sets occurrence=IN_FLIGHT

PROPOSE_ACTION while IN_FLIGHT:
    REFUSE_IN_FLIGHT

PROPOSE_ACTION while AMBIGUOUS:
    REFUSE_AMBIGUOUS

RESULT_SUCCESS / RESULT_FAILURE from IN_FLIGHT:
    occurrence=IDLE and settlement records SUCCESS / FAILURE

RESULT_AMBIGUOUS from IN_FLIGHT:
    occurrence=AMBIGUOUS and settlement=NONE

SETTLE_SUCCESS / SETTLE_FAILURE from AMBIGUOUS:
    occurrence=IDLE and settlement records SUCCESS / FAILURE
```

Ambiguous outcomes do not refund budget and do not authorize blind retry. Settlement is an explicit event. Event ordering is fully explicit; for example, `PROPOSE_ACTION` with `lease=1` differs from `TICK` followed by `PROPOSE_ACTION`.

## Arithmetic Construction

The core arithmetic primitive is saturating decrement over little-endian bits:

```text
dec_n(x) = max(x - 1, 0)
```

Implementation in `src/compiled_bits.py` uses a ripple-borrow subtract-one circuit:

```text
borrow_0 = 1
raw_bit_i = xor(x_i, borrow_i)
borrow_(i+1) = borrow_i AND NOT x_i
nonzero = OR_i x_i
dec = raw if nonzero else 0
```

The zero and nonzero tests are direct Boolean reductions over the same bit vector. Construction cost is linear in bit width:

```text
xor gates:        n
borrow AND gates: n
mux bits:         n
ripple depth:     n
```

This is not a lookup table over counter values. The 16-bit decrementer uses the same construction as the 8-bit decrementer with twice the ripple length.

## Transformer Realization

`src/recurrent_compiled.py` implements the transition as a deterministic synthesized tensor program over fixed slots:

```text
bit tests
ReLU threshold gates for AND/OR/XOR/equals
hard multiplexers for conditional state-field selection
one-hot output logits with fixed margin
```

This is best interpreted as a hard/discrete transformer-program MLP block over a fixed token/slot representation. It is not a trained model and it does not call the Python reference transition inside the compiled step.

This pass did not instantiate every gate as explicit `nn.Linear` weight matrices and did not attempt the finite-softmax equivalent of the bit-level arithmetic. Those are separate backend-realization questions. The semantic computation itself is still performed by synthesized PyTorch tensor operations, not by a Python FSM wrapped around a model.

## Recurrent Execution Model

The recurrent harness does only:

```text
encode state_t
encode event_t
invoke compiled transition
decode/check state_(t+1) and output_t
feed emitted state bits to the next invocation
```

The harness does not supply old execution history and does not carry hidden state beyond the declared 29-bit governance state. The logical trace may be much longer than the physical input width because each transition consumes the previous fixed-width state, not the whole transcript.

## Correctness Strategy

The full syntactic state space is too large for direct enumeration:

```text
2 authority values * 2^16 leases * 2^8 budgets * 3 occurrences * 3 settlements
= 301,989,888 valid states
```

Therefore this pass uses compositional verification instead of full state enumeration:

```text
1. exhaustively verify dec8 over all 256 inputs
2. exhaustively verify dec16 over all 65,536 inputs
3. verify zero/nonzero and boundary behavior through the decrement checks
4. test composed governance transitions on edge states and large random valid batches
5. test long recurrent traces where emitted bits are fed back as the next state
6. test invalid enum encodings and fault-injection cases
```

The intended induction argument is:

```text
if each compiled one-step transition equals the reference transition for every valid fixed-width state and event,
and emitted valid states preserve the representation invariant,
then arbitrary finite logical traces are equivalent by induction,
within the declared machine and numeric assumptions.
```

The arithmetic primitives are exhaustively checked for their full bit width. The composed governance transition is characterized by explicit Boolean/arithmetic decomposition plus edge/random/property tests, not by enumerating all 301,989,888 states.

## Numerical Assumptions

The hard/discrete construction assumes valid state and event inputs are exact binary values in `{0,1}`. The gates are threshold-style tensor operations that return exact `0` or `1` on those inputs in the tested dtypes. Output decoding is final `argmax` over logits with fixed margin.

This pass tests `float64` CPU and CUDA `float32`, `float16`, and `bfloat16` for the 16-bit decrement primitive where supported. It does not establish a finite-temperature softmax bound for recurrent arithmetic.

## Relation To Previous History Retrieval

The previous history-retrieval controllers had strong state-token and finite-softmax results, but physical context grew with logical execution history. This recurrent construction changes the substrate:

```text
previous: history -> retrieve latest state -> transition
this pass: fixed state + current event -> transition
```

The tradeoff is that correctness must now be argued compositionally over arithmetic and Boolean gates rather than by exhaustive enumeration of a tiny abstract transition graph.

## Limits

Not established here:

```text
ordinary finite-softmax implementation of the bit-level recurrent transition
raw-weight transformer instantiation of every Boolean gate
SMT/SAT equivalence of the full composed 29-bit transition relation
fault tolerance against valid-but-wrong state bit flips
production relevance over a deterministic reference monitor
```
