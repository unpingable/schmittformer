# Recurrent Softmax Analysis

This pass starts from:

```text
cb6ce94 recurrent: add fixed-state counter governance
```

It preserves the hard/discrete recurrent result and does not modify:

```text
RECURRENT_ANALYSIS.md
RECURRENT_RESULTS.md
results/recurrent/*
```

## Questions

This pass separates three questions:

```text
Q1: finite-softmax lowering of the existing fixed-width recurrent machine
Q2: stock transformer checkpoint realization
Q3: bounded one-step transition equivalence by SMT
```

A success or failure in one layer does not automatically transfer to the others.

## Stage 0 Audit

The current recurrent implementation in `src/recurrent_compiled.py` is not a stock transformer checkpoint. It is a deterministic hard/discrete tensor circuit over fixed slots.

| Operation | Current implementation | Classification |
| --- | --- | --- |
| State/event bit encoding | Python tensor construction, enum one-hot | EXTERNAL_ENCODING |
| State/event field split | Tensor slicing by fixed offsets | CUSTOM_TENSOR_CIRCUIT |
| Latest state retrieval | Not used in recurrent backend | not applicable |
| Bit copy | Direct tensor carry / concatenation | CUSTOM_TENSOR_CIRCUIT |
| NOT | `1 - x` | CUSTOM_TENSOR_CIRCUIT, MLP-realizable |
| AND | `relu(a + b - 1)` | CUSTOM_TENSOR_CIRCUIT, MLP-realizable |
| OR | clipped sum via ReLU | CUSTOM_TENSOR_CIRCUIT, MLP-realizable |
| XOR | `relu(a-b)+relu(b-a)` | CUSTOM_TENSOR_CIRCUIT, MLP-realizable |
| Equality test | fixed bit comparisons plus AND tree | CUSTOM_TENSOR_CIRCUIT |
| Zero/nonzero test | OR reduction over counter bits | CUSTOM_TENSOR_CIRCUIT |
| Borrow propagation | explicit Python loop unrolling gates by bit index | CUSTOM_TENSOR_CIRCUIT |
| Saturating decrement | ripple-borrow plus mux | CUSTOM_TENSOR_CIRCUIT |
| Conditional select | algebraic hard mux `c*t+(1-c)*f` | CUSTOM_TENSOR_CIRCUIT |
| Refusal/admission selection | ordered hard mux over one-hot outputs | CUSTOM_TENSOR_CIRCUIT |
| Final state/output decode | Python rounding and argmax in harness | EXTERNAL_DECODING |
| Reference comparison | Python oracle | TEST_ONLY |

The important correction is that the hard/discrete recurrent checkpoint should not be described as a conventional transformer. It is transformer-program adjacent: fixed-width vector slots, synthesized gates, final decode, and no learned weights.

## Primitive Obligations

The recurrent machine depends on these logical primitives:

```text
bit copy
bit NOT
bit AND / OR / XOR
conditional select
zero/nonzero test
borrow propagation
saturating decrement
small enum decode
output one-hot/logit decode
```

Most of these do not require attention. They are local Boolean/threshold computations naturally expressible by feed-forward layers over fixed slots. The only selection-like operation introduced in this pass is fixed-slot retrieval from the state/event input boundary.

## Finite-Softmax Lowering

`src/recurrent_softmax.py` adds a finite-temperature softmax slot-gather step before the existing recurrent transition circuit.

For a vector of width `W`, head `j` scores slot `j` with `gap` and every other slot with `0`:

```text
weights_j = softmax([0, ..., gap at j, ..., 0])
retrieved_j = sum_i weights_j[i] * slot_i
```

This is ordinary softmax-weighted averaging. It does not use hardmax, post-softmax zeroing, straight-through estimators, or Python winner selection.

The total losing mass for each slot is:

```text
epsilon = (W - 1) / (exp(gap) + W - 1)
        = (W - 1) exp(-gap) / (1 + (W - 1) exp(-gap))
```

For the recurrent governance input width `W=43` and `gap=8`, this is about:

```text
0.0138936764
```

For a binary input slot, retrieval error is at most `epsilon` because all values are in `[0,1]`. The downstream hard/discrete circuit is then tested and margin-measured under this bounded perturbation.

## Numerical Target

This pass targets decoded semantic exactness:

```text
retrieved activations may be analog
internal Boolean/counter activations may be analog
final emitted bits/logits must decode to the reference next state/output
```

It does not target latent exactness. In the successful non-saturated configuration, softmax losing mass is nonzero and intermediate values are not exactly the hard circuit's values.

## Error Reset Boundary

The main recurrent execution mode is:

```text
softmax step -> decoded state bits -> canonical re-encode -> next step
```

This is a recurrent symbolic transducer with a finite-softmax one-step implementation. Analog retrieval error is reset at every state boundary if the decoded bits are correct.

A diagnostic analog-carry mode also exists:

```text
softmax step -> raw next-state activations -> next step
```

This is a continuous recurrent dynamical system. It is not the primary semantic model. The smoke run already showed analog carry drifts quickly, so the semantic guarantee depends on the canonical decode/re-encode boundary.

## Stock Transformer Attempt

`src/stock_transformer_recurrent.py` synthesizes a standard `torch.nn.MultiheadAttention` layer that performs the same fixed-slot softmax gather:

```text
ordinary Q/K/V projections: yes
ordinary softmax attention: yes
ordinary attention mask: yes, masks only the compute token from attending to itself
synthesized weights: yes
save/load state_dict: yes
```

However, the governance transition arithmetic after retrieval still calls the custom hard/discrete tensor circuit. Therefore the stock realization is only a stock attention front-end, not a full stock transformer implementation of the recurrent governance machine.

## SMT Strategy

`src/transition_smt.py` encodes two one-step transition relations over bounded bit-vectors:

```text
reference transition:
    source-level event cases and bounded arithmetic

compiled logical transition:
    bit-level ripple decrement, Boolean gates, and mux schedule
```

The solver query asks whether there exists a valid bounded state/event such that the two next states or outputs differ.

The SMT domain is:

```text
authority: boolean
lease: uint16
action_budget: uint8
occurrence: {IDLE, IN_FLIGHT, AMBIGUOUS}
settlement: {NONE, SUCCESS, FAILURE}
event: 0..13
```

Invalid enum encodings are excluded from the SMT equivalence claim. They remain handled empirically by the hard/discrete invalid-state tests.

The solver does not model floating-point softmax. SMT establishes logical circuit equivalence only.

## Claims Chain

A strong arbitrary-trace claim requires these separate premises:

```text
1. logical compiled transition equals reference transition for every valid state/event
2. finite-softmax numerical realization decodes the same one-step transition in the tested/assumed regime
3. each successful step re-establishes canonical fixed-width state bits
4. the next invocation receives only that canonical state plus the current event
```

If any premise fails, the strong semantic claim does not follow. In particular, analog carry without re-encoding is not covered.

## Architecture Assumptions

The current finite-softmax result assumes:

```text
fixed 43-slot state/event input
ordinary finite softmax slot retrieval with the recorded score gap
existing hard/discrete recurrent Boolean/counter circuit after retrieval
valid binary state/event encodings at each recurrent boundary
final bit rounding and output argmax decode
canonical decode/re-encode between recurrent steps
fixed dtype/backend regime as measured
```

It does not establish a model composed entirely of standard transformer layers.
