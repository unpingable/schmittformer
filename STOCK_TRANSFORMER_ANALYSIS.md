# Stock Transformer Closure Analysis

This pass starts from:

```text
a446c00 recurrent: add finite-softmax lowering audit
```

It preserves the recurrent hard/discrete and recurrent finite-softmax result sets. New artifacts live under:

```text
results/stock_transformer/
```

## Target

The target transition is unchanged:

```text
29-bit fixed governance state
    authority
    uint16 lease_remaining
    uint8 action_budget
    occurrence
    settlement

14-way current event

-> 29 next-state bits + 8 governance-output logits
```

The physical model input is fixed at 43 slots. No old execution history is supplied to the one-step model. Recurrent execution still uses the explicit canonical latch:

```text
decode next bits -> canonical state -> re-encode for next step
```

## Boundary Audit

| operation | classification |
| --- | --- |
| input state/event encoding | external encoding |
| fixed-slot retrieval | `torch.nn.MultiheadAttention` finite softmax |
| Q/K/V projections | ordinary synthesized linear projections |
| bit copy | generated standard `Linear/ReLU/Linear` blocks |
| NOT | generated FFN affine output |
| AND/OR/XOR | generated ReLU threshold/absolute-value formulas |
| MUX / conditional select | generated ReLU Boolean formula |
| zero test | generated OR/NOT FFN circuit |
| borrow propagation | generated ripple-borrow FFN circuit |
| saturating decrement | generated FFN circuit, not lookup |
| governance gates/refusal selection | generated FFN circuit |
| state/output logits | generated final linear readout |
| state decode/re-encode | external canonical latch |
| trace checker/reference oracle | test only |

The final stock model forward path does not call `compiled_counter.py`, `compiled_bits.py`, `recurrent_compiled.py`, or the reference transition. Those modules are still used by tests/experiments as oracles or historical baselines.

## ReLU Boolean Construction

All formulas are exact for binary inputs.

```text
NOT(x)      = 1 - x
AND(x,y)   = ReLU(x + y - 1)
AND(xs)    = ReLU(sum(xs) - (n - 1))
OR(xs)     = sum(xs) - ReLU(sum(xs) - 1)
XOR(x,y)   = ReLU(x - y) + ReLU(y - x)
```

The mux is implemented without multiplication:

```text
AND(c, t)       = ReLU(c + t - 1)
AND(NOT c, f)   = ReLU(f - c)
MUX(c,t,f)      = OR(AND(c,t), AND(NOT c,f))
```

The generated network is a sequence of ordinary frozen blocks:

```text
Linear -> ReLU -> Linear
```

Each block appends new wires while preserving existing nonnegative wires.

## Counter Construction

For little-endian bits and initial `borrow_0 = 1`:

```text
raw_i      = XOR(bit_i, borrow_i)
borrow_i+1 = ReLU(borrow_i - bit_i)
nonzero    = OR(all input bits)
dec_i      = MUX(nonzero, raw_i, 0)
```

This implements:

```text
dec(x) = max(x - 1, 0)
```

The construction cost scales with bit width. It is not a 256-entry or 65,536-entry lookup.

## Stock Architecture

The model is:

```text
StockSoftmaxGather(torch.nn.MultiheadAttention)
    ->
SynthesizedReLUCircuit(118 Linear/ReLU/Linear blocks)
    ->
final Linear readout
```

LayerNorm and residual paths are not used in the critical circuit. This is deliberate: normalization would make exact Boolean margins harder to control and was unnecessary for the closure question. The result is therefore a stock-operations transformer realization, not a standard LLM block layout.

## Correctness Chain

Logical:

```text
reference transition
    == existing SMT result
logical Boolean/ripple circuit
    == exact ReLU primitive formulas on binary inputs
stock FFN realization
```

Numerical:

```text
finite-softmax slot retrieval at gap 8
    introduces bounded analog leakage
generated FFN circuit
    preserves decoded bits/output margins in tested regimes
final argmax/decode
    recovers the reference state/output
```

The pass does not prove a full floating-point theorem for every possible CUDA/CPU implementation. It empirically validates the complete stock model over the same transition populations used by the recurrent experiments and records the observed margins.

## Assumptions

The guarantee remains conditional on:

```text
fixed 43-slot state/event input representation
valid binary state bits and one-hot event encoding
score gap 8 finite-softmax slot retrieval
synthesized weights unchanged
tested dtype/backend regime
final deterministic argmax/decode
canonical discrete state re-encode between recurrent invocations
valid enum state/event domain for the SMT equivalence result
```

Raw analog recurrent carry is not part of the claim. The prior recurrent-softmax experiment showed analog carry fails quickly; this pass preserves the canonical latch.

## Limits

The stock realization closes the architectural asterisk but at a real cost:

```text
118 FFN blocks
max FFN width 743
29,835,345 parameters
checkpoint about 119.5 MB
single-trajectory feedback about 138 steps/sec on RTX 5060 Ti
```

This is not a production recommendation. It demonstrates that the transition can live in ordinary transformer operations, not that it is the sensible way to enforce governance.
