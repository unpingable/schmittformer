# Recurrent Softmax Results

This result set is under:

```text
results/recurrent_softmax/
```

It starts from the pushed recurrent fixed-state checkpoint:

```text
cb6ce94 recurrent: add fixed-state counter governance
```

The hard/discrete recurrent result remains historical and unchanged. This pass does not overwrite `results/recurrent/*`.

## Reproduce

```bash
.venv-cuda/bin/python -m experiments.run_recurrent_softmax \
  --out-dir results/recurrent_softmax \
  --device cuda \
  --score-gap 8 \
  --random-samples 200000 \
  --max-long-steps 1000000 \
  --force

.venv-solver/bin/python -m experiments.run_transition_solver \
  --out results/recurrent_softmax/solver.json \
  --timeout-ms 120000

.venv-cuda/bin/python -m experiments.run_recurrent_softmax \
  --out-dir results/recurrent_softmax \
  --device cuda \
  --score-gap 8 \
  --random-samples 200000 \
  --max-long-steps 1000000
```

The last command refreshes the manifest after the solver artifact is written. Runners are artifact-resumable unless `--force` is passed.


## Validation

```text
JSON artifacts: all results/recurrent_softmax/*.json parsed successfully
CPU full suite: 113 passed, 1 warning
CUDA focused recurrent-softmax suite: 8 passed
solver: UNSAT in .venv-solver
historical results/recurrent artifacts: unchanged
```

The warning is the existing PyTorch nested-tensor warning from `tests/test_circuit_learned.py::test_circuit_learned_forward_shapes`.

## Environment

```text
CUDA env: .venv-cuda
Python: 3.12.3
PyTorch: 2.11.0+cu128
torch CUDA runtime: 12.8
GPU: NVIDIA GeForce RTX 5060 Ti, 16 GB
NVIDIA driver: 570.211.01
solver env: .venv-solver
z3-solver: 5.0.0.0, reports Z3 5.0.0
```

## Logically / Solver Established

SMT result:

```text
query:
    exists valid bounded state/event where reference transition differs from
    compiled ripple/Boolean transition

valid domain:
    authority bool
    lease uint16
    budget uint8
    occurrence in {IDLE, IN_FLIGHT, AMBIGUOUS}
    settlement in {NONE, SUCCESS, FAILURE}
    event in 0..13

result: UNSAT
runtime: 0.031 s
```

This proves one-step logical equivalence between the reference transition and the compiled hard/discrete ripple/Boolean transition for every valid bounded state/event in the stated domain. It does not model floating-point softmax.

## Derived Under Numerical Assumptions

For fixed-slot softmax retrieval over width `W`, score gap `g` gives losing mass:

```text
epsilon = (W - 1) / (exp(g) + W - 1)
```

At the governance input boundary:

```text
W = 43
g = 8
epsilon = 0.0138936764
correct mass = 0.9861064
```

Because state/event slots are binary, input retrieval error is bounded by `epsilon`. The downstream circuit then relies on decision margins staying above the perturbation induced by this leakage. This pass measures those margins rather than proving a global real-arithmetic perturbation theorem.

The successful configuration is not effectively hard:

```text
max losing weight at gap 8: 0.00033080185
max losing mass at gap 8:   0.0138936788
softmax losing weights:     nonzero
```

## Exhaustively Verified

Finite-softmax counter primitives, CUDA float32:

| width | gap | checked | result | classification | max losing mass | min output margin |
| ---: | ---: | ---: | --- | --- | ---: | ---: |
| 8 | 4 | 256 | pass | PASS_EXACT | 0.113640 | 0.240252 |
| 8 | 8 | 256 | pass | PASS_EXACT | 0.002343 | 0.494645 |
| 16 | 4 | 65,536 | fail | SEMANTIC_FAILURE | 0.215523 | 0.002887 |
| 16 | 6 | 65,536 | pass | PASS_EXACT | 0.035848 | 0.347047 |
| 16 | 8 | 65,536 | pass | PASS_EXACT | 0.005007 | 0.478638 |

The `gap=4` 16-bit failure is useful: it shows the construction has a real leakage boundary. The successful `gap=8` result is non-saturated finite softmax, not numerical hardmax.

Precision matrix for `dec16`, gap 8:

| precision | device | checked | result | effectively hard | min output margin |
| --- | --- | ---: | --- | --- | ---: |
| float64 | CPU | 65,536 | pass | false | 0.478638 |
| float32 | CPU | 65,536 | pass | false | 0.478638 |
| float32 | CUDA | 65,536 | pass | false | 0.478638 |
| float16 | CUDA | 65,536 | pass | false | 0.477051 |
| bfloat16 | CUDA | 65,536 | pass | false | 0.494141 |

## Empirically Observed

Composed governance at gap 8, CUDA float32:

```text
targeted edge transitions:       10,584 / 10,584 passed
adversarial transitions:          21,168 / 21,168 passed
random valid transitions:        200,000 / 200,000 passed
main semantic counterexamples:   0
reference invariant violations:  0
```

Margins:

| set | max losing mass | min next-bit margin | max next-bit error | min output margin |
| --- | ---: | ---: | ---: | ---: |
| edge | 0.0138937 | 0.372640 | 0.127360 | 24.714840 |
| adversarial | 0.0138937 | 0.372640 | 0.127360 | 24.714840 |
| random | 0.0138937 | 0.388837 | 0.111163 | 25.845074 |

Discrete decode/re-encode recurrent long-run, gap 8, CPU float64:

| logical steps | result | steps/sec | min next-bit margin | max next-bit error |
| ---: | --- | ---: | ---: | ---: |
| 10 | pass | 551.53 | 0.407315 | 0.092685 |
| 100 | pass | 598.51 | 0.390661 | 0.109339 |
| 1,000 | pass | 600.47 | 0.390661 | 0.109339 |
| 10,000 | pass | 595.00 | 0.377385 | 0.122615 |
| 100,000 | pass | 604.45 | 0.377385 | 0.122615 |
| 1,000,000 | pass | 603.07 | 0.372640 | 0.127360 |

Physical input width stayed fixed at 43 slots.

Analog-carry diagnostic:

```text
raw activation carry, no canonical re-encode: fails at step 5
failure: lease decodes to 65528 where reference expects 65532
```

This is important. The finite-softmax recurrent guarantee is a symbolic recurrent guarantee with canonical decode/re-encode after each step. It is not a guarantee for carrying raw analog activations indefinitely.

## Stock Model Attempt

A synthesized `torch.nn.MultiheadAttention` layer implements the same fixed-slot softmax gather:

```text
slot count: 43
parameters in stock gather wrapper: 29,928
stock gather vs direct softmax max abs diff: 2.98e-7
checkpoint size: 122,547 bytes
save/load: passed
```

But the recurrent governance arithmetic after retrieval still calls the custom hard/discrete tensor circuit. The saved checkpoint therefore does not demonstrate that the whole governed transition lives inside stock transformer layers.

## Perturbation Sweep

A small perturbation sweep over the stock-gather parameters checked 128 edge-derived transitions per sigma:

```text
sigma: 0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2
failures: 0 in this small sweep
```

This is only exploratory. It does not prove robustness of the full recurrent machine.

## Not Established

This pass does not establish:

```text
a full stock transformer implementation of the recurrent arithmetic circuit
finite-softmax correctness for analog carry without decode/re-encode
floating-point proof over every valid state/event under softmax leakage
Hugging Face compatibility
invalid-enum SMT equivalence
production value over a deterministic reference monitor
```

## Comparison Table

| property | hard/discrete recurrent | finite-softmax recurrent | stock model attempt |
| --- | --- | --- | --- |
| real uint8/uint16 counters | yes | yes | retrieval only; arithmetic custom |
| fixed physical width | yes | yes | yes |
| million-step recurrence | yes | yes, with discrete re-encode | not as full stock arithmetic |
| exact decoded semantics | yes | yes in tested bounded regime | smoke only, custom arithmetic |
| non-saturated attention | not applicable | yes at gap 8 | yes for gather |
| SMT logical equivalence | now yes for logical circuit | logical circuit yes; softmax not modeled | no full stock proof |
| standard transformer ops | no | softmax retrieval plus custom circuit | MHA retrieval only |
| save/load checkpoint | not meaningful | not full stock | gather wrapper save/load passed |
| fp32 | pass | pass | gather smoke pass |
| fp16/bf16 | hard primitive pass | dec16 primitive pass | not established |

## Rulings

### Q1 - finite-softmax recurrent lowering

**B. Bounded numerical success.**

Ordinary non-saturated finite softmax can retrieve the fixed state/event slots with enough margin for exact decoded recurrent behavior in the tested regime. The 29-bit governed state, uint16 lease, and uint8 budget all survive the gap-8 configuration, including a million-step discrete decode/re-encode trace.

This is not promoted to the strongest category because the softmax perturbation margin is measured and extensively tested, but not mechanically proved over every valid state/event as a real-arithmetic theorem.

### Q2 - stock transformer realization

**C. Custom tensor circuit only.**

A stock `nn.MultiheadAttention` layer can realize the softmax slot retrieval and can be saved/loaded. But the actual governance arithmetic and transition logic still run through custom tensor-circuit code. Calling the complete machine a normal stock transformer checkpoint would be misleading.

### Q3 - formal transition equivalence

**A. Full SMT equivalence.**

For valid bounded states/events, Z3 found the transition-disequality query UNSAT. This establishes logical one-step equivalence between the reference transition and the compiled ripple/Boolean transition. It does not include floating-point softmax in the theorem.

## Final Synthesis

At this checkpoint, Schmittformer is not yet legitimately demonstrating the whole recurrent governance machine as synthesized sequential computation in a conventional transformer checkpoint.

The honest description is:

```text
ordinary finite-softmax slot retrieval
    +
custom synthesized Boolean/arithmetic tensor circuit
    +
canonical symbolic decode/re-encode recurrence
```

The word `transformer` earns its keep for the finite-softmax retrieval front-end, and the logical circuit is now SMT-equivalent to the reference transition. But the full arithmetic/governance transition has not been lowered into stock transformer layers.

## Required Assumptions

The semantic guarantee depends on:

```text
architecture: fixed 43-slot state/event input and existing recurrent tensor circuit
weights/scores: score gap 8 for softmax slot retrieval
dtype/backend: tested CPU float64/float32 and CUDA float32 for composed checks; dec16 tested in CUDA fp16/bf16
softmax regime: finite non-saturated softmax, losing mass about 0.0138937
state encoding: valid binary 29-bit governance state
event encoding: valid one-hot event over 14 events
recurrence boundary: final state bits are decoded and canonically re-encoded each step
valid domain: SMT equivalence excludes invalid occurrence/settlement enum encodings
final decode: bit rounding and output argmax
```

If raw analog activations are carried instead of canonical state bits, the diagnostic fails at step 5.

## Constellation Note

This does not change the practical recommendation:

```text
explicit typed governance state
+ deterministic implementation
+ conformance corpus
+ receipts/provenance
```

The research result clarifies the computational boundary. It does not make neural arithmetic a better production governance boundary than an ordinary deterministic monitor.
