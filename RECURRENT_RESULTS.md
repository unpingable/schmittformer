# Recurrent Fixed-State Results

This result set is under:

```text
results/recurrent/
```

It starts from the pushed semantic-register checkpoint:

```text
f58b429a96d397f122f1ba477ff9d012591e1b93
projection: add explicit semantic register experiment
```

Historical hysteresis, circuit-breaker, finite-softmax, projection-loss, latent-autopsy, semantic-register, and governance-semantic-core results are preserved. This experiment does not rewrite their rulings.

## Reproduce

```bash
.venv-cuda/bin/python -m experiments.run_counter_verification --out-dir results/recurrent --device cuda --force
.venv-cuda/bin/python -m experiments.run_recurrent_governance --out-dir results/recurrent --device cuda --random-samples 200000 --max-long-steps 1000000 --force
.venv/bin/python -m pytest -q
.venv-cuda/bin/python -m pytest tests/test_compiled_bits.py tests/test_counter8.py tests/test_counter16.py tests/test_recurrent_reference.py tests/test_recurrent_compiled.py tests/test_recurrent_invariants.py tests/test_recurrent_longrun.py tests/test_recurrent_invalid_state.py tests/test_recurrent_resume.py -q
```

The experiment runners support artifact-level resume by default; pass `--force` to recompute existing files.


## Validation

```text
JSON artifacts: all results/recurrent/*.json parsed successfully
CPU full suite: 104 passed, 1 warning
CUDA recurrent-focused suite: 17 passed
counterexamples.json count: 0
```

The warning is the pre-existing PyTorch nested-tensor warning from `tests/test_circuit_learned.py::test_circuit_learned_forward_shapes`; it is not introduced by the recurrent experiment.

## Environment

```text
CUDA environment: .venv-cuda
Python: 3.12.3
PyTorch: 2.11.0+cu128
torch CUDA runtime: 12.8
GPU: NVIDIA GeForce RTX 5060 Ti, 16 GB
NVIDIA driver: 570.211.01
```

Batched transition checks ran on CUDA. The single-trajectory recurrent feedback loop ran on CPU because tiny per-step CUDA launches were slower for batch size 1.

## Structural / By Construction

The compiled recurrent transition has constant physical shape:

```text
state width: 29 bits
event width: 14 one-hot slots
physical input width: 43 slots
output width: 29 next-state bits + 8 output logits
```

The runner supplies no old execution history and no growing KV cache. Each step receives only:

```text
state_t + event_t
```

and emits:

```text
state_(t+1) + governance output_t
```

The counter construction is ripple-borrow saturating decrement, not value lookup. Gate/depth cost scales linearly with bit width:

| width | exhaustive | checked | ripple depth | xor gates | borrow AND gates | mux bits |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 4 | yes | 16 | 4 | 4 | 4 | 4 |
| 8 | yes | 256 | 8 | 8 | 8 | 8 |
| 16 | yes | 65,536 | 16 | 16 | 16 | 16 |
| 32 | no | 200,013 | 32 | 32 | 32 | 32 |

## Exhaustively Verified

Counter primitives:

```text
dec8(x)  == max(x - 1, 0) for all 256 uint8 values
dec16(x) == max(x - 1, 0) for all 65,536 uint16 values
zero/nonzero tests matched through the same exhaustive checks
failures: 0
```

16-bit decrement precision checks:

| dtype | device | checked | failures |
| --- | --- | ---: | ---: |
| float64 | CPU | 65,536 | 0 |
| float32 | CUDA | 65,536 | 0 |
| float16 | CUDA | 65,536 | 0 |
| bfloat16 | CUDA | 65,536 | 0 |

The minimum observed bit margin to the `0.5` decode threshold was `0.5` in all primitive checks: outputs remained exact binary values.

## Empirically Observed

Composed governance transition checks:

```text
edge state/event transitions checked: 10,584
random valid state/event transitions checked: 200,000
composition scenarios: passed
invalid enum state cases: passed
reference invariant violations: 0
compiled failures: 0
```

Full lease countdown:

```text
initial lease: 65,535
TICK steps: 65,536
final lease: 0
first divergence: none
invariant violations: 0
physical input width: 43
```

Budget stress:

```text
initial budgets checked: 0..255
expected admitted actions: initial budget
failures: 0
```

Long recurrent feedback traces:

| logical steps | passed | steps/sec | final lease | final budget | final occurrence |
| ---: | :---: | ---: | ---: | ---: | ---: |
| 10 | yes | 581.35 | 65,532 | 253 | 2 |
| 100 | yes | 597.85 | 65,507 | 237 | 0 |
| 1,000 | yes | 596.41 | 65,242 | 79 | 0 |
| 10,000 | yes | 588.58 | 65,001 | 114 | 0 |
| 100,000 | yes | 594.87 | 65,037 | 88 | 1 |
| 1,000,000 | yes | 597.76 | 65,367 | 180 | 2 |

The one-million-step trace used the same 43-slot input on every invocation.

History equivalence:

```text
history A length: 3
history B length: 1,000,000
same final fixed state: true
common suffix compiled/reference checks: 5
violations: 0
```

This is deliberately almost trivial in the recurrent architecture: prior history is not available to the step function once the fixed state is equal.

Fault injection:

```text
invalid occurrence code -> REFUSE_INVALID_STATE and safe state
invalid settlement code -> REFUSE_INVALID_STATE and safe state
ordinary valid bit flips may silently change authority/counter values
```

The last point is expected. This pass does not implement parity, provenance, or fault-tolerant state storage.

## Solver-Verified

No SMT/SAT equivalence check was implemented in this pass. The `solver.json` artifact records z3 availability if present and explicitly marks solver verification as not attempted.

## Not Established

This pass does not establish:

```text
ordinary finite-temperature softmax realization of recurrent binary arithmetic
raw nn.Linear weight matrices for every Boolean gate
SMT proof of full 301,989,888-state composed transition equivalence
fault tolerance against valid-but-wrong state bit flips
production value over an ordinary deterministic governance monitor
```

The hard/discrete recurrent construction assumes valid binary state/event encodings and deterministic final argmax decoding. Dtype/backend changes can matter if they stop preserving exact `0/1` gate outputs; the tested primitive dtypes did not.

## Comparison To History Retrieval

| property | history-retrieval controllers | fixed recurrent controller |
| --- | --- | --- |
| persistent state source | latest state token in growing history | explicit 29-bit state input |
| physical context | grows with logical trace | constant 43 input slots |
| counter representation | tiny finite enums in prior governance kernel | actual 16-bit and 8-bit binary counters |
| verification style | finite reachable-state enumeration | primitive exhaustive checks + compositional argument + long traces |
| softmax result | established bounded-margin regime | not attempted here |

## Answers To Claude's Objections

**Counter objection:** yes for this pass. The experiment uses actual bounded binary counters: uint16 lease and uint8 budget. The decrement construction scales linearly with bit width and was not implemented as a 65,536-entry or 16,777,216-entry lookup table.

**Horizon objection:** yes under the recurrent execution model. The longest trace checked was 1,000,000 logical transitions while the physical input stayed fixed at 43 slots. Logical time is carried only by the 29-bit state, not by accumulated context history.

**Numerical objection:** partially. The hard/discrete tensor circuit preserved exact decoded bits in `float64`, `float32`, `float16`, and `bfloat16` for the exhaustive 16-bit primitive checks. This does not automatically transfer to finite-softmax attention or to arbitrary backend/compiler changes.

## Constellation Relevance

This remains research-track work. It does not change the practical recommendation from the governance semantic-core pass:

```text
explicit typed state + deterministic governance/reference monitor
```

The useful lesson for practical constellation infrastructure is that fixed typed state is the clean boundary. The transformer realization did not demonstrate a production advantage over a conventional deterministic monitor.

## Primary Ruling

**A. Strong recurrent success under hard/discrete fixed-slot assumptions.**

A fixed-width synthesized transition implements bounded binary counters and the selected governance semantics with exact decoded behavior under the stated assumptions. The arithmetic primitives were exhaustively verified for 8-bit and 16-bit counters, the composed transition matched the reference on edge and large random state/event tests, representation preservation held in long feedback traces, and logical execution length was decoupled from physical transformer context.

The ruling is intentionally scoped: finite-softmax recurrent arithmetic and full SMT equivalence of the composed 29-bit transition relation remain unestablished.

## Strongest Falsification Found

The strongest negative result is performance and backend scope, not semantic failure:

```text
single-trajectory hard/discrete recurrent execution: about 595 steps/sec on CPU
CUDA batch verification: useful for large independent state batches
CUDA single-step recurrence: slower due tiny launch overhead
finite-softmax recurrent arithmetic: not established
```

This suggests the construction is scientifically useful for the counter/horizon objections, but not a practical runtime recommendation.
