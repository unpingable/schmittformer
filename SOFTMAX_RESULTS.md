# Softmax Results

This experiment adds finite-temperature softmax versions of the compiled hysteresis and circuit-breaker controllers. It preserves the hard-attention baseline at:

```text
cd5a9867175c73ca186a8c4cb67d0338fbf474ff
hard-attention-success
```

## Reproduce

```bash
.venv/bin/python -m pytest
.venv/bin/python -m experiments.run_softmax --hysteresis-max-len 6 --batch-size 8192
```

The run used `torch==2.13.0+cpu`. The host has an RTX 5060 Ti, but the installed PyTorch wheel is CPU-only.

## Files

- `results/softmax_theory.json`
- `results/softmax_sweep.json`
- `results/softmax_hysteresis.json`
- `results/softmax_circuit.json`
- `results/softmax_precision.json`
- `results/softmax_equivalence.json`
- `results/softmax_counterexamples.json`
- `results/softmax_summary.json`

## PROVED / DERIVED UNDER ASSUMPTIONS

For the state-token construction, if adjacent obsolete state records lose score gap `Delta`, stale-state unnormalized mass is bounded by:

```text
S <= exp(-Delta) / (1 - exp(-Delta))
```

With one-hot state values, zero-valued non-state tokens, deterministic transition projection, and final argmax decoding, decoded semantics are guaranteed when:

```text
S < 1
```

Equivalently:

```text
Delta > ln 2
```

This is a real-score decoded-semantics result. It does not require latent one-hot attention. It also does not require softmax saturation.

Analog leakage does not accumulate across steps because every successful step emits a clean discrete state token.

## EXHAUSTIVELY VERIFIED

Main non-saturated configuration:

```text
state_record_gap Delta = 2.0
non_state_penalty beta = 4.0
dtype = float64
```

Hysteresis:

- all state/input transitions passed
- exhaustive traces through length 6 for both initial states passed
- sequences checked: 2,222,220
- tokens checked: 13,086,420
- failures: 0
- effectively hard: false
- max stale-state mass: 0.1269
- max non-state mass: 0.0474
- min decision margin: about 0.6988

Circuit breaker:

- reachable states: 48
- transition histories checked: 384
- state/input checks: 1152
- failures: 0
- effectively hard: false
- max stale-state mass: 0.1289
- max non-state mass: 0.0474
- min decision margin: about 0.6947

Circuit long traces remained exact with zero invariant violations:

| scenario | length |
| --- | ---: |
| healthy | 4096 |
| sustained UNKNOWN | 1024 |
| sustained FAILURE | 256 |
| repeated trip/recover | 1280 |
| threshold edge | 1024 |

## HISTORY EQUIVALENCE

Semantic equivalence passed while latent equivalence failed, as expected.

Hysteresis:

- comparisons: 35
- semantic violations: 0
- max latent state-mass difference: 0.1135

Circuit breaker:

- groups checked: 48
- comparisons: 1680
- semantic violations: 0
- max latent state-mass difference: 0.1289

This directly supports the distinction between latent exactness and decoded semantic exactness.

## EMPIRICALLY OBSERVED

The phase sweep covered:

```text
controllers: hysteresis, circuit
contexts: 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384
state gaps: 0.25, 0.5, 0.69, 0.7, 1.0, 1.5, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0
dtypes: float64, float32, bfloat16, float16 where supported
```

Classification counts in `results/softmax_sweep.json`:

- `PASS_EXACT`: 407
- `SEMANTIC_FAILURE`: 317
- `EFFECTIVELY_HARD`: 174
- `NUMERIC_FAILURE`: 38

First non-failing synthetic sweep margin through context 16384:

| controller | dtype | first passing gap |
| --- | --- | ---: |
| hysteresis | float64 | 0.7 |
| hysteresis | float32 | 0.7 |
| circuit | float64 | 0.7 |
| circuit | float32 | 0.7 |
| hysteresis | bfloat16 | none through 16384 |
| hysteresis | float16 | none through 16384 |
| circuit | bfloat16 | none through 16384 |
| circuit | float16 | none through 16384 |

Low precision still passed smaller contexts in many cases, but failed at long contexts due absolute-position score resolution.

## COUNTEREXAMPLES

Counterexamples below `ln 2` are saved in `results/softmax_counterexamples.json`.

Examples:

- hysteresis, `Delta=0.25`, first failing synthetic context: 3 updates
- hysteresis, `Delta=0.5`, first failing synthetic context: 4 updates
- hysteresis, `Delta=0.69`, first failing synthetic context: 9 updates
- circuit, same margins produced analogous failures

These are adversarial synthetic state-token histories designed to maximize obsolete-state contamination. They confirm the analytical boundary rather than contradict it.

## NUMERICAL HARDENING

Large score gaps can make finite softmax numerically one-hot. These cases are useful engineering points but are weaker conceptually.

The main `Delta=2` result is not numerically hard: stale-state and non-state masses are nonzero and measured.

## NOT ESTABLISHED

- Arbitrary-context correctness for this exact absolute-position implementation in every finite dtype.
- A guarantee for `bfloat16` or `float16` through 16384 contexts with the current absolute-position scoring.
- Latent exactness.
- Any claim about learned baselines improving under retraining; they were not retrained.
- A finite-temperature result for an implementation without explicit state tokens and final discrete decoding.

## Ruling

```text
B. Bounded-margin success
```

Finite softmax works exactly over a clear context/precision/margin regime, and the real-score leakage argument is stronger than empirical robustness: `Delta > ln 2` is sufficient for decoded semantic correctness in the state-token construction.

The concrete PyTorch implementation still has a finite-precision boundary because it uses absolute-position scores. In float32/float64, the non-saturated softmax regime passed through context 16384. In bfloat16/float16, long-context failures occur when position-score resolution collapses. Therefore hard attention was not conceptually essential, but finite-softmax correctness depends on explicit numerical margins and representation assumptions.
