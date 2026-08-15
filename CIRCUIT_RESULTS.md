# Circuit Breaker Results

This is the second experiment, added without changing the first hysteresis pass.

## Reproduce

```bash
.venv/bin/python -m pytest
.venv/bin/python -m experiments.run_circuit \
  --natural-steps 1500 \
  --balanced-steps 1200 \
  --classifier-steps 1000 \
  --e2e-steps 1500 \
  --train-len 64
```

Environment for this run:

- `torch==2.13.0+cpu`
- CUDA unavailable to PyTorch in the installed wheel
- host GPU reported by `nvidia-smi`: `NVIDIA GeForce RTX 5060 Ti, 16311 MiB, driver 570.211.01`

The CUDA PyTorch wheel download failed during the earlier setup, so this serious pass ran on CPU. That affects speed, not the semantics tested here.

## Reference Semantics

The reference controller uses an explicit hashable `CircuitState`:

```text
mode, failure_window, cooldown_remaining, consecutive_successes, probe_budget
```

Authoritative semantic choices:

- `UNKNOWN` does not enter or advance the CLOSED failure window.
- Cooldown is checked at the start of each input tick. `OPEN(c>0)` remains OPEN and decrements; `OPEN(0)` transitions to `HALF_OPEN` and ignores that tick's observation for recovery.
- `SUCCESS, UNKNOWN, SUCCESS` in `HALF_OPEN` counts as two consecutive relevant successes.
- Entering `CLOSED` resets the CLOSED failure window.
- The probe-budget exhaustion rule is implemented, but is redundant for reachable states under the immediate-reopen-on-failure rule: reachable HALF_OPEN states are only `(successes=0,budget=3)` and `(successes=1,budget=2)`.

Reachable finite graph:

- normalized syntactic states: 80
- reachable states from initial CLOSED: 48
- reachable transitions: 144

Saved graph: `results/circuit_graph.json`.

## Compiled Controller

Representation:

- 3 input tokens: `SUCCESS`, `FAILURE`, `UNKNOWN`
- 48 generated complete-state tokens, one per reachable logical state
- token overhead: one generated state token per input, plus initial state token
- hard attention: select the latest prior state token
- lookup behavior: deterministic table over `reachable_state_id x event_id`
- decoding: greedy argmax emits the next complete state token

The runtime history is autoregressive:

```text
STATE(s0), INPUT(x1), STATE(s1), INPUT(x2), STATE(s2), ...
```

The Python driver appends generated tokens, but the transition itself is computed by the compiled module's hard latest-state selection plus transition table. The reference transition is not called during compiled decoding.

### Exhaustive/Finite Checks

Compiled transition verification checked every reachable state and every input, using multiple histories per state rather than only canonical histories:

- histories checked: 384
- transitions checked: 1152
- failures: 0

History-equivalence testing:

- equivalent-state groups checked: 48
- comparisons: 1680
- `history_equivalence_violation`: 0

Long traces all matched exactly with zero invariant violations:

| scenario | length | exact | invariant violations |
| --- | ---: | ---: | ---: |
| healthy | 4096 | yes | 0 |
| sustained UNKNOWN | 1024 | yes | 0 |
| sustained FAILURE | 256 | yes | 0 |
| repeated trip/recover | 1280 | yes | 0 |
| threshold edge | 1024 | yes | 0 |

Measured compiled decode throughput for those traces was about 2744 input tokens/s on CPU. This is not optimized.

## Induction Argument

This experiment supports an induction-style correctness argument under the hard-attention assumptions:

1. Every reachable abstract controller state has exactly one generated state-token representation.
2. For every reachable abstract state and every input symbol, transition verification shows that the compiled module emits the reference next state token.
3. The emitted token is again one of the 48 valid reachable state tokens, so the representation invariant is preserved.
4. The compiled step attends only to the latest state token, so older history is irrelevant once the latest complete state token is fixed.

Therefore arbitrary finite traces are correct under these assumptions: exact token representation, deterministic transition lookup, hard argmax attention, and greedy decoding. This is stronger than bounded testing alone. It is still not a finite-temperature softmax result.

## Learned Transformers

Two small causal transformer variants were trained to infer the complete controller state from input history only. They were not given reference state variables as inputs.

Common architecture:

- 2 transformer layers
- `d_model=48`
- 4 attention heads
- `d_ff=96`
- sinusoidal positions, max evaluation length 4096
- AdamW, learning rate `3e-4`
- train length 64, batch size 128

### Natural Training Distribution

Distribution: mostly `SUCCESS`, occasional `UNKNOWN`, uncommon `FAILURE`.

Training:

- steps: 1500
- wall time: 58.5 s CPU
- final training state accuracy: 0.9847
- final training mode accuracy: 0.9976

Evaluation found serious failures:

| scenario | length | state acc | mode acc | illegal steps | first mode divergence |
| --- | ---: | ---: | ---: | ---: | ---: |
| natural 32 | 32 | 1.000 | 1.000 | 0 | none |
| natural 64 | 64 | 0.969 | 1.000 | 3 | none |
| healthy 128 | 128 | 1.000 | 1.000 | 0 | none |
| sustained UNKNOWN | 256 | 0.453 | 1.000 | 51 | none |
| sustained FAILURE | 128 | 0.008 | 0.016 | 127 | 2 |
| repeated trip/recover | 260 | 0.004 | 0.296 | 144 | 2 |
| threshold edge | 256 | 0.012 | 0.023 | 189 | 6 |
| adversarial mix | 1054 | 0.016 | 0.532 | 546 | 6 |

History-equivalence test:

- groups checked before hitting cap: 3
- comparisons: 53
- `history_equivalence_violation`: 50

The model learned the common CLOSED/healthy path but did not learn robust abstract-state equivalence. It also changed hidden failure-window state on `UNKNOWN` in many predicted trajectories.

### Adversarial/Balanced Training Variant

Distribution: adversarial trip/recovery and edge patterns with injected noise.

Training:

- steps: 1200
- wall time: 46.5 s CPU
- final training state accuracy: 0.8733
- final training mode accuracy: 0.9613

It improved some rare transitions but degraded common ones and still failed equivalence:

| scenario | length | state acc | mode acc | illegal steps | first mode divergence |
| --- | ---: | ---: | ---: | ---: | ---: |
| natural 32 | 32 | 0.500 | 1.000 | 19 | none |
| natural 64 | 64 | 0.375 | 0.984 | 32 | 53 |
| healthy 128 | 128 | 0.992 | 1.000 | 1 | none |
| sustained UNKNOWN | 256 | 0.766 | 1.000 | 4 | none |
| sustained FAILURE | 128 | 0.281 | 0.562 | 82 | 25 |
| repeated trip/recover | 260 | 0.150 | 0.354 | 174 | 3 |
| threshold edge | 256 | 0.172 | 0.504 | 223 | 3 |
| adversarial mix | 1054 | 0.120 | 0.379 | 688 | 3 |

History-equivalence test:

- groups checked before hitting cap: 4
- comparisons: 63
- `history_equivalence_violation`: 50

Counterexamples are saved in:

- `results/circuit_learned_natural.json`
- `results/circuit_learned_adversarial.json`

The minimized examples are not guaranteed globally minimal; they are greedy deletion reductions.

## Hybrid

Hybrid structure:

```text
noisy scalar observation -> learned event classifier -> compiled circuit breaker -> state/action
```

Classifier training was balanced over the three event classes. The end-to-end comparison model was a continuous causal transformer trained directly from noisy observations to controller state.

Training:

- classifier steps: 1000, final accuracy 0.922, wall time 0.84 s CPU
- end-to-end steps: 1500, final state accuracy 0.920, final mode accuracy 0.999, wall time 59.8 s CPU

Evaluation separates world-relative correctness from belief-relative semantic correctness:

| scenario | classifier acc | hybrid world state acc | e2e world state acc | hybrid semantic violations vs belief | hybrid violations vs true world | e2e violations vs true world |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| in distribution | 0.907 | 0.893 | 0.799 | 0 | 2005 | 10009 |
| shifted noise | 0.764 | 0.710 | 0.643 | 0 | 13455 | 40624 |
| adversarial events | 0.747 | 0.350 | 0.092 | 0 | 16179 | 94972 |
| long duration | 0.806 | 0.351 | 0.048 | 0 | 14103 | 117352 |

The hybrid makes wrong world-relative decisions when perception is wrong. But given the classifier's discrete event beliefs, the compiled controller made zero semantic violations in every scenario. The end-to-end learned controller often produced trajectories that were illegal relative to the true latent event sequence.

## What Was Demonstrated

- The richer controller has a small finite reachable state graph: 48 states.
- A synthesized hard-attention transformer-like module can implement the full finite transition graph exactly.
- Multiple histories denoting the same complete logical state are observationally equivalent for the compiled controller under tested suffixes.
- Because the representation invariant is preserved for every reachable transition, correctness extends by induction to arbitrary finite decoded traces under the stated hard-attention assumptions.
- Learned end-to-end transformers can fit common training behavior while failing rare transitions, cooldown/recovery discipline, `UNKNOWN` invariance, and history equivalence.
- A learned perception front-end can be wrong about the world while the compiled controller still preserves belief-relative control semantics exactly.

## What Was Only Empirically Observed

- Learned model failures depend on seed, architecture, and training distribution.
- Hybrid world-relative accuracy depends on the synthetic observation/noise model.
- Throughput numbers are rough CPU measurements, not optimized performance claims.

## Assumptions

- Hard argmax attention is the exactness primitive.
- State-token decoding is greedy and deterministic.
- State tokens encode complete logical controller state.
- Input symbols are discrete and correctly represented after classification for belief-relative claims.
- The result does not solve finite-temperature softmax exactness.

## Ruling

```text
A. Strong success
```

The compiled transformer implements the complete circuit-breaker semantics, history-equivalence holds, and correctness beyond tested trace length is justified by the finite reachable-state transition check plus the preserved state-token representation invariant.

This is not a claim that ordinary learned transformers will discover this policy, nor that softmax attention is exact. It is a claim that uncertain learned perception can feed a synthesized temporal/control mechanism whose legal state evolution is structural, independently testable, and invariant to irrelevant history.
