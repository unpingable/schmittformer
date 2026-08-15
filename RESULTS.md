# Results

First serious pass completed on the local host. The installed PyTorch build is CPU-only (`torch==2.13.0+cpu`) because the initial CUDA wheel download failed while fetching cuDNN. The machine still reports `NVIDIA GeForce RTX 5060 Ti, 16311 MiB, driver 570.211.01`.

## Reproduce

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest
.venv/bin/python -m experiments.run_all --compiled-max-len 6 --learned-steps 800 --hybrid-steps 800 --classifier-steps 600 --train-len 16
```

## Test Status

`pytest` passes: 11 tests.

## Compiled Controller

The compiled hard-attention controller passed the bounded exhaustive check for both initial states through length 6:

- sequences: 2,222,220
- tokens: 13,086,420
- failures: 0
- elapsed: about 1.0 s on CPU

The reachable transition check passed for fp32 and fp64: 40 checks, no failures. This is every `(initial realization, previous state, input)` case used to realize the 20 transition-table entries.

Additional compiled scenarios were exact with zero illegal transitions:

- deadband OFF length 1536
- deadband ON length 1536
- repeated threshold crossings length 1000
- near thresholds length 1024
- long random length 10000

This is bounded verification of the implementation, not a proof for arbitrary sequence length or arbitrary floating-point/compiler settings.

## Learned Transformer

Architecture/training: 2 causal transformer layers, `d_model=32`, 2 heads, `d_ff=64`, train length 16, batch size 128, seed 7, 800 steps. Training took about 6.6 s on CPU. Final training batch accuracy was 0.9985.

Despite that, exhaustive length <= 4 found a failure from initial OFF:

```text
inputs:   5 6 4 5
expected: O O O O
actual:   O O O N
```

That is an illegal deadband transition. Initial ON length <= 4 happened to pass. Longer/adversarial traces degraded sharply: deadband OFF token accuracy 0.698, repeated threshold crossings 0.700, near thresholds 0.781, and long random length 512 0.793.

## Hybrid

Hybrid setup: learned scalar level classifier -> discrete symbol -> compiled hysteresis controller. Compared against a learned end-to-end continuous transformer.

Key result: the hybrid had zero illegal transitions relative to its predicted discrete belief in every scenario, even when the classifier was wrong. It still made mistakes relative to the true latent levels when perception failed.

Scenario summary:

| scenario | classifier acc | hybrid state acc | e2e state acc | hybrid illegal vs belief | hybrid illegal vs true | e2e illegal vs true |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| in distribution | 0.797 | 0.978 | 0.847 | 0 | 2046 | 18842 |
| shifted noise | 0.482 | 0.915 | 0.826 | 0 | 16716 | 44936 |
| near thresholds | 0.519 | 0.904 | 0.792 | 0 | 18398 | 53294 |
| long duration | 0.639 | 0.940 | 0.809 | 0 | 11361 | 47754 |

The hybrid result supports the intended distinction: perception errors can corrupt beliefs, but the compiled controller constrains how those beliefs can change state. The end-to-end transformer has no such structural invariant and produced many transition violations.

## Ruling

```text
B. It works, but only in a weaker/awkward sense.
```

The synthesized transformer machinery implements the desired hysteresis semantics exactly within the tested finite bounds and explicit assumptions: discrete inputs, explicit initial-state token, hard causal argmax attention, deterministic event/state tables, and greedy OFF/ON decoding.

It is not outcome A because the exactness primitive is hard attention/argmax rather than an ordinary finite-temperature softmax-only transformer. A softmax variant is included as a margin-stability check in tests, but the strong compiled result relies on hard selection.
