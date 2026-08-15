# Projection-Loss Governance Results

Authoritative result directory for this pass:

```text
results/projection_context/
```

Calibration/smoke directories also exist, but the ruling below is based on the 8-seed context sweep.

## Reproduce

```bash
.venv/bin/python -m pytest tests/test_projection_channels.py tests/test_projection_impossibility.py tests/test_projection_baselines.py tests/test_latent_gate.py tests/test_interventions.py tests/test_projection_resume.py tests/test_projection_runner.py -q
.venv/bin/python -m experiments.run_projection_sweep --profile smoke --out-dir results/projection_context_smoke --force
.venv/bin/python -m experiments.run_projection_sweep --profile overnight --out-dir results/projection_context --force
.venv/bin/python -m pytest -q
```

Environment recorded by the run:

```text
schmittformer revision: cc355f12a83fa9bcc19f98790ab16e518dd9f26d
governance semantic digest: 1926060d9a20ff1f19d4e67d31c0c0cf4725a11b1a3f401419fd6c033333cf8c
Python: 3.12.3
PyTorch: 2.13.0+cpu
PyTorch CUDA available: false
GPU visible to nvidia-smi: NVIDIA GeForce RTX 5060 Ti, 16311 MiB, driver 570.211.01
```

The final sweep used 8 upstream seeds (`101..108`), context lengths `64, 256, 1024`, 18 projection/noise points, and 4 nuisance-shift conditions. Sum of per-seed elapsed times was about 2516 seconds on CPU, roughly 42 minutes.

Machine-readable artifacts:

```text
results/projection_context/manifest.json
results/projection_context/aggregate.json
results/projection_context/bayes_bounds.json
results/projection_context/interventions.json
results/projection_context/counterexamples.json
results/projection_context/runs/*.json
results/projection_context/checkpoints/*.pt
results/projection_context/figures/*.svg
```

## DERIVED / INFORMATION-THEORETIC

Complete erasure (`P0_COMPLETE_ERASURE`) creates true policy collisions. For example:

```text
proposal=REMEDIATE_A, witness=VALID,   scope=A -> ADMIT_A
proposal=REMEDIATE_A, witness=INVALID, scope=A -> REFUSE_INVALID_WITNESS
```

Both serialize identically when only proposal survives. No downstream function of that projection can implement both decisions. Under the experiment distribution, the Bayes-optimal error for `P0_COMPLETE_ERASURE` is 0.45.

Full trusted export (`P3_FULL_TRUSTED_EXPORT`) has Bayes-optimal error 0.0. The ordinary deterministic monitor with faithful trusted metadata is also exact. This is the main control: latent enforcement has no intrinsic advantage when an external monitor receives equivalent trusted information.

For noisy witness/scope export (`P1_NOISY_EXPORT`) at context 64, IID nuisance, Bayes error and observed token-boundary errors were:

| bit flip | Bayes error | token reference error | token learned error | token reference false admit |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.01 | 0.0135 | 0.0131 | 0.0144 | 0.0045 |
| 0.05 | 0.0664 | 0.0652 | 0.0668 | 0.0219 |
| 0.10 | 0.1305 | 0.1298 | 0.1328 | 0.0425 |
| 0.20 | 0.2520 | 0.2506 | 0.2561 | 0.0816 |
| 0.35 | 0.4174 | 0.4168 | 0.4338 | 0.1308 |
| 0.50 | 0.4500 | 0.5619 | 0.4498 | 0.1725 |

The token reference follows the observed noisy metadata, so it is not Bayes-optimal at 0.50; the learned token model learns to ignore uninformative bits and approaches the Bayes floor.

## CAUSALLY SUPPORTED

At the trained 64-token context, the final-layer synthesized latent gate was exact across all nuisance-shift conditions. Final-layer intervention results:

| intervention | synthesized consistency | learned latent consistency | proposal preservation | nuisance control effect | random control consistency |
| --- | ---: | ---: | ---: | ---: | ---: |
| witness | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| scope | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |

This is the cleanest positive result. The synthesized gate uses variable-aligned coordinates in the intended way, while the learned latent decision probe can be perfectly accurate yet fail this intervention test. Probe accuracy alone was not causal evidence.

## EMPIRICALLY OBSERVED

At context length 64 under IID nuisance:

| boundary | regime | accuracy | false admission rate | refusal reason accuracy |
| --- | --- | ---: | ---: | ---: |
| TOKEN_ONLY_REFERENCE | P0_COMPLETE_ERASURE | 0.1004 | 0.0000 | 0.1294 |
| TOKEN_LEARNED | P0_COMPLETE_ERASURE | 0.5508 | 0.0000 | 0.7114 |
| TOKEN_ONLY_REFERENCE | P3_FULL_TRUSTED_EXPORT | 1.0000 | 0.0000 | 1.0000 |
| TOKEN_PLUS_TRUSTED_METADATA | P0_COMPLETE_ERASURE | 1.0000 | 0.0000 | 1.0000 |
| TOKEN_LEARNED | P3_FULL_TRUSTED_EXPORT | 1.0000 | 0.0000 | 1.0000 |
| LATENT_SYNTHESIZED final layer | PRE_PROJECTION | 1.0000 | 0.0000 | 1.0000 |
| LATENT_LEARNED final layer | PRE_PROJECTION | 1.0000 | 0.0000 | 1.0000 |
| END_TO_END_LEARNED | FULL_CONTEXT | 1.0000 | 0.0000 | 1.0000 |

Layerwise synthesized performance at context 64 under IID nuisance:

| layer | policy accuracy | false admission rate | witness probe | scope probe |
| --- | ---: | ---: | ---: | ---: |
| embedding | 0.4244 | 0.2986 | 0.5019 | 0.5016 |
| layer_1 | 0.9664 | 0.0088 | 0.9617 | 1.0000 |
| layer_2 | 0.9938 | 0.0016 | 0.9922 | 1.0000 |
| layer_3 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| layer_4 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |

Nuisance-only export (`P5_SPURIOUS_EXPORT`) shows shortcut fragility for token-boundary learning:

| condition | TOKEN_LEARNED accuracy | TOKEN_LEARNED false admit | LATENT_SYNTHESIZED final accuracy |
| --- | ---: | ---: | ---: |
| IID | 0.7411 | 0.0743 | 1.0000 |
| WEAKENED_NUISANCE | 0.5052 | 0.1010 | 1.0000 |
| INDEPENDENT_NUISANCE | 0.4388 | 0.1083 | 1.0000 |
| REVERSED_NUISANCE | 0.1344 | 0.1384 | 1.0000 |

The important negative result is long-context degradation. The model and guards trained at length 64 did not preserve the latent policy abstraction at 256 or 1024:

| context | LATENT_SYNTHESIZED accuracy | LATENT_SYNTHESIZED false admit | LATENT_LEARNED accuracy | END_TO_END accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| 256 | 0.7861 | 0.0662 | 0.8113 | 0.8499 |
| 1024 | 0.6602 | 0.0820 | 0.6211 | 0.6250 |

That falsifies an unbounded latent-governance claim for this construction. The deterministic policy composition did not drift; the upstream representation presented to it stopped carrying stable decoded policy variables at longer contexts.

## NOT ESTABLISHED

- No claim is made that latent synthesized governance remains correct beyond the trained 64-token context. It does not in this run.
- The variable extraction part of `LATENT_SYNTHESIZED` is learned and empirical. Only the policy composition is deterministic.
- This does not show that a neural governance boundary is better than a conventional monitor with trusted metadata. The trusted metadata monitor is exact.
- This does not compile the 912-state governance kernel or its 35,568 transitions.
- This does not establish production suitability for latent enforcement.

## Ruling

Primary ruling: **B. Information-boundary success, latent synthesis weak**.

The information-boundary result is clean: lossy serialization provably destroys distinctions that downstream enforcement needs, and trusted metadata restores ordinary deterministic enforcement. At the trained context, the synthesized latent gate gives a causally supported pre-projection policy boundary and avoids the probe-accuracy trap.

The stronger neural claim fails under long-context evaluation in this pass. The policy-relevant latent variables are stable enough at length 64, but not at 256 or 1024. Secondary observations: `D. Metadata eliminates the advantage` applies when faithful metadata is exported, and `E. Probe/causal mismatch` applies to the learned latent decision probe despite perfect ordinary accuracy at context 64.
