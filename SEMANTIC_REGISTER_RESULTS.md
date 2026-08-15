# Semantic Register Results

This result set is under:

```text
results/semantic_register/
```

It was produced from the pushed latent-autopsy checkpoint:

```text
bb29471 projection: autopsy latent gate context drift
```

The previous projection and latent-autopsy rulings are preserved. This experiment does not rewrite them.

## Run Summary

```text
CUDA environment: .venv-cuda
GPU: NVIDIA GeForce RTX 5060 Ti, 16 GB
PyTorch: 2.11.0+cu128
CUDA runtime reported by torch: 12.8
seeds: 101..116
encodings: binary_pair, grouped_one_hot, joint_one_hot
training context: 64
evaluation contexts: 64, 256, 1024, 4096
training steps per run: 1500
completed runs: 48
```

Runtime receipt:

```text
full training+evaluation observed before metric refresh: 1549.37 s
metric refresh from checkpoints: 573.59 s
total including refresh: 2122.96 s, about 0.59 GPU-hours
```

A metric bug in the first aggregate omitted NOOP precedence from the register-relative checker. The checkpoints were not retrained; evaluations were recomputed from saved checkpoints after fixing the checker.

## Structural / By Construction

The fixed ABI makes these semantics structural:

```text
register coordinates/code points have fixed meanings
invalid code points refuse instead of silently selecting a policy state
valid register contents are decoded by a deterministic policy gate
internal synthesized gate and external metadata monitor agree when given the same register contents
```

Register-relative synthesized governance was exact in the aggregate:

```text
all encodings, all contexts, all nuisance regimes: 1.0000
```

Metadata equivalence was exact:

```text
192 internal-vs-external register comparisons
minimum exact match rate: 1.0000
maximum exact match rate: 1.0000
```

## Empirically Observed

IID world-relative performance for the synthesized register gate:

| encoding | ctx 64 | ctx 256 | ctx 1024 | ctx 4096 |
| --- | ---: | ---: | ---: | ---: |
| binary_pair | 1.0000 | 0.5319 | 0.4341 | 0.3125 |
| grouped_one_hot | 1.0000 | 0.4404 | 0.2959 | 0.2656 |
| joint_one_hot | 1.0000 | 0.3932 | 0.1694 | 0.2344 |

IID register writer joint accuracy:

| encoding | ctx 64 | ctx 256 | ctx 1024 | ctx 4096 |
| --- | ---: | ---: | ---: | ---: |
| binary_pair | 1.0000 | 0.4088 | 0.2549 | 0.1484 |
| grouped_one_hot | 1.0000 | 0.2859 | 0.1543 | 0.1250 |
| joint_one_hot | 1.0000 | 0.2628 | 0.0532 | 0.0938 |

For the best encoding here, `binary_pair`, IID comparison with learned heads:

| context | synthesized register gate | learned register gate | end-to-end learned head | register joint accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 1.0000 | 0.9016 | 1.0000 | 1.0000 |
| 256 | 0.5319 | 0.5103 | 0.8547 | 0.4088 |
| 1024 | 0.4341 | 0.3345 | 0.6250 | 0.2549 |
| 4096 | 0.3125 | 0.2578 | 0.6094 | 0.1484 |

Nuisance reversal did not explain the main failure. Binary-pair synthesized accuracy under reversed nuisance was similar:

```text
ctx 64:   1.0000
ctx 256:  0.5447
ctx 1024: 0.4307
ctx 4096: 0.2969
```

Position effects remained substantial. For binary-pair governance:

| context | scaled | fixed_absolute | fixed_distance |
| ---: | ---: | ---: | ---: |
| 64 | 1.0000 | 1.0000 | 0.8984 |
| 256 | 0.6563 | 0.7578 | 0.5130 |
| 1024 | 0.5781 | 0.6563 | 0.4714 |
| 4096 | 0.4896 | 0.5260 | 0.4609 |

This points to writer/context/position generalization failure, not a drifting gate coordinate system.

## Causally Supported

Register-level semantic interventions behaved as intended:

```text
witness intervention semantic consistency: 1.0000
scope intervention semantic consistency:   1.0000
nuisance intervention semantic consistency:1.0000
```

This held across the tested contexts and encodings. Random small register perturbations were less stable at long context because long-context writer outputs had smaller margins and were often closer to invalid or wrong code regions.

## Fault Injection

Detected as invalid and refused:

```text
zero vector
NaN
large out-of-domain vector
```

Valid but wrong code points are not detectable by this ABI alone:

```text
bit_flip_witness: valid alias rate 1.0000
bit_flip_scope:   valid alias rate 1.0000
partial stale scope A: valid alias rate 0.5000
```

This is expected. A valid code for the wrong semantic state is indistinguishable from a true state without provenance/freshness/redundancy mechanisms.

## Not Established

This experiment does not establish that a learned transformer can reliably write a correct semantic register outside its training context.

It does not establish that internal placement is superior to exporting trusted metadata. Given the same register contents, the external deterministic monitor is exactly equivalent.

It does not establish tamper resistance or protection against valid but stale/wrong register codes.

It does not establish that the end-to-end learned head is safer. At long context it can be more accurate world-relative than the register gate, but it lacks the register-relative structural guarantee.

## Primary Ruling

**B. Stable register, unreliable writer.**

The explicit register fixed the coordinate-interface problem for the governance gate. The decoder and policy semantics transferred directly across context and seed with no affine alignment. But the learned writer trained at context 64 did not reliably populate the register at 256, 1024, or 4096.

## Secondary Rulings

**D. Metadata-equivalent.** Faithfully exporting the same witness/scope state as trusted metadata makes an ordinary deterministic monitor fully equivalent for policy decisions.

**G, limited fault-model warning.** Normal invalid continuous corruptions are detected, but valid wrong code points silently alias real semantic states. That is not a bug in the decoder; it is a missing provenance/freshness/fault-tolerance layer.

## Answer To The Previous Failure Question

The emergent latent failure was mostly a failure to establish a stable interface for a recoverable semantic state. This experiment shows that a stable interface can be defined. However, defining the interface moves the hard problem to the learned writer: the model must still infer and maintain the correct semantic value at the interface.

## Next Kill Test

The next separate experiment should not try to rescue this writer by scaling. The stronger next test is:

```text
fixed-width recurrent governance state with actual bounded binary counters / leases
```

That asks whether governance state can advance for arbitrarily long logical traces without ever-growing context retrieval or final-token positional extrapolation.
