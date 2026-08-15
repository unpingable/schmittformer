# Latent Governance Autopsy

This pass analyzes the pushed projection-loss result at:

```text
0defef1 projection: add lossy-boundary governance experiment
```

It does not modify or overwrite `PROJECTION_ANALYSIS.md`, `PROJECTION_RESULTS.md`, or `results/projection_context/*`. The original ruling remains:

```text
B. Information-boundary success, latent synthesis weak
```

The autopsy asks why a synthesized latent governance gate that was exact at context 64 failed at contexts 256 and 1024.

Authoritative autopsy artifacts:

```text
results/latent_autopsy/manifest.json
results/latent_autopsy/checkpoint_inventory.json
results/latent_autopsy/layer_metrics.json
results/latent_autopsy/context_alignment.json
results/latent_autopsy/intervention_metrics.json
results/latent_autopsy/gate_margins.json
results/latent_autopsy/representation_geometry.json
results/latent_autopsy/position_effects.json
results/latent_autopsy/seed_transfer.json
results/latent_autopsy/counterexamples.json
results/latent_autopsy/aggregate.json
results/latent_autopsy/figures/*.svg
```

## Observed Failure

The original projection experiment found:

```text
LATENT_SYNTHESIZED:
    context 64:   1.0000
    context 256:  0.7861
    context 1024: 0.6602

learned latent probe at context 64:
    ordinary accuracy = 1.0000
    final-layer intervention consistency = 0.0
```

The autopsy reuses the preserved upstream checkpoints. It does not retrain the upstream transformers. It fits only lightweight closed-form ridge linear readouts on frozen representations for diagnostics.

## Checkpoint Inventory

All eight original upstream checkpoints are present:

```text
results/projection_context/checkpoints/upstream_overnight_seed_101_L64_D96_S900.pt
...
results/projection_context/checkpoints/upstream_overnight_seed_108_L64_D96_S900.pt
```

The model architecture for every seed is:

```text
training context: 64
max context: 1024
d_model: 96
layers: 4 transformer layers plus embedding representation
heads: 4
d_ff: 192
training steps: 900
training nuisance correlation: 0.95
```

The saved training logs report final training accuracy of 1.0 for proposal, witness, scope, nuisance, and decision heads.

## Upstream Task Integrity

The upstream learned heads themselves degrade at longer contexts, even though the hidden representation remains linearly recoverable with context-specific probes. On sampled evaluation batches:

| context | proposal head | witness head | scope head | decision head | false admit |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| 256 | 1.0000 | 0.8735 | 0.7959 | 0.8350 | 0.0303 |
| 1024 | 1.0000 | 0.8008 | 0.5794 | 0.6458 | 0.1250 |

This is not pure information loss in the hidden state: context-specific linear readouts recover the policy variables exactly on the balanced autopsy set. It is, however, task-head extrapolation failure. The original final heads were trained at context 64 and do not remain reliable at longer contexts.

## Layerwise Decodability

Same-context closed-form linear readouts were fitted separately for each context and layer. Final-layer same-context results:

| context | synthesized accuracy | false admit | witness probe | scope probe | decision probe | synthesized margin mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 256 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 1024 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9996 |

Layerwise, embedding representations are not sufficient, but layer 1 and above are linearly decodable in this balanced diagnostic:

| context | embedding synth | layer_1 synth | layer_2 synth | layer_3 synth | layer_4 synth |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.5000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 256 | 0.5000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 1024 | 0.5000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Interpretation: the policy variables are still present in a linearly decodable form at long context, at least for the balanced finite state set. The context-64 gate fails because the 64-trained coordinate system does not transfer, not because no linear separator exists at 1024.

## Context Transfer

Final-layer synthesized gate transfer matrix:

```text
                 EVALUATE
BUILT AT        64      256     1024
64            1.0000  0.8437  0.7292
256           0.9792  1.0000  0.7604
1024          0.9062  0.9167  1.0000
```

The 64-built final-layer gate loses witness and especially scope accuracy at longer contexts:

| transfer | synthesized accuracy | witness accuracy | scope accuracy | margin mean |
| --- | ---: | ---: | ---: | ---: |
| 64 -> 64 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 64 -> 256 | 0.8437 | 0.8854 | 0.7552 | 0.5773 |
| 64 -> 1024 | 0.7292 | 0.7344 | 0.5417 | 0.2855 |

This is direct evidence of context-dependent coordinates. A gate built at 1024 works at 1024; the 64 gate does not.

## Seed Transfer

Final-layer context-64 gates do not transfer directly across independently trained seeds:

```text
diagonal same-seed accuracy:        1.0000
off-diagonal unaligned accuracy:   0.2262
off-diagonal affine aligned:       1.0000
off-diagonal Procrustes aligned:   1.0000
```

So the semantic variables are not naturally in a shared coordinate system across model instances. They can be aligned simply, but the alignment is per-model. A governance mechanism that depends on these coordinates must either be calibrated per checkpoint or the architecture must deliberately provide a shared register.

## Representation Geometry

Final-layer variable centroids become less compact by the single-variable grouping at long context. Between/within ratios for witness and scope shrink with context:

| context | witness between/within | scope between/within | decision between/within |
| ---: | ---: | ---: | ---: |
| 64 | 1.143 | 0.983 | 2.024 |
| 256 | 0.919 | 0.449 | 2.500 |
| 1024 | 0.502 | 0.186 | 3.019 |

This is not a compact semantic-region picture for `witness` or `scope` alone. The variables remain linearly decodable, but their simple class centroids are increasingly dominated by other factors. The representation is better described as a context-dependent, compositional code than as a stable scalar semantic register.

## Causal Interventions

Final-layer intervention consistency, using context-specific readouts:

| context | witness intervention | scope intervention | nuisance intervention target-preserving | learned-decision witness intervention | random control |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.8750 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 256 | 0.6875 | 1.0000 | 0.8125 | 0.1250 | 0.0000 |
| 1024 | 0.5625 | 1.0000 | 0.6875 | 0.1250 | 0.0000 |

The synthesized gate has some causal alignment, especially for scope, but witness localization weakens with context. Nuisance-direction interventions increasingly perturb synthesized decisions at long context, which is evidence that the variable directions are not cleanly independent semantic controls.

The learned decision probe remains the clearest probe illusion. It can be accurate but its decision output is not controlled by swapping the witness/scope variable direction. That preserves the original observation that ordinary probe accuracy overstated semantic usefulness.

## Gate-Margin Analysis

The 64-built final-layer gate's mean synthesized decision margin shrinks as context grows:

```text
64 -> 64:     1.0000
64 -> 256:    0.5773
64 -> 1024:   0.2855
```

Failure examples often appear well before 1024 when padding is minimized under the scaled-position generator. Examples saved in `counterexamples.json` include minimized failing lengths around:

```text
98, 129, 197, 216, 783, 833
```

So the issue is not a magic 1024-token cliff. It is a gradual coordinate/margin drift that can cross the fixed 64 decision boundary much earlier for some states and seeds.

## Position And Distance Effects

The 64-built final-layer gate is sensitive to where the witness and scope tokens appear when the context is extended:

| mode | ctx 64 | ctx 256 | ctx 1024 | interpretation |
| --- | ---: | ---: | ---: | --- |
| scaled | 1.0000 | 0.8437 | 0.7292 | original generator; both absolute position and distance change |
| fixed_absolute | 1.0000 | 0.9740 | 0.8750 | preserving early absolute positions helps |
| fixed_distance | 0.9271 | 0.6354 | 0.5729 | late absolute positions hurt despite short distance |
| early | 1.0000 | 0.9531 | 0.8281 | early evidence is more stable |
| middle | 0.9688 | 0.7865 | 0.5729 | mid-context placement drifts |
| late | 0.8281 | 0.6302 | 0.5729 | late absolute placement is worst |

This implicates positional/context coordinate drift more strongly than simple memory distance. Fixed absolute positions remain much better than fixed distance from the decision token.

## Counterexamples

`results/latent_autopsy/counterexamples.json` records representative failures. A typical failure is:

```text
seed: 102
expected: ADMIT_A
predicted: REFUSE_SCOPE
proposal: REMEDIATE_A
witness: VALID
scope: A
original length: 256
first failing coarse length: 128
minimized length if monotonic: 98
margin: -0.6436
```

This kind of example is a fixed 64-gate boundary error, not an oracle ambiguity. A context-specific gate at the same long context can classify the state correctly.

## Position Encoding Audit

The upstream model uses sinusoidal positional encodings with `max_len=1024`. Evaluation at 256 and 1024 is architecturally defined and numerically plausible. However, training occurred only at context 64. Therefore the long-context test is an extrapolation of the learned use of positional structure, not an undefined position-embedding lookup.

The evidence points to a positional/extrapolation contribution, but not a pure positional-encoding implementation bug. Fixed absolute placement improves accuracy substantially, and affine context alignment restores the gate, which suggests systematic representation drift rather than random numerical failure.

## CUDA Side Quest

The current environment remains CPU-only for PyTorch:

```text
PyTorch: 2.13.0+cpu
torch.cuda.is_available(): false
GPU visible to nvidia-smi: NVIDIA GeForce RTX 5060 Ti, 16 GB
```

Cause recorded in `environment.json`: `requirements.txt` pins `torch==2.13.0+cpu` from the PyTorch CPU wheel index. No CUDA environment was created in this pass because the autopsy was runnable on CPU and changing the environment would add reproducibility risk.

## Hypothesis Rulings

H1 - Coordinate drift: strongly supported. Context-specific gates are exact at long contexts, the 64 gate fails there, and affine alignment restores final-layer accuracy to about `0.9987` at 1024.

H2 - Representation redistribution: partly supported. Variable class centroids become less compact, nuisance interventions increasingly affect decisions, and witness intervention consistency declines. The information is not simply a stable scalar moved by a rigid transform.

H3 - Information loss: not supported as the main hidden-state diagnosis. Context-specific linear readouts recover witness, scope, and synthesized policy exactly in the balanced diagnostic. Upstream output heads do degrade, but the hidden representation still contains recoverable policy information.

H4 - Positional/extrapolation artifact: materially supported. The model uses sinusoidal positions, so long contexts are defined, but training was only length 64. Fixed absolute positions are much more stable than late/fixed-distance placements.

H5 - Probe illusion: supported for the learned decision probe. It can be accurate while intervention consistency remains near zero. The synthesized gate is less illusory because variable-direction interventions affect it, but witness causality weakens with context.

H6 - No stable semantic register: supported across seeds without alignment, and partly supported across contexts. The representation can be calibrated into a useful coordinate system, but no shared, naturally stable semantic register emerged.

## Primary Diagnosis

**G. Mixed result**, with coordinate drift as the dominant mechanism.

The context-64 representation was not a durable semantic register. It was a context-specific linear coordinate system over a representation that still contains the policy variables. At longer contexts, the hidden state can be re-read with a context-specific gate or mapped back with a simple affine alignment, so the information is not gone. But the original synthesized gate depends on coordinates that drift with context, position, and seed.

More compactly:

```text
The policy state survived.
The 64-coordinate readout did not.
The causal localization was weaker than the ordinary probe accuracy suggested.
```

## Implications

The 64-token success fooled us because it combined three facts that are not equivalent:

```text
information exists somewhere in the representation
information is linearly decodable
a particular learned coordinate is causally usable
that coordinate is stable across context
that coordinate is stable across seeds
a deterministic governance gate can rely on it
```

Only the first two were robustly true. The third was partially true. The fourth and fifth were false without alignment.

## NEXT RESEARCH

1. **Deliberately engineer an explicit semantic register.** This is the strongest next experiment. The autopsy suggests emergent coordinates are calibratable but not naturally stable. A reserved state token/subspace for witness/scope would test whether stability can be made structural.

2. **Test recurrent/fixed-width governance state.** A fixed-width state interface may avoid dependence on drifting full-context final-token geometry.

3. **Investigate better causal alignment methods.** Useful only if the goal remains latent enforcement over emergent representations. The current affine result says simple alignment helps, but it is still per-model/per-context calibration.

4. **Return to external trusted metadata/reference monitors when the metadata can be exported.** The previous projection result remains: trusted metadata makes ordinary deterministic governance exact.

5. **Do not pursue bigger emergent latent gates as the next step.** Scaling or retraining to improve the 1024 number would be a rescue attempt, not an autopsy.
