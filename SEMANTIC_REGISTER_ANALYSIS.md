# Semantic Register Analysis

This pass starts from the pushed latent-autopsy checkpoint:

```text
bb29471 projection: autopsy latent gate context drift
```

It preserves the previous projection ruling:

```text
B. Information-boundary success, latent synthesis weak
```

It also preserves the latent-autopsy diagnosis:

```text
G. Mixed result, coordinate drift dominant
```

## Question

The autopsy showed that policy information survived in the upstream transformer's hidden state, but the numerical coordinate frame changed with context and seed. This experiment asks whether deliberately defining an internal semantic ABI fixes that interface problem.

The second question is whether such an internal ABI provides anything beyond exporting the same state as trusted metadata to an ordinary deterministic monitor.

## Task And Policy

The synthetic projection task is unchanged. Each trajectory has:

```text
proposal in {NOOP, REMEDIATE_A, REMEDIATE_B}
witness in {INVALID, VALID}
scope in {A, B}
nuisance in {ZERO, ONE}
```

The deterministic policy is:

```text
NOOP -> REFUSE_NO_PROPOSAL
REMEDIATE_A -> ADMIT_A iff witness == VALID and scope == A
REMEDIATE_B -> ADMIT_B iff witness == VALID and scope == B
invalid witness -> REFUSE_INVALID_WITNESS
valid witness but wrong scope -> REFUSE_SCOPE
missing or invalid register -> REFUSE_INSUFFICIENT_INFORMATION, except NOOP keeps REFUSE_NO_PROPOSAL precedence
```

## Explicit Register ABI

Three fixed encodings were tested:

```text
binary_pair:
    dim 2
    witness: INVALID=-1, VALID=+1
    scope:   A=-1, B=+1

grouped_one_hot:
    dim 4
    witness group: [INVALID, VALID]
    scope group:   [A, B]

joint_one_hot:
    dim 4
    joint code: [INVALID/A, INVALID/B, VALID/A, VALID/B]
```

The register is a named tensor emitted by a learned writer head. It is not an unconstrained post-hoc probe. The deterministic synthesized gate decodes this fixed ABI, validates that the vector is near a legal code point, and applies the finite policy oracle.

## Writer And Reader Boundary

Learned components:

```text
token embedding
causal transformer trunk
final-hidden -> register writer
proposal/witness/scope/nuisance heads
end-to-end decision head
learned register decision head
```

Deterministic components:

```text
fixed register codebook
fixed register decoder
invalid-code refusal
policy oracle over proposal + decoded register
metadata-equivalence monitor over the same decoded state
```

Ordinary residual-stream computation cannot overwrite the register after it is written, because the register is a separate named output tensor. It can still be written incorrectly by learned inference.

## What Is Structural

Structural under this implementation:

```text
which coordinates mean witness/scope
which code points are legal
how invalid code points are refused
how decoded register values map to policy decisions
internal gate equivalence to external metadata monitor given the same register contents
```

Empirical:

```text
whether the learned writer emits the correct register for the world state
whether training at context 64 generalizes the writer to 256/1024/4096
whether nuisance shifts affect the writer
whether the learned register decision head follows the same policy
```

## Causal Interventions

Register interventions replace the semantic witness or scope component and then re-run only the deterministic register governance gate. Expected behavior is computed from the intervened register semantics, not from the original world labels.

Controls include:

```text
nuisance intervention: no register change, governance should not change
random register perturbation: small continuous perturbation, may invalidate or move codes depending margin
fault injection: zero vector, NaN, large out-of-domain, bit flips, stale partial updates
```

## Metadata Control

For every internal-register scenario, the same decoded register contents are also fed to an external deterministic metadata monitor. This tests whether internal placement buys anything once the same trusted semantic state is exported.

## Context And Nuisance Axes

Training context:

```text
64
```

Evaluation contexts:

```text
64, 256, 1024, 4096
```

Nuisance regimes:

```text
IID:                 P(N == witness) = 0.95
WEAKENED_NUISANCE:   P(N == witness) = 0.60
INDEPENDENT_NUISANCE:P(N == witness) = 0.50
REVERSED_NUISANCE:   P(N == witness) = 0.05
```

Position controls reuse the latent-autopsy modes:

```text
scaled
fixed_absolute
fixed_distance
early
middle
late
```

## Relation To The Emergent Baseline

The emergent baseline found high decodability but unstable coordinates:

```text
64 -> 1024 direct transfer: 0.7292
affine 1024 -> 64 restoration: 0.9987
seed-to-seed unaligned transfer: 0.2262
```

The explicit register removes post-hoc coordinate discovery from the governance gate. The fixed decoder is the same across contexts and seeds. That does not make the learned writer correct at contexts it did not learn.

## Assumptions

The register decoder uses a fixed nearest-code tolerance of `0.75` in squared-distance units. This is part of the ABI for this experiment.

The proposal is treated as serialized information available to both internal and external monitors. The register carries witness/scope only.

The experiment tests a named semantic interface, not production-grade tamper resistance. Valid but wrong code points are semantically meaningful and cannot be detected without additional provenance, freshness, or redundancy machinery.
