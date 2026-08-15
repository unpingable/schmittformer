# Projection-Loss Governance Analysis

This research-track pass does not modify the governance semantic core. It uses the current semantic digest as context:

```text
1926060d9a20ff1f19d4e67d31c0c0cf4725a11b1a3f401419fd6c033333cf8c
```

The experiment asks where a governance boundary can enforce a policy when policy-relevant information exists before serialization but is partly or completely absent afterward. The policy is synthetic, not a reduction of the full 912-state governance kernel. It preserves one source-derived distinction: a proposal is not evidence. A remediation proposal is admissible only when separate witness and scope facts are available.

## Task

Each trajectory contains a temporal record:

```text
witness in {VALID, INVALID}
scope in {A, B}
nuisance in {ZERO, ONE}
proposal in {NOOP, REMEDIATE_A, REMEDIATE_B}
```

The witness appears early in the sequence, scope appears later, nuisance events are interleaved, and the proposal appears at the final token. The upstream model is a causal transformer trained at context length 64 and evaluated at lengths 64, 256, and 1024.

The nuisance variable is correlated with witness during training. Evaluation uses four correlations:

```text
IID: 0.95
WEAKENED_NUISANCE: 0.60
INDEPENDENT_NUISANCE: 0.50
REVERSED_NUISANCE: 0.05
```

## Policy Oracle

The deterministic oracle is:

```text
NOOP -> REFUSE_NO_PROPOSAL
REMEDIATE_A -> ADMIT_A iff witness == VALID and scope == A
REMEDIATE_B -> ADMIT_B iff witness == VALID and scope == B
otherwise invalid witness wins before scope mismatch:
    REFUSE_INVALID_WITNESS
    REFUSE_SCOPE
```

The explicit output alphabet is:

```text
ADMIT_A
ADMIT_B
REFUSE_NO_PROPOSAL
REFUSE_INVALID_WITNESS
REFUSE_SCOPE
REFUSE_INSUFFICIENT_INFORMATION
```

`REFUSE_INSUFFICIENT_INFORMATION` is used by conservative token-boundary monitors when serialization omits required policy inputs. It is not a true-world oracle output; it is an enforcement-boundary refusal.

## Projection Channels

The projection regimes are finite channels, implemented in `src/projection_channels.py`:

```text
P0_COMPLETE_ERASURE:
    exports proposal only. witness, scope, nuisance are absent.

P1_NOISY_EXPORT:
    exports proposal plus witness and scope through independent bit-flip channels.

P2_PARTIAL_EXPORT:
    exports proposal and witness. scope is absent.

P3_FULL_TRUSTED_EXPORT:
    exports proposal, witness, and scope faithfully.

P4_REDUNDANT_EXPORT:
    exports proposal plus three noisy witness carriers and three noisy scope carriers.

P5_SPURIOUS_EXPORT:
    exports proposal plus nuisance. witness and scope are absent.
```

The sweep uses bit-flip probabilities:

```text
0.00, 0.01, 0.05, 0.10, 0.20, 0.35, 0.50
```

## Information-Theoretic Bound

For each projection regime, the code enumerates all 24 latent states and the exact channel law. For projected value `z`, the Bayes-optimal downstream decision is the most probable policy decision under `P(decision | z)`. Therefore:

```text
accuracy* = sum_z max_d P(d, z)
error* = 1 - accuracy*
```

The same enumeration computes `I(policy_decision ; serialized_projection)` in bits. This is not an asymptotic estimate; it is exact for the finite generative distribution and channel model.

For complete erasure, there are states with identical projection and different required decisions. Example:

```text
proposal=REMEDIATE_A, witness=VALID,   scope=A -> ADMIT_A
proposal=REMEDIATE_A, witness=INVALID, scope=A -> REFUSE_INVALID_WITNESS
```

Both serialize to the same proposal-only record under `P0_COMPLETE_ERASURE`. No downstream function of that serialized record can be correct for both states. This is an information-limited boundary, not a model failure.

## Models And Boundaries

The upstream model is `ProjectionCausalTransformer` in `src/projection_model.py`:

```text
causal transformer
d_model=96 in the main run
n_layers=4
n_heads=4
d_ff=192
training length=64
max evaluated length=1024
auxiliary heads: proposal, witness, scope, nuisance, decision
sinusoidal positions for variable-length evaluation
```

Compared boundaries:

```text
TOKEN_ONLY_REFERENCE:
    deterministic monitor over serialized projection only. It refuses when required metadata is absent.

TOKEN_PLUS_TRUSTED_METADATA:
    deterministic oracle over projection plus trusted witness/scope metadata.

TOKEN_LEARNED:
    learned MLP over serialized projection features only.

LATENT_LEARNED:
    learned linear decision probe over a frozen upstream hidden representation.

LATENT_SYNTHESIZED:
    learned variable decoders for proposal/witness/scope, followed by the deterministic policy oracle.
    The policy composition is synthesized; variable extraction is empirical.

END_TO_END_LEARNED:
    upstream transformer's learned decision head over the full temporal input.
```

The synthesized latent gate should not be read as a production reference monitor. It is a pre-projection research boundary: it can only help when the policy-relevant latent coordinates are available and stable.

## Latent Alignment

For each layer, the run trains variable probes for proposal, witness, scope, and nuisance. The synthesized gate composes decoded proposal/witness/scope with the deterministic policy. This avoids training a policy classifier and then calling it synthesized, but it does not make variable identification formal. The learned variable decoders remain empirical components.

This distinction is load-bearing:

```text
learned extraction of policy variables
        +
deterministic policy composition
        =
LATENT_SYNTHESIZED in this experiment
```

## Causal Interventions

The intervention test constructs matched examples with the same proposal and controlled nuisance. For witness interventions:

```text
A: proposal=REMEDIATE_A/B, witness=VALID,   matching scope -> admit
B: proposal=REMEDIATE_A/B, witness=INVALID, matching scope -> refuse
```

For scope interventions:

```text
A: proposal matches scope, witness=VALID -> admit
B: proposal mismatches scope, witness=VALID -> refuse
```

At each layer, the test takes the binary probe direction for witness or scope and swaps only the scalar component along that direction between paired representations. It then asks whether the governance decision changes to the paired example's policy decision while proposal decoding stays fixed.

Negative controls swap nuisance direction and random direction. These should not reproduce the policy-specific intervention effect.

This is a causal-alignment check for the guard representation, not a proof that the whole upstream transformer implements a clean causal abstraction.

## What External Monitors Can And Cannot Do

A deterministic external monitor with trusted witness/scope metadata is fully sufficient for this finite policy. The latent mechanism has no special advantage over an ordinary monitor with equivalent information.

The difference appears only when the serialization boundary omits or corrupts policy-relevant information. In that case, a downstream monitor is bounded by the projection channel. A pre-projection mechanism can do better only if it has access to stable policy-relevant representation before the lossy channel.

## Assumptions

- The finite policy oracle is the intended semantics for this synthetic task.
- Serialization channels are trusted as specified by the experiment.
- `TOKEN_PLUS_TRUSTED_METADATA` receives faithful metadata, not model assertions.
- `LATENT_SYNTHESIZED` depends on empirical variable decoders; only the final policy composition is deterministic.
- Long-context evaluation uses sinusoidal positions, but training occurs at length 64.
- The installed PyTorch wheel is CPU-only, so the RTX 5060 Ti was visible to `nvidia-smi` but unavailable to PyTorch.

## Relation To Prior Work

This experiment does not claim that hidden states contain more information than output tokens, nor that transformers can implement finite policies. Those are settled points. The relevant connection is to causal abstraction and interchange intervention: probe accuracy alone is not enough. A variable is more credible as a policy coordinate when interventions on that coordinate produce the counterfactual policy effect and negative controls do not.

The experiment also separates ordinary reference monitoring from neural placement. If trusted metadata crosses the serialization boundary, a conventional deterministic monitor regains the same information as the latent gate. The interesting question is the information boundary, not whether a neural guard is intrinsically superior.
