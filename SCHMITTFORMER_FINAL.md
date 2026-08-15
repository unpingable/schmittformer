# Schmittformer Final Synthesis

Schmittformer started as a falsifiable toy experiment about compiled stateful transformer behavior and ended as a broader study of semantic boundaries: what should be learned, what should be explicit, and what can be structurally enforced.

## Research Arc

| step | question | result | strongest claim | main limitation |
| --- | --- | --- | --- | --- |
| 1. Hysteresis | Can a synthesized transformer implement stateful deadband control? | yes, hard-attention version exact in bounded tests | state is not just current input; compiled mechanism preserves hysteresis | first result used hard argmax attention |
| 2. Circuit breaker | Can richer temporal policy with windows/cooldown/recovery be compiled? | yes | 48 reachable states, 144 transitions, 0 compiled failures, history equivalence held | hard latest-state retrieval |
| 3. Finite softmax | Is hard attention conceptually essential? | no for tested controllers | finite-temperature non-saturated softmax gives exact decoded semantics under margins | bounded numerical/position assumptions |
| 4. Governance core | Can AG-ng doctrine yield an implementation-neutral finite semantics? | yes | 912 reachable states, 35,568 transitions, conformance corpus and digest | finite reduction omits production AG-ng concerns |
| 5. Projection loss | What happens when serialization erases policy state? | downstream enforcement hits an information limit | complete erasure has Bayes error 0.45; trusted metadata restores exact monitoring | latent synthesis weakened at long context |
| 6. Latent autopsy | Why did emergent latent gates fail at long context? | mostly coordinate drift | context-specific gates exact; affine alignment mostly restores transfer | emergent coordinates are not a durable ABI |
| 7. Explicit register | Can an engineered semantic ABI fix coordinate drift? | yes, relative to the register | register-relative governance exact; internal register equals trusted metadata for policy | learned writer remains empirical failure source |
| 8. Fixed recurrent state | Can logical horizon detach from context length? | yes under hard/discrete fixed slots | uint16/uint8 counters, 1M logical steps, fixed 43-slot input | not finite softmax at first |
| 9. Recurrent finite softmax + SMT | Can recurrent counters survive finite softmax? | bounded numerical success | gap 8 non-saturated success; full SMT transition equivalence UNSAT | custom tensor circuit remained |
| 10. Stock closure | Can the custom circuit lower into stock transformer ops? | yes | MHA + generated Linear/ReLU FFNs execute the transition; save/load works | 118 FFN blocks, 29.8M params, external latch remains |

## What Schmittformer Establishes

Under explicit assumptions, synthesized transformer computation can implement nontrivial stateful control/governance transitions with exact decoded behavior.

The strongest final machine has:

```text
fixed 29-bit governance state
uint16 lease counter
uint8 budget counter
14-event alphabet
ordinary finite-softmax slot retrieval
generated standard Linear/ReLU FFN transition circuit
ordinary saved PyTorch checkpoint
canonical decode/re-encode recurrent latch
```

It passed:

```text
dec8 exhaustive: 256 / 256
dec16 exhaustive: 65,536 / 65,536
edge transitions: 10,584 / 10,584
adversarial transitions: 21,168 / 21,168
random transitions: 200,000 / 200,000
100,000 recurrent logical steps at fixed 43-slot input
full prior SMT one-step equivalence over valid bounded states/events
```

## What It Does Not Establish

Schmittformer does not show that neural governance should replace ordinary deterministic governance. It does not prove floating-point correctness for every backend. It does not make raw analog recurrent carry safe; the recurrent guarantee uses a discrete latch. It does not show that emergent latent state is naturally stable across contexts or seeds. It does not claim novelty for compiling programs into transformers.

## What Is Prior Art

The general idea that transformer programs can implement algorithms or finite automata is prior art: RASP/Tracr-style compilation, transformer/FSA work, hard-to-soft attention simulation, and related neural rule systems all matter here. Torchwright/Torchdoom was the practical inspiration for hand-synthesized transformer computation.

## Interesting Delta

The useful delta is not "transformers are universal computers." It is:

```text
semantic policy can be treated as an explicit object,
compiled or tested independently of learned perception,
and placed at different information boundaries.
```

The projection and register experiments were especially clarifying. If serialization destroys policy-relevant information, no downstream monitor can recover it. If the same state is exported as trusted metadata, a boring deterministic monitor regains the same capability. Internal neural placement only matters when it has access to information or timing that the external boundary lacks.

## Constellation Value

The practical value is the governance semantic core:

```text
explicit typed state
deterministic transition semantics
conformance corpus
semantic digest
receipts/provenance hooks
```

The transformer backend is research. The production recommendation remains a deterministic reference monitor or generated conventional runtime, not neural arithmetic.

## What Should Not Be Deployed

Do not deploy the stock transformer counter circuit as a production governance boundary merely because it works. It is large, slow for single recurrent feedback, sensitive to explicit numerical assumptions, and inferior to ordinary typed code for operational governance.

## Final Judgment

Schmittformer was genuinely about transformers in the narrow computational-realization sense: the final transition executes through ordinary finite-softmax attention and standard Linear/ReLU modules with synthesized weights.

But the more important lesson is substrate-independent. The work repeatedly found that explicit semantic interfaces, canonical state boundaries, conformance tests, and information availability did the real assurance work. The transformer result closes a research asterisk; it does not overturn the practical architecture.

Recommended next action: **WRITE UP**, then stop active expansion. The marginal value of adding more toy controllers is now low. A future experiment should only proceed if it attacks a new boundary, not another variant of "compile this finite state machine too."
