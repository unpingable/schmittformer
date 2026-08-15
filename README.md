# Schmittformer

Schmittformer is an experimental compiler/synthesis project exploring whether explicit temporal and governance semantics can be realized inside transformer computation rather than learned from examples.

The final prototype synthesizes a fixed-width recurrent governance machine into ordinary finite-temperature softmax attention and generated standard `Linear/ReLU` transformer FFNs. The machine includes real `uint16` and `uint8` bounded counters, uses a constant 43-slot physical input, and preserves decoded semantics under explicit numerical and discrete-latch assumptions.

This is deliberately inefficient research machinery. It is not a production governance runtime.

## What This Is

Schmittformer is a falsification-oriented research repository. It asks whether learned neural systems can contain deliberately synthesized regions whose temporal/control semantics are structural, testable, and separable from learned perception or proposal generation.

The project proceeds through small controllers, finite-softmax margin tests, an implementation-neutral governance semantic core, projection-loss experiments, latent representation autopsies, explicit semantic registers, fixed-width recurrent counters, SMT equivalence, and a final stock-transformer closure.

## What This Is Not

Schmittformer is not:

- a claim that compiling programs into transformers is new;
- a claim that transformers are a sensible production substrate for governance;
- a complete implementation of AG-ng governance doctrine;
- a solution to epistemic correctness or perception errors;
- evidence that latent probes imply causal semantic variables;
- a raw continuous recurrent system with unbounded analog stability.

For production governance, conventional typed state and deterministic monitors remain the default recommendation.

## Headline Result

The final closure experiment implements this recurrent transducer:

```text
fixed governance state + current event
                |
                v
      finite-softmax slot routing
                |
                v
   synthesized transformer FFN circuit
                |
                v
       next-state/output logits
                |
                v
       canonical discrete latch
                |
                +------> next invocation
```

The logical trace length can grow while the physical model input remains fixed. The canonical discrete latch is part of the guarantee: every step decodes the next state bits and re-encodes a clean state for the next invocation.

Final stock-transformer closure metrics:

```text
dec8:                  256 / 256 exhaustive, 0 failures
dec16:                 65,536 / 65,536 exhaustive, 0 failures
edge transitions:      10,584, 0 failures
adversarial transitions: 21,168, 0 failures
random transitions:    200,000, 0 failures
stock recurrent trace: 100,000 logical steps, fixed 43-slot input, 0 divergences
SMT:                   full bounded logical transition equivalence UNSAT
```

Stock architecture:

```text
torch.nn.MultiheadAttention
+ generated Linear/ReLU/Linear FFNs
118 FFN blocks
max FFN width 743
29,835,345 parameters
~119.5 MB generated checkpoint
```

Finite-softmax boundary:

```text
gaps 2 / 4 / 6: semantic failures
gap 8: non-saturated finite-softmax success
```

Important caveat: raw analog carry failed at step 5 in the earlier recurrent-softmax audit. Schmittformer establishes decoded semantic exactness with canonical discrete decode/re-encode, not stable unbounded analog recurrence.

## Why This Exists

Stateful policies fail in ways that aggregate sequence accuracy hides. Hysteresis, circuit breakers, leases, budgets, and settlement rules have rare transitions, forbidden transitions, and history-equivalence properties. The project asks whether those semantics can be specified, synthesized, and checked independently of learned components.

The practical question is not “can a transformer count?” The practical question is where the semantic boundary lives and what information is available there.

## Results At A Glance

| Experiment | Question | Result | Strongest claim | Main limitation |
| --- | --- | --- | --- | --- |
| Hysteresis | Can deadband control be synthesized instead of learned? | yes | compiled controller exactly matches bounded exhaustive checks | first version used hard attention |
| Circuit breaker | Does richer temporal state survive? | yes | 48 reachable states, 144 transitions, history equivalence held | hard latest-state retrieval |
| Finite-softmax margin | Is hard attention essential? | no for tested controllers | decoded exactness under explicit softmax margins | bounded numerical/position assumptions |
| Governance semantic core | Can AG-ng-like doctrine be extracted implementation-neutrally? | yes | 912 reachable states, 35,568 transitions, conformance corpus and digest | finite reduction omits production concerns |
| Projection loss | What if serialization erases policy state? | information boundary is real | complete erasure has Bayes error 0.45; trusted metadata restores exact monitoring | latent synthesis weakens at long context |
| Latent autopsy | Why did emergent latent gates fail at long context? | coordinate drift dominates | context-specific gates exact; affine alignment mostly restores transfer | emergent coordinates are not durable semantic ABI |
| Explicit semantic register | Can a deliberate ABI fix coordinate drift? | yes relative to register | register-relative governance exact across tested regimes | learned writer remains empirical failure source |
| Fixed-width recurrent counters | Can logical horizon detach from context length? | yes under hard/discrete slots | `uint16`/`uint8` counters, 1M logical steps, fixed 43-slot input | initially not finite softmax |
| Finite-softmax recurrent lowering | Can recurrent counters survive softmax retrieval? | bounded numerical success | gap 8 non-saturated success; canonical latch prevents drift | custom tensor circuit remained |
| SMT equivalence | Can one-step logic be mechanically checked? | yes | full valid bounded state/event disequality query UNSAT | does not model floating-point softmax |
| Stock transformer closure | Can the custom circuit lower into stock ops? | yes | ordinary MHA + generated `Linear/ReLU` FFNs execute the transition | large generated model; external latch remains |

See [RESULTS_INDEX.md](RESULTS_INDEX.md) for the full document/artifact map.

## Architecture

There are three different implementation styles in the repository:

- **Reference semantics:** ordinary deterministic Python oracles such as [src/recurrent_reference.py](src/recurrent_reference.py).
- **Custom synthesized circuits:** direct tensor circuits using exact Boolean-style operations, such as [src/recurrent_compiled.py](src/recurrent_compiled.py).
- **Stock transformer realization:** ordinary `MultiheadAttention` plus generated frozen `Linear/ReLU` FFNs in [src/stock_governance_transformer.py](src/stock_governance_transformer.py).

The final stock model receives only:

```text
state_t bits + event_t one-hot
```

and emits:

```text
state_(t+1) bits + governance-output logits
```

The recurrent runner may feed decoded state bits back into the next invocation. It may not use old history or hidden Python governance logic to compute the transition.

## Why The Discrete Latch Matters

Softmax attention and FFNs produce analog activations. Schmittformer does not require those activations to be identical to symbolic state. The target is decoded semantic exactness: after final bit/logit decoding, the abstract state and output match the reference semantics.

The canonical latch:

1. decodes the emitted state bits;
2. checks/uses the discrete state;
3. re-encodes a clean canonical state for the next step.

This prevents analog error from accumulating across recurrent invocations. Without that latch, raw analog carry failed quickly in the recurrent-softmax audit. Schmittformer is best understood as a synthesized sequential machine with explicit state latching, not as a continuous recurrent dynamical system.

## Research Arc

The experiments were ordered as falsification tests:

1. Could hysteresis be synthesized rather than learned?
2. Does the result survive richer temporal semantics?
3. Is hard attention essential?
4. Can governance semantics be extracted independently of transformer backend?
5. What happens when policy-relevant information is lost at serialization?
6. Are emergent latent policy coordinates stable?
7. Does an explicit semantic register fix coordinate drift?
8. Can state be recurrent and fixed-width instead of history-retrieved?
9. Can real binary counters be synthesized without FSM enumeration?
10. Can the logical machine be mechanically checked?
11. Can the entire transition execute through stock transformer operations?

The final answer is mixed but coherent: yes, explicit temporal/governance semantics can be compiled into transformer computation under assumptions; no, this does not make transformers the right production governance substrate.

## Quick Start

CPU/reference setup:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

Small demos:

```bash
.venv/bin/python examples/hysteresis_demo.py
.venv/bin/python examples/recurrent_demo.py
```

The recurrent demo constructs the stock transformer on CPU and prints a short governed trace with counter changes and margins.

## Reproducing The Experiments

Use [REPRODUCIBILITY.md](REPRODUCIBILITY.md). It separates:

```text
SMOKE                 cheap demos and tests
CORE                  main CPU/CUDA validation commands
FULL                  serious result regeneration
OPTIONAL / EXPENSIVE  long sweeps, CUDA closure, SMT
```

Environment split:

- [requirements.txt](requirements.txt): historical CPU/reference environment, `torch==2.13.0+cpu`.
- [requirements-cuda.txt](requirements-cuda.txt): CUDA experiment environment, `torch==2.11.0+cu128`.
- [requirements-solver.txt](requirements-solver.txt): optional Z3 dependency for SMT checks.
- [CUDA_ENVIRONMENT.md](CUDA_ENVIRONMENT.md): CUDA setup notes for the RTX 5060 Ti host.

## Repository Layout

```text
src/          reference implementations, synthesized circuits, stock transformer closure
experiments/  reproducible experiment runners
examples/     small human-facing demonstrations
tests/        unit, exhaustive, regression, and invariant tests
results/      committed machine-readable evidence; large generated checkpoint ignored
*.md          analyses, results, audits, and rulings
```

Start with:

- [SCHMITTFORMER_FINAL.md](SCHMITTFORMER_FINAL.md) for the overall synthesis;
- [STOCK_TRANSFORMER_RESULTS.md](STOCK_TRANSFORMER_RESULTS.md) for the final closure;
- [RESULTS_INDEX.md](RESULTS_INDEX.md) for the artifact map.

## Assumptions And Limitations

The strongest final claim assumes:

```text
fixed 43-slot state/event encoding
valid binary state bits and one-hot events
finite-softmax slot retrieval at score gap 8
synthesized weights unchanged
tested dtype/backend regime
final deterministic argmax/bit decoding
canonical decode/re-encode latch between recurrent invocations
valid enum state/event domain for SMT equivalence
```

It does not establish backend-independent floating-point proof for arbitrary kernels. It does not establish raw analog recurrence. It does not establish that internal governance beats an external deterministic monitor with equivalent trusted metadata.

## Prior Art

See [PRIOR_ART.md](PRIOR_ART.md). The short version: Schmittformer does not claim to invent transformer-program synthesis, automata in transformers, softmax simulation of hard attention, reference monitors, or shielded policies. The interesting delta is applying these ideas to assurance-oriented temporal/governance boundaries and carrying the experiment through information-boundary, representation-stability, recurrent-state, SMT, and stock-transformer closure tests.

## Why Not Just Use Rust?

For production governance, you probably should use ordinary code.

The useful practical artifacts here are explicit semantics, conformance vectors, state/interface design lessons, and projection-boundary analysis. Matrix multiplication is not the sensible way to decrement a lease counter in production. It is useful here because it tests what it means for a transformer computation to contain structural semantics rather than merely learn to imitate them.

## Constellation / Practical Implications

The practical result is the implementation-neutral governance semantic core and conformance corpus:

```text
typed policy state
+ authenticated/provenance-aware transport
+ deterministic governance/reference monitor
+ conformance vectors
+ receipts
```

Do not deploy the stock transformer governance machine as the production enforcement boundary. The transformer backend is research-only.

## License And Citation

No explicit license is currently present. Normal copyright defaults apply until the repository owner chooses one.

Citation metadata is in [CITATION.cff](CITATION.cff). It intentionally does not invent a DOI, publication venue, or formal paper title beyond this repository checkpoint.

## Status

Research checkpoint complete: `schmittformer-research-v0` at commit `3568392`.

Recommended next action: **write up**, not feature accumulation. Further experiments should only happen if they test a genuinely new boundary.
