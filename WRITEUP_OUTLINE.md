# Writeup Outline

This is a short outline for a possible paper or technical report. It is not the paper.

## 1. Motivation

Learned sequence models can look competent while failing rare stateful transitions. Temporal governance rules have forbidden transitions, refusal reasons, leases, budgets, and settlement semantics that are poorly summarized by aggregate accuracy.

## 2. Synthesized Baseline

Start with hysteresis and a circuit breaker. Show why current-input classifiers are insufficient, why history-equivalence matters, and how hard-attention state-token machinery gives the first exact decoded result.

## 3. Finite Softmax And Decoded Exactness

Move from hard attention to ordinary finite-temperature softmax. Derive leakage bounds and distinguish latent exactness from decoded semantic exactness.

## 4. Governance Semantics As The Object

Extract an implementation-neutral governance semantic core from AG-ng. Emphasize that the semantics are useful independently of any transformer backend.

## 5. Information Boundaries

Use the projection-loss experiment to show that downstream monitors are bounded by serialized information. Trusted metadata restores deterministic external enforcement.

## 6. Representation Boundaries

Analyze emergent latent gates. Show coordinate drift across context/seed, probe/causal mismatch, and why learned decodability is weaker than a stable semantic ABI.

## 7. Explicit Semantic Registers

Introduce an engineered internal semantic register. Separate writer accuracy from register-relative governance correctness and compare with equivalent trusted metadata.

## 8. Fixed-Width Recurrent State

Replace growing history retrieval with fixed state plus current event. Implement real `uint16` and `uint8` counters using structural bitwise arithmetic.

## 9. Mechanical Equivalence

Use SMT to prove full bounded one-step logical transition equivalence between the reference semantics and compiled logical circuit.

## 10. Stock Transformer Closure

Lower the Boolean/arithmetic circuit into ordinary finite-softmax attention and generated standard `Linear/ReLU` FFNs. Report closure metrics and the model-size/performance caveat.

## 11. Discussion

The important object is the semantic interface and explicit state boundary. The transformer realization closes a research asterisk, but production governance should remain deterministic and provenance-aware.

## Related Work Buckets

- RASP / Tracr / transformer-program compilation
- transformers and automata / formal languages
- hard-to-soft attention simulation
- transformer-program verification
- neuro-symbolic and compiled neural rules
- reference monitors, shield synthesis, runtime governance
- causal abstraction and interchange interventions
- Torchwright / Torchdoom
