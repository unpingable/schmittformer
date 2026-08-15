# Prior Art Hygiene

Schmittformer does not claim to invent program synthesis for transformers, automata in transformers, finite-softmax approximations to hard attention, reference monitors, or governance shielding. This file situates the project enough to keep the claim narrow.

## Transformer Programs And Compilation

**RASP.** RASP-style work treats transformers as a programming substrate with sequence operations such as selection and aggregation. Schmittformer inherits the idea that some transformer computations are better understood as programs than as learned black boxes.

**Tracr.** Tracr compiles RASP programs into transformer weights. It is one of the closest conceptual ancestors: Schmittformer similarly synthesizes weights/operations rather than training them, but focuses on temporal control/governance semantics and explicit decoded invariants.

**C-RASP.** C-RASP and related work extend the expressivity and compilation story for transformer-like programs. It is relevant because Schmittformer repeatedly asks which parts of the computation need attention, feed-forward circuits, or explicit state.

**ALTA and transformer-program analysis.** Work on analyzing learned transformers as algorithmic programs is adjacent. Schmittformer is mostly constructive rather than interpretive: it deliberately builds the program and then tests/characterizes it.

## Transformers, Automata, And State Tracking

**Transformers and finite automata.** There is extensive work showing that transformers can recognize or simulate formal languages and automata under different positional, precision, and attention assumptions. Schmittformer is not a universality claim; it asks what guarantees survive in a concrete synthesized governance setting.

**Flip-Flop/state-tracking language modeling.** Synthetic state-tracking tasks such as flip-flop languages are relevant because they separate sequence accuracy from persistent latent state. Schmittformer’s projection and latent-autopsy experiments make a similar distinction between decodability, causal control, and stable semantic coordinates.

## Hard Attention, Soft Attention, And Numerical Margins

**Hard-attention to soft-attention simulation.** Prior work already studies when soft attention can approximate or simulate hard selection. Schmittformer’s finite-softmax experiments are not a discovery that softmax can approximate hardmax; the useful part is the decoded semantic margin analysis and the explicit failure boundary when leakage is too large.

**Transformer-program verification.** Verification-oriented transformer work is relevant to the SMT and bounded-margin parts of this repository. Schmittformer separates logical circuit equivalence from floating-point execution evidence.

## Compiled Neural Rules And Neuro-Symbolic Systems

**Compiled neural rule systems.** CoNN-style and broader neuro-symbolic work compiles rules or constraints into neural structures. Schmittformer is in that neighborhood, but its focus is sequential operational semantics, canonical state latches, and policy admissibility rather than generic rule injection.

## Governance, Runtime Enforcement, And Shielding

**Reference monitors and policy enforcement.** Runtime monitors, capability systems, and policy engines long predate this project. The practical governance conclusion here aligns with that tradition: production systems should prefer explicit typed state and deterministic monitors.

**Shield synthesis / constrained policies.** Shielding in control and reinforcement learning constrains actions to a safe language or transition relation. Schmittformer’s governance kernel is analogous in spirit: arbitrary upstream proposals are filtered through an independently specified admissible transition language.

**Agent runtime governance.** Agent systems often need authority, evidence, lease, budget, refusal, and settlement semantics. Schmittformer’s governance semantic core is a small finite extraction from AG-ng-like doctrine, not a complete operational governance platform.

## Representation And Causal Abstraction

**Causal abstraction and interchange interventions.** The projection-loss and latent-autopsy experiments rely on the distinction between statistical decodability and causal semantic control. A probe can decode a variable without identifying a stable intervention target.

**Semantic interfaces / registers.** The explicit-register experiment is related to architectural interface design: if a learned model writes a named semantic register, deterministic downstream governance can be exact relative to that register even when the learned writer is wrong about the world.

## Immediate Inspiration

**Torchwright / Torchdoom.** Torchwright/Torchdoom was the immediate practical inspiration for constructing computation directly in PyTorch transformer-like weights. Schmittformer applies that style to tiny temporal controllers, governance semantics, recurrent counters, and stock-transformer closure.

## Narrow Claim

A cautious statement of Schmittformer’s contribution is:

> Existing transformer-program synthesis ideas can be used to construct assurance-oriented temporal/governance kernels whose decoded operational transitions are structural rather than learned, while exposing exactly where information boundaries, representation drift, numerical margins, and explicit state latches matter.

That is different from claiming novelty for transformer expressivity, softmax approximation, or runtime governance itself.
