# Notes

## 1. What Torchwright/Torchdoom Demonstrates

Torchwright is a compiler that takes a Python-defined computation graph and
constructs transformer weights directly, with no gradient training. Its author
first describes the compiler as allocating graph values into residual-stream
subspaces, using attention and FFN primitives for operations such as equality,
selection, table lookup, and sequence lookup. The later Torchdoom experiment
uses that machinery to compile a substantial part of Doom's renderer into a
stock decoder-only Phi-3 style Hugging Face checkpoint.

Torchdoom is not a learned imitation of rendered frames. The prompt contains
scene data and player state; generated tokens encode drawing commands and
intermediate records. The host program mechanically replays a small set of draw
commands. The key state representation is append-only history: instead of
mutating Doom data structures, the transformer emits records, and later steps
use attention to retrieve the most recent matching record or aggregate over all
matching records. This is the relevant idea for hysteresis.

Sources read:

- Robert Porter, "Introducing torchwright", Out of Distribution, July 2026:
  https://ood.dev/posts/torchwright-intro/
- Robert Porter, "Doom, compiled into a transformer", Out of Distribution,
  August 2026: https://ood.dev/posts/doom/
- Torchdoom source repository: https://github.com/physicsrob/torchwright_doom

## 2. Can Exact Finite-State Behavior Be Synthesized Similarly?

For this hysteresis machine, yes, in a small and technically sensible way. The
state after time `t` is equivalent to the latest threshold event in the prefix:

- inputs `0..3` are RESET events, selecting OFF;
- inputs `7..9` are SET events, selecting ON;
- inputs `4..6` are neutral and should not change the state;
- if no event has appeared, the initial state token supplies the answer.

That can be lowered into:

1. a deterministic token-to-event classifier;
2. causal attention that selects the most recent non-neutral event in the
   prefix;
3. a deterministic event-to-state output projection.

This is close to the append-only-history pattern in Torchdoom, but vastly
smaller. It is also close in spirit to RASP/Tracr, where selectors and
aggregates are compiled into attention heads and sequence operations are
represented in residual-stream subspaces.

Relevant prior art:

- Weiss, Goldberg, and Yahav, "Thinking Like Transformers" / RASP, ICML 2021:
  https://arxiv.org/abs/2106.06981
- Lindner et al., "Tracr: Compiled Transformers as a Laboratory for
  Interpretability", NeurIPS 2023: https://arxiv.org/abs/2301.05062
- Tracr repository: https://github.com/google-deepmind/tracr

## 3. Simplest Persistent-State Representation

The simplest honest representation is autoregressive history plus an explicit
initial-state token. There is no mutable hidden state. Each input token is
classified as SET, RESET, or neutral, and the current state is recovered by
attending to the most recent SET/RESET record in the causal prefix.

This is simpler than an external recurrent state because it keeps the mechanism
inside a sequence model. It is also easier to test exhaustively than a learned
state vector because the reachable transition surface is only:

```text
2 previous states * 10 input symbols = 20 transitions
```

The implementation also supports sequences beginning in either initial state.

## 4. What "Exact" Can Honestly Mean

There are several levels:

- Exact reference semantics: the Python oracle over integers is exact.
- Exact compiled semantics under hard argmax attention: for finite integer input
  tokens, deterministic event tables, causal masks, and argmax selection, the
  compiled controller returns the same OFF/ON sequence as the reference. This is
  the strong claim tested here.
- Softmax approximation: a conventional softmax attention version can be made
  margin-stable over a bounded maximum length, but it is not mathematically
  exact. It relies on finite floating-point margins and argmax decoding.
- Low precision: fp16/bf16 should not be assumed exact unless separately tested.
  Torchwright's author also notes fp32 as the practical target for compiled
  models.

So the honest claim for this prototype is bounded exactness for discrete inputs
under hard attention/argmax and fp32/fp64 arithmetic. That is useful, but weaker
than claiming arbitrary-length exactness in a stock softmax-only checkpoint.

## 5. Smallest Architecture

The smallest demonstrator is not a large pretrained model. It is a tiny
decoder-style PyTorch module with:

- vocabulary: ten input symbols plus configurable initial state;
- one deterministic event-classifier embedding/table;
- one causal recency attention head selecting the latest SET/RESET event;
- one deterministic event-to-state output projection;
- greedy argmax decoding over OFF/ON logits.

The learned baseline is a one- or two-layer tiny causal transformer trained on
generated traces. The hybrid uses a learned scalar-observation classifier in
front of the compiled controller, then compares against an end-to-end learned
continuous transformer.

Fundamental blocker: none found. The caveat is architectural: the exact
compiled controller uses hard attention rather than a fully standard softmax
head. This makes the first-pass result a weaker but still meaningful version of
the hypothesis.
