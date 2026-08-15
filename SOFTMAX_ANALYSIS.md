# Softmax Analysis

This note attacks the hard-attention caveat in the completed hysteresis and circuit-breaker experiments.

Historical baseline checkpoint:

```text
cd5a9867175c73ca186a8c4cb67d0338fbf474ff
hard-attention-success
```

That checkpoint preserves the prior result: circuit breaker has 48 reachable abstract states, 144 reachable transitions, 1152 compiled transition checks across 384 histories, 0 failures, 1680 history-equivalence comparisons, 0 violations, exact long traces through length 4096, and an induction-style representation-preservation argument under hard causal argmax attention.

## Exactness Distinction

Finite-temperature softmax should not be expected to reproduce hard retrieval latently.

```text
latent exactness:
    retrieved activation equals the hard-selected value exactly

decoded semantic exactness:
    retrieved activation contains bounded leakage, but the final decoded
    abstract state is identical to the reference controller state
```

The construction here targets decoded semantic exactness. Latent exactness generally fails and is measured as state-mass differences in `results/softmax_equivalence.json`.

## Simple Bound

Let the correct record score be `s*`. Let there be `N - 1` competitors, each with score at most `s* - Delta`.

Softmax probability on the correct record is:

```text
p* = exp(s*) / (exp(s*) + sum_i exp(s_i))
```

Since `s_i <= s* - Delta`:

```text
sum_i exp(s_i) <= (N - 1) exp(s* - Delta)
```

Therefore:

```text
p* >= exp(s*) / (exp(s*) + (N - 1) exp(s* - Delta))
   = 1 / (1 + (N - 1) exp(-Delta))
```

The total incorrect attention mass is at most:

```text
1 - p* <= ((N - 1) exp(-Delta)) / (1 + (N - 1) exp(-Delta))
```

This bound is verified numerically in `tests/test_softmax_attention.py` and serialized examples are in `results/softmax_theory.json`.

The drawback is that it grows with the number of obsolete records.

## Geometric Recency Bound

The softmax construction uses alternating autoregressive records:

```text
STATE(s0), INPUT(x1), STATE(s1), INPUT(x2), STATE(s2), ...
```

For a next-step query, the latest state record is one token before the current input. State-record scores are:

```text
score(token_i) = alpha * absolute_position_i - beta * is_non_state_i
```

Adjacent state records are two token positions apart. Define:

```text
Delta = 2 alpha
```

Then an obsolete state record `k` state updates older has score at most:

```text
s* - k Delta
```

The unnormalized stale-state mass is bounded by the geometric series:

```text
S <= sum_{k>=1} exp(-k Delta)
  = exp(-Delta) / (1 - exp(-Delta))
```

This is independent of sequence length in the real-score model.

## Decision Margin

State-token values are one-hot state vectors. Non-state input tokens have zero value. The downstream transition map is linear for a fixed current input:

```text
state_mass -> next_state_logits
```

Let the latest state be `s*`, current input be `x`, and correct next state be:

```text
y* = transition(s*, x)
```

After subtracting the latest-state score, the correct state record has unnormalized mass `1`. Let stale-state unnormalized mass be `S`, and non-state unnormalized mass be `U`. The denominator is:

```text
D = 1 + S + U
```

The correct next-state logit is at least:

```text
1 / D
```

A wrong next-state logit is at most:

```text
S / D
```

because only stale state records can vote for wrong abstract next states. Non-state tokens dilute all state logits but do not vote for any state.

Therefore decoded argmax is guaranteed correct if:

```text
S < 1
```

For the geometric stale bound, this is:

```text
exp(-Delta) / (1 - exp(-Delta)) < 1
exp(-Delta) < 1/2
Delta > ln 2
```

A lower bound on the correct-vs-runner-up margin is:

```text
margin >= (1 - S) / (1 + S + U)
```

For the main configuration:

```text
Delta = 2.0
beta = 4.0
S <= 0.1565176427
U <= 0.0575796229
margin >= about 0.6949
```

Observed minimum margins were close to this:

- hysteresis transition check: 0.6988
- circuit transition check: 0.6947

## Non-State Contamination

The current input has one-position advantage over the latest state, but also pays non-state penalty `beta`. Its gap below the latest state is:

```text
beta - alpha
```

Older input records then decay geometrically. The implementation records non-state mass instead of ignoring it. With `Delta=2`, `beta=4`, measured max non-state mass was about `0.0474`.

In exact arithmetic, non-state mass does not affect argmax correctness because non-state values are zero. In floating-point arithmetic, excessive non-state mass can reduce margins and produce ties or underflow/rounding problems. Low precision exposed this failure mode.

## Representation Preservation

The softmax construction resets analog error at every decoded state-token boundary:

1. A valid history ends in a clean discrete state token.
2. Softmax retrieval produces a mixed state-mass vector, not a clean latent state.
3. The transition layer maps that mixture to next-state logits.
4. Final argmax emits a clean discrete next-state token.
5. The next step attends to that clean token, not to the previous analog mixture.

Thus analog leakage does not accumulate across decoded steps, provided every step's argmax is correct.

## Induction Shape

Under real-score assumptions, the following supports arbitrary finite decoded traces:

1. Every abstract state has a clean state-token representation.
2. Every valid history has a latest state token denoting the current abstract state.
3. If `Delta > ln 2`, stale-state unnormalized mass is `< 1` for any number of older state records.
4. Therefore, for every current input, the correct next-state logit beats any wrong next-state logit.
5. Greedy decoding emits the correct clean next-state token.
6. The representation invariant is preserved, so induction continues.

This is a decoded semantic argument, not latent exactness.

## Finite Precision Boundary

The implemented score uses absolute positions. In low precision, large absolute positions lose enough resolution that adjacent positions and type penalties are no longer represented reliably.

Observed examples:

- `bfloat16`, `Delta=2`, hysteresis synthetic probe failed at 1024 updates: non-state mass reached `0.5` and the decision tied.
- `float16`, `Delta=2`, hysteresis synthetic probe failed at 8192 updates.
- `float32` and `float64` passed the synthetic sweep through 16384 updates at `Delta >= 0.7` without becoming numerically hard.

This means the mathematical leakage bound is context-independent, but the concrete absolute-position floating-point implementation is not context-independent for all dtypes.

## Saturation

Large gaps can make softmax numerically one-hot through underflow or rounding. The phase diagram labels these cases `EFFECTIVELY_HARD`.

Examples:

- `float32` became effectively hard by `Delta=32` in the sweep.
- `float64` remained non-saturated through `Delta=32` but became effectively hard by larger gaps in tested cases.
- Low precision saturated much earlier in some context regimes.

The main result does not rely on saturation: `Delta=2`, `beta=4` passed in float32/float64 with nonzero stale and non-state attention mass.

## Standardness Audit

- ordinary causal softmax attention: yes, one query over the causal prefix
- ordinary softmax, no hard max: yes
- ordinary linear score form: yes, scalar QK score equivalent to `[alpha, beta] dot [position, -is_non_state]`
- ordinary causal mask/history prefix: yes
- explicit handcrafted weights: yes
- explicit state tokens: yes
- deterministic transition lookup/linear projection: yes
- final argmax decode: yes
- custom hard attention: no
- straight-through estimator: no
- external recurrent state: no
- external interpreter for transition semantics: no; Python only runs the autoregressive decode loop and appends emitted tokens

The unusual parts are synthesized weights, explicit state tokens, and final symbolic decoding. The attention operation itself is ordinary finite-temperature softmax.
