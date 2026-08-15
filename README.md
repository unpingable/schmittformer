# Stateful Control in Tiny Transformers

This repository is a small falsification-oriented prototype inspired by
Torchwright/Torchdoom.

Question:

> Can a deliberately synthesized transformer implement exact stateful control
> semantics such as hysteresis, and can that compiled mechanism be combined with
> learned transformer computation in a way that gives stronger behavioral
> guarantees than learning the entire behavior end-to-end?

The controller is:

```text
state in {OFF, ON}
input in {0, 1, ..., 9}

OFF -> ON   iff input >= 7
ON  -> OFF  iff input <= 3
otherwise retain state
```

The repository contains:

- `src/reference.py`: deterministic oracle state machine.
- `src/learned.py`: tiny trained causal transformer baseline.
- `src/compiled.py`: deterministic synthesized transformer-like controller.
- `src/hybrid.py`: learned noisy-level classifier plus compiled controller, and
  a learned end-to-end continuous transformer comparison.
- `experiments/run_all.py`: saves machine-readable results under `results/`.
- `tests/`: focused tests for the oracle, compiled controller, exhaustive
  bounded checks, and hybrid invariants.

## Setup

Use a local virtual environment. The checked-in requirements pin the CPU PyTorch
wheel used for the first pass because the CUDA wheel download was interrupted on
this host. The code itself uses CUDA automatically if you install a CUDA-enabled
PyTorch build.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

## Run Tests

```bash
.venv/bin/python -m pytest
```

## Run Experiments

```bash
.venv/bin/python -m experiments.run_all --compiled-max-len 6
```

For a faster smoke run:

```bash
.venv/bin/python -m experiments.run_all \
  --compiled-max-len 4 \
  --learned-steps 200 \
  --hybrid-steps 200 \
  --classifier-steps 200
```

Results are written as JSON files in `results/`. `RESULTS.md` summarizes the
first serious pass.


## Circuit Breaker Experiment

The second experiment adds a richer finite-state controller with a failure window, cooldown counter, half-open recovery probes, and history-equivalence checks. See `CIRCUIT_RESULTS.md` and run:

```bash
.venv/bin/python -m experiments.run_circuit --natural-steps 1500 --balanced-steps 1200 --classifier-steps 1000 --e2e-steps 1500 --train-len 64
```

## Current Ruling

The intended ruling for this first pass is **B**, not **A**.

The synthesized mechanism implements the hysteresis semantics exactly under the
stated bounded, discrete, hard-argmax assumptions. It is weaker than a fully
standard softmax-only transformer checkpoint because the compiled controller
uses explicit hard attention/argmax selection as its exactness primitive. The
experiment is still useful because the state update is executed inside the
synthesized PyTorch module by attention over the token history, not by a Python
state-machine loop hidden around the model.
