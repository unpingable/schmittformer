# Reproducibility

This repository contains both cheap checks and expensive historical sweeps. Do not assume `pytest` alone reproduces every headline result; several exhaustive or adversarial checks are experiment scripts that write JSON artifacts.

The experimental closure checkpoint is:

```text
commit: 3568392 stock: close recurrent transformer realization
tag:    schmittformer-research-v0
```

## Environments

### CPU / Historical Reference

Used by early hysteresis, circuit, governance-core, projection, and ordinary tests.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

This pins `torch==2.13.0+cpu`.

### CUDA

Used by semantic-register, recurrent, recurrent-softmax, and stock-transformer closure runs.

```bash
python3 -m venv .venv-cuda
.venv-cuda/bin/python -m pip install --upgrade pip
.venv-cuda/bin/python -m pip install -r requirements-cuda.txt
.venv-cuda/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

See [CUDA_ENVIRONMENT.md](CUDA_ENVIRONMENT.md) for the recorded RTX 5060 Ti setup.

### Solver

Used for SMT transition equivalence.

```bash
python3 -m venv .venv-solver
.venv-solver/bin/python -m pip install --upgrade pip
.venv-solver/bin/python -m pip install -r requirements-solver.txt
```

The recorded solver was `z3-solver==5.0.0.0`, reporting Z3 5.0.0.

## Smoke

Cheap checks for a fresh reader.

```bash
.venv/bin/python examples/hysteresis_demo.py
.venv/bin/python examples/recurrent_demo.py
.venv/bin/python -m pytest tests/test_relu_boolean.py tests/test_ffn_counter.py tests/test_stock_governance_transformer.py -q
```

The recurrent demo constructs the final stock model on CPU and runs a small trace. It does not regenerate the full result set.

## Core

Run the full ordinary CPU test suite:

```bash
.venv/bin/python -m pytest -q
```

At closeout this reported:

```text
120 passed, 1 warning
```

The warning is the existing PyTorch nested-tensor warning from the learned circuit test.

Focused CUDA closure tests:

```bash
.venv-cuda/bin/python -m pytest tests/test_relu_boolean.py tests/test_ffn_counter.py tests/test_stock_governance_transformer.py -q
```

At closeout this reported:

```text
7 passed
```

Validate committed JSON artifacts:

```bash
.venv/bin/python -c "import json; from pathlib import Path; [json.loads(p.read_text()) for p in Path('results').glob('**/*.json')]; print('json ok')"
```

## Full / Serious Regeneration

These commands regenerate the main result directories. They may take minutes to hours depending on hardware.

### Hysteresis

```bash
.venv/bin/python -m experiments.run_all --compiled-max-len 6
```

### Circuit Breaker

```bash
.venv/bin/python -m experiments.run_circuit --natural-steps 1500 --balanced-steps 1200 --classifier-steps 1000 --e2e-steps 1500 --train-len 64
```

### Finite Softmax Controllers

```bash
.venv/bin/python -m experiments.run_softmax --hysteresis-max-len 6 --batch-size 8192
```

### Governance Semantic Core

```bash
.venv/bin/python experiments/run_governance.py --out-dir results
```

### Projection-Loss Sweep

```bash
.venv/bin/python -m experiments.run_projection_sweep --profile overnight --out-dir results/projection_context --force
```

### Latent Autopsy

```bash
.venv/bin/python -m experiments.run_latent_autopsy --out-dir results/latent_autopsy --force
```

### Explicit Semantic Register

```bash
.venv-cuda/bin/python -m experiments.run_semantic_register_sweep --out-dir results/semantic_register --device cuda --force
```

### Fixed Recurrent Governance

```bash
.venv-cuda/bin/python -m experiments.run_counter_verification --out-dir results/recurrent --device cuda --force
.venv-cuda/bin/python -m experiments.run_recurrent_governance --out-dir results/recurrent --device cuda --random-samples 200000 --max-long-steps 1000000 --force
```

### Recurrent Finite Softmax

```bash
.venv-cuda/bin/python -m experiments.run_recurrent_softmax --out-dir results/recurrent_softmax --device cuda --score-gap 8 --random-samples 200000 --max-long-steps 1000000 --force
```

### SMT Equivalence

```bash
.venv-solver/bin/python -m experiments.run_transition_solver --out results/recurrent_softmax/solver.json --timeout-ms 120000
```

### Stock Transformer Closure

```bash
.venv-cuda/bin/python -m experiments.run_stock_closure --out-dir results/stock_transformer --device cuda --random-samples 200000 --long-steps 10000 --force
```

A separate capped 100k recurrent stock trace was added during closeout:

```bash
.venv-cuda/bin/python -c "import json, torch; from pathlib import Path; from experiments.run_stock_closure import longrun_report, atomic_write_json; base=Path('results/stock_transformer'); extra=longrun_report(torch.device('cuda'), [100000]); current=json.loads((base/'longrun.json').read_text()); by_len={row['requested_logical_steps']: row for row in current['rows']}; [by_len.__setitem__(row['requested_logical_steps'], row) for row in extra['rows']]; current['rows']=[by_len[k] for k in sorted(by_len)]; atomic_write_json(base/'longrun.json', current)"
```

## Generated Checkpoint Policy

`results/stock_transformer/stock_governance.pt` is generated locally by the stock closure run and intentionally git-ignored because it is about 119.5 MB.

The committed artifact [results/stock_transformer/checkpoint.json](results/stock_transformer/checkpoint.json) records:

```text
checkpoint_size_bytes: 119,504,135
post-load validation: 2,048 / 2,048 passed
```

Regenerate it with the stock closure command above.

## What Was Exhaustive

Exhaustive checks include:

- `dec8`: all 256 values;
- `dec16`: all 65,536 values;
- finite reachable graphs for hysteresis/circuit/governance semantic core;
- SMT one-step logical equivalence over every valid bounded recurrent state/event.

## What Was Random Or Adversarial

Random/adversarial checks include:

- learned baselines and hybrid traces;
- projection/latent/register sweeps;
- recurrent governance random valid state/event batches;
- stock transformer closure random 200,000 transition set;
- adversarial edge states around leases, budgets, authority, occurrence, settlement, and borrow chains.

The relevant JSON artifacts preserve counts, seeds where applicable, and first counterexamples when found.
