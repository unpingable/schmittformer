# Results Index

This file maps the human-readable result documents to the committed machine-readable artifacts.

## First Hysteresis Pass

- Document: [RESULTS.md](RESULTS.md)
- Main code: [src/reference.py](src/reference.py), [src/compiled.py](src/compiled.py), [src/learned.py](src/learned.py), [src/hybrid.py](src/hybrid.py)
- Runner: [experiments/run_all.py](experiments/run_all.py)
- Artifacts: `results/compiled.json`, `results/learned.json`, `results/hybrid.json`, `results/summary.json`

## Circuit Breaker

- Document: [CIRCUIT_RESULTS.md](CIRCUIT_RESULTS.md)
- Main code: [src/circuit_reference.py](src/circuit_reference.py), [src/circuit_compiled.py](src/circuit_compiled.py), [src/circuit_learned.py](src/circuit_learned.py), [src/circuit_hybrid.py](src/circuit_hybrid.py)
- Runner: [experiments/run_circuit.py](experiments/run_circuit.py)
- Artifacts: `results/circuit_*.json`, especially `results/circuit_graph.json` and `results/circuit_summary.json`

## Finite Softmax Controllers

- Documents: [SOFTMAX_ANALYSIS.md](SOFTMAX_ANALYSIS.md), [SOFTMAX_RESULTS.md](SOFTMAX_RESULTS.md)
- Main code: [src/softmax_attention.py](src/softmax_attention.py), [src/hysteresis_softmax.py](src/hysteresis_softmax.py), [src/circuit_softmax.py](src/circuit_softmax.py)
- Runner: [experiments/run_softmax.py](experiments/run_softmax.py)
- Artifacts: `results/softmax_*.json`

## Governance Semantic Core

- Documents: [GOVERNANCE_SOURCE.md](GOVERNANCE_SOURCE.md), [GOVERNANCE_RESULTS.md](GOVERNANCE_RESULTS.md), [GOVERNANCE_FIDELITY.md](GOVERNANCE_FIDELITY.md), [CONSTELLATION_USE.md](CONSTELLATION_USE.md)
- Main code: [src/governance_reference.py](src/governance_reference.py), [src/governance_admissibility.py](src/governance_admissibility.py), [src/governance_conformance.py](src/governance_conformance.py)
- Runner: [experiments/run_governance.py](experiments/run_governance.py)
- Artifacts: `results/governance_states.json`, `results/governance_transitions.json`, `results/governance_conformance.json`, `results/governance_summary.json`
- Digest: `1926060d9a20ff1f19d4e67d31c0c0cf4725a11b1a3f401419fd6c033333cf8c`

## Projection-Loss Governance Boundary

- Documents: [PROJECTION_ANALYSIS.md](PROJECTION_ANALYSIS.md), [PROJECTION_RESULTS.md](PROJECTION_RESULTS.md)
- Main code: [src/projection_task.py](src/projection_task.py), [src/projection_channels.py](src/projection_channels.py), [src/projection_model.py](src/projection_model.py), [src/latent_guard.py](src/latent_guard.py), [src/synthesized_latent_gate.py](src/synthesized_latent_gate.py)
- Runner: [experiments/run_projection_sweep.py](experiments/run_projection_sweep.py)
- Authoritative artifacts: `results/projection_context/`

## Latent Autopsy

- Document: [LATENT_AUTOPSY.md](LATENT_AUTOPSY.md)
- Main code: [src/latent_autopsy.py](src/latent_autopsy.py), [src/causal_interventions.py](src/causal_interventions.py)
- Runner: [experiments/run_latent_autopsy.py](experiments/run_latent_autopsy.py)
- Artifacts: `results/latent_autopsy/`

## Explicit Semantic Register

- Documents: [SEMANTIC_REGISTER_ANALYSIS.md](SEMANTIC_REGISTER_ANALYSIS.md), [SEMANTIC_REGISTER_RESULTS.md](SEMANTIC_REGISTER_RESULTS.md)
- Main code: [src/semantic_register.py](src/semantic_register.py), [src/explicit_register_model.py](src/explicit_register_model.py), [src/register_governance.py](src/register_governance.py), [src/register_interventions.py](src/register_interventions.py)
- Runner: [experiments/run_semantic_register_sweep.py](experiments/run_semantic_register_sweep.py)
- Artifacts: `results/semantic_register/`

## Fixed-Width Recurrent Governance

- Documents: [RECURRENT_ANALYSIS.md](RECURRENT_ANALYSIS.md), [RECURRENT_RESULTS.md](RECURRENT_RESULTS.md)
- Main code: [src/fixed_state.py](src/fixed_state.py), [src/recurrent_reference.py](src/recurrent_reference.py), [src/recurrent_compiled.py](src/recurrent_compiled.py), [src/compiled_bits.py](src/compiled_bits.py), [src/compiled_counter.py](src/compiled_counter.py)
- Runners: [experiments/run_counter_verification.py](experiments/run_counter_verification.py), [experiments/run_recurrent_governance.py](experiments/run_recurrent_governance.py)
- Artifacts: `results/recurrent/`

## Recurrent Finite Softmax And SMT

- Documents: [RECURRENT_SOFTMAX_ANALYSIS.md](RECURRENT_SOFTMAX_ANALYSIS.md), [RECURRENT_SOFTMAX_RESULTS.md](RECURRENT_SOFTMAX_RESULTS.md)
- Main code: [src/recurrent_softmax.py](src/recurrent_softmax.py), [src/softmax_counter.py](src/softmax_counter.py), [src/stock_transformer_recurrent.py](src/stock_transformer_recurrent.py), [src/transition_smt.py](src/transition_smt.py)
- Runners: [experiments/run_recurrent_softmax.py](experiments/run_recurrent_softmax.py), [experiments/run_transition_solver.py](experiments/run_transition_solver.py)
- Artifacts: `results/recurrent_softmax/`

## Stock Transformer Closure

- Documents: [STOCK_TRANSFORMER_ANALYSIS.md](STOCK_TRANSFORMER_ANALYSIS.md), [STOCK_TRANSFORMER_RESULTS.md](STOCK_TRANSFORMER_RESULTS.md)
- Main code: [src/relu_boolean.py](src/relu_boolean.py), [src/ffn_counter.py](src/ffn_counter.py), [src/stock_governance_transformer.py](src/stock_governance_transformer.py)
- Runner: [experiments/run_stock_closure.py](experiments/run_stock_closure.py)
- Artifacts: `results/stock_transformer/`
- Generated checkpoint: `results/stock_transformer/stock_governance.pt`, intentionally git-ignored

## Overall Synthesis

- Document: [SCHMITTFORMER_FINAL.md](SCHMITTFORMER_FINAL.md)
- Final research tag: `schmittformer-research-v0`
