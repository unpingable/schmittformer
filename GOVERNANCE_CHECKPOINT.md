# Governance Semantic-Core Checkpoint Receipt

## Checkpoint Identity

- schmittformer checkpoint commit: `6f11f3681479e2c929e20a84a539944c7d1954b9`
- checkpoint tag: `governance-semantic-core-v0`
- tag object: `de8abadfeb4cf764012c3654abe9d86d8743b945`
- tag target: `6f11f3681479e2c929e20a84a539944c7d1954b9`
- commit subject: `governance: add implementation-neutral semantic core`
- created on branch: `main`

The branch push attempted during the follow-up audit pass failed because the configured GitHub remote denied access to user `beckness`. The local commit and annotated tag remain intact.

## Source Provenance

- AG-ng source path: `/home/jbeck/ag_ng`
- AG-ng source revision: `aab771b636d0e7f09b5e281fa2104d94dde7a595`
- AG-ng status at extraction/audit: clean
- AG-ng governed-loop source test run: `cargo test -p ag-campaign --test governed_loop`, `12 passed`

AG-ng contains a formal-calculus crosswalk pinned to Lean revision `ff491b808ebeab2a132d9ade46d234cf85dcfbe9`, release 14.0.0. This checkpoint does not claim that the Python model inherits a Lean theorem.

## Runtime Environment

- Python: `3.12.3`
- PyTorch: `2.13.0+cpu`
- PyTorch CUDA available: `false`
- PyTorch CUDA runtime: none
- host GPU reported by `nvidia-smi`: `NVIDIA GeForce RTX 5060 Ti, 16311 MiB, driver 570.211.01`

No learned training or GPU computation is involved in this checkpoint.

## Reproduction Commands

```bash
.venv/bin/python experiments/run_governance.py --out-dir results
.venv/bin/python -m pytest tests/test_governance_reference.py tests/test_governance_invariants.py tests/test_governance_reachable.py -q
.venv/bin/python -m pytest -q
```

Additional audit/conformance commands added after the checkpoint:

```bash
.venv/bin/python -m pytest tests/test_governance_conformance.py tests/test_governance_reference.py tests/test_governance_invariants.py tests/test_governance_reachable.py -q
cargo test -p ag-campaign --test governed_loop   # run in /home/jbeck/ag_ng
```

## Checkpoint Counts

| measure | value |
| --- | ---: |
| normalized syntactic states | 3,840 |
| reachable states | 912 |
| event alphabet size | 39 |
| reachable transitions | 35,568 |
| admitted action transitions | 144 |
| refusal transitions | 30,724 |
| invariant violations | 0 |
| history-equivalence comparisons | 21,888 |
| history-equivalence violations | 0 |
| adversarial trace violations | 0 |

## Semantic Digest

Audit pass canonical transition digest:

```text
1926060d9a20ff1f19d4e67d31c0c0cf4725a11b1a3f401419fd6c033333cf8c
```

This digest is computed over the compact canonical reachable transition relation and state list. It identifies the intended finite semantics; it does not prove that any backend implements them.

## Result Files

- `results/governance_states.json`
- `results/governance_transitions.json`
- `results/governance_summary.json`
- `results/governance_conformance.json` added during the audit pass

## Test Warning

The full schmittformer suite passed with one PyTorch warning from `tests/test_circuit_learned.py`:

```text
UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
```

This warning is unrelated to the governance semantic kernel. It concerns the existing learned circuit-transformer test path.
