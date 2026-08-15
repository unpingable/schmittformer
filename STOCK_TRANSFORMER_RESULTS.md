# Stock Transformer Closure Results

Authoritative result directory:

```text
results/stock_transformer/
```

## Reproduce

```bash
.venv-cuda/bin/python -m experiments.run_stock_closure \
  --out-dir results/stock_transformer \
  --device cuda \
  --random-samples 200000 \
  --long-steps 10000 \
  --force

.venv-cuda/bin/python - <<'PY'
import torch
from pathlib import Path
from experiments.run_stock_closure import longrun_report, atomic_write_json
import json
base = Path("results/stock_transformer")
extra = longrun_report(torch.device("cuda"), [100000])
current = json.loads((base / "longrun.json").read_text())
by_len = {row["requested_logical_steps"]: row for row in current["rows"]}
for row in extra["rows"]:
    by_len[row["requested_logical_steps"]] = row
current["rows"] = [by_len[k] for k in sorted(by_len)]
atomic_write_json(base / "longrun.json", current)
PY

.venv/bin/python -m pytest -q
.venv-cuda/bin/python -m pytest \
  tests/test_relu_boolean.py \
  tests/test_ffn_counter.py \
  tests/test_stock_governance_transformer.py \
  -q
```

The second command adds the capped 100k recurrent trace. A 1M stock feedback trace would take roughly two hours at the measured single-trajectory throughput and was not run in this pass.

## Environment

```text
CUDA env: .venv-cuda
Python: 3.12.3
PyTorch: 2.11.0+cu128
torch CUDA runtime: 12.8
GPU: NVIDIA GeForce RTX 5060 Ti, 16 GB
NVIDIA driver: 570.211.01
```

## Architecture

```text
attention: torch.nn.MultiheadAttention
FFN: generated standard Linear/ReLU/Linear blocks
state width: 29 bits
event slots: 14
physical input width: 43
output: 29 next-state bits + 8 logits
score gap: 8
FFN blocks: 118
max FFN width: 743
parameters: 29,835,345
checkpoint size: 119,504,135 bytes (generated locally; the .pt file is git-ignored to avoid pushing a >100 MB artifact)
```

## Logically Established

The prior recurrent-softmax checkpoint already established full SMT equivalence:

```text
reference transition != logical compiled transition
over any valid bounded state/event
=> UNSAT
Z3 5.0.0
```

This pass lowers that same logical circuit into exact ReLU formulas over binary inputs. It does not rerun SMT over floating-point transformer execution.

## Exact On Discrete Domain By Construction

The generated FFN primitives implement:

```text
NOT, AND, OR, XOR, MUX, zero-test, borrow propagation
```

with explicit ReLU formulas. The decrementer is a ripple-borrow circuit:

```text
dec8  block count: 14
dec16 block count: 22
```

No transition table over counter values is used.

## Exhaustively Verified

Counter primitives:

| primitive | dtype/device | checked | failures |
| --- | --- | ---: | ---: |
| dec8 | float64 CPU | 256 | 0 |
| dec16 | float64 CPU | 65,536 | 0 |
| dec8 | float32 CUDA | 256 | 0 |
| dec16 | float32 CUDA | 65,536 | 0 |
| dec8 | float16 CUDA | 256 | 0 |
| dec16 | float16 CUDA | 65,536 | 0 |
| dec8 | bfloat16 CUDA | 256 | 0 |
| dec16 | bfloat16 CUDA | 65,536 | 0 |

Primitive outputs remained exact binary values in these checks:

```text
min bit margin to decode threshold: 0.5
max abs error from binary: 0
```

## Empirically Observed

Stock model, gap 8, CUDA float32:

```text
edge transitions:        10,584 / 10,584 passed
adversarial transitions: 21,168 / 21,168 passed
random transitions:     200,000 / 200,000 passed
semantic failures:      0
```

Margins:

| set | min state-bit margin | max state error before decode | min output margin |
| --- | ---: | ---: | ---: |
| edge | 0.365694 | 0.134306 | 25.352203 |
| adversarial | 0.365694 | 0.134306 | 25.352203 |
| random 200k | 0.392820 | 0.107180 | 26.622498 |

Gap sweep:

| gap | classification |
| ---: | --- |
| 2 | semantic failure |
| 4 | semantic failure |
| 6 | semantic failure |
| 8 | non-saturated softmax success |

The gap-8 success is not effectively hard. The retrieved slots contain visible analog leakage: max observed state error before decode is about `0.134`.

Precision matrix at gap 8:

| dtype/device | edge | adversarial | random 50k |
| --- | --- | --- | --- |
| float64 CPU | pass | pass | pass |
| float32 CUDA | pass | pass | pass |
| float16 CUDA | pass | pass | pass |
| bfloat16 CUDA | pass | pass | pass |

Checkpoint closure:

```text
save ordinary state_dict payload: passed
fresh load: passed
post-load validation: 2,048 / 2,048 passed
load path: reconstructs generic Linear/ReLU layer shapes from state_dict, then loads weights
```

Recurrent feedback with canonical decode/re-encode:

| logical steps | result | physical input width | steps/sec | min state-bit margin |
| ---: | --- | ---: | ---: | ---: |
| 10 | pass | 43 | 122.97 | 0.403405 |
| 100 | pass | 43 | 135.00 | 0.385542 |
| 1,000 | pass | 43 | 135.08 | 0.385542 |
| 10,000 | pass | 43 | 134.44 | 0.370988 |
| 100,000 | pass | 43 | 137.73 | 0.370988 |

The physical input stays fixed. The long-run result depends on the canonical latch, not analog carry.

## Not Established

This pass does not establish:

```text
full floating-point proof for every backend/kernel
raw analog recurrent-state correctness
Hugging Face save/load with trust_remote_code=False
production advantage over deterministic governance
efficiency comparable to the custom tensor circuit
```

LayerNorm/residual-heavy LLM-style blocks were not used. The model uses stock transformer operations, but the architecture is a generated circuit-shaped transformer, not a normal trained language model.

## Backend Comparison

| property | custom circuit | stock transformer |
| --- | ---: | ---: |
| finite-softmax retrieval | yes | yes |
| real uint8/uint16 counters | yes | yes |
| exact decoded semantics at gap 8 | yes | yes |
| SMT-equivalent logical core | yes | via same logical circuit |
| standard FFN implementation | no | yes |
| custom semantic ops in forward | yes | no |
| save/load checkpoint closure | partial | yes |
| 100k recurrent trace | yes | yes |
| 1M recurrent trace | yes | not run |
| parameter count | small/custom | 29.8M |

## Ruling

Primary ruling: **A. Full stock-transformer closure**.

The complete fixed-width recurrent governance machine, including real bounded binary counters, executes through ordinary finite-softmax attention and generated standard transformer FFNs with synthesized weights. No custom semantic tensor operation remains in the model forward path. Save/load preserves behavior. Decoded semantics are exact in the tested regime under the stated numerical and latch assumptions.

Important caveat: this is a stock-operations closure, not an efficiency or deployment result. The architecture is large for the tiny state machine, and the single-trajectory recurrent path is slow.
