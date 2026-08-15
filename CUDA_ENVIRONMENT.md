# CUDA Environment

This repository preserves the historical CPU environment used for earlier results:

```text
.venv/
requirements.txt
    torch==2.13.0+cpu
```

This pass adds a separate CUDA-capable environment. The CPU venv and `requirements.txt` were not modified.

## Machine Inspection

```text
GPU: NVIDIA GeForce RTX 5060 Ti, 16311 MiB
Driver: 570.211.01
nvidia-smi CUDA version: 12.8
PCI ID: NVIDIA Corporation Device 2d04 (rev a1)
System nvcc: CUDA 12.0, V12.0.140
Python: 3.12.3
```

The system CUDA toolkit is not used by the PyTorch wheel. The installed PyTorch wheel provides its own CUDA 12.8 runtime.

The historical CPU venv reports:

```text
torch: 2.13.0+cpu
torch.version.cuda: None
torch.cuda.is_available(): False
```

## Package Choice

Official PyTorch install documentation lists CUDA 12.8 as a supported pip compute platform and recommends selecting the CUDA version suited to the machine:

```text
https://pytorch.org/get-started/locally/
```

The previous-versions page lists official CUDA 12.8 wheel indexes for recent releases:

```text
https://pytorch.org/get-started/previous-versions/
```

This environment uses the official CUDA 12.8 index:

```text
--extra-index-url https://download.pytorch.org/whl/cu128
torch==2.11.0+cu128
```

## Setup

```bash
python3 -m venv .venv-cuda
.venv-cuda/bin/python -m pip install --upgrade pip
.venv-cuda/bin/python -m pip install -r requirements-cuda.txt
```

Installed core versions:

```text
Python: 3.12.3
torch: 2.11.0+cu128
torch CUDA runtime: 12.8
CUDA available: true
GPU: NVIDIA GeForce RTX 5060 Ti
compute capability: (12, 0)
```

## Verification

Raw CUDA computation:

```python
import torch
assert torch.cuda.is_available()
x = torch.randn(4096, 4096, device="cuda")
y = x @ x
torch.cuda.synchronize()
print(torch.cuda.get_device_name())
print(y.device)
```

Observed:

```text
device: NVIDIA GeForce RTX 5060 Ti
y_device: cuda:0
matmul_seconds: 0.0954
peak_allocated_mib: 136.1
```

Project model CUDA forward/backward:

```text
loss: 5.3471
param_device: cuda:0
peak_allocated_mib: 105.0
```

## Performance Sanity Check

Benchmark command shape:

```bash
.venv/bin/python -m experiments.run_semantic_register_sweep   --out-dir results/semantic_register --benchmark-only --device cpu

.venv-cuda/bin/python -m experiments.run_semantic_register_sweep   --out-dir results/semantic_register --benchmark-only --device cuda
```

Observed short workload:

| environment | torch | device | steps/sec | samples/sec | peak GPU MiB |
| --- | --- | --- | ---: | ---: | ---: |
| CPU | 2.13.0+cpu | cpu | 3.26 | 626.0 | n/a |
| CUDA | 2.11.0+cu128 | cuda | 11.55 | 2216.9 | 278.9 |

The benchmark is small and includes Python/model overhead, but it confirms that the CUDA environment is materially useful.

## Reproduction Commands

Focused CUDA tests:

```bash
.venv-cuda/bin/python -m pytest   tests/test_semantic_register.py   tests/test_register_interventions.py   tests/test_semantic_register_runner.py -q
```

Full semantic-register sweep:

```bash
.venv-cuda/bin/python -m experiments.run_semantic_register_sweep   --out-dir results/semantic_register   --seeds 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116   --encodings binary_pair grouped_one_hot joint_one_hot   --contexts 64 256 1024 4096   --steps 1500   --batch-size 256   --eval-batch-size 512   --eval-batches 4
```

The runner is resumable. Completed `(encoding, seed)` configurations are recorded in `results/semantic_register/manifest.json` and skipped unless `--force` is supplied.
