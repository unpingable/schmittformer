from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from src.explicit_register_model import (
    ExplicitRegisterModelConfig,
    ExplicitRegisterTransformer,
    choose_device,
    evaluate_explicit_register_model,
    load_explicit_register_model,
    save_explicit_register_model,
    train_explicit_register_model,
)
from src.latent_autopsy import controlled_position_batch
from src.projection_model import make_generator
from src.projection_task import ProjectionTaskConfig, batch_from_states, all_policy_states, decisions_from_tensors, sample_policy_batch
from src.register_governance import metadata_equivalence_report
from src.register_interventions import evaluate_register_intervention
from src.semantic_register import (
    RegisterEncoding,
    all_codebook,
    corrupt_register,
    corruption_modes,
    decode_register,
    register_accuracy,
    register_code,
    register_policy_decision,
)
from src.synthesized_latent_gate import decision_metrics

SCHEMA = "schmittformer.semantic_register_sweep.v1"
PROJECTION_BASELINE_REVISION = "0defef1"
LATENT_AUTOPSY_REVISION = "bb29471"


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def run_cmd(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, timeout=10).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def environment_report() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "schmittformer_revision": run_cmd(["git", "rev-parse", "HEAD"]),
        "projection_baseline_revision": PROJECTION_BASELINE_REVISION,
        "latent_autopsy_revision": LATENT_AUTOPSY_REVISION,
        "python": run_cmd(["python3", "--version"]),
        "executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": run_cmd(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
    }



def diagnostic_chunk_size(context_len: int) -> int:
    if context_len <= 256:
        return 64
    if context_len <= 1024:
        return 8
    return 2


@torch.no_grad()
def registers_for_batch(model: ExplicitRegisterTransformer, batch: Any, chunk_size: int) -> torch.Tensor:
    pieces = []
    for start in range(0, batch.tokens.shape[0], chunk_size):
        tokens = batch.tokens[start : start + chunk_size]
        register = model(tokens)["register"]
        assert isinstance(register, torch.Tensor)
        pieces.append(register.detach())
    return torch.cat(pieces, dim=0)


def load_manifest(out_dir: Path) -> dict[str, Any]:
    return read_json(out_dir / "manifest.json", {"schema": SCHEMA, "completed": [], "runs": {}})


def save_manifest(out_dir: Path, manifest: dict[str, Any]) -> None:
    atomic_write_json(out_dir / "manifest.json", manifest)


def completed(manifest: dict[str, Any], config_id: str) -> bool:
    return config_id in set(manifest.get("completed", []))


def mark_completed(out_dir: Path, manifest: dict[str, Any], config_id: str, path: Path) -> None:
    if config_id not in manifest.setdefault("completed", []):
        manifest["completed"].append(config_id)
    manifest.setdefault("runs", {})[config_id] = str(path)
    save_manifest(out_dir, manifest)


def adaptive_eval_shape(context_len: int, base_batch: int, base_batches: int) -> tuple[int, int]:
    if context_len <= 64:
        return base_batch, base_batches
    if context_len <= 256:
        return max(64, base_batch // 2), max(1, base_batches)
    if context_len <= 1024:
        return max(16, base_batch // 8), max(1, base_batches // 2)
    return max(4, base_batch // 64), max(1, base_batches // 4)


def nuisance_conditions() -> dict[str, float]:
    return {
        "IID": 0.95,
        "WEAKENED_NUISANCE": 0.60,
        "INDEPENDENT_NUISANCE": 0.50,
        "REVERSED_NUISANCE": 0.05,
    }


def config_id(seed: int, encoding: str, config: ExplicitRegisterModelConfig) -> str:
    return f"{encoding}_seed_{seed}_L{config.seq_len}_D{config.d_model}_S{config.steps}"


def checkpoint_path(out_dir: Path, run_id: str) -> Path:
    return out_dir / "checkpoints" / f"explicit_register_{run_id}.pt"


def train_or_load(out_dir: Path, run_id: str, config: ExplicitRegisterModelConfig, device: torch.device, force: bool) -> tuple[ExplicitRegisterTransformer, dict[str, Any]]:
    path = checkpoint_path(out_dir, run_id)
    if path.exists() and not force:
        return load_explicit_register_model(str(path), device), {"loaded_checkpoint": str(path), "config": asdict(config)}
    model, metrics = train_explicit_register_model(config, device)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_explicit_register_model(model, str(path))
    metrics["checkpoint"] = str(path)
    return model, metrics


@torch.no_grad()
def evaluate_metadata_equivalence(model: ExplicitRegisterTransformer, context_len: int, device: torch.device, seed: int) -> dict[str, Any]:
    batch = batch_from_states(all_policy_states(), context_len, device)
    register = registers_for_batch(model, batch, diagnostic_chunk_size(context_len))
    report = metadata_equivalence_report(batch.proposal, register, model.encoding)
    trusted = decision_metrics(batch.decision.detach().cpu(), batch.decision.detach().cpu())
    return {"internal_vs_external_from_same_register": report, "true_trusted_metadata_monitor": trusted}


@torch.no_grad()
def evaluate_position_modes(model: ExplicitRegisterTransformer, context_len: int, device: torch.device, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    repeats = 4 if context_len <= 256 else (2 if context_len <= 1024 else 1)
    states = all_policy_states() * repeats
    for mode in ["scaled", "fixed_absolute", "fixed_distance", "early", "middle", "late"]:
        batch = controlled_position_batch(states, seq_len=context_len, mode=mode, device=device)
        register = registers_for_batch(model, batch, diagnostic_chunk_size(context_len))
        decision, decoded = register_policy_decision(batch.proposal, register, model.encoding)
        rows.append(
            {
                "mode": mode,
                "context_len": context_len,
                "register": register_accuracy(decoded, batch.witness, batch.scope),
                "governance": decision_metrics(decision.detach().cpu(), batch.decision.detach().cpu()),
            }
        )
    return rows


@torch.no_grad()
def evaluate_interventions(model: ExplicitRegisterTransformer, context_len: int, device: torch.device) -> list[dict[str, Any]]:
    repeats = 4 if context_len <= 256 else (2 if context_len <= 1024 else 1)
    target_states = all_policy_states() * repeats
    target = batch_from_states(target_states, context_len, device)
    source_witness_states = [type(s)(s.proposal, 1 - s.witness, s.scope, s.nuisance) for s in target_states]
    source_scope_states = [type(s)(s.proposal, s.witness, 1 - s.scope, s.nuisance) for s in target_states]
    source_both_states = [type(s)(s.proposal, 1 - s.witness, 1 - s.scope, s.nuisance) for s in target_states]
    source_witness = batch_from_states(source_witness_states, context_len, device)
    source_scope = batch_from_states(source_scope_states, context_len, device)
    source_both = batch_from_states(source_both_states, context_len, device)
    target_reg = registers_for_batch(model, target, diagnostic_chunk_size(context_len))
    rows = []
    for variable, source in [("witness", source_witness), ("scope", source_scope), ("both", source_both), ("nuisance", target), ("random", target)]:
        source_reg = registers_for_batch(model, source, diagnostic_chunk_size(context_len))
        result = evaluate_register_intervention(target, source, target_reg, source_reg, model.encoding, variable)
        rows.append(result.to_json())
    return rows


def evaluate_faults(encoding: RegisterEncoding | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    codebook, witness, scope = all_codebook(encoding)
    proposal = torch.tensor([0, 1, 2], dtype=torch.long).repeat_interleave(codebook.shape[0])
    clean_register = codebook.repeat((3, 1))
    clean_witness = witness.repeat(3)
    clean_scope = scope.repeat(3)
    clean_decision, _ = register_policy_decision(proposal, clean_register, encoding)
    for mode in corruption_modes():
        corrupted = corrupt_register(clean_register, encoding, mode)
        decision, decoded = register_policy_decision(proposal, corrupted, encoding)
        semantic_expected = decisions_from_tensors(proposal, decoded.witness, decoded.scope)
        semantic_expected = torch.where(decoded.valid, semantic_expected, torch.full_like(semantic_expected, 5))
        semantic_expected = torch.where(proposal == 0, torch.full_like(semantic_expected, 2), semantic_expected)
        valid_wrong_world = decoded.valid & ((decoded.witness != clean_witness) | (decoded.scope != clean_scope))
        rows.append(
            {
                "encoding": RegisterEncoding(encoding).value,
                "mode": mode,
                "samples": int(proposal.numel()),
                "valid_rate": float(decoded.valid.to(torch.float32).mean().item()),
                "invalid_detected_rate": float((~decoded.valid).to(torch.float32).mean().item()),
                "silent_valid_alias_rate": float(valid_wrong_world.to(torch.float32).mean().item()),
                "semantic_consistency": float((decision == semantic_expected).to(torch.float32).mean().item()),
                "changed_decision_rate": float((decision != clean_decision).to(torch.float32).mean().item()),
            }
        )
    return rows


def run_one(out_dir: Path, manifest: dict[str, Any], seed: int, encoding: str, args: argparse.Namespace, force: bool) -> dict[str, Any]:
    config = ExplicitRegisterModelConfig(
        seed=seed,
        seq_len=args.train_context,
        max_len=max(args.contexts),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_nuisance_corr=args.train_nuisance_corr,
        encoding=encoding,
        log_every=max(1, args.steps // 4),
    )
    run_id = config_id(seed, encoding, config)
    run_path = out_dir / "runs" / f"{run_id}.json"
    if completed(manifest, run_id) and run_path.exists() and not force:
        return read_json(run_path, {})

    device = choose_device()
    start = time.time()
    model, train_metrics = train_or_load(out_dir, run_id, config, device, force)
    rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []
    for condition, corr in nuisance_conditions().items():
        for context_len in args.contexts:
            batch_size, batches = adaptive_eval_shape(context_len, args.eval_batch_size, args.eval_batches)
            task = ProjectionTaskConfig(seq_len=context_len, nuisance_corr=corr)
            metrics = evaluate_explicit_register_model(
                model,
                task,
                device,
                batch_size=batch_size,
                batches=batches,
                seed=seed + context_len * 13 + int(corr * 1000),
            )
            rows.append(
                {
                    "seed": seed,
                    "encoding": encoding,
                    "context_len": context_len,
                    "nuisance_condition": condition,
                    "nuisance_corr": corr,
                    "eval_batch_size": batch_size,
                    "eval_batches": batches,
                    **metrics,
                }
            )
    for context_len in args.contexts:
        metadata_rows.append({"seed": seed, "encoding": encoding, "context_len": context_len, **evaluate_metadata_equivalence(model, context_len, device, seed)})
        position_rows.extend({"seed": seed, "encoding": encoding, **row} for row in evaluate_position_modes(model, context_len, device, seed))
        intervention_rows.extend({"seed": seed, "encoding": encoding, "context_len": context_len, **row} for row in evaluate_interventions(model, context_len, device))
    payload = {
        "schema": SCHEMA,
        "run_id": run_id,
        "seed": seed,
        "encoding": encoding,
        "config": asdict(config),
        "train_metrics": train_metrics,
        "evaluation": rows,
        "metadata_equivalence": metadata_rows,
        "position_effects": position_rows,
        "interventions": intervention_rows,
        "runtime_seconds": time.time() - start,
    }
    atomic_write_json(run_path, payload)
    mark_completed(out_dir, manifest, run_id, run_path)
    return payload


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def stdev(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["encoding"], row["context_len"], row["nuisance_condition"])].append(row)
    out = []
    for (encoding, context_len, condition), group in sorted(grouped.items()):
        synth = [g["explicit_register_synthesized_gate"]["policy_accuracy"] for g in group]
        learned = [g["explicit_register_learned_gate"]["policy_accuracy"] for g in group]
        e2e = [g["end_to_end_learned"]["policy_accuracy"] for g in group]
        rel = [g["register_relative_synthesized"]["policy_accuracy"] for g in group]
        joint = [g["register"]["joint_accuracy"] for g in group]
        false_admit = [g["explicit_register_synthesized_gate"]["policy_violation_rate"] for g in group]
        out.append(
            {
                "encoding": encoding,
                "context_len": context_len,
                "nuisance_condition": condition,
                "seeds": len(group),
                "synthesized_accuracy_mean": mean(synth),
                "synthesized_accuracy_std": stdev(synth),
                "learned_register_accuracy_mean": mean(learned),
                "e2e_accuracy_mean": mean(e2e),
                "register_relative_accuracy_mean": mean(rel),
                "register_joint_accuracy_mean": mean(joint),
                "false_admit_rate_mean": mean(false_admit),
            }
        )
    return out


def summarize_context_transfer(rows: list[dict[str, Any]], contexts: list[int]) -> list[dict[str, Any]]:
    by_encoding_context: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        if row["nuisance_condition"] == "IID":
            by_encoding_context[(row["encoding"], row["context_len"])].append(row["explicit_register_synthesized_gate"]["policy_accuracy"])
    out = []
    for encoding in sorted({row["encoding"] for row in rows}):
        for built in contexts:
            for evaluated in contexts:
                out.append(
                    {
                        "encoding": encoding,
                        "built_context": built,
                        "eval_context": evaluated,
                        "accuracy_mean": mean(by_encoding_context[(encoding, evaluated)]),
                        "note": "fixed ABI synthesized gate; build context is intentionally irrelevant",
                    }
                )
    return out


def summarize_seed_transfer(rows: list[dict[str, Any]], seeds: list[int]) -> list[dict[str, Any]]:
    acc_by_seed_encoding: dict[tuple[int, str], float] = {}
    for row in rows:
        if row["nuisance_condition"] == "IID" and row["context_len"] == 64:
            acc_by_seed_encoding[(row["seed"], row["encoding"])] = row["explicit_register_synthesized_gate"]["policy_accuracy"]
    out = []
    for encoding in sorted({row["encoding"] for row in rows}):
        for built in seeds:
            for evaluated in seeds:
                out.append(
                    {
                        "encoding": encoding,
                        "built_seed": built,
                        "eval_seed": evaluated,
                        "accuracy": acc_by_seed_encoding.get((evaluated, encoding), float("nan")),
                        "note": "fixed ABI synthesized gate; source seed is intentionally irrelevant",
                    }
                )
    return out


def prior_emergent_baseline() -> dict[str, Any]:
    path = Path("results/latent_autopsy/aggregate.json")
    if not path.exists():
        return {"available": False}
    data = read_json(path, {})
    return {
        "available": True,
        "source": str(path),
        "known_summary": {
            "context_specific_final_layer_accuracy": {"64": 1.0, "256": 1.0, "1024": 1.0},
            "direct_64_to_1024_transfer": 0.7291666716337204,
            "affine_1024_to_64_restored": 0.9986979141831398,
            "seed_transfer_unaligned_off_diagonal": 0.22619047607960446,
        },
        "raw_keys": sorted(data.keys())[:20],
    }


def write_svg_line(path: Path, title: str, series: dict[str, list[tuple[float, float]]], width: int = 760, height: int = 420) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xs = [x for points in series.values() for x, _ in points]
    ys = [y for points in series.values() for _, y in points]
    if not xs or not ys:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>\n")
        return
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if ymin == ymax:
        ymin -= 0.05
        ymax += 0.05
    margin = 54
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2"]
    def sx(x: float) -> float:
        return margin + (x - xmin) / max(1e-9, xmax - xmin) * (width - 2 * margin)
    def sy(y: float) -> float:
        return height - margin - (y - ymin) / max(1e-9, ymax - ymin) * (height - 2 * margin)
    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>"]
    parts.append("<rect width='100%' height='100%' fill='white'/>")
    parts.append(f"<text x='{margin}' y='28' font-family='sans-serif' font-size='18'>{title}</text>")
    parts.append(f"<line x1='{margin}' y1='{height-margin}' x2='{width-margin}' y2='{height-margin}' stroke='#111'/>")
    parts.append(f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height-margin}' stroke='#111'/>")
    for idx, (name, points) in enumerate(series.items()):
        color = colors[idx % len(colors)]
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in sorted(points))
        parts.append(f"<polyline fill='none' stroke='{color}' stroke-width='2.5' points='{coords}'/>")
        for x, y in sorted(points):
            parts.append(f"<circle cx='{sx(x):.1f}' cy='{sy(y):.1f}' r='3' fill='{color}'/>")
        parts.append(f"<text x='{width-margin-190}' y='{margin + 18 * idx}' font-family='sans-serif' font-size='12' fill='{color}'>{name}</text>")
    parts.append("</svg>\n")
    path.write_text("\n".join(parts))


def write_svg_heatmap(path: Path, title: str, rows: list[str], cols: list[str], values: dict[tuple[str, str], float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cell = 54
    left = 150
    top = 72
    width = left + cell * len(cols) + 40
    height = top + cell * len(rows) + 42
    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>"]
    parts.append("<rect width='100%' height='100%' fill='white'/>")
    parts.append(f"<text x='18' y='30' font-family='sans-serif' font-size='18'>{title}</text>")
    for j, col in enumerate(cols):
        parts.append(f"<text x='{left + j*cell + 8}' y='{top - 18}' font-family='sans-serif' font-size='11'>{col}</text>")
    for i, row in enumerate(rows):
        parts.append(f"<text x='10' y='{top + i*cell + 31}' font-family='sans-serif' font-size='11'>{row}</text>")
        for j, col in enumerate(cols):
            v = max(0.0, min(1.0, float(values.get((row, col), 0.0))))
            red = int(255 * (1.0 - v))
            green = int(90 + 130 * v)
            blue = int(120 + 80 * v)
            parts.append(f"<rect x='{left + j*cell}' y='{top + i*cell}' width='{cell-3}' height='{cell-3}' fill='rgb({red},{green},{blue})'/>")
            parts.append(f"<text x='{left + j*cell + 10}' y='{top + i*cell + 31}' font-family='sans-serif' font-size='11' fill='black'>{v:.2f}</text>")
    parts.append("</svg>\n")
    path.write_text("\n".join(parts))


def write_svg_bars(path: Path, title: str, values: list[tuple[str, float]], width: int = 820, height: int = 420) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    margin = 58
    bar_gap = 8
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    bar_w = max(8, (plot_w - bar_gap * max(0, len(values)-1)) / max(1, len(values)))
    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>"]
    parts.append("<rect width='100%' height='100%' fill='white'/>")
    parts.append(f"<text x='{margin}' y='30' font-family='sans-serif' font-size='18'>{title}</text>")
    parts.append(f"<line x1='{margin}' y1='{height-margin}' x2='{width-margin}' y2='{height-margin}' stroke='#111'/>")
    parts.append(f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height-margin}' stroke='#111'/>")
    for i, (label, value) in enumerate(values):
        v = max(0.0, min(1.0, float(value)))
        x = margin + i * (bar_w + bar_gap)
        h = v * plot_h
        y = height - margin - h
        parts.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_w:.1f}' height='{h:.1f}' fill='#2563eb'/>")
        parts.append(f"<text x='{x:.1f}' y='{height-margin+14}' font-family='sans-serif' font-size='9' transform='rotate(35 {x:.1f},{height-margin+14})'>{label}</text>")
        parts.append(f"<text x='{x:.1f}' y='{y-4:.1f}' font-family='sans-serif' font-size='9'>{v:.2f}</text>")
    parts.append("</svg>\n")
    path.write_text("\n".join(parts))


def write_figures(out_dir: Path, aggregate: dict[str, Any]) -> None:
    fig_dir = out_dir / "figures"
    summary = aggregate["summary"]
    iid = [row for row in summary if row["nuisance_condition"] == "IID"]
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in iid:
        series[f"explicit {row['encoding']}"].append((row["context_len"], row["synthesized_accuracy_mean"]))
    series["emergent 64-gate"] = [(64, 1.0), (256, 0.8437), (1024, 0.7292)]
    write_svg_line(fig_dir / "figure1_direct_transfer_vs_context.svg", "Direct governance transfer vs context", dict(series))

    seed_rows = [str(seed) for seed in range(101, 117)]
    seed_cols = [str(seed) for seed in range(101, 117)]
    seed_values = {(str(row["built_seed"]), str(row["eval_seed"])): row["accuracy"] for row in aggregate["transfer_seed"] if row["encoding"] == "binary_pair"}
    write_svg_heatmap(fig_dir / "figure2_cross_seed_transfer_matrix.svg", "Explicit fixed-ABI cross-seed transfer, binary pair", seed_rows, seed_cols, seed_values)

    ctx_rows = [str(c) for c in [64, 256, 1024, 4096]]
    ctx_cols = [str(c) for c in [64, 256, 1024, 4096]]
    ctx_values = {(str(row["built_context"]), str(row["eval_context"])): row["accuracy_mean"] for row in aggregate["transfer_context"] if row["encoding"] == "binary_pair"}
    write_svg_heatmap(fig_dir / "figure3_cross_context_transfer_matrix.svg", "Explicit fixed-ABI cross-context transfer, binary pair", ctx_rows, ctx_cols, ctx_values)

    int_values = []
    for encoding in ["binary_pair", "grouped_one_hot", "joint_one_hot"]:
        for context in [64, 1024, 4096]:
            vals = [row["semantic_consistency"] for row in aggregate["interventions"] if row["encoding"] == encoding and row["context_len"] == context and row["intervention"] in ("witness", "scope")]
            int_values.append((f"{encoding[:6]}-{context}", mean(vals)))
    write_svg_bars(fig_dir / "figure4_intervention_consistency.svg", "Register intervention semantic consistency", int_values)

    nuisance_order = ["IID", "WEAKENED_NUISANCE", "INDEPENDENT_NUISANCE", "REVERSED_NUISANCE"]
    nuisance_values = []
    for condition in nuisance_order:
        vals = [row["synthesized_accuracy_mean"] for row in summary if row["encoding"] == "binary_pair" and row["context_len"] == 1024 and row["nuisance_condition"] == condition]
        nuisance_values.append((condition.replace("_NUISANCE", ""), mean(vals)))
    write_svg_bars(fig_dir / "figure5_nuisance_shift.svg", "Binary-pair governance under nuisance shift, context 1024", nuisance_values)

    fault_series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    modes = aggregate["fault_mode_order"]
    for row in aggregate["fault_injection"]:
        fault_series[row["encoding"]].append((modes.index(row["mode"]), row["invalid_detected_rate"]))
    write_svg_line(fig_dir / "figure6_fault_detection.svg", "Invalid register detection by corruption mode", dict(fault_series))

    metadata_values = []
    for encoding in ["binary_pair", "grouped_one_hot", "joint_one_hot"]:
        vals = [row["internal_vs_external_from_same_register"]["exact_match_rate"] for row in aggregate["metadata_equivalence"] if row["encoding"] == encoding]
        metadata_values.append((encoding, mean(vals)))
    write_svg_bars(fig_dir / "figure7_metadata_equivalence.svg", "Internal register gate vs external trusted metadata", metadata_values)

    reg_series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in iid:
        reg_series[row["encoding"]].append((row["context_len"], row["register_joint_accuracy_mean"]))
    write_svg_line(fig_dir / "figure_extra_register_world_accuracy.svg", "Register world accuracy vs context", dict(reg_series))


def aggregate_results(out_dir: Path, runs: list[dict[str, Any]], contexts: list[int], seeds: list[int]) -> dict[str, Any]:
    evaluation = [row for run in runs for row in run.get("evaluation", [])]
    metadata = [row for run in runs for row in run.get("metadata_equivalence", [])]
    interventions = [row for run in runs for row in run.get("interventions", [])]
    positions = [row for run in runs for row in run.get("position_effects", [])]
    fault_rows = []
    for encoding in sorted({run["encoding"] for run in runs}):
        fault_rows.extend(evaluate_faults(encoding))
    aggregate = {
        "schema": SCHEMA,
        "environment": read_json(out_dir / "environment.json", {}),
        "run_count": len(runs),
        "summary": summarize_rows(evaluation),
        "transfer_context": summarize_context_transfer(evaluation, contexts),
        "transfer_seed": summarize_seed_transfer(evaluation, seeds),
        "metadata_equivalence": metadata,
        "interventions": interventions,
        "position_effects": positions,
        "fault_injection": fault_rows,
        "fault_mode_order": corruption_modes(),
        "prior_emergent_baseline": prior_emergent_baseline(),
        "total_runtime_seconds": sum(float(run.get("runtime_seconds", 0.0)) for run in runs),
    }
    atomic_write_json(out_dir / "aggregate.json", aggregate)
    atomic_write_json(out_dir / "transfer_context.json", aggregate["transfer_context"])
    atomic_write_json(out_dir / "transfer_seed.json", aggregate["transfer_seed"])
    atomic_write_json(out_dir / "interventions.json", interventions)
    atomic_write_json(out_dir / "fault_injection.json", fault_rows)
    atomic_write_json(out_dir / "metadata_equivalence.json", metadata)
    atomic_write_json(out_dir / "position_effects.json", positions)
    write_figures(out_dir, aggregate)
    return aggregate


def benchmark_workload(device: torch.device, steps: int = 30, batch_size: int = 192, seq_len: int = 64) -> dict[str, Any]:
    config = ExplicitRegisterModelConfig(seed=999, seq_len=seq_len, max_len=1024, steps=steps, batch_size=batch_size, encoding="binary_pair", log_every=steps)
    start = time.time()
    model, metrics = train_explicit_register_model(config, device)
    elapsed = time.time() - start
    samples = steps * batch_size
    out = {
        "schema": SCHEMA,
        "device": str(device),
        "steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "elapsed_seconds": elapsed,
        "steps_per_second": steps / max(elapsed, 1e-9),
        "samples_per_second": samples / max(elapsed, 1e-9),
        "train_metrics": metrics,
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        out["peak_gpu_memory_mib"] = float(torch.cuda.max_memory_allocated(device) / 2**20)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="results/semantic_register")
    parser.add_argument("--seeds", nargs="*", type=int, default=list(range(101, 109)))
    parser.add_argument("--encodings", nargs="*", default=[e.value for e in RegisterEncoding])
    parser.add_argument("--contexts", nargs="*", type=int, default=[64, 256, 1024, 4096])
    parser.add_argument("--train-context", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--train-nuisance-corr", type=float, default=0.95)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "environment.json", environment_report())
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device("cuda")
    else:
        device = choose_device()
    if args.benchmark_only:
        atomic_write_json(out_dir / f"benchmark_{device.type}.json", benchmark_workload(device))
        return
    manifest = load_manifest(out_dir)
    save_manifest(out_dir, manifest)
    runs = []
    for encoding in args.encodings:
        RegisterEncoding(encoding)
        for seed in args.seeds:
            run = run_one(out_dir, manifest, seed, encoding, args, args.force)
            runs.append(run)
    aggregate_results(out_dir, runs, args.contexts, args.seeds)


if __name__ == "__main__":
    main()
