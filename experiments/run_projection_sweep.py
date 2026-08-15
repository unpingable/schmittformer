from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import torch

from src.causal_interventions import InterventionConfig, evaluate_interventions
from src.latent_guard import (
    GuardTrainingConfig,
    evaluate_layer_guards,
    evaluate_upstream_policy_head,
    train_layer_guards,
)
from src.projection_baselines import (
    TokenGuardConfig,
    evaluate_deterministic_boundaries,
    evaluate_token_guard,
    train_token_guard,
)
from src.projection_channels import ProjectionRegime, bayes_bounds, regime_information_description
from src.projection_model import (
    ProjectionModelConfig,
    choose_device,
    load_projection_model,
    save_projection_model,
    train_projection_model,
)
from src.projection_task import ProjectionTaskConfig


GOVERNANCE_DIGEST = "1926060d9a20ff1f19d4e67d31c0c0cf4725a11b1a3f401419fd6c033333cf8c"
SCHEMA = "schmittformer.projection_sweep.v1"


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def environment_report() -> dict[str, Any]:
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
            timeout=5,
        ).strip()
    except Exception as exc:
        gpu = f"unavailable: {exc}"
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, timeout=5).strip()
    except Exception as exc:
        revision = f"unavailable: {exc}"
    return {
        "schema": SCHEMA,
        "schmittformer_revision": revision,
        "governance_semantic_digest": GOVERNANCE_DIGEST,
        "python": subprocess.check_output(["python3", "--version"], text=True).strip(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": gpu,
    }


def projection_points(noise_values: Iterable[float]) -> list[dict[str, Any]]:
    points = [
        {"regime": ProjectionRegime.P0_COMPLETE_ERASURE.value, "noise": 0.0},
        {"regime": ProjectionRegime.P2_PARTIAL_EXPORT.value, "noise": 0.0},
        {"regime": ProjectionRegime.P3_FULL_TRUSTED_EXPORT.value, "noise": 0.0},
        {"regime": ProjectionRegime.P5_SPURIOUS_EXPORT.value, "noise": 0.0},
    ]
    for noise in noise_values:
        points.append({"regime": ProjectionRegime.P1_NOISY_EXPORT.value, "noise": float(noise)})
    for noise in noise_values:
        points.append({"regime": ProjectionRegime.P4_REDUNDANT_EXPORT.value, "noise": float(noise)})
    seen = set()
    unique = []
    for point in points:
        key = (point["regime"], point["noise"])
        if key not in seen:
            seen.add(key)
            unique.append(point)
    return unique


def nuisance_conditions() -> dict[str, float]:
    return {
        "IID": 0.95,
        "WEAKENED_NUISANCE": 0.60,
        "INDEPENDENT_NUISANCE": 0.50,
        "REVERSED_NUISANCE": 0.05,
    }


def run_id(seed: int, profile: str, model_config: ProjectionModelConfig) -> str:
    return f"{profile}_seed_{seed}_L{model_config.seq_len}_D{model_config.d_model}_S{model_config.steps}"


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


def model_checkpoint_path(out_dir: Path, config_id: str) -> Path:
    return out_dir / "checkpoints" / f"upstream_{config_id}.pt"


def train_or_load_upstream(
    out_dir: Path,
    config_id: str,
    config: ProjectionModelConfig,
    device: torch.device,
    force: bool,
) -> tuple[Any, dict[str, Any]]:
    path = model_checkpoint_path(out_dir, config_id)
    if path.exists() and not force:
        model = load_projection_model(str(path), device)
        return model, {"loaded_checkpoint": str(path), "config": asdict(config)}
    model, metrics = train_projection_model(config, device)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_projection_model(model, str(path))
    metrics["checkpoint"] = str(path)
    return model, metrics


def task_config_for(seq_len: int, nuisance_corr: float) -> ProjectionTaskConfig:
    return ProjectionTaskConfig(seq_len=seq_len, nuisance_corr=nuisance_corr)


def scaled_eval_shape(train_len: int, context_len: int, batch_size: int, batches: int) -> tuple[int, int]:
    scale = max(1, context_len // max(1, train_len))
    scaled_batch = max(16, batch_size // scale)
    scaled_batches = max(1, batches // max(1, scale // 2))
    return scaled_batch, scaled_batches


def run_one_seed(
    out_dir: Path,
    manifest: dict[str, Any],
    profile: str,
    seed: int,
    model_config: ProjectionModelConfig,
    guard_config: GuardTrainingConfig,
    token_config: TokenGuardConfig,
    points: list[dict[str, Any]],
    eval_batches: int,
    eval_batch_size: int,
    context_lengths: list[int],
    force: bool,
) -> dict[str, Any]:
    config_id = run_id(seed, profile, model_config)
    run_path = out_dir / "runs" / f"{config_id}.json"
    if completed(manifest, config_id) and run_path.exists() and not force:
        return read_json(run_path, {})

    device = choose_device()
    start = time.time()
    model, upstream_metrics = train_or_load_upstream(out_dir, config_id, model_config, device, force)
    guards, guard_metrics = train_layer_guards(model, guard_config, device)

    token_guards = {}
    token_train_metrics = {}
    train_task = task_config_for(model_config.seq_len, token_config.train_nuisance_corr)
    for point in points:
        key = f"{point['regime']}@{point['noise']:.3f}"
        guard, metrics = train_token_guard(point["regime"], point["noise"], train_task, token_config, device)
        token_guards[key] = guard
        token_train_metrics[key] = metrics

    evaluation_rows: list[dict[str, Any]] = []
    bayes_rows: list[dict[str, Any]] = []
    latent_rows_by_condition: dict[str, list[dict[str, Any]]] = {}
    e2e_rows_by_condition: dict[str, dict[str, Any]] = {}
    for condition, corr in nuisance_conditions().items():
        latent_rows_by_condition[condition] = []
        e2e_rows_by_condition[condition] = {}
        for context_len in context_lengths:
            task_config = task_config_for(context_len, corr)
            context_batch_size, context_batches = scaled_eval_shape(model_config.seq_len, context_len, eval_batch_size, eval_batches)
            for point in points:
                bounds = bayes_bounds(task_config, point["regime"], point["noise"])
                bayes_rows.append({"context_len": context_len, "nuisance_condition": condition, "nuisance_corr": corr, **bounds})
                deterministic = evaluate_deterministic_boundaries(
                    point["regime"],
                    point["noise"],
                    task_config,
                    device,
                    batch_size=context_batch_size,
                    batches=context_batches,
                    seed=seed + 21000 + context_len,
                )
                for boundary, metrics in deterministic.items():
                    evaluation_rows.append(
                        {
                            "seed": seed,
                            "boundary": boundary.upper(),
                            "regime": point["regime"],
                            "noise": point["noise"],
                            "context_len": context_len,
                            "nuisance_condition": condition,
                            "nuisance_corr": corr,
                            "layer": None,
                            **metrics,
                        }
                    )
                token_key = f"{point['regime']}@{point['noise']:.3f}"
                token_metrics = evaluate_token_guard(
                    token_guards[token_key],
                    point["regime"],
                    point["noise"],
                    task_config,
                    device,
                    batch_size=context_batch_size,
                    batches=context_batches,
                    seed=seed + 22000 + context_len,
                )
                evaluation_rows.append(
                    {
                        "seed": seed,
                        "boundary": "TOKEN_LEARNED",
                        "regime": point["regime"],
                        "noise": point["noise"],
                        "context_len": context_len,
                        "nuisance_condition": condition,
                        "nuisance_corr": corr,
                        "layer": None,
                        **token_metrics,
                    }
                )

            latent_rows = evaluate_layer_guards(
                model,
                guards,
                task_config,
                device,
                batch_size=context_batch_size,
                batches=context_batches,
                seed=seed + 23000 + context_len,
            )
            latent_rows_by_condition[condition].append({"context_len": context_len, "rows": latent_rows})
            for row in latent_rows:
                evaluation_rows.append(
                    {
                        "seed": seed,
                        "boundary": "LATENT_LEARNED",
                        "regime": "PRE_PROJECTION",
                        "noise": 0.0,
                        "context_len": context_len,
                        "nuisance_condition": condition,
                        "nuisance_corr": corr,
                        "layer": row["layer"],
                        "layer_index": row["layer_index"],
                        **row["latent_learned"],
                    }
                )
                evaluation_rows.append(
                    {
                        "seed": seed,
                        "boundary": "LATENT_SYNTHESIZED",
                        "regime": "PRE_PROJECTION",
                        "noise": 0.0,
                        "context_len": context_len,
                        "nuisance_condition": condition,
                        "nuisance_corr": corr,
                        "layer": row["layer"],
                        "layer_index": row["layer_index"],
                        **row["latent_synthesized"],
                        "probe_accuracy": row["probe_accuracy"],
                    }
                )
            e2e_metrics = evaluate_upstream_policy_head(
                model,
                task_config,
                device,
                batch_size=context_batch_size,
                batches=context_batches,
                seed=seed + 24000 + context_len,
            )
            e2e_rows_by_condition[condition][str(context_len)] = e2e_metrics
            evaluation_rows.append(
                {
                    "seed": seed,
                    "boundary": "END_TO_END_LEARNED",
                    "regime": "FULL_CONTEXT",
                    "noise": 0.0,
                    "context_len": context_len,
                    "nuisance_condition": condition,
                    "nuisance_corr": corr,
                    "layer": "final_head",
                    **e2e_metrics,
                }
            )

    intervention_rows = evaluate_interventions(
        model,
        guards,
        InterventionConfig(seq_len=model_config.seq_len, batch_size=min(eval_batch_size, 512), seed=seed + 25000),
        device,
    )
    for row in intervention_rows:
        row["seed"] = seed

    payload = {
        "schema": SCHEMA,
        "profile": profile,
        "seed": seed,
        "config_id": config_id,
        "environment": environment_report(),
        "projection_points": points,
        "context_lengths": context_lengths,
        "projection_regime_descriptions": [regime_information_description(point["regime"]) for point in points],
        "upstream_training": upstream_metrics,
        "guard_training": guard_metrics,
        "token_guard_training": token_train_metrics,
        "evaluation_rows": evaluation_rows,
        "bayes_rows": bayes_rows,
        "latent_rows_by_condition": latent_rows_by_condition,
        "end_to_end_rows_by_condition": e2e_rows_by_condition,
        "intervention_rows": intervention_rows,
        "elapsed_seconds": time.time() - start,
    }
    atomic_write_json(run_path, payload)
    mark_completed(out_dir, manifest, config_id, run_path)
    return payload


def mean_std_ci(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "ci95": float("nan"), "n": 0}
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    ci95 = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": ci95, "n": len(values)}


def aggregate_runs(out_dir: Path) -> dict[str, Any]:
    run_payloads = []
    for path in sorted((out_dir / "runs").glob("*.json")):
        run_payloads.append(read_json(path, {}))

    eval_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    intervention_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    bayes_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for payload in run_payloads:
        for row in payload.get("evaluation_rows", []):
            key = (
                row["boundary"],
                row["regime"],
                float(row["noise"]),
                int(row.get("context_len", 64)),
                row["nuisance_condition"],
                row.get("layer"),
            )
            eval_groups[key].append(row)
        for row in payload.get("intervention_rows", []):
            key = (row["intervention"], row["layer"], row["layer_index"])
            intervention_groups[key].append(row)
        for row in payload.get("bayes_rows", []):
            key = (row["regime"], float(row["noise"]), int(row.get("context_len", 64)), row["nuisance_condition"])
            bayes_groups[key].append(row)

    numeric_metrics = [
        "policy_accuracy",
        "policy_violation_rate",
        "admit_false_positive_rate",
        "refuse_false_positive_rate",
        "refusal_reason_accuracy",
        "admit_rate",
        "true_admit_rate",
    ]
    aggregate_eval = []
    for key, rows in sorted(eval_groups.items(), key=lambda item: str(item[0])):
        boundary, regime, noise, context_len, condition, layer = key
        out = {
            "boundary": boundary,
            "regime": regime,
            "noise": noise,
            "context_len": context_len,
            "nuisance_condition": condition,
            "layer": layer,
        }
        if rows and rows[0].get("layer_index") is not None:
            out["layer_index"] = rows[0]["layer_index"]
        for metric in numeric_metrics:
            if metric in rows[0]:
                out[metric] = mean_std_ci([float(row[metric]) for row in rows])
        if rows and rows[0].get("probe_accuracy"):
            for probe_name in ("proposal", "witness", "scope", "nuisance"):
                out[f"probe_{probe_name}_accuracy"] = mean_std_ci(
                    [float(row["probe_accuracy"][probe_name]) for row in rows if row.get("probe_accuracy")]
                )
        aggregate_eval.append(out)

    intervention_metrics = [
        "latent_synthesized_intervention_consistency",
        "latent_learned_intervention_consistency",
        "proposal_preservation",
        "nuisance_intervention_effect_synthesized",
        "nuisance_intervention_effect_learned",
        "random_direction_consistency_synthesized",
        "random_direction_consistency_learned",
    ]
    aggregate_interventions = []
    for key, rows in sorted(intervention_groups.items(), key=lambda item: str(item[0])):
        intervention, layer, layer_index = key
        out = {"intervention": intervention, "layer": layer, "layer_index": layer_index}
        for metric in intervention_metrics:
            out[metric] = mean_std_ci([float(row[metric]) for row in rows])
        aggregate_interventions.append(out)

    aggregate_bayes = []
    for key, rows in sorted(bayes_groups.items(), key=lambda item: str(item[0])):
        regime, noise, context_len, condition = key
        first = rows[0]
        aggregate_bayes.append(
            {
                "regime": regime,
                "noise": noise,
                "context_len": context_len,
                "nuisance_condition": condition,
                "bayes_optimal_accuracy": first["bayes_optimal_accuracy"],
                "bayes_optimal_error": first["bayes_optimal_error"],
                "mutual_information_policy_decision_bits": first["mutual_information_policy_decision_bits"],
                "ambiguous_projection_keys": first["ambiguous_projection_keys"],
                "collision_example": first["collision_example"],
            }
        )

    aggregate = {
        "schema": SCHEMA,
        "runs": len(run_payloads),
        "evaluation": aggregate_eval,
        "interventions": aggregate_interventions,
        "bayes": aggregate_bayes,
    }
    atomic_write_json(out_dir / "aggregate.json", aggregate)
    atomic_write_json(out_dir / "bayes_bounds.json", aggregate_bayes)
    atomic_write_json(out_dir / "interventions.json", aggregate_interventions)
    return aggregate


def _metric(aggregate: dict[str, Any], **filters: Any) -> float | None:
    for row in aggregate["evaluation"]:
        if all(row.get(key) == value for key, value in filters.items()):
            metric = row.get("policy_violation_rate")
            if isinstance(metric, dict):
                return float(metric["mean"])
    return None


def _line_svg(
    title: str,
    x_label: str,
    y_label: str,
    series: list[dict[str, Any]],
    path: Path,
    width: int = 760,
    height: int = 460,
) -> None:
    margin_left, margin_right, margin_top, margin_bottom = 72, 26, 40, 64
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    xs = [x for item in series for x in item["x"]]
    ys = [y for item in series for y in item["y"]]
    if not xs or not ys:
        return
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = 0.0, max(max(ys), 0.01)
    if x_min == x_max:
        x_max = x_min + 1.0
    y_max *= 1.15

    def sx(x: float) -> float:
        return margin_left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return margin_top + (y_max - y) / (y_max - y_min) * plot_h

    colors = ["#174ea6", "#c5221f", "#188038", "#b06000", "#6f42c1", "#00796b"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-family="sans-serif" font-size="16">{title}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#333"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#333"/>',
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="12">{x_label}</text>',
        f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-family="sans-serif" font-size="12">{y_label}</text>',
    ]
    for tick in range(6):
        y = y_min + (y_max - y_min) * tick / 5
        parts.append(f'<line x1="{margin_left-4}" y1="{sy(y):.2f}" x2="{margin_left}" y2="{sy(y):.2f}" stroke="#333"/>')
        parts.append(f'<text x="{margin_left-8}" y="{sy(y)+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="10">{y:.3f}</text>')
    for tick in range(6):
        x = x_min + (x_max - x_min) * tick / 5
        parts.append(f'<line x1="{sx(x):.2f}" y1="{margin_top + plot_h}" x2="{sx(x):.2f}" y2="{margin_top + plot_h + 4}" stroke="#333"/>')
        parts.append(f'<text x="{sx(x):.2f}" y="{margin_top + plot_h + 18}" text-anchor="middle" font-family="sans-serif" font-size="10">{x:.2f}</text>')
    for index, item in enumerate(series):
        color = colors[index % len(colors)]
        points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(item["x"], item["y"]))
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>')
        for x, y in zip(item["x"], item["y"]):
            parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3" fill="{color}"/>')
        legend_y = margin_top + 18 + index * 18
        parts.append(f'<rect x="{width-230}" y="{legend_y-10}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{width-212}" y="{legend_y}" font-family="sans-serif" font-size="11">{item["label"]}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts))


def write_figures(out_dir: Path, aggregate: dict[str, Any]) -> None:
    fig_dir = out_dir / "figures"
    eval_rows = aggregate["evaluation"]
    bayes_rows = aggregate["bayes"]
    plot_context_len = 64

    def rows_for(boundary: str, regime: str, condition: str, layer: str | None = None, context_len: int = 64) -> list[dict[str, Any]]:
        rows = [
            row
            for row in eval_rows
            if row["boundary"] == boundary
            and row["regime"] == regime
            and row["nuisance_condition"] == condition
            and int(row.get("context_len", 64)) == context_len
        ]
        if layer is not None:
            rows = [row for row in rows if row["layer"] == layer]
        return sorted(rows, key=lambda row: float(row["noise"]))

    series = []
    for boundary in ("TOKEN_ONLY_REFERENCE", "TOKEN_LEARNED", "TOKEN_PLUS_TRUSTED_METADATA"):
        rows = rows_for(boundary, ProjectionRegime.P1_NOISY_EXPORT.value, "IID")
        series.append(
            {
                "label": boundary,
                "x": [float(row["noise"]) for row in rows],
                "y": [float(row["policy_violation_rate"]["mean"]) for row in rows],
            }
        )
    _line_svg(
        "Figure 1: policy violation vs noisy export",
        "bit flip probability",
        "false admission rate per sample",
        series,
        fig_dir / "figure1_violation_vs_projection_noise.svg",
    )

    layer_series = []
    for boundary in ("LATENT_LEARNED", "LATENT_SYNTHESIZED"):
        rows = [
            row
            for row in eval_rows
            if row["boundary"] == boundary
            and row["regime"] == "PRE_PROJECTION"
            and row["nuisance_condition"] == "IID"
            and int(row.get("context_len", 64)) == plot_context_len
        ]
        rows = sorted(rows, key=lambda row: row.get("layer_index", 0))
        layer_series.append(
            {
                "label": boundary,
                "x": [float(row.get("layer_index", 0)) for row in rows],
                "y": [float(row["policy_accuracy"]["mean"]) for row in rows],
            }
        )
    _line_svg("Figure 2: governance accuracy by layer", "layer index", "policy accuracy", layer_series, fig_dir / "figure2_layer_performance.svg")

    condition_x = {"IID": 0.0, "WEAKENED_NUISANCE": 1.0, "INDEPENDENT_NUISANCE": 2.0, "REVERSED_NUISANCE": 3.0}
    latent_final_layer = None
    latent_final_index = -1
    for row in eval_rows:
        if (
            row["boundary"] in ("LATENT_LEARNED", "LATENT_SYNTHESIZED")
            and row["regime"] == "PRE_PROJECTION"
            and int(row.get("context_len", 64)) == plot_context_len
        ):
            index = int(row.get("layer_index", -1))
            if index > latent_final_index:
                latent_final_index = index
                latent_final_layer = row["layer"]
    shift_series = []
    for boundary, regime, layer in (
        ("TOKEN_LEARNED", ProjectionRegime.P5_SPURIOUS_EXPORT.value, None),
        ("LATENT_LEARNED", "PRE_PROJECTION", latent_final_layer),
        ("LATENT_SYNTHESIZED", "PRE_PROJECTION", latent_final_layer),
        ("END_TO_END_LEARNED", "FULL_CONTEXT", "final_head"),
    ):
        rows = [
            row
            for row in eval_rows
            if row["boundary"] == boundary
            and row["regime"] == regime
            and int(row.get("context_len", 64)) == plot_context_len
            and (layer is None or row["layer"] == layer)
        ]
        rows = sorted(rows, key=lambda row: condition_x[row["nuisance_condition"]])
        shift_series.append(
            {
                "label": boundary,
                "x": [condition_x[row["nuisance_condition"]] for row in rows],
                "y": [float(row["policy_violation_rate"]["mean"]) for row in rows],
            }
        )
    _line_svg("Figure 3: nuisance shift", "IID, weakened, independent, reversed", "false admission rate per sample", shift_series, fig_dir / "figure3_nuisance_shift.svg")

    intervention_series = []
    for metric, label in (
        ("latent_synthesized_intervention_consistency", "LATENT_SYNTHESIZED"),
        ("latent_learned_intervention_consistency", "LATENT_LEARNED"),
        ("random_direction_consistency_synthesized", "RANDOM_CONTROL"),
    ):
        rows = [row for row in aggregate["interventions"] if row["intervention"] == "witness"]
        rows = sorted(rows, key=lambda row: row["layer_index"])
        intervention_series.append(
            {
                "label": label,
                "x": [float(row["layer_index"]) for row in rows],
                "y": [float(row[metric]["mean"]) for row in rows],
            }
        )
    _line_svg("Figure 4: witness intervention consistency", "layer index", "consistency", intervention_series, fig_dir / "figure4_intervention_consistency.svg")

    floor_rows = sorted(
        [
            row
            for row in bayes_rows
            if row["regime"] == ProjectionRegime.P1_NOISY_EXPORT.value
            and row["nuisance_condition"] == "IID"
            and int(row.get("context_len", 64)) == plot_context_len
        ],
        key=lambda row: row["noise"],
    )
    token_rows = rows_for("TOKEN_LEARNED", ProjectionRegime.P1_NOISY_EXPORT.value, "IID")
    reference_rows = rows_for("TOKEN_ONLY_REFERENCE", ProjectionRegime.P1_NOISY_EXPORT.value, "IID")
    _line_svg(
        "Figure 5: token-boundary error vs Bayes floor",
        "bit flip probability",
        "policy error",
        [
            {"label": "BAYES_FLOOR", "x": [row["noise"] for row in floor_rows], "y": [row["bayes_optimal_error"] for row in floor_rows]},
            {"label": "TOKEN_LEARNED", "x": [row["noise"] for row in token_rows], "y": [1.0 - row["policy_accuracy"]["mean"] for row in token_rows]},
            {"label": "TOKEN_ONLY_REFERENCE", "x": [row["noise"] for row in reference_rows], "y": [1.0 - row["policy_accuracy"]["mean"] for row in reference_rows]},
        ],
        fig_dir / "figure5_bayes_floor.svg",
    )


def profile_defaults(profile: str) -> dict[str, Any]:
    if profile == "smoke":
        return {
            "seeds": [101],
            "upstream_steps": 40,
            "guard_steps": 40,
            "token_steps": 40,
            "batch_size": 96,
            "eval_batch_size": 128,
            "eval_batches": 2,
            "d_model": 48,
            "n_layers": 2,
            "d_ff": 96,
            "noise": [0.0, 0.10, 0.50],
            "context_lengths": [64, 256],
        }
    if profile == "overnight":
        return {
            "seeds": [101, 102, 103, 104, 105, 106, 107, 108],
            "upstream_steps": 900,
            "guard_steps": 350,
            "token_steps": 300,
            "batch_size": 192,
            "eval_batch_size": 512,
            "eval_batches": 8,
            "d_model": 96,
            "n_layers": 4,
            "d_ff": 192,
            "noise": [0.0, 0.01, 0.05, 0.10, 0.20, 0.35, 0.50],
            "context_lengths": [64, 256, 1024],
        }
    return {
        "seeds": [101, 102, 103, 104],
        "upstream_steps": 300,
        "guard_steps": 150,
        "token_steps": 140,
        "batch_size": 128,
        "eval_batch_size": 256,
        "eval_batches": 4,
        "d_model": 64,
        "n_layers": 3,
        "d_ff": 128,
        "noise": [0.0, 0.01, 0.05, 0.10, 0.20, 0.35, 0.50],
        "context_lengths": [64, 256, 1024],
    }


def parse_seeds(raw: str | None, default: list[int]) -> list[int]:
    if raw is None:
        return default
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_context_lengths(raw: str | None, default: list[int]) -> list[int]:
    if raw is None:
        return default
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("at least one context length is required")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("results/projection"))
    parser.add_argument("--profile", choices=["smoke", "standard", "overnight"], default="standard")
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--context-lengths", type=str, default=None)
    args = parser.parse_args()

    defaults = profile_defaults(args.profile)
    seeds = parse_seeds(args.seeds, defaults["seeds"])
    context_lengths = parse_context_lengths(args.context_lengths, defaults["context_lengths"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.out_dir / "environment.json", environment_report())

    points = projection_points(defaults["noise"])
    atomic_write_json(
        args.out_dir / "projection_channels.json",
        {
            "schema": SCHEMA,
            "points": points,
            "descriptions": [regime_information_description(point["regime"]) for point in points],
            "context_lengths": context_lengths,
        },
    )
    manifest = load_manifest(args.out_dir)
    save_manifest(args.out_dir, manifest)

    for seed in seeds:
        model_config = ProjectionModelConfig(
            seed=seed,
            seq_len=args.seq_len,
            max_len=max(context_lengths),
            d_model=defaults["d_model"],
            n_layers=defaults["n_layers"],
            n_heads=4 if defaults["d_model"] % 4 == 0 else 2,
            d_ff=defaults["d_ff"],
            steps=defaults["upstream_steps"],
            batch_size=defaults["batch_size"],
            train_nuisance_corr=0.95,
        )
        guard_config = GuardTrainingConfig(
            seed=seed + 700,
            steps=defaults["guard_steps"],
            batch_size=defaults["batch_size"],
            train_nuisance_corr=0.95,
        )
        token_config = TokenGuardConfig(
            seed=seed + 1300,
            steps=defaults["token_steps"],
            batch_size=max(defaults["batch_size"], 256),
            train_nuisance_corr=0.95,
        )
        run_one_seed(
            args.out_dir,
            manifest,
            args.profile,
            seed,
            model_config,
            guard_config,
            token_config,
            points,
            defaults["eval_batches"],
            defaults["eval_batch_size"],
            context_lengths,
            args.force,
        )
        manifest = load_manifest(args.out_dir)
        aggregate = aggregate_runs(args.out_dir)
        write_figures(args.out_dir, aggregate)

    aggregate = aggregate_runs(args.out_dir)
    write_figures(args.out_dir, aggregate)
    atomic_write_json(
        args.out_dir / "summary.json",
        {
            "schema": SCHEMA,
            "profile": args.profile,
            "seeds": seeds,
            "runs": aggregate["runs"],
            "projection_points": points,
            "context_lengths": context_lengths,
            "result_files": {
                "aggregate": str(args.out_dir / "aggregate.json"),
                "bayes_bounds": str(args.out_dir / "bayes_bounds.json"),
                "interventions": str(args.out_dir / "interventions.json"),
                "manifest": str(args.out_dir / "manifest.json"),
            },
        },
    )


if __name__ == "__main__":
    main()
