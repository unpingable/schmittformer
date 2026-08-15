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
from typing import Any

import torch
from torch import Tensor

from src.latent_autopsy import (
    VariableReadouts,
    alignment_quality,
    apply_affine_alignment,
    apply_orthogonal_procrustes,
    centroid_geometry,
    classification_margin,
    collect_representations,
    controlled_position_batch,
    evaluate_readouts,
    fit_affine_alignment,
    fit_orthogonal_procrustes,
    fit_variable_readouts,
    margin_summary,
    minimize_padding_failure,
    representative_states,
    sample_representation_set,
    synthesized_decision_scores,
)
from src.projection_model import ProjectionCausalTransformer, choose_device, layer_names, load_projection_model, make_generator
from src.projection_task import (
    Decision,
    Nuisance,
    PolicyState,
    ProjectionBatch,
    ProjectionTaskConfig,
    Proposal,
    Scope,
    Witness,
    decisions_from_tensors,
    sample_policy_batch,
    tokens_from_latents,
)
from src.synthesized_latent_gate import decision_metrics


SCHEMA = "schmittformer.latent_autopsy.v1"
PROJECTION_REVISION = "0defef1"
PROJECTION_RESULTS_DIR = Path("results/projection_context")
CONTEXTS = [64, 256, 1024]
GOVERNANCE_DIGEST = "1926060d9a20ff1f19d4e67d31c0c0cf4725a11b1a3f401419fd6c033333cf8c"


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def read_json(path: Path) -> Any:
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
    try:
        pip_show = subprocess.check_output([".venv/bin/python", "-m", "pip", "show", "torch"], text=True, timeout=5)
    except Exception as exc:
        pip_show = f"unavailable: {exc}"
    return {
        "schema": SCHEMA,
        "schmittformer_revision": revision,
        "projection_checkpoint_revision": PROJECTION_REVISION,
        "governance_semantic_digest": GOVERNANCE_DIGEST,
        "python": subprocess.check_output([".venv/bin/python", "--version"], text=True).strip(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": gpu,
        "torch_pip_show": pip_show,
        "cuda_diagnosis": (
            "requirements.txt pins torch==2.13.0+cpu from the PyTorch CPU wheel index; "
            "the GPU is visible to nvidia-smi but this virtualenv has no CUDA-enabled torch build."
        ),
    }


def checkpoint_inventory(results_dir: Path = PROJECTION_RESULTS_DIR) -> list[dict[str, Any]]:
    summary = read_json(results_dir / "summary.json")
    runs_by_seed: dict[int, dict[str, Any]] = {}
    for run_path in sorted((results_dir / "runs").glob("*.json")):
        payload = read_json(run_path)
        runs_by_seed[int(payload["seed"])] = payload
    inventory = []
    for seed in summary["seeds"]:
        seed = int(seed)
        ckpt = results_dir / "checkpoints" / f"upstream_overnight_seed_{seed}_L64_D96_S900.pt"
        run = runs_by_seed.get(seed)
        config = run["upstream_training"]["config"] if run else None
        final_training = run["upstream_training"]["history"][-1] if run and run["upstream_training"].get("history") else None
        inventory.append(
            {
                "seed": seed,
                "checkpoint_path": str(ckpt),
                "checkpoint_exists": ckpt.exists(),
                "training_context": config.get("seq_len") if config else None,
                "architecture": {
                    key: config.get(key)
                    for key in ("d_model", "n_layers", "n_heads", "d_ff", "max_len", "steps", "batch_size")
                }
                if config
                else None,
                "final_training_metrics": final_training,
                "projection_contexts": summary["context_lengths"],
                "projection_config_id": run.get("config_id") if run else None,
            }
        )
    return inventory


def matched_latent_batch(
    seed: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    nuisance_corr: float = 0.95,
) -> ProjectionBatch:
    generator = make_generator(seed, device)
    base = sample_policy_batch(batch_size, ProjectionTaskConfig(seq_len=64, nuisance_corr=nuisance_corr), device, generator)
    tokens = tokens_from_latents(base.proposal, base.witness, base.scope, base.nuisance, seq_len)
    return ProjectionBatch(tokens, base.proposal, base.witness, base.scope, base.nuisance, base.decision)


def balanced_batch(seq_len: int, repeats: int, device: torch.device) -> ProjectionBatch:
    states = representative_states(repeats)
    proposal = torch.tensor([state.proposal for state in states], dtype=torch.long, device=device)
    witness = torch.tensor([state.witness for state in states], dtype=torch.long, device=device)
    scope = torch.tensor([state.scope for state in states], dtype=torch.long, device=device)
    nuisance = torch.tensor([state.nuisance for state in states], dtype=torch.long, device=device)
    decision = decisions_from_tensors(proposal, witness, scope)
    tokens = tokens_from_latents(proposal, witness, scope, nuisance, seq_len)
    return ProjectionBatch(tokens, proposal, witness, scope, nuisance, decision)


def upstream_head_metrics(model: ProjectionCausalTransformer, batch: ProjectionBatch, device: torch.device) -> dict[str, Any]:
    batch = batch.to(device)
    with torch.no_grad():
        outputs = model(batch.tokens)
    def acc(name: str, labels: Tensor) -> float:
        return float((outputs[f"{name}_logits"].argmax(dim=-1).detach().cpu() == labels.detach().cpu()).float().mean().item())
    return {
        "proposal_accuracy": acc("proposal", batch.proposal),
        "witness_accuracy": acc("witness", batch.witness),
        "scope_accuracy": acc("scope", batch.scope),
        "nuisance_accuracy": acc("nuisance", batch.nuisance),
        "decision": decision_metrics(outputs["decision_logits"].argmax(dim=-1).detach().cpu(), batch.decision.detach().cpu()),
        "proposal_margin": margin_summary(classification_margin(outputs["proposal_logits"].detach().cpu(), batch.proposal.detach().cpu())),
        "witness_margin": margin_summary(classification_margin(outputs["witness_logits"].detach().cpu(), batch.witness.detach().cpu())),
        "scope_margin": margin_summary(classification_margin(outputs["scope_logits"].detach().cpu(), batch.scope.detach().cpu())),
        "decision_margin": margin_summary(classification_margin(outputs["decision_logits"].detach().cpu(), batch.decision.detach().cpu())),
    }


def fit_readouts_for_contexts(
    train_sets: dict[int, Any],
    layer_count: int,
) -> dict[tuple[int, int], VariableReadouts]:
    readouts = {}
    for context_len, data in train_sets.items():
        for layer_index in range(layer_count):
            readouts[(context_len, layer_index)] = fit_variable_readouts(data, layer_index)
    return readouts


def direction_cosines(readouts: VariableReadouts) -> dict[str, float]:
    witness = (readouts.witness.weights[:-1, 1] - readouts.witness.weights[:-1, 0]).to(torch.float32)
    scope = (readouts.scope.weights[:-1, 1] - readouts.scope.weights[:-1, 0]).to(torch.float32)
    nuisance = (readouts.nuisance.weights[:-1, 1] - readouts.nuisance.weights[:-1, 0]).to(torch.float32)
    decision = readouts.decision.weights[:-1].to(torch.float32)
    refuse_invalid = decision[:, int(Decision.REFUSE_INVALID_WITNESS)]
    admit_a = decision[:, int(Decision.ADMIT_A)]
    admit_b = decision[:, int(Decision.ADMIT_B)]
    return {
        "witness_scope_cosine": float(torch.nn.functional.cosine_similarity(witness, scope, dim=0).item()),
        "witness_nuisance_cosine": float(torch.nn.functional.cosine_similarity(witness, nuisance, dim=0).item()),
        "scope_nuisance_cosine": float(torch.nn.functional.cosine_similarity(scope, nuisance, dim=0).item()),
        "witness_refuse_invalid_cosine": float(torch.nn.functional.cosine_similarity(witness, refuse_invalid, dim=0).item()),
        "scope_admit_a_cosine": float(torch.nn.functional.cosine_similarity(scope, admit_a, dim=0).item()),
        "scope_admit_b_cosine": float(torch.nn.functional.cosine_similarity(scope, admit_b, dim=0).item()),
    }


def directional_swap(source: Tensor, target: Tensor, direction: Tensor) -> Tensor:
    direction = direction.to(target.dtype)
    unit = direction / torch.clamp(torch.linalg.norm(direction), min=1.0e-8)
    return target + ((source @ unit) - (target @ unit)).unsqueeze(1) * unit.unsqueeze(0)


def paired_intervention_data(kind: str, seq_len: int, batch_size: int, device: torch.device) -> ProjectionBatch:
    states = []
    for index in range(batch_size):
        if kind == "witness_valid":
            proposal = Proposal.REMEDIATE_A if index % 2 == 0 else Proposal.REMEDIATE_B
            scope = Scope.A if proposal == Proposal.REMEDIATE_A else Scope.B
            states.append(PolicyState(int(proposal), int(Witness.VALID), int(scope), int(Nuisance.ONE)))
        elif kind == "witness_invalid":
            proposal = Proposal.REMEDIATE_A if index % 2 == 0 else Proposal.REMEDIATE_B
            scope = Scope.A if proposal == Proposal.REMEDIATE_A else Scope.B
            states.append(PolicyState(int(proposal), int(Witness.INVALID), int(scope), int(Nuisance.ONE)))
        elif kind == "scope_match":
            proposal = Proposal.REMEDIATE_A if index % 2 == 0 else Proposal.REMEDIATE_B
            scope = Scope.A if proposal == Proposal.REMEDIATE_A else Scope.B
            states.append(PolicyState(int(proposal), int(Witness.VALID), int(scope), int(Nuisance.ONE)))
        elif kind == "scope_mismatch":
            proposal = Proposal.REMEDIATE_A if index % 2 == 0 else Proposal.REMEDIATE_B
            scope = Scope.B if proposal == Proposal.REMEDIATE_A else Scope.A
            states.append(PolicyState(int(proposal), int(Witness.VALID), int(scope), int(Nuisance.ONE)))
        elif kind == "nuisance_one":
            proposal = Proposal.REMEDIATE_A if index % 2 == 0 else Proposal.REMEDIATE_B
            scope = Scope.A if proposal == Proposal.REMEDIATE_A else Scope.B
            states.append(PolicyState(int(proposal), int(Witness.VALID), int(scope), int(Nuisance.ONE)))
        elif kind == "nuisance_zero":
            proposal = Proposal.REMEDIATE_A if index % 2 == 0 else Proposal.REMEDIATE_B
            scope = Scope.A if proposal == Proposal.REMEDIATE_A else Scope.B
            states.append(PolicyState(int(proposal), int(Witness.VALID), int(scope), int(Nuisance.ZERO)))
        else:
            raise ValueError(kind)
    return controlled_position_batch(states, seq_len, "scaled", device)


def intervention_metrics_for_context(
    model: ProjectionCausalTransformer,
    readouts_by_context: dict[tuple[int, int], VariableReadouts],
    context_len: int,
    layer_labels: list[str],
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    rows = []
    pairs = {
        "witness": ("witness_valid", "witness_invalid", "witness"),
        "scope": ("scope_match", "scope_mismatch", "scope"),
        "nuisance": ("nuisance_one", "nuisance_zero", "nuisance"),
    }
    for name, (source_name, target_name, direction_name) in pairs.items():
        source_batch = paired_intervention_data(source_name, context_len, batch_size, device)
        target_batch = paired_intervention_data(target_name, context_len, batch_size, device)
        source = collect_representations(model, source_batch, device)
        target = collect_representations(model, target_batch, device)
        for layer_index, layer in enumerate(layer_labels):
            readouts = readouts_by_context[(context_len, layer_index)]
            random_direction = torch.randn(readouts.witness.weights.shape[0] - 1)
            if direction_name == "witness":
                direction = readouts.witness.weights[:-1, 1] - readouts.witness.weights[:-1, 0]
            elif direction_name == "scope":
                direction = readouts.scope.weights[:-1, 1] - readouts.scope.weights[:-1, 0]
            else:
                direction = readouts.nuisance.weights[:-1, 1] - readouts.nuisance.weights[:-1, 0]
            source_rep = source.representations[layer_index]
            target_rep = target.representations[layer_index]
            target_scores = synthesized_decision_scores(
                readouts.proposal.logits(target_rep),
                readouts.witness.logits(target_rep),
                readouts.scope.logits(target_rep),
            )
            target_correct = (target_scores.argmax(dim=-1) == target.decision).float().mean().item()
            for control_name, active_direction in ((direction_name, direction), ("random", random_direction)):
                swapped = directional_swap(source_rep, target_rep, active_direction)
                proposal_logits = readouts.proposal.logits(swapped)
                witness_logits = readouts.witness.logits(swapped)
                scope_logits = readouts.scope.logits(swapped)
                synth_scores = synthesized_decision_scores(proposal_logits, witness_logits, scope_logits)
                synth_pred = synth_scores.argmax(dim=-1)
                learned_pred = readouts.decision.predict(swapped)
                proposal_preserved = (proposal_logits.argmax(dim=-1) == target.proposal).float().mean().item()
                rows.append(
                    {
                        "context_len": context_len,
                        "layer_index": layer_index,
                        "layer": layer,
                        "intervention": name,
                        "direction": control_name,
                        "latent_synthesized_intervention_consistency": float((synth_pred == source.decision).float().mean().item()),
                        "latent_learned_intervention_consistency": float((learned_pred == source.decision).float().mean().item()),
                        "target_stability_without_intervention": float(target_correct),
                        "proposal_preservation": float(proposal_preserved),
                        "samples": int(batch_size),
                    }
                )
    return rows


def classify_probe(row: dict[str, Any], intervention_consistency: float | None) -> str:
    acc = row["probe_accuracy"]["witness"]
    if acc < 0.80:
        return "NOT_DECODABLE"
    if acc < 0.95:
        return "UNSTABLE"
    if intervention_consistency is not None and intervention_consistency >= 0.80:
        return "DECODABLE_AND_CAUSAL"
    return "DECODABLE_NOT_CAUSAL"


def representative_eval_batch(seq_len: int, repeats: int, device: torch.device, mode: str = "scaled") -> ProjectionBatch:
    states = representative_states(repeats)
    if mode == "scaled":
        return balanced_batch(seq_len, repeats, device)
    return controlled_position_batch(states, seq_len, mode, device)


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "ci95": float("nan"), "n": 0}
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0, "n": len(values)}


def aggregate_rows(rows: list[dict[str, Any]], group_keys: list[str], metric_paths: dict[str, list[str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in group_keys)].append(row)
    out = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        item = {name: value for name, value in zip(group_keys, key)}
        for metric_name, path in metric_paths.items():
            values = []
            for row in group:
                value: Any = row
                for part in path:
                    value = value[part]
                values.append(float(value))
            item[metric_name] = summarize(values)
        out.append(item)
    return out


def run_seed(
    seed: int,
    checkpoint_path: Path,
    out_dir: Path,
    train_repeats: int,
    eval_repeats: int,
    intervention_batch_size: int,
    force: bool,
) -> dict[str, Any]:
    run_path = out_dir / "runs" / f"latent_autopsy_seed_{seed}.json"
    if run_path.exists() and not force:
        return read_json(run_path)

    device = choose_device()
    model = load_projection_model(str(checkpoint_path), device)
    labels = layer_names(model.config)
    start = time.time()

    train_sets = {}
    eval_sets = {}
    sampled_sets = {}
    upstream_rows = []
    for context_len in CONTEXTS:
        train_sets[context_len] = collect_representations(model, balanced_batch(context_len, train_repeats, device), device)
        eval_batch = balanced_batch(context_len, eval_repeats, device)
        eval_sets[context_len] = collect_representations(model, eval_batch, device)
        sampled_batch = matched_latent_batch(seed + 9000 + context_len, 256 if context_len < 1024 else 96, context_len, device)
        sampled_sets[context_len] = collect_representations(model, sampled_batch, device)
        upstream_rows.append(
            {
                "seed": seed,
                "context_len": context_len,
                "balanced": upstream_head_metrics(model, eval_batch, device),
                "sampled": upstream_head_metrics(model, sampled_batch, device),
            }
        )

    readouts = fit_readouts_for_contexts(train_sets, len(labels))

    layer_rows = []
    transfer_rows = []
    geometry_rows = []
    margin_examples = []
    for built_context in CONTEXTS:
        for eval_context in CONTEXTS:
            eval_data = eval_sets[eval_context]
            for layer_index, layer in enumerate(labels):
                row = evaluate_readouts(readouts[(built_context, layer_index)], eval_data.representations[layer_index], eval_data)
                enriched = {
                    "seed": seed,
                    "built_context": built_context,
                    "eval_context": eval_context,
                    "layer": layer,
                    "layer_index": layer_index,
                    **row,
                }
                transfer_rows.append(enriched)
                if built_context == eval_context:
                    layer_rows.append(enriched)
                    for variable, classes, label_tensor in (
                        ("witness", 2, eval_data.witness),
                        ("scope", 2, eval_data.scope),
                        ("nuisance", 2, eval_data.nuisance),
                        ("decision", len(Decision), eval_data.decision),
                    ):
                        geometry_rows.append(
                            {
                                "seed": seed,
                                "context_len": eval_context,
                                "layer": layer,
                                "layer_index": layer_index,
                                "variable": variable,
                                **centroid_geometry(eval_data.representations[layer_index], label_tensor, classes),
                            }
                        )
                    if layer_index == len(labels) - 1 and eval_context in (256, 1024):
                        scores = synthesized_decision_scores(
                            readouts[(64, layer_index)].proposal.logits(eval_data.representations[layer_index]),
                            readouts[(64, layer_index)].witness.logits(eval_data.representations[layer_index]),
                            readouts[(64, layer_index)].scope.logits(eval_data.representations[layer_index]),
                        )
                        pred = scores.argmax(dim=-1)
                        bad = (pred != eval_data.decision).nonzero(as_tuple=False).flatten()
                        if bad.numel():
                            index = int(bad[0].item())
                            margin_examples.append(
                                {
                                    "seed": seed,
                                    "eval_context": eval_context,
                                    "layer": layer,
                                    "index": index,
                                    "proposal": int(eval_data.proposal[index].item()),
                                    "witness": int(eval_data.witness[index].item()),
                                    "scope": int(eval_data.scope[index].item()),
                                    "nuisance": int(eval_data.nuisance[index].item()),
                                    "expected": int(eval_data.decision[index].item()),
                                    "predicted": int(pred[index].item()),
                                    "margin": float(classification_margin(scores[index : index + 1], eval_data.decision[index : index + 1])[0].item()),
                                }
                            )

    intervention_rows = []
    for context_len in CONTEXTS:
        for row in intervention_metrics_for_context(model, readouts, context_len, labels, device, intervention_batch_size):
            row["seed"] = seed
            intervention_rows.append(row)

    alignment_rows = []
    for source_context in (256, 1024):
        for layer_index, layer in enumerate(labels):
            fit_source = collect_representations(model, matched_latent_batch(seed + 12000 + source_context, 192, source_context, device), device)
            fit_target = collect_representations(model, matched_latent_batch(seed + 12000 + source_context, 192, 64, device), device)
            eval_source = collect_representations(model, matched_latent_batch(seed + 13000 + source_context, 192, source_context, device), device)
            eval_target = collect_representations(model, matched_latent_batch(seed + 13000 + source_context, 192, 64, device), device)
            source_x = fit_source.representations[layer_index]
            target_x = fit_target.representations[layer_index]
            eval_source_x = eval_source.representations[layer_index]
            eval_target_x = eval_target.representations[layer_index]
            gate64 = readouts[(64, layer_index)]
            baseline = evaluate_readouts(gate64, eval_source_x, eval_source)
            target = evaluate_readouts(gate64, eval_target_x, eval_target)
            affine = fit_affine_alignment(source_x, target_x)
            affine_mapped = apply_affine_alignment(eval_source_x, affine)
            procrustes = fit_orthogonal_procrustes(source_x, target_x)
            procrustes_mapped = apply_orthogonal_procrustes(eval_source_x, procrustes)
            for method, mapped in (("affine", affine_mapped), ("orthogonal_procrustes", procrustes_mapped)):
                restored = evaluate_readouts(gate64, mapped, eval_source)
                alignment_rows.append(
                    {
                        "seed": seed,
                        "source_context": source_context,
                        "target_context": 64,
                        "layer": layer,
                        "layer_index": layer_index,
                        "method": method,
                        "baseline_source_accuracy": baseline["latent_synthesized"]["policy_accuracy"],
                        "target_64_accuracy": target["latent_synthesized"]["policy_accuracy"],
                        "restored_accuracy": restored["latent_synthesized"]["policy_accuracy"],
                        "baseline_source_margin_mean": baseline["synthesized_decision_margin"]["mean"],
                        "restored_margin_mean": restored["synthesized_decision_margin"]["mean"],
                        "alignment_quality": alignment_quality(mapped, eval_target_x),
                    }
                )

    seed_transfer_rows = []
    # Filled at aggregate level where all seed representations are available.

    position_rows = []
    final_layer = len(labels) - 1
    gate64 = readouts[(64, final_layer)]
    for context_len in CONTEXTS:
        for mode in ("scaled", "fixed_absolute", "fixed_distance", "early", "middle", "late"):
            batch = representative_eval_batch(context_len, max(2, eval_repeats // 2), device, mode)
            data = collect_representations(model, batch, device)
            row = evaluate_readouts(gate64, data.representations[final_layer], data)
            position_rows.append(
                {
                    "seed": seed,
                    "context_len": context_len,
                    "mode": mode,
                    "witness_pos_description": mode,
                    "layer": labels[final_layer],
                    "layer_index": final_layer,
                    **row,
                }
            )

    counterexamples = []
    for candidate in margin_examples[:2]:
        state = PolicyState(candidate["proposal"], candidate["witness"], candidate["scope"], candidate["nuisance"])

        def fails(length: int) -> bool:
            batch = controlled_position_batch([state], length, "scaled", device)
            data = collect_representations(model, batch, device)
            pred = synthesized_decision_scores(
                gate64.proposal.logits(data.representations[final_layer]),
                gate64.witness.logits(data.representations[final_layer]),
                gate64.scope.logits(data.representations[final_layer]),
            ).argmax(dim=-1)
            return bool(int(pred[0].item()) != int(data.decision[0].item()))

        coarse = [64, 80, 96, 128, 160, 192, 224, 256, 320, 384, 512, 768, 1024]
        first = next((length for length in coarse if fails(length)), None)
        minimized = minimize_padding_failure(fails, 64, first) if first else None
        counterexamples.append(
            {
                **candidate,
                "original_length": candidate["eval_context"],
                "first_failing_coarse_length": first,
                "minimized_length_if_monotonic": minimized,
            }
        )

    probe_illusion_rows = []
    intervention_lookup = {
        (row["context_len"], row["layer_index"], row["intervention"], row["direction"]): row for row in intervention_rows
    }
    for row in layer_rows:
        key = (row["eval_context"], row["layer_index"], "witness", "witness")
        intervention = intervention_lookup.get(key)
        readout = readouts[(row["built_context"], row["layer_index"])]
        probe_illusion_rows.append(
            {
                "seed": seed,
                "context_len": row["eval_context"],
                "layer": row["layer"],
                "layer_index": row["layer_index"],
                "classification": classify_probe(row, intervention["latent_synthesized_intervention_consistency"] if intervention else None),
                "witness_probe_accuracy": row["probe_accuracy"]["witness"],
                "synthesized_intervention_consistency": intervention["latent_synthesized_intervention_consistency"] if intervention else None,
                "direction_cosines": direction_cosines(readout),
            }
        )

    payload = {
        "schema": SCHEMA,
        "seed": seed,
        "checkpoint_path": str(checkpoint_path),
        "layer_names": labels,
        "contexts": CONTEXTS,
        "upstream_task": upstream_rows,
        "layer_metrics": layer_rows,
        "cross_context_transfer": transfer_rows,
        "representation_geometry": geometry_rows,
        "intervention_metrics": intervention_rows,
        "context_alignment": alignment_rows,
        "seed_transfer": seed_transfer_rows,
        "position_effects": position_rows,
        "gate_margin_counterexamples": margin_examples,
        "counterexamples": counterexamples,
        "probe_illusion": probe_illusion_rows,
        "elapsed_seconds": time.time() - start,
    }
    atomic_write_json(run_path, payload)
    return payload


def seed_transfer_analysis(run_payloads: list[dict[str, Any]], out_dir: Path, eval_repeats: int) -> list[dict[str, Any]]:
    # This pass evaluates whether a final-layer context-64 gate fitted on one
    # seed transfers to another seed. It intentionally uses fresh readouts fitted
    # from the preserved checkpoints rather than serialized guard weights from
    # the projection sweep, because those were not retained.
    device = choose_device()
    seed_models = {}
    seed_train = {}
    seed_eval = {}
    seed_readouts = {}
    for payload in run_payloads:
        seed = int(payload["seed"])
        model = load_projection_model(payload["checkpoint_path"], device)
        seed_models[seed] = model
        seed_train[seed] = collect_representations(model, balanced_batch(64, 12, device), device)
        seed_eval[seed] = collect_representations(model, balanced_batch(64, eval_repeats, device), device)
        final_layer = len(layer_names(model.config)) - 1
        seed_readouts[seed] = fit_variable_readouts(seed_train[seed], final_layer)

    rows = []
    for source_seed, readout in seed_readouts.items():
        source_model = seed_models[source_seed]
        source_fit = seed_train[source_seed]
        final_layer = len(layer_names(source_model.config)) - 1
        for target_seed, target_model in seed_models.items():
            target_eval = seed_eval[target_seed]
            baseline = evaluate_readouts(readout, target_eval.representations[final_layer], target_eval)
            item = {
                "source_seed": source_seed,
                "target_seed": target_seed,
                "context_len": 64,
                "layer": layer_names(target_model.config)[final_layer],
                "layer_index": final_layer,
                "unaligned_accuracy": baseline["latent_synthesized"]["policy_accuracy"],
                "unaligned_margin_mean": baseline["synthesized_decision_margin"]["mean"],
            }
            if source_seed != target_seed:
                # Fit target->source affine/procrustes maps on matched latent examples.
                target_fit = seed_train[target_seed]
                affine = fit_affine_alignment(target_fit.representations[final_layer], source_fit.representations[final_layer])
                mapped_affine = apply_affine_alignment(target_eval.representations[final_layer], affine)
                restored_affine = evaluate_readouts(readout, mapped_affine, target_eval)
                proc = fit_orthogonal_procrustes(target_fit.representations[final_layer], source_fit.representations[final_layer])
                mapped_proc = apply_orthogonal_procrustes(target_eval.representations[final_layer], proc)
                restored_proc = evaluate_readouts(readout, mapped_proc, target_eval)
                item.update(
                    {
                        "affine_accuracy": restored_affine["latent_synthesized"]["policy_accuracy"],
                        "affine_quality": alignment_quality(mapped_affine, source_fit.representations[final_layer][: mapped_affine.shape[0]]),
                        "orthogonal_procrustes_accuracy": restored_proc["latent_synthesized"]["policy_accuracy"],
                        "orthogonal_procrustes_quality": alignment_quality(mapped_proc, source_fit.representations[final_layer][: mapped_proc.shape[0]]),
                    }
                )
            else:
                item.update(
                    {
                        "affine_accuracy": baseline["latent_synthesized"]["policy_accuracy"],
                        "orthogonal_procrustes_accuracy": baseline["latent_synthesized"]["policy_accuracy"],
                    }
                )
            rows.append(item)
    atomic_write_json(out_dir / "seed_transfer.json", rows)
    return rows


def write_figures(out_dir: Path, aggregate: dict[str, Any]) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def svg_line(title: str, x_label: str, y_label: str, series: list[dict[str, Any]], path: Path) -> None:
        width, height = 780, 460
        left, right, top, bottom = 74, 28, 38, 62
        plot_w, plot_h = width - left - right, height - top - bottom
        xs = [x for s in series for x in s["x"]]
        ys = [y for s in series for y in s["y"]]
        if not xs or not ys:
            return
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(0.0, min(ys)), max(ys)
        if x_min == x_max:
            x_max += 1.0
        if y_min == y_max:
            y_max += 1.0
        pad = (y_max - y_min) * 0.08
        y_min -= pad
        y_max += pad
        def sx(x: float) -> float:
            return left + (x - x_min) / (x_max - x_min) * plot_w
        def sy(y: float) -> float:
            return top + (y_max - y) / (y_max - y_min) * plot_h
        colors = ["#174ea6", "#c5221f", "#188038", "#b06000", "#6f42c1", "#00796b", "#444"]
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="{width/2}" y="24" text-anchor="middle" font-family="sans-serif" font-size="16">{title}</text>',
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
            f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="12">{x_label}</text>',
            f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-family="sans-serif" font-size="12">{y_label}</text>',
        ]
        for index, item in enumerate(series):
            color = colors[index % len(colors)]
            points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(item["x"], item["y"]))
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>')
            for x, y in zip(item["x"], item["y"]):
                parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3" fill="{color}"/>')
            legend_y = top + 18 + index * 18
            parts.append(f'<rect x="{width-250}" y="{legend_y-10}" width="12" height="12" fill="{color}"/>')
            parts.append(f'<text x="{width-232}" y="{legend_y}" font-family="sans-serif" font-size="11">{item["label"]}</text>')
        parts.append("</svg>")
        path.write_text("\n".join(parts))

    layer_rows = aggregate["layer_metrics"]
    for variable in ("witness", "scope"):
        series = []
        for context in CONTEXTS:
            rows = [r for r in layer_rows if r["eval_context"] == context]
            rows = sorted(rows, key=lambda r: r["layer_index"])
            series.append(
                {
                    "label": f"{variable} ctx {context}",
                    "x": [r["layer_index"] for r in rows],
                    "y": [r[f"{variable}_accuracy"]["mean"] for r in rows],
                }
            )
        svg_line(f"Figure 1: {variable} decodability vs layer", "layer", "accuracy", series, fig_dir / f"figure1_{variable}_decodability.svg")

    rows = sorted(aggregate["layer_metrics"], key=lambda r: (r["eval_context"], r["layer_index"]))
    series = [
        {
            "label": f"accuracy ctx {context}",
            "x": [r["layer_index"] for r in rows if r["eval_context"] == context],
            "y": [r["synth_accuracy"]["mean"] for r in rows if r["eval_context"] == context],
        }
        for context in CONTEXTS
    ]
    svg_line("Figure 2a: synthesized-gate accuracy", "layer", "accuracy", series, fig_dir / "figure2a_gate_accuracy.svg")
    series = [
        {
            "label": f"margin ctx {context}",
            "x": [r["layer_index"] for r in rows if r["eval_context"] == context],
            "y": [r["synth_margin_mean"]["mean"] for r in rows if r["eval_context"] == context],
        }
        for context in CONTEXTS
    ]
    svg_line("Figure 2b: synthesized-gate margin", "layer", "decision margin", series, fig_dir / "figure2b_gate_margin.svg")

    transfer = [r for r in aggregate["cross_context_transfer"] if r["layer"] == "layer_4"]
    series = []
    for built in CONTEXTS:
        rows = sorted([r for r in transfer if r["built_context"] == built], key=lambda r: r["eval_context"])
        series.append({"label": f"built {built}", "x": [r["eval_context"] for r in rows], "y": [r["synth_accuracy"]["mean"] for r in rows]})
    svg_line("Figure 3: cross-context gate transfer", "evaluation context", "synthesized accuracy", series, fig_dir / "figure3_cross_context_transfer.svg")

    interventions = aggregate["intervention_metrics"]
    series = []
    for context in CONTEXTS:
        rows = sorted(
            [r for r in interventions if r["context_len"] == context and r["intervention"] == "witness" and r["direction"] == "witness"],
            key=lambda r: r["layer_index"],
        )
        series.append({"label": f"witness ctx {context}", "x": [r["layer_index"] for r in rows], "y": [r["synth_intervention"]["mean"] for r in rows]})
    svg_line("Figure 4: witness intervention consistency", "layer", "consistency", series, fig_dir / "figure4_intervention_consistency.svg")

    alignment = [r for r in aggregate["context_alignment"] if r["method"] == "affine" and r["layer"] == "layer_4"]
    series = []
    for source in (256, 1024):
        rows = sorted([r for r in alignment if r["source_context"] == source], key=lambda r: r["source_context"])
        if rows:
            series.append({"label": f"baseline {source}", "x": [source], "y": [rows[0]["baseline_accuracy"]["mean"]]})
            series.append({"label": f"affine restored {source}", "x": [source], "y": [rows[0]["restored_accuracy"]["mean"]]})
    svg_line("Figure 5: affine alignment restoration", "source context", "accuracy", series, fig_dir / "figure5_alignment_restoration.svg")

    position = [r for r in aggregate["position_effects"] if r["mode"] in ("scaled", "fixed_absolute", "fixed_distance") and r["layer"] == "layer_4"]
    series = []
    for mode in ("scaled", "fixed_absolute", "fixed_distance"):
        rows = sorted([r for r in position if r["mode"] == mode], key=lambda r: r["context_len"])
        series.append({"label": mode, "x": [r["context_len"] for r in rows], "y": [r["synth_accuracy"]["mean"] for r in rows]})
    svg_line("Figure 6: position/distance effect", "context length", "accuracy", series, fig_dir / "figure6_position_distance.svg")


def aggregate(run_payloads: list[dict[str, Any]], seed_transfer: list[dict[str, Any]]) -> dict[str, Any]:
    layer_rows = [row for payload in run_payloads for row in payload["layer_metrics"]]
    transfer_rows = [row for payload in run_payloads for row in payload["cross_context_transfer"]]
    geometry_rows = [row for payload in run_payloads for row in payload["representation_geometry"]]
    intervention_rows = [row for payload in run_payloads for row in payload["intervention_metrics"]]
    alignment_rows = [row for payload in run_payloads for row in payload["context_alignment"]]
    position_rows = [row for payload in run_payloads for row in payload["position_effects"]]
    upstream_rows = [row for payload in run_payloads for row in payload["upstream_task"]]
    probe_illusion_rows = [row for payload in run_payloads for row in payload["probe_illusion"]]
    counterexamples = [row for payload in run_payloads for row in payload["counterexamples"]]

    layer_metrics = aggregate_rows(
        layer_rows,
        ["built_context", "eval_context", "layer", "layer_index"],
        {
            "proposal_accuracy": ["probe_accuracy", "proposal"],
            "witness_accuracy": ["probe_accuracy", "witness"],
            "scope_accuracy": ["probe_accuracy", "scope"],
            "nuisance_accuracy": ["probe_accuracy", "nuisance"],
            "decision_probe_accuracy": ["probe_accuracy", "decision"],
            "synth_accuracy": ["latent_synthesized", "policy_accuracy"],
            "synth_false_admit": ["latent_synthesized", "policy_violation_rate"],
            "learned_policy_accuracy": ["latent_learned", "policy_accuracy"],
            "synth_margin_mean": ["synthesized_decision_margin", "mean"],
            "synth_margin_min": ["synthesized_decision_margin", "min"],
        },
    )
    same_context = [row for row in layer_metrics if row["built_context"] == row["eval_context"]]

    transfer_metrics = aggregate_rows(
        transfer_rows,
        ["built_context", "eval_context", "layer", "layer_index"],
        {
            "witness_accuracy": ["probe_accuracy", "witness"],
            "scope_accuracy": ["probe_accuracy", "scope"],
            "synth_accuracy": ["latent_synthesized", "policy_accuracy"],
            "synth_false_admit": ["latent_synthesized", "policy_violation_rate"],
            "synth_margin_mean": ["synthesized_decision_margin", "mean"],
        },
    )
    intervention_metrics_agg = aggregate_rows(
        intervention_rows,
        ["context_len", "layer", "layer_index", "intervention", "direction"],
        {
            "synth_intervention": ["latent_synthesized_intervention_consistency"],
            "learned_intervention": ["latent_learned_intervention_consistency"],
            "target_stability": ["target_stability_without_intervention"],
            "proposal_preservation": ["proposal_preservation"],
        },
    )
    alignment_metrics = aggregate_rows(
        alignment_rows,
        ["source_context", "target_context", "layer", "layer_index", "method"],
        {
            "baseline_accuracy": ["baseline_source_accuracy"],
            "target_accuracy": ["target_64_accuracy"],
            "restored_accuracy": ["restored_accuracy"],
            "baseline_margin": ["baseline_source_margin_mean"],
            "restored_margin": ["restored_margin_mean"],
            "alignment_mean_cosine": ["alignment_quality", "mean_cosine"],
            "alignment_rmse": ["alignment_quality", "rmse"],
        },
    )
    position_metrics = aggregate_rows(
        position_rows,
        ["context_len", "mode", "layer", "layer_index"],
        {
            "synth_accuracy": ["latent_synthesized", "policy_accuracy"],
            "synth_false_admit": ["latent_synthesized", "policy_violation_rate"],
            "synth_margin_mean": ["synthesized_decision_margin", "mean"],
        },
    )
    seed_transfer_metrics = aggregate_rows(
        seed_transfer,
        ["source_seed", "target_seed", "context_len", "layer", "layer_index"],
        {
            "unaligned_accuracy": ["unaligned_accuracy"],
            "affine_accuracy": ["affine_accuracy"],
            "orthogonal_procrustes_accuracy": ["orthogonal_procrustes_accuracy"],
        },
    )
    upstream_metrics = aggregate_rows(
        upstream_rows,
        ["context_len"],
        {
            "upstream_proposal_accuracy": ["sampled", "proposal_accuracy"],
            "upstream_witness_accuracy": ["sampled", "witness_accuracy"],
            "upstream_scope_accuracy": ["sampled", "scope_accuracy"],
            "upstream_decision_accuracy": ["sampled", "decision", "policy_accuracy"],
            "upstream_false_admit": ["sampled", "decision", "policy_violation_rate"],
        },
    )
    geometry_metrics = aggregate_rows(
        geometry_rows,
        ["context_len", "layer", "layer_index", "variable"],
        {
            "within_class_distance": ["within_class_distance_mean"],
            "between_centroid_distance": ["between_centroid_distance_mean"],
            "between_centroid_min": ["between_centroid_distance_min"],
            "centroid_cosine_mean": ["centroid_cosine_offdiag_mean"],
        },
    )

    return {
        "schema": SCHEMA,
        "runs": len(run_payloads),
        "contexts": CONTEXTS,
        "upstream_task": upstream_metrics,
        "layer_metrics": same_context,
        "cross_context_transfer": transfer_metrics,
        "representation_geometry": geometry_metrics,
        "intervention_metrics": intervention_metrics_agg,
        "context_alignment": alignment_metrics,
        "seed_transfer": seed_transfer_metrics,
        "position_effects": position_metrics,
        "probe_illusion": probe_illusion_rows,
        "counterexamples": counterexamples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("results/latent_autopsy"))
    parser.add_argument("--projection-results", type=Path, default=PROJECTION_RESULTS_DIR)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--train-repeats", type=int, default=16)
    parser.add_argument("--eval-repeats", type=int, default=12)
    parser.add_argument("--intervention-batch-size", type=int, default=96)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.out_dir / "environment.json", environment_report())
    inventory = checkpoint_inventory(args.projection_results)
    if args.seeds:
        requested = {int(part.strip()) for part in args.seeds.split(",") if part.strip()}
        inventory = [item for item in inventory if int(item["seed"]) in requested]
    atomic_write_json(args.out_dir / "checkpoint_inventory.json", inventory)
    missing = [item for item in inventory if not item["checkpoint_exists"]]
    if missing:
        raise SystemExit(f"missing checkpoints: {missing}")

    manifest_path = args.out_dir / "manifest.json"
    manifest = {"schema": SCHEMA, "completed": [], "runs": {}}
    if manifest_path.exists() and not args.force:
        manifest = read_json(manifest_path)

    run_payloads = []
    for item in inventory:
        seed = int(item["seed"])
        run_name = f"latent_autopsy_seed_{seed}"
        run_path = args.out_dir / "runs" / f"{run_name}.json"
        if run_name in manifest.get("completed", []) and run_path.exists() and not args.force:
            payload = read_json(run_path)
        else:
            payload = run_seed(
                seed,
                Path(item["checkpoint_path"]),
                args.out_dir,
                args.train_repeats,
                args.eval_repeats,
                args.intervention_batch_size,
                args.force,
            )
            if run_name not in manifest["completed"]:
                manifest["completed"].append(run_name)
            manifest["runs"][run_name] = str(run_path)
            atomic_write_json(manifest_path, manifest)
        run_payloads.append(payload)

    seed_transfer = seed_transfer_analysis(run_payloads, args.out_dir, args.eval_repeats)
    aggregate_payload = aggregate(run_payloads, seed_transfer)
    atomic_write_json(args.out_dir / "aggregate.json", aggregate_payload)
    atomic_write_json(args.out_dir / "layer_metrics.json", aggregate_payload["layer_metrics"])
    atomic_write_json(args.out_dir / "context_alignment.json", aggregate_payload["context_alignment"])
    atomic_write_json(args.out_dir / "intervention_metrics.json", aggregate_payload["intervention_metrics"])
    atomic_write_json(args.out_dir / "gate_margins.json", aggregate_payload["layer_metrics"])
    atomic_write_json(args.out_dir / "representation_geometry.json", aggregate_payload["representation_geometry"])
    atomic_write_json(args.out_dir / "counterexamples.json", aggregate_payload["counterexamples"])
    atomic_write_json(args.out_dir / "position_effects.json", aggregate_payload["position_effects"])
    write_figures(args.out_dir, aggregate_payload)


if __name__ == "__main__":
    main()
