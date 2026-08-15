from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .latent_guard import LayerGuardSet
from .projection_model import ProjectionCausalTransformer, final_representations, make_generator
from .projection_task import (
    Nuisance,
    PolicyState,
    Proposal,
    Scope,
    Witness,
    batch_from_states,
)
from .synthesized_latent_gate import synthesize_policy_from_variables


@dataclass(frozen=True)
class InterventionConfig:
    seq_len: int = 64
    batch_size: int = 256
    seed: int = 1701


def directional_swap(source: Tensor, target: Tensor, direction: Tensor) -> Tensor:
    direction = direction.to(target.device, target.dtype)
    norm = torch.linalg.norm(direction)
    if float(norm.item()) == 0.0:
        raise ValueError("cannot swap along a zero direction")
    unit = direction / norm
    source_scalar = source @ unit
    target_scalar = target @ unit
    return target + (source_scalar - target_scalar).unsqueeze(-1) * unit.unsqueeze(0)


def binary_probe_direction(weight: Tensor) -> Tensor:
    if weight.shape[0] != 2:
        raise ValueError("binary probe direction requires a 2-class head")
    return (weight[1] - weight[0]).detach()


def _paired_states(kind: str, batch_size: int) -> tuple[list[PolicyState], list[PolicyState]]:
    states_a: list[PolicyState] = []
    states_b: list[PolicyState] = []
    for index in range(batch_size):
        if kind == "witness":
            proposal = Proposal.REMEDIATE_A if index % 2 == 0 else Proposal.REMEDIATE_B
            scope = Scope.A if proposal == Proposal.REMEDIATE_A else Scope.B
            states_a.append(PolicyState(int(proposal), int(Witness.VALID), int(scope), int(Nuisance.ONE)))
            states_b.append(PolicyState(int(proposal), int(Witness.INVALID), int(scope), int(Nuisance.ONE)))
        elif kind == "scope":
            proposal = Proposal.REMEDIATE_A if index % 2 == 0 else Proposal.REMEDIATE_B
            good_scope = Scope.A if proposal == Proposal.REMEDIATE_A else Scope.B
            bad_scope = Scope.B if proposal == Proposal.REMEDIATE_A else Scope.A
            states_a.append(PolicyState(int(proposal), int(Witness.VALID), int(good_scope), int(Nuisance.ONE)))
            states_b.append(PolicyState(int(proposal), int(Witness.VALID), int(bad_scope), int(Nuisance.ONE)))
        elif kind == "nuisance":
            proposal = Proposal.REMEDIATE_A if index % 2 == 0 else Proposal.REMEDIATE_B
            scope = Scope.A if proposal == Proposal.REMEDIATE_A else Scope.B
            states_a.append(PolicyState(int(proposal), int(Witness.VALID), int(scope), int(Nuisance.ONE)))
            states_b.append(PolicyState(int(proposal), int(Witness.VALID), int(scope), int(Nuisance.ZERO)))
        else:
            raise ValueError(f"unknown intervention kind {kind}")
    return states_a, states_b


def _decision_for_swapped(
    rep_source: Tensor,
    rep_target: Tensor,
    direction: Tensor,
    variable_probe,
    decision_probe,
) -> tuple[Tensor, Tensor, Tensor]:
    swapped = directional_swap(rep_source, rep_target, direction)
    variable_outputs = variable_probe(swapped)
    synth = synthesize_policy_from_variables(
        variable_outputs["proposal"],
        variable_outputs["witness"],
        variable_outputs["scope"],
    )
    learned = decision_probe(swapped).argmax(dim=-1)
    proposal = variable_outputs["proposal"].argmax(dim=-1)
    return synth, learned, proposal


@torch.no_grad()
def evaluate_interventions(
    model: ProjectionCausalTransformer,
    guards: LayerGuardSet,
    config: InterventionConfig,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    for module in guards.modules_for_optim():
        module.eval()

    rows: list[dict[str, Any]] = []
    generator = make_generator(config.seed, device)
    for kind in ("witness", "scope"):
        states_a, states_b = _paired_states(kind, config.batch_size)
        batch_a = batch_from_states(states_a, config.seq_len, device)
        batch_b = batch_from_states(states_b, config.seq_len, device)
        reps_a = final_representations(model, batch_a)
        reps_b = final_representations(model, batch_b)
        expected_a = batch_a.decision
        expected_b = batch_b.decision
        proposal_a = batch_a.proposal
        proposal_b = batch_b.proposal
        for layer_index, layer in enumerate(guards.layer_labels):
            variable_probe = guards.variable_probes[layer_index]
            decision_probe = guards.decision_probes[layer_index]
            if kind == "witness":
                direction = binary_probe_direction(variable_probe.witness.weight)
            else:
                direction = binary_probe_direction(variable_probe.scope.weight)

            synth_b_from_a, learned_b_from_a, proposal_b_after = _decision_for_swapped(
                reps_a[layer_index], reps_b[layer_index], direction, variable_probe, decision_probe
            )
            synth_a_from_b, learned_a_from_b, proposal_a_after = _decision_for_swapped(
                reps_b[layer_index], reps_a[layer_index], direction, variable_probe, decision_probe
            )
            synth_consistency = torch.cat(
                [(synth_b_from_a == expected_a), (synth_a_from_b == expected_b)],
                dim=0,
            ).float()
            learned_consistency = torch.cat(
                [(learned_b_from_a == expected_a), (learned_a_from_b == expected_b)],
                dim=0,
            ).float()
            proposal_preserved = torch.cat(
                [(proposal_b_after == proposal_b), (proposal_a_after == proposal_a)],
                dim=0,
            ).float()

            nuisance_direction = binary_probe_direction(variable_probe.nuisance.weight)
            synth_nuis_b_from_a, learned_nuis_b_from_a, _ = _decision_for_swapped(
                reps_a[layer_index], reps_b[layer_index], nuisance_direction, variable_probe, decision_probe
            )
            synth_nuis_a_from_b, learned_nuis_a_from_b, _ = _decision_for_swapped(
                reps_b[layer_index], reps_a[layer_index], nuisance_direction, variable_probe, decision_probe
            )
            nuisance_synth_changed = torch.cat(
                [(synth_nuis_b_from_a != expected_b), (synth_nuis_a_from_b != expected_a)],
                dim=0,
            ).float()
            nuisance_learned_changed = torch.cat(
                [(learned_nuis_b_from_a != expected_b), (learned_nuis_a_from_b != expected_a)],
                dim=0,
            ).float()

            random_direction = torch.randn(direction.shape, device=device, generator=generator)
            synth_rand_b_from_a, learned_rand_b_from_a, _ = _decision_for_swapped(
                reps_a[layer_index], reps_b[layer_index], random_direction, variable_probe, decision_probe
            )
            synth_rand_a_from_b, learned_rand_a_from_b, _ = _decision_for_swapped(
                reps_b[layer_index], reps_a[layer_index], random_direction, variable_probe, decision_probe
            )
            random_synth_consistency = torch.cat(
                [(synth_rand_b_from_a == expected_a), (synth_rand_a_from_b == expected_b)],
                dim=0,
            ).float()
            random_learned_consistency = torch.cat(
                [(learned_rand_b_from_a == expected_a), (learned_rand_a_from_b == expected_b)],
                dim=0,
            ).float()

            rows.append(
                {
                    "intervention": kind,
                    "layer": layer,
                    "layer_index": layer_index,
                    "latent_synthesized_intervention_consistency": float(synth_consistency.mean().item()),
                    "latent_learned_intervention_consistency": float(learned_consistency.mean().item()),
                    "proposal_preservation": float(proposal_preserved.mean().item()),
                    "nuisance_intervention_effect_synthesized": float(nuisance_synth_changed.mean().item()),
                    "nuisance_intervention_effect_learned": float(nuisance_learned_changed.mean().item()),
                    "random_direction_consistency_synthesized": float(random_synth_consistency.mean().item()),
                    "random_direction_consistency_learned": float(random_learned_consistency.mean().item()),
                    "samples": int(2 * config.batch_size),
                }
            )
    return rows
