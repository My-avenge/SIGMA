"""Paper-faithful support-first, complement-on-demand SIGMA routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from .assembly import apply_plausibility_constraint, assemble_residuals, support_only
from .config import RoutingConfig
from .uncertainty import (
    CandidateState,
    TriggerDecision,
    apply_positive_calibration,
    candidate_state,
    dynamic_alpha,
    needs_complement,
    trigger_decision,
)


@dataclass(frozen=True)
class RoutePreparation:
    trigger: TriggerDecision
    calibrated_global: torch.Tensor
    support_only_logits: torch.Tensor
    dynamic_alpha: float
    global_state: CandidateState
    support_state: CandidateState
    needs_complement: bool
    complement_flags: Dict[str, bool]


@dataclass(frozen=True)
class StepResult:
    final_logits: torch.Tensor
    triggered: bool
    used_support: bool
    used_complement: bool
    trigger: TriggerDecision
    dynamic_alpha: float
    complement_flags: Dict[str, bool]


def prepare_route(
    raw_global_logits: torch.Tensor,
    support_logits: torch.Tensor,
    candidate_ids: torch.Tensor,
    positive_candidate_ids: torch.Tensor,
    config: RoutingConfig,
    require_two_complement_flags: bool = False,
) -> RoutePreparation:
    trigger = trigger_decision(raw_global_logits, candidate_ids, config)
    if not trigger.triggered:
        raise ValueError("prepare_route must only be called for a triggered step")
    calibrated_global = apply_positive_calibration(
        raw_global_logits,
        positive_candidate_ids,
        trigger.candidate_entropy,
        config,
    )
    alpha = dynamic_alpha(
        config.alpha,
        trigger.full_entropy,
        trigger.candidate_entropy,
        int(candidate_ids.numel()),
    )
    support_logits_only = support_only(
        calibrated_global,
        support_logits,
        alpha,
        config.support_weight,
    )
    global_state = candidate_state(calibrated_global, candidate_ids, positive_candidate_ids)
    support_state = candidate_state(support_logits_only, candidate_ids, positive_candidate_ids)
    use_complement, flags = needs_complement(
        global_state,
        support_state,
        config,
        require_two_flags=require_two_complement_flags,
    )
    return RoutePreparation(
        trigger=trigger,
        calibrated_global=calibrated_global,
        support_only_logits=support_logits_only,
        dynamic_alpha=alpha,
        global_state=global_state,
        support_state=support_state,
        needs_complement=use_complement,
        complement_flags=flags,
    )


def finish_route(
    prepared: RoutePreparation,
    support_logits: torch.Tensor,
    complement_logits: Optional[torch.Tensor],
    config: RoutingConfig,
) -> StepResult:
    if prepared.needs_complement:
        if complement_logits is None:
            raise ValueError("complement logits are required by the scheduler")
        assembled = assemble_residuals(
            prepared.calibrated_global,
            support_logits,
            complement_logits,
            prepared.dynamic_alpha,
            config.support_weight,
            config.complement_weight,
        )
    else:
        assembled = prepared.support_only_logits
    final_logits = apply_plausibility_constraint(
        prepared.calibrated_global,
        assembled,
        config.plausibility_threshold,
        config.plausibility_penalty,
    )
    return StepResult(
        final_logits=final_logits,
        triggered=True,
        used_support=True,
        used_complement=prepared.needs_complement,
        trigger=prepared.trigger,
        dynamic_alpha=prepared.dynamic_alpha,
        complement_flags=prepared.complement_flags,
    )


def route_step(
    raw_global_logits: torch.Tensor,
    candidate_ids: torch.Tensor,
    positive_candidate_ids: torch.Tensor,
    config: RoutingConfig,
    support_logits: Optional[torch.Tensor] = None,
    complement_logits: Optional[torch.Tensor] = None,
    require_two_complement_flags: bool = False,
) -> StepResult:
    trigger = trigger_decision(raw_global_logits, candidate_ids, config)
    if not trigger.triggered:
        return StepResult(
            final_logits=raw_global_logits,
            triggered=False,
            used_support=False,
            used_complement=False,
            trigger=trigger,
            dynamic_alpha=0.0,
            complement_flags={},
        )
    if support_logits is None:
        raise ValueError("support logits are required for a triggered step")
    prepared = prepare_route(
        raw_global_logits,
        support_logits,
        candidate_ids,
        positive_candidate_ids,
        config,
        require_two_complement_flags=require_two_complement_flags,
    )
    return finish_route(prepared, support_logits, complement_logits, config)
