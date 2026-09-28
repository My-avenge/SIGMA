"""Model-agnostic SIGMA mechanism.

The package intentionally contains no model, dataset, or benchmark code.
Backbone adapters provide visual relevance tensors, stream logits, and task
candidate IDs; this package owns only the SIGMA equations and routing policy.
"""

from .assembly import apply_plausibility_constraint, assemble_residuals, support_only
from .config import MaskConfig, RoutingConfig, SigmaConfig, SUBMITTED_CONFIG
from .decoder import BranchState, CandidateSpec, synchronize_branch
from .masks import (
    DualMaskOutput,
    MultiEntityMaskOutput,
    build_dual_masks,
    build_multi_entity_masks,
    polarity_corrected_relevance,
    sparsemax,
)
from .routing import RoutePreparation, StepResult, finish_route, prepare_route, route_step
from .vision import reweight_patch_tokens

__all__ = [
    "MaskConfig",
    "RoutingConfig",
    "SigmaConfig",
    "SUBMITTED_CONFIG",
    "DualMaskOutput",
    "MultiEntityMaskOutput",
    "build_dual_masks",
    "build_multi_entity_masks",
    "polarity_corrected_relevance",
    "sparsemax",
    "support_only",
    "assemble_residuals",
    "apply_plausibility_constraint",
    "RoutePreparation",
    "StepResult",
    "prepare_route",
    "finish_route",
    "route_step",
    "BranchState",
    "CandidateSpec",
    "synchronize_branch",
    "reweight_patch_tokens",
]
