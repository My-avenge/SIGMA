"""Self-vision dual-mask equations from SIGMA Sections 2.2 and Appendix B."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .config import MaskConfig


def _check_patch_map(x: torch.Tensor, name: str) -> None:
    if x.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, patches], got {tuple(x.shape)}")
    if x.shape[-1] < 1:
        raise ValueError(f"{name} must contain at least one patch")
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} contains non-finite values")


def standardize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Patch-wise population standardization."""
    _check_patch_map(x, "x")
    stable = x.float()
    mean = stable.mean(dim=-1, keepdim=True)
    std = stable.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
    return (stable - mean) / std


def polarity_corrected_relevance(raw_scores: torch.Tensor, polarity: int = -1) -> torch.Tensor:
    """Eq. (2): sigmoid(Std(gamma_E * score))."""
    if polarity not in {-1, 1}:
        raise ValueError("polarity must be -1 or +1")
    return torch.sigmoid(standardize(raw_scores * float(polarity)))


def support_seed(
    image_relevance: torch.Tensor,
    query_relevance: Optional[torch.Tensor],
    question_weight: float,
) -> torch.Tensor:
    """Eq. (3), with image-only fallback when no resolvable query exists."""
    _check_patch_map(image_relevance, "image_relevance")
    if (image_relevance < 0.0).any() or (image_relevance > 1.0).any():
        raise ValueError("image_relevance must be bounded by [0, 1]")
    image_relevance = image_relevance.float()
    if not 0.0 <= question_weight <= 1.0:
        raise ValueError("question_weight must be in [0, 1]")
    if query_relevance is None:
        return image_relevance.clamp(0.0, 1.0)
    _check_patch_map(query_relevance, "query_relevance")
    if (query_relevance < 0.0).any() or (query_relevance > 1.0).any():
        raise ValueError("query_relevance must be bounded by [0, 1]")
    query_relevance = query_relevance.float()
    if query_relevance.shape != image_relevance.shape:
        raise ValueError("image and query relevance must have the same shape")
    agreement = torch.sqrt((image_relevance * query_relevance).clamp_min(0.0))
    seed = (1.0 - question_weight) * agreement + question_weight * query_relevance
    return seed.clamp(0.0, 1.0)


def smooth_patch_grid(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    _check_patch_map(x, "x")
    if kernel_size <= 1:
        return x
    grid = math.isqrt(x.shape[-1])
    if grid * grid != x.shape[-1]:
        return x
    x_grid = x.reshape(x.shape[0], 1, grid, grid)
    pad = kernel_size // 2
    x_grid = F.pad(x_grid, (pad, pad, pad, pad), mode="replicate")
    return F.avg_pool2d(x_grid, kernel_size=kernel_size, stride=1).reshape_as(x)


def sparsemax(x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Sparsemax over the last dimension with explicit temperature scaling."""
    _check_patch_map(x, "x")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    z = x.float() / temperature
    z_sorted, _ = torch.sort(z, dim=-1, descending=True)
    z_cumsum = z_sorted.cumsum(dim=-1)
    ranks = torch.arange(1, z.shape[-1] + 1, device=z.device, dtype=z.dtype)
    support = 1.0 + ranks * z_sorted > z_cumsum
    support_size = support.sum(dim=-1, keepdim=True).clamp_min(1)
    tau = (z_cumsum.gather(-1, support_size - 1) - 1.0) / support_size.to(z.dtype)
    return torch.clamp(z - tau, min=0.0)


def normalize_max(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    _check_patch_map(x, "x")
    return x / x.amax(dim=-1, keepdim=True).clamp_min(eps)


@dataclass(frozen=True)
class DualMaskOutput:
    support_mask: torch.Tensor
    complement_mask: torch.Tensor
    support_seed: torch.Tensor
    support_score: torch.Tensor
    complement_score: torch.Tensor
    context_prior: torch.Tensor


def _finalize_from_seed(
    seed: torch.Tensor,
    cls_patch_similarity: torch.Tensor,
    config: MaskConfig,
) -> DualMaskOutput:
    _check_patch_map(seed, "seed")
    _check_patch_map(cls_patch_similarity, "cls_patch_similarity")
    if seed.shape != cls_patch_similarity.shape:
        raise ValueError("seed and cls_patch_similarity must have the same shape")

    anti_query = (1.0 - seed).clamp(0.0, 1.0)
    context_prior = torch.sigmoid(standardize(cls_patch_similarity)) * anti_query

    support_score = seed * (1.0 - config.context_weight * context_prior)
    complement_score = anti_query + config.context_weight * context_prior
    support_score = smooth_patch_grid(support_score.clamp(0.0, 1.0), config.smooth_kernel_size)
    complement_score = smooth_patch_grid(
        complement_score.clamp(0.0, 1.0), config.smooth_kernel_size
    )

    support_mask = normalize_max(sparsemax(support_score, config.temperature))
    complement_mask = normalize_max(sparsemax(complement_score, config.temperature))
    return DualMaskOutput(
        support_mask=support_mask,
        complement_mask=complement_mask,
        support_seed=seed,
        support_score=support_score,
        complement_score=complement_score,
        context_prior=context_prior,
    )


def build_dual_masks(
    image_relevance: torch.Tensor,
    query_relevance: Optional[torch.Tensor],
    cls_patch_similarity: torch.Tensor,
    config: MaskConfig = MaskConfig(),
) -> DualMaskOutput:
    """Build one support/complement pair from calibrated relevance maps."""
    seed = support_seed(image_relevance, query_relevance, config.question_weight)
    return _finalize_from_seed(seed, cls_patch_similarity, config)


def noisy_or(values: torch.Tensor) -> torch.Tensor:
    """Union probability over the leading entity dimension."""
    if values.ndim != 3:
        raise ValueError("values must have shape [entities, batch, patches]")
    return 1.0 - torch.prod((1.0 - values).clamp(0.0, 1.0), dim=0)


@dataclass(frozen=True)
class MultiEntityMaskOutput:
    combined: DualMaskOutput
    per_entity_seeds: torch.Tensor


def build_multi_entity_masks(
    image_relevance: torch.Tensor,
    query_relevances: torch.Tensor,
    cls_patch_similarity: torch.Tensor,
    config: MaskConfig = MaskConfig(),
    combine: str = "noisy_or",
) -> MultiEntityMaskOutput:
    """Combine entity seeds, then apply Eq. (4) exactly once.

    This avoids double-applying the CLS context prior when entity-level views
    are aggregated.
    """
    if query_relevances.ndim != 3:
        raise ValueError("query_relevances must have shape [entities, batch, patches]")
    if query_relevances.shape[0] < 1:
        raise ValueError("query_relevances must contain at least one entity")
    if query_relevances.shape[1:] != image_relevance.shape:
        raise ValueError("query_relevances and image_relevance shapes are incompatible")
    seeds = torch.stack(
        [support_seed(image_relevance, query, config.question_weight) for query in query_relevances],
        dim=0,
    )
    if combine == "noisy_or":
        combined_seed = noisy_or(seeds)
    elif combine == "max":
        combined_seed = seeds.max(dim=0).values
    else:
        raise ValueError("combine must be 'noisy_or' or 'max'")
    return MultiEntityMaskOutput(
        combined=_finalize_from_seed(combined_seed, cls_patch_similarity, config),
        per_entity_seeds=seeds,
    )
