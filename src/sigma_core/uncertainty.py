"""Uncertainty, calibration, and complement scheduling for SIGMA."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable

import torch
import torch.nn.functional as F

from .config import RoutingConfig


def _check_logits(logits: torch.Tensor) -> None:
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError("SIGMA routing currently requires logits with shape [1, vocabulary]")
    if not torch.isfinite(logits).all():
        raise ValueError("logits contain non-finite values")


def _check_ids(ids: torch.Tensor, vocabulary_size: int, name: str) -> None:
    if ids.ndim != 1 or ids.numel() < 1:
        raise ValueError(f"{name} must be a non-empty 1D tensor")
    if ids.dtype != torch.long:
        raise ValueError(f"{name} must use torch.long")
    if torch.unique(ids).numel() != ids.numel():
        raise ValueError(f"{name} must not contain duplicates")
    if ids.min().item() < 0 or ids.max().item() >= vocabulary_size:
        raise ValueError(f"{name} contains an out-of-range token ID")


def entropy_from_logits(logits: torch.Tensor) -> float:
    _check_logits(logits)
    probs = F.softmax(logits.float(), dim=-1)
    return float((-(probs * probs.clamp_min(1e-10).log()).sum(dim=-1))[0].item())


def candidate_entropy(logits: torch.Tensor, candidate_ids: torch.Tensor) -> float:
    _check_logits(logits)
    _check_ids(candidate_ids, logits.shape[-1], "candidate_ids")
    return entropy_from_logits(logits.index_select(-1, candidate_ids))


@dataclass(frozen=True)
class TriggerDecision:
    triggered: bool
    full_entropy: float
    candidate_entropy: float


def trigger_decision(
    raw_global_logits: torch.Tensor,
    candidate_ids: torch.Tensor,
    config: RoutingConfig,
) -> TriggerDecision:
    """Eq. (5), intentionally evaluated before positive calibration."""
    full = entropy_from_logits(raw_global_logits)
    candidate = candidate_entropy(raw_global_logits, candidate_ids)
    return TriggerDecision(
        triggered=(full > config.full_entropy_threshold)
        or (candidate > config.candidate_entropy_threshold),
        full_entropy=full,
        candidate_entropy=candidate,
    )


def dynamic_alpha(
    base_alpha: float,
    full_entropy: float,
    candidate_entropy_value: float,
    candidate_count: int,
) -> float:
    """Algorithm 2 line 15 with the actual candidate-set cardinality."""
    if candidate_count < 2:
        candidate_term = 0.0
    else:
        candidate_term = candidate_entropy_value / math.log(float(candidate_count))
    factor = 0.5 + 0.5 * max(min(full_entropy / 8.0, 1.0), min(candidate_term, 1.0))
    return base_alpha * factor


def apply_positive_calibration(
    raw_global_logits: torch.Tensor,
    positive_candidate_ids: torch.Tensor,
    candidate_entropy_value: float,
    config: RoutingConfig,
) -> torch.Tensor:
    """Eq. (6), applied at most once and only after correction is triggered."""
    _check_logits(raw_global_logits)
    _check_ids(positive_candidate_ids, raw_global_logits.shape[-1], "positive_candidate_ids")
    calibrated = raw_global_logits.clone()
    if candidate_entropy_value >= config.calibration_threshold and config.calibration_strength > 0.0:
        calibrated[:, positive_candidate_ids] += config.calibration_strength
    return calibrated


@dataclass(frozen=True)
class CandidateState:
    entropy: float
    margin: float
    top_choice: int
    positive_support: float


def candidate_state(
    logits: torch.Tensor,
    candidate_ids: torch.Tensor,
    positive_candidate_ids: torch.Tensor,
) -> CandidateState:
    _check_logits(logits)
    _check_ids(candidate_ids, logits.shape[-1], "candidate_ids")
    _check_ids(positive_candidate_ids, logits.shape[-1], "positive_candidate_ids")
    if not (positive_candidate_ids[:, None] == candidate_ids[None, :]).any(dim=1).all():
        raise ValueError("positive_candidate_ids must be a subset of candidate_ids")
    candidate_logits = logits.index_select(-1, candidate_ids).float()
    probs = F.softmax(candidate_logits, dim=-1)
    entropy = float((-(probs * probs.clamp_min(1e-10).log()).sum(dim=-1))[0].item())
    top_values, top_positions = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1)
    margin = 1.0 if probs.shape[-1] == 1 else float((top_values[0, 0] - top_values[0, 1]).abs().item())
    top_choice = int(candidate_ids[top_positions[0, 0]].item())
    is_positive = (candidate_ids[:, None] == positive_candidate_ids[None, :]).any(dim=1)
    positive_support = float(probs[0, is_positive].sum().item())
    return CandidateState(entropy, margin, top_choice, positive_support)


def needs_complement(
    global_state: CandidateState,
    support_state: CandidateState,
    config: RoutingConfig,
    require_two_flags: bool = False,
) -> tuple[bool, Dict[str, bool]]:
    flags = {
        "disagreement": global_state.top_choice != support_state.top_choice,
        "high_entropy": support_state.entropy >= config.complement_entropy_threshold,
        "low_margin": support_state.margin <= config.complement_margin_threshold,
        "unsupported_positive": (
            global_state.positive_support >= config.global_positive_support_threshold
            and support_state.positive_support <= config.support_positive_support_threshold
        ),
    }
    votes = sum(int(value) for value in flags.values())
    return (votes >= 2 if require_two_flags else votes >= 1), flags
