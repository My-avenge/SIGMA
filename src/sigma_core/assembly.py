"""Structured residual assembly from SIGMA Eqs. (7) and (8)."""

from __future__ import annotations

import torch


def support_only(
    calibrated_global: torch.Tensor,
    support_logits: torch.Tensor,
    alpha: float,
    support_weight: float,
) -> torch.Tensor:
    return calibrated_global + alpha * support_weight * (support_logits - calibrated_global)


def assemble_residuals(
    calibrated_global: torch.Tensor,
    support_logits: torch.Tensor,
    complement_logits: torch.Tensor,
    alpha: float,
    support_weight: float,
    complement_weight: float,
) -> torch.Tensor:
    return calibrated_global + alpha * (
        support_weight * (support_logits - calibrated_global)
        - complement_weight * (complement_logits - calibrated_global)
    )


def apply_plausibility_constraint(
    calibrated_global: torch.Tensor,
    assembled_logits: torch.Tensor,
    threshold: float,
    penalty: float,
) -> torch.Tensor:
    if threshold < 0.0 or penalty < 0.0:
        raise ValueError("threshold and penalty must be non-negative")
    global_probs = torch.softmax(calibrated_global.float(), dim=-1)
    invalid = global_probs <= threshold
    return assembled_logits - invalid.to(assembled_logits.dtype) * penalty
