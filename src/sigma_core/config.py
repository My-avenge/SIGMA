"""Validated SIGMA configuration.

Defaults are the submitted-paper balanced operating point, not the stale
defaults retained by historical E-AGLA function signatures.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MaskConfig:
    top_k: int = 5
    question_weight: float = 0.10
    context_weight: float = 0.10
    temperature: float = 35.0
    smooth_kernel_size: int = 3
    min_keep_ratio: float = 0.15
    num_masked_layers: int = 2
    polarity: int = -1

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if not 0.0 <= self.question_weight <= 1.0:
            raise ValueError("question_weight must be in [0, 1]")
        if not 0.0 <= self.context_weight <= 1.0:
            raise ValueError("context_weight must be in [0, 1]")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.smooth_kernel_size < 1 or self.smooth_kernel_size % 2 == 0:
            raise ValueError("smooth_kernel_size must be a positive odd integer")
        if not 0.0 <= self.min_keep_ratio <= 1.0:
            raise ValueError("min_keep_ratio must be in [0, 1]")
        if self.num_masked_layers < 1:
            raise ValueError("num_masked_layers must be positive")
        if self.polarity not in {-1, 1}:
            raise ValueError("polarity must be -1 or +1")


@dataclass(frozen=True)
class RoutingConfig:
    alpha: float = 0.50
    full_entropy_threshold: float = 0.20
    candidate_entropy_threshold: float = 0.28
    calibration_threshold: float = 0.28
    calibration_strength: float = 0.12
    support_weight: float = 0.35
    complement_weight: float = 0.80
    plausibility_threshold: float = 0.01
    plausibility_penalty: float = 1.0e4
    complement_entropy_threshold: float = 0.24
    complement_margin_threshold: float = 0.08
    global_positive_support_threshold: float = 0.75
    support_positive_support_threshold: float = 0.60

    def __post_init__(self) -> None:
        nonnegative = {
            "alpha": self.alpha,
            "full_entropy_threshold": self.full_entropy_threshold,
            "candidate_entropy_threshold": self.candidate_entropy_threshold,
            "calibration_threshold": self.calibration_threshold,
            "calibration_strength": self.calibration_strength,
            "support_weight": self.support_weight,
            "complement_weight": self.complement_weight,
            "plausibility_threshold": self.plausibility_threshold,
            "plausibility_penalty": self.plausibility_penalty,
            "complement_entropy_threshold": self.complement_entropy_threshold,
            "complement_margin_threshold": self.complement_margin_threshold,
        }
        for name, value in nonnegative.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        for name, value in {
            "global_positive_support_threshold": self.global_positive_support_threshold,
            "support_positive_support_threshold": self.support_positive_support_threshold,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class SigmaConfig:
    mask: MaskConfig = field(default_factory=MaskConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)


SUBMITTED_CONFIG = SigmaConfig()
