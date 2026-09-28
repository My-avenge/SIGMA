"""Backbone-neutral operation used by tri-stream self-vision injection."""

from __future__ import annotations

import torch


def reweight_patch_tokens(
    hidden_states: torch.Tensor,
    mask: torch.Tensor,
    min_keep_ratio: float = 0.15,
    prefix_tokens: int = 1,
) -> torch.Tensor:
    """Algorithm 1 line 4: softly reweight patches and preserve prefix tokens.

    A backbone adapter is responsible for calling this operation immediately
    before self-attention in each of the selected upper vision layers.
    """
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [batch, tokens, channels]")
    if not hidden_states.is_floating_point():
        raise ValueError("hidden_states must be floating point")
    if mask.ndim != 2:
        raise ValueError("mask must have shape [batch, patches]")
    if hidden_states.shape[0] != mask.shape[0]:
        raise ValueError("hidden_states and mask batch sizes differ")
    if prefix_tokens < 0 or prefix_tokens > hidden_states.shape[1]:
        raise ValueError("prefix_tokens is outside the token sequence")
    if hidden_states.shape[1] - prefix_tokens != mask.shape[1]:
        raise ValueError("mask length does not match the number of patch tokens")
    if not 0.0 <= min_keep_ratio <= 1.0:
        raise ValueError("min_keep_ratio must be in [0, 1]")
    if not torch.isfinite(mask).all() or (mask < 0.0).any() or (mask > 1.0).any():
        raise ValueError("mask must be finite and bounded by [0, 1]")

    keep = min_keep_ratio + (1.0 - min_keep_ratio) * mask
    output = hidden_states.clone()
    output[:, prefix_tokens:, :] *= keep.to(output.dtype).unsqueeze(-1)
    return output
