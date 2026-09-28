"""Model-adapter contracts and cache-safe selective branch synchronization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Tuple

import torch


@dataclass(frozen=True)
class CandidateSpec:
    candidate_ids: torch.Tensor
    positive_candidate_ids: torch.Tensor


@dataclass
class BranchState:
    cache: Any = None
    consumed_tokens: int = 0


class StreamBackend(Protocol):
    def forward_stream(
        self,
        stream: str,
        input_ids: torch.Tensor,
        cache: Any,
    ) -> Tuple[torch.Tensor, Any]:
        """Return last-position logits and an updated cache."""


def synchronize_branch(
    backend: StreamBackend,
    stream: str,
    generated_ids: torch.Tensor,
    state: BranchState,
) -> torch.Tensor:
    """Advance a selectively used branch through every unconsumed token.

    A branch can be skipped for several decoding steps. On reactivation, its KV
    cache must consume the entire missing suffix, not only the newest token.
    """
    if generated_ids.ndim != 2 or generated_ids.shape[0] != 1:
        raise ValueError("generated_ids must have shape [1, sequence]")
    sequence_length = int(generated_ids.shape[1])
    if state.consumed_tokens < 0 or state.consumed_tokens > sequence_length:
        raise ValueError("branch cache cursor is inconsistent with the generated prefix")
    missing = generated_ids[:, state.consumed_tokens :]
    if missing.shape[1] == 0:
        raise ValueError("branch is already synchronized to this prefix")
    logits, updated_cache = backend.forward_stream(stream, missing, state.cache)
    state.cache = updated_cache
    state.consumed_tokens = sequence_length
    return logits
