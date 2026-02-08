from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
import torch

@dataclass(frozen=True)
class SparseMask:
    """
    Canonical mask representation contract between SDE3 (mask) and SDE2 (sparse forward).

    Recommended: store as per-token neighbor indices (ragged) or COO indices.
    Avoid dense [T,T] masks.
    """
    # Example placeholder fields;
    indices: Any  # TODO: define exact type (e.g., LongTensor [nnz, 2] or list[list[int]])
    shape: tuple[int, int]  # (T, T)

def build_sparse_mask(
    T: int,
    *,
    window_size: int,
    expander_degree: int,
    device: torch.device,
    seed: int = 0,
    cache: Optional[dict] = None,
) -> SparseMask:
    """
    Build a deterministic causal structured sparse mask combining:
    - local sliding window of size W
    - expander neighbors of degree d

    Requirements (Epic 2):
    - causal (no future edges)
    - deterministic for same (T, W, d, seed)
    - O(T*(W+d)) complexity
    """
    raise NotImplementedError("Not implemented.")
