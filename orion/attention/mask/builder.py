from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    cache: dict | None = None,
) -> SparseMask:
    """
    Build a deterministic causal structured sparse mask combining:
    - local sliding window of size W
    - expander neighbors of degree d

    Requirements:
    - causal (no future edges)
    - deterministic for same (T, W, d, seed)
    - O(T*(W+d)) complexity

    Returns SparseMask with COO indices format [2, nnz] where each row is (query_pos, key_pos).
    """
    # Check cache first
    if cache is not None:
        cache_key = (T, window_size, expander_degree, seed)
        if cache_key in cache:
            indices = cache[cache_key]
            return SparseMask(indices=indices.to(device), shape=(T, T))

    # Deterministic RNG for expander links
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    edges_q = []
    edges_k = []

    # Build window edges efficiently
    for q in range(T):
        window_start = max(0, q - window_size)
        window_end = q + 1
        window_len = window_end - window_start

        # Add window edges
        edges_q.extend([q] * window_len)
        edges_k.extend(range(window_start, window_end))

        # Add expander edges
        if expander_degree > 0 and q > 0:
            num_samples = min(expander_degree, q)
            if num_samples > 0:
                sampled = torch.randperm(q, generator=rng)[:num_samples].tolist()
                edges_q.extend([q] * num_samples)
                edges_k.extend(sampled)

    # Convert to COO indices tensor
    if edges_q:
        indices = torch.stack(
            [
                torch.tensor(edges_q, dtype=torch.long, device=device),
                torch.tensor(edges_k, dtype=torch.long, device=device),
            ]
        )  # [2, nnz]
    else:
        indices = torch.zeros((2, 0), dtype=torch.long, device=device)

    # Cache the result
    if cache is not None:
        cache_key = (T, window_size, expander_degree, seed)
        cache[cache_key] = indices.cpu()

    return SparseMask(indices=indices, shape=(T, T))
