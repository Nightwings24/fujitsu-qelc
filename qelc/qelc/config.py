"""Deterministic seed derivation shared by every stochastic component.

All randomness in the package flows from a single master seed through
``np.random.SeedSequence.spawn``-style keyed derivation, so any run is exactly
reproducible and independent components never share a stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def derive_seed(master_seed: int, *key: int) -> int:
    """Derive a 32-bit child seed from a master seed and an integer key path.

    Stable across processes and platforms (unlike hash()).
    """
    ss = np.random.SeedSequence(entropy=master_seed, spawn_key=tuple(key))
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def rng_for(master_seed: int, *key: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(entropy=master_seed, spawn_key=tuple(key)))


@dataclass
class RunConfig:
    """Knobs for a full QELC run. Every field has a reproducible default."""

    master_seed: int = 42

    # problem construction
    k_cycles: int = 20
    length_bound: int = 5
    penalty_eps_frac: float = 0.25
    log_weights: bool = False  # experiment flag only; headline results use linear weights

    # QAOA
    p: int = 2
    n_starts: int = 4
    maxiter: int = 100
    n_shots: int = 10_000
    cvar_alpha: float | None = None

    # peeling
    max_iters: int = 30
    extras: dict = field(default_factory=dict)
