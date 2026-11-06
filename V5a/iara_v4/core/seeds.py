"""Seeded, independent random streams and the blind seed pools.

The protocol uses three disjoint pools:

* ``tune``       0..19   -- controller/planner tuning only
* ``validation`` 20..39  -- model selection and sanity checks
* ``test``       40..89  -- frozen final evaluation only

No episode may draw from the global ``random`` module or from NumPy's legacy
global state.  Every stochastic component receives a ``numpy.random.Generator``
derived from an explicit ``SeedSequence``.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np

SEED_POOLS: dict[str, range] = {
    "tune": range(0, 20),
    "validation": range(20, 40),
    "test": range(40, 90),
}


def pool(name: str) -> range:
    try:
        return SEED_POOLS[name]
    except KeyError as exc:
        raise KeyError(f"unknown seed pool {name!r}; choose from {sorted(SEED_POOLS)}") from exc


def all_pool_seeds() -> set[int]:
    return {seed for seeds in SEED_POOLS.values() for seed in seeds}


def validate_pools() -> None:
    """Raise if the three protocol pools overlap."""

    names = list(SEED_POOLS)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = set(SEED_POOLS[a]) & set(SEED_POOLS[b])
            if overlap:
                raise ValueError(f"seed pools {a!r} and {b!r} overlap: {sorted(overlap)}")


def stable_hash(text: str) -> int:
    """Deterministic 32-bit identifier, independent of PYTHONHASHSEED."""

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def rng_for(seed: int, *streams: int) -> np.random.Generator:
    """Create an independent generator for ``(seed, stream...)``.

    ``SeedSequence`` mixes the entropy and guarantees that changing one stream
    does not perturb the others.  Use a stable condition identifier as a stream
    when the same scenario must receive the same random draws across conditions.
    """

    entropy = [int(seed), *(int(s) for s in streams)]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def planner_rng(planner: str, scenario: str, seed: int) -> np.random.Generator:
    """Planner stream shared by every condition that uses the same planner.

    The stream deliberately excludes the condition name.  Consequently the
    RRT* tree (and therefore the global path) is identical for ``rrt_star_pp``,
    ``rrt_star_vfh_pp`` and ``rrt_star_pd`` at a given ``(scenario, seed)``.
    The VFH+ and controller ablations then differ only in the component under
    test, not in a different random global plan.
    """

    return rng_for(seed, stable_hash(f"planner:{planner}:{scenario}"))


def validate_seed_set(seeds: Iterable[int], phase: str) -> None:
    allowed = set(pool(phase))
    bad = sorted(set(seeds) - allowed)
    if bad:
        raise ValueError(f"seeds {bad} are outside the {phase!r} pool {sorted(allowed)}")
