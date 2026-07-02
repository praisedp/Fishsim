"""The fish steering rules, vectorized over the whole population.

Same math as classic per-boid Reynolds steering, but every function takes arrays
covering all fish at once. Each rule builds a *desired* velocity per fish, then
:func:`steer_towards` turns it into a force: scale the desired direction to that
fish's top speed, subtract its current velocity, clamp to ``max_force``. This
keeps motion smooth instead of snapping.

The flocking rules consume the flat pair arrays from
:func:`neighbors.perception_pairs`; per-fish sums fall out of ``np.bincount``
over the pair's fish index (one bincount per vector component).
"""

from __future__ import annotations

import numpy as np

from config import Config
from genome import F_FEAR, F_MAXSPD, F_WALI, F_WCOH, F_WFLEE, F_WSEP

# Guards divisions by ~zero lengths only; never changes which rows get clamped
# or which pairs count (those cuts are made on exact comparisons beforehand).
EPS = 1e-12


def limit_norm(v: np.ndarray, max_len: np.ndarray | float) -> np.ndarray:
    """Rows of ``v`` ``(n, 2)`` shortened to at most ``max_len`` (never lengthened).

    ``max_len`` may be a scalar or per-row ``(n,)``.
    """
    norm = np.sqrt(np.einsum("nc,nc->n", v, v))
    scale = np.minimum(1.0, max_len / np.maximum(norm, EPS))
    return v * scale[:, None]


def steer_towards(
    desired: np.ndarray,
    vel: np.ndarray,
    max_speed: np.ndarray,
    max_force: float,
) -> np.ndarray:
    """Reynolds steering, row-wise: aim at ``desired`` (scaled to each fish's
    ``max_speed``), clamp the correction to ``max_force``.

    Rows with a zero desired vector produce zero force (a rule with nothing to
    say applies no correction).
    """
    d2 = np.einsum("nc,nc->n", desired, desired)
    norm = np.sqrt(d2)
    aimed = desired * (max_speed / np.maximum(norm, EPS))[:, None]
    steer = limit_norm(aimed - vel, max_force)
    steer[d2 == 0.0] = 0.0
    return steer


def flock_desires(
    i: np.ndarray,
    j: np.ndarray,
    rel: np.ndarray,
    d2: np.ndarray,
    vel: np.ndarray,
    n: int,
    separation_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three classic flocking desires per fish: ``(separation, alignment,
    cohesion)``, each ``(n, 2)``, from the flat neighbour-pair arrays.

    - **separation**: mean of ``-rel / d²`` over *close* neighbours — direction
      away from each, weighted inverse-squared so nearer fish push harder.
    - **alignment**: mean neighbour velocity.
    - **cohesion**: ``mean(pos_j) - pos_i``, computed as ``mean(rel)`` — same
      thing, and correct across the wrap seam since ``rel`` is minimum-imaged.

    Fish with no (close) neighbours get a zero desire, which steers to nothing.
    """

    def per_fish_mean(pair_values: np.ndarray, divisor: np.ndarray) -> np.ndarray:
        sums = np.stack(
            [np.bincount(i, weights=pair_values[:, c], minlength=n) for c in (0, 1)],
            axis=1,
        )
        return sums / divisor[:, None]

    cnt = np.maximum(np.bincount(i, minlength=n), 1)
    alignment = per_fish_mean(vel[j], cnt)
    cohesion = per_fish_mean(rel, cnt)

    close = (d2 > 0.0) & (d2 < separation_radius**2)
    away = np.where(close[:, None], -rel / np.maximum(d2, EPS)[:, None], 0.0)
    close_cnt = np.maximum(np.bincount(i, weights=close.astype(np.float64), minlength=n), 1.0)
    separation = per_fish_mean(away, close_cnt)
    return separation, alignment, cohesion


def flee_desire(
    offset: np.ndarray,
    d2: np.ndarray,
    fear: np.ndarray,
    flee_radius: float,
) -> np.ndarray:
    """Desire away from each fish's nearest shark — stronger up close.

    ``offset`` points from the shark to the fish. The trigger range is the base
    ``flee_radius`` scaled by each fish's ``fear`` gene, so timid fish bolt early
    and bold fish hold their nerve. ``offset / d²`` is the unit direction over
    distance: closer shark, larger desire (steering renormalises magnitude).
    """
    radius2 = (flee_radius * fear) ** 2
    active = (d2 > 0.0) & (d2 <= radius2)
    return np.where(active[:, None], offset / np.maximum(d2, EPS)[:, None], 0.0)


def fish_accelerations(
    vel: np.ndarray,
    genes: np.ndarray,
    i: np.ndarray,
    j: np.ndarray,
    rel: np.ndarray,
    d2: np.ndarray,
    flee_offset: np.ndarray | None,
    flee_d2: np.ndarray | None,
    cfg: Config,
) -> np.ndarray:
    """The whole fish brain for one tick: the four rules, steered and summed with
    each fish's own genome weights. ``flee_offset`` / ``flee_d2`` are None when no
    sharks are alive (nothing to flee)."""
    n = len(vel)
    sep, ali, coh = flock_desires(i, j, rel, d2, vel, n, cfg.separation_radius)
    max_speed = genes[:, F_MAXSPD]

    accel = genes[:, F_WSEP, None] * steer_towards(sep, vel, max_speed, cfg.max_force)
    accel += genes[:, F_WALI, None] * steer_towards(ali, vel, max_speed, cfg.max_force)
    accel += genes[:, F_WCOH, None] * steer_towards(coh, vel, max_speed, cfg.max_force)
    if flee_offset is not None:
        flee = flee_desire(flee_offset, flee_d2, genes[:, F_FEAR], cfg.flee_radius)
        accel += genes[:, F_WFLEE, None] * steer_towards(flee, vel, max_speed, cfg.max_force)
    return accel
