"""Derived body physics for fish — pure functions of ``size`` and the config.

A fish's ``size`` is a state variable (it grows toward the ``adult_size`` gene),
and everything physical follows from it. Splitting these formulas out means the
engine (:mod:`sim`) and the tests share one definition — the tests use them as an
oracle rather than re-deriving the numbers.

All functions are vectorized: pass a scalar or an ``(n,)`` size array and get the
same shape back. The trade-offs they encode:

- **small fish are fast** (``max_speed_of``) but **store little** (``energy_cap_of``),
- **large fish are sluggish** (heavy ``mass``, so acceleration ``= force / mass``
  is small — momentum) and **cost more to live** (``metabolism`` scales with mass),
  but can hoard energy and, once grown, out-reproduce.
"""

from __future__ import annotations

import numpy as np

from config import Config


def mass_of(size):
    """Body mass, taken as area-like (``size**2``) for a 2-D creature."""
    return size * size


def mass_rel_of(size, cfg: Config):
    """Mass relative to a reference-size fish, i.e. ``(size / size_ref)**2``.

    Used as the ``acceleration = force / mass`` divisor so a reference fish has
    unit responsiveness and larger fish turn/accelerate proportionally slower.
    """
    return (size / cfg.size_ref) ** 2


def max_speed_of(size, cfg: Config):
    """Top speed: smaller fish are faster, clamped to a sane band.

    ``speed_ref * (size_ref / size)**speed_alpha`` — inverse in size, so a tiny
    juvenile darts and a big adult lumbers, both bounded by the speed caps.
    """
    speed = cfg.speed_ref * (cfg.size_ref / size) ** cfg.speed_alpha
    return np.clip(speed, cfg.speed_min_cap, cfg.speed_max_cap)


def energy_cap_of(size, cfg: Config):
    """Maximum energy a fish of this size can store (bigger body, bigger tank)."""
    return cfg.energy_cap_k * size * size


def metabolism(size, speed, perception, cfg: Config):
    """Energy burned this tick: a basal cost, a movement cost, and a sight cost.

    Basal cost scales with mass raised to ``metab_basal_exp`` — sub-quadratic
    (Kleiber's law, ~0.75, gives bigger animals a lower cost *per unit mass*), so
    being large isn't ruinously expensive just to exist. Movement cost is the
    kinetic ``½mv²`` shape (∝ mass·speed²), which genuinely penalises heavy, fast
    motion — the momentum trade-off. Perception is a flat sensory tax that keeps
    big eyes from being free.
    """
    mass = size * size
    return (
        cfg.metab_basal_k * mass**cfg.metab_basal_exp
        + cfg.metab_move_k * mass * speed * speed
        + cfg.metab_perc_k * perception
    )
