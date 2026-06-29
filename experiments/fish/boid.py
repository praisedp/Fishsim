"""The Boid (fish) and its four steering rules.

Pure vector logic — no pygame display, no global state. Each rule returns a
*steering* vector using Reynolds' convention: build a desired velocity of length
``max_speed``, then return ``desired - velocity`` clamped to ``max_force``. This
keeps motion smooth instead of snapping.

These functions take the already-gathered list of neighbours (from the spatial
grid) so the math stays independent of how neighbours are found.
"""

from __future__ import annotations

from pygame.math import Vector2

from config import Config
from genome import Genome


def _limit(vec: Vector2, max_len: float) -> Vector2:
    """Return ``vec`` shortened to at most ``max_len`` (never lengthened)."""
    if vec.length_squared() > max_len * max_len:
        return vec.normalize() * max_len
    return vec


def _steer_towards(
    desired: Vector2, velocity: Vector2, max_speed: float, cfg: Config
) -> Vector2:
    """Reynolds steering: aim at ``desired`` (scaled to max_speed), clamp force.

    ``max_speed`` is per-fish (from its genome); ``max_force`` stays global.
    """
    if desired.length_squared() == 0:
        return Vector2(0, 0)
    desired = desired.normalize() * max_speed
    return _limit(desired - velocity, cfg.max_force)


class Boid:
    """A single fish. Its behaviour is fully determined by its ``genome``."""

    __slots__ = ("pos", "vel", "accel", "genome", "age")

    def __init__(self, pos: Vector2, vel: Vector2, genome: Genome):
        self.pos = pos
        self.vel = vel
        self.accel = Vector2(0, 0)  # scratch for the two-pass update
        self.genome = genome
        self.age = 0  # ticks lived; drives reproduction maturity

    # --- The three classic rules -------------------------------------------

    def separation(self, neighbours: list["Boid"], cfg: Config) -> Vector2:
        """Steer away from neighbours that are too close (weighted by closeness)."""
        steer = Vector2(0, 0)
        count = 0
        for other in neighbours:
            offset = self.pos - other.pos
            dist_sq = offset.length_squared()
            if 0 < dist_sq < cfg.separation_radius * cfg.separation_radius:
                # Push harder the closer they are: divide by distance.
                steer += offset.normalize() / (dist_sq ** 0.5)
                count += 1
        if count == 0:
            return Vector2(0, 0)
        steer /= count
        return _steer_towards(steer, self.vel, self.genome.max_speed, cfg)

    def alignment(self, neighbours: list["Boid"], cfg: Config) -> Vector2:
        """Steer toward the average heading of neighbours."""
        avg_vel = Vector2(0, 0)
        count = 0
        for other in neighbours:
            avg_vel += other.vel
            count += 1
        if count == 0:
            return Vector2(0, 0)
        avg_vel /= count
        return _steer_towards(avg_vel, self.vel, self.genome.max_speed, cfg)

    def cohesion(self, neighbours: list["Boid"], cfg: Config) -> Vector2:
        """Steer toward the local centre of mass."""
        center = Vector2(0, 0)
        count = 0
        for other in neighbours:
            center += other.pos
            count += 1
        if count == 0:
            return Vector2(0, 0)
        center /= count
        return _steer_towards(center - self.pos, self.vel, self.genome.max_speed, cfg)

    # --- Predator avoidance -------------------------------------------------

    def flee(self, shark_pos: Vector2 | None, cfg: Config) -> Vector2:
        """Flee the nearest shark when within range — stronger up close.

        The trigger range is the base ``flee_radius`` scaled by the fish's ``fear``
        gene, so timid fish bolt early and bold fish hold their nerve. ``shark_pos``
        is ``None`` when no sharks are left alive, in which case there's nothing to
        flee.
        """
        if shark_pos is None:
            return Vector2(0, 0)
        radius = cfg.flee_radius * self.genome.fear
        offset = self.pos - shark_pos
        dist_sq = offset.length_squared()
        if dist_sq == 0 or dist_sq > radius * radius:
            return Vector2(0, 0)
        dist = dist_sq ** 0.5
        # Closer shark -> larger desired magnitude before steering normalises it.
        desired = offset.normalize() / dist
        return _steer_towards(desired, self.vel, self.genome.max_speed, cfg)

    # --- Combine ------------------------------------------------------------

    def compute(
        self, neighbours: list["Boid"], shark_pos: Vector2 | None, cfg: Config
    ) -> None:
        """Read-only pass: accumulate steering into ``self.accel`` (don't apply).

        The four rule weights now come from this fish's genome, not the config.
        ``shark_pos`` is the nearest shark (or ``None`` if none are alive).
        """
        g = self.genome
        self.accel = (
            g.w_sep * self.separation(neighbours, cfg)
            + g.w_ali * self.alignment(neighbours, cfg)
            + g.w_coh * self.cohesion(neighbours, cfg)
            + g.w_flee * self.flee(shark_pos, cfg)
        )

    def apply(self, cfg: Config) -> None:
        """Write pass: integrate velocity (speed-clamped) then position; age one tick."""
        self.vel = _limit(self.vel + self.accel, self.genome.max_speed)
        # Keep a minimum speed so isolated fish keep drifting instead of freezing.
        speed_sq = self.vel.length_squared()
        if 0 < speed_sq < cfg.min_speed * cfg.min_speed:
            self.vel = self.vel.normalize() * cfg.min_speed
        self.pos = self.pos + self.vel
        self.age += 1
