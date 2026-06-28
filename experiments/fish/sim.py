"""The simulation world: holds the fish + shark and runs one update step.

This module is deliberately free of any pygame *display* code (it only uses
``pygame.math.Vector2``, which needs no window) so the simulation can be stepped
and unit-tested headlessly.
"""

from __future__ import annotations

import math
import random

from pygame.math import Vector2

from boid import Boid
from config import Config
from grid import SpatialGrid


class Shark:
    """The predator. Its target (the mouse) is pushed in each frame; it steers
    toward the target capped at ``shark_max_speed`` for a laggy, natural chase."""

    def __init__(self, pos: Vector2):
        self.pos = pos
        self.vel = Vector2(0, 0)

    def update(self, target: Vector2, cfg: Config) -> None:
        desired = target - self.pos
        if desired.length_squared() > cfg.shark_max_speed * cfg.shark_max_speed:
            desired.scale_to_length(cfg.shark_max_speed)
        self.vel = desired
        self.pos = self.pos + self.vel


class Simulation:
    def __init__(self, cfg: Config, seed: int | None = None):
        self.cfg = cfg
        self.grid = SpatialGrid(cfg.cell_size)
        rng = random.Random(seed)
        self.fish: list[Boid] = [
            self._random_fish(rng) for _ in range(cfg.num_fish)
        ]
        self.shark = Shark(Vector2(cfg.width / 2, cfg.height / 2))

    def _random_fish(self, rng: random.Random) -> Boid:
        pos = Vector2(rng.uniform(0, self.cfg.width), rng.uniform(0, self.cfg.height))
        angle = rng.uniform(0, 2 * math.pi)
        speed = rng.uniform(self.cfg.min_speed, self.cfg.max_speed)
        vel = Vector2(math.cos(angle), math.sin(angle)) * speed
        return Boid(pos, vel)

    # --- One simulation tick -----------------------------------------------

    def step(self, shark_target: Vector2) -> None:
        cfg = self.cfg

        # Predator moves first so fish react to its new position.
        self.shark.update(shark_target, cfg)

        # Rebuild the grid from current positions.
        self.grid.build(self.fish)

        # Pass 1 — compute (read-only): every fish reads neighbours' *current*
        # state, so update order can't bias the result.
        for f in self.fish:
            near = self.grid.neighbours(f, cfg.perception_radius)
            f.compute(near, self.shark.pos, cfg)

        # Pass 2 — apply (write): integrate, then resolve world edges.
        for f in self.fish:
            f.apply(cfg)
            self._handle_edges(f)

    def _handle_edges(self, f: Boid) -> None:
        cfg = self.cfg
        if cfg.wrap_edges:
            # Toroidal wrap-around: off one edge -> back on the opposite.
            f.pos.x %= cfg.width
            f.pos.y %= cfg.height
        else:
            # Soft bounce: steer back inward near the borders.
            if f.pos.x < cfg.edge_margin:
                f.vel.x += cfg.edge_turn
            elif f.pos.x > cfg.width - cfg.edge_margin:
                f.vel.x -= cfg.edge_turn
            if f.pos.y < cfg.edge_margin:
                f.vel.y += cfg.edge_turn
            elif f.pos.y > cfg.height - cfg.edge_margin:
                f.vel.y -= cfg.edge_turn
