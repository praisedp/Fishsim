"""The simulation world: a coevolving ecosystem of fish and sharks.

This is the genetic-algorithm engine for *both* species. Each tick:

- **Sharks hunt** the nearest fish they can see (limited by their ``perception``
  gene), aiming at a predicted intercept (their ``lead`` gene) and chasing at their
  ``max_speed`` gene. Eating a fish restores energy; every tick burns some
  (faster/keener-sighted sharks burn more). Out of energy → the shark starves and
  dies; well-fed → it spawns a mutated child, splitting its energy.
- **Fish flee** the nearest shark and flock, then mature ones occasionally breed
  mutated copies of themselves.

There are no population floors: if the sharks can't catch enough they go extinct and
the fish fill the sea; if they're too good they eat everything and then starve. Good
genes — dodging for fish, hunting for sharks — spread on their own. No generations,
no resets.

Deliberately free of any pygame *display* code (only ``pygame.math.Vector2``, which
needs no window) so the simulation can be stepped and unit-tested headlessly.
"""

from __future__ import annotations

import math
import random

from pygame.math import Vector2

from boid import Boid
from config import Config
from genome import (
    GENE_NAMES,
    SHARK_GENE_NAMES,
    Genome,
    SharkGenome,
)
from grid import SpatialGrid


class Shark:
    """A predator: position, velocity, a hunting ``genome``, and an energy reserve
    that food tops up and metabolism drains. It steers toward a target capped at its
    own ``max_speed`` gene."""

    __slots__ = ("pos", "vel", "genome", "energy", "age")

    def __init__(self, pos: Vector2, genome: SharkGenome, energy: float):
        self.pos = pos
        self.vel = Vector2(0, 0)
        self.genome = genome
        self.energy = energy
        self.age = 0

    def update(self, target: Vector2, cfg: Config) -> None:
        """Steer toward ``target``, capped at this shark's ``max_speed`` gene."""
        desired = target - self.pos
        ms = self.genome.max_speed
        if desired.length_squared() > ms * ms:
            desired.scale_to_length(ms)
        self.vel = desired
        self.pos = self.pos + self.vel


class Simulation:
    def __init__(self, cfg: Config, seed: int | None = None):
        self.cfg = cfg
        self.grid = SpatialGrid(cfg.cell_size)
        self.rng = random.Random(seed)
        self.fish: list[Boid] = [self._random_fish() for _ in range(cfg.num_fish)]
        self.sharks: list[Shark] = [
            self._random_shark() for _ in range(cfg.num_sharks)
        ]

        # Lifetime tallies + a per-step snapshot for the HUD.
        self.births = 0
        self.deaths = 0
        self.shark_births = 0
        self.shark_deaths = 0
        self.stats: dict[str, float] = {}
        self._update_stats()

    # --- Spawning -----------------------------------------------------------

    def _random_fish(self) -> Boid:
        cfg = self.cfg
        pos = Vector2(self.rng.uniform(0, cfg.width), self.rng.uniform(0, cfg.height))
        fish = self._spawn_fish_at(pos, Genome.random(self.rng))
        # Seed a spread of ages so a founding population isn't uniformly immature —
        # some can breed right away instead of all waiting out maturity together.
        fish.age = self.rng.randint(0, cfg.maturity_age)
        return fish

    def _spawn_fish_at(self, pos: Vector2, genome: Genome) -> Boid:
        """A fish at ``pos`` with a random heading at its genome's top speed."""
        angle = self.rng.uniform(0, 2 * math.pi)
        speed = self.rng.uniform(self.cfg.min_speed, genome.max_speed)
        vel = Vector2(math.cos(angle), math.sin(angle)) * speed
        return Boid(pos, vel, genome)

    def _random_shark(self) -> Shark:
        cfg = self.cfg
        pos = Vector2(self.rng.uniform(0, cfg.width), self.rng.uniform(0, cfg.height))
        return Shark(pos, SharkGenome.random(self.rng), cfg.shark_start_energy)

    # --- One simulation tick -----------------------------------------------

    def step(self, manual_target: Vector2 | None = None) -> None:
        """Advance the world one tick.

        ``manual_target`` (the mouse) makes the *nearest* shark chase that point;
        every other shark hunts autonomously.
        """
        cfg = self.cfg

        # Grid is built from current fish positions; both sharks (prey-finding) and
        # fish (neighbour-finding) read it this tick.
        self.grid.build(self.fish)

        # Sharks move first so fish react to their new positions.
        manual_shark = (
            self._nearest_shark(manual_target) if manual_target is not None else None
        )
        for s in self.sharks:
            target = manual_target if s is manual_shark else self._shark_target(s)
            s.update(target, cfg)
            self._wrap_shark(s)

        # Pass 1 — fish compute (read-only): flock + flee the nearest shark.
        for f in self.fish:
            near = self.grid.neighbours(f, f.genome.perception)
            f.compute(near, self._nearest_shark_pos(f), cfg)

        # Pass 2 — fish apply (write): integrate, then resolve world edges.
        for f in self.fish:
            f.apply(cfg)
            self._handle_edges(f)

        # Genetic algorithm: predation feeds sharks, then births and deaths.
        self._predate()
        self._shark_metabolise_and_die()
        self._shark_reproduce()
        self._reproduce()
        self._update_stats()

    # --- Shark targeting ----------------------------------------------------

    def _shark_target(self, shark: Shark) -> Vector2:
        """Where this shark aims: a predicted intercept of the nearest visible fish,
        or a gentle wander when nothing is in sight (still burning energy)."""
        prey = self.grid.neighbours_at(
            shark.pos.x, shark.pos.y, shark.genome.perception
        )
        if not prey:
            return self._wander_target(shark)
        nearest = min(prey, key=lambda f: shark.pos.distance_squared_to(f.pos))
        # Lead the target: aim where it will be by the time we could reach it.
        dist = shark.pos.distance_to(nearest.pos)
        lead_time = shark.genome.lead * dist / max(shark.genome.max_speed, 1e-6)
        return nearest.pos + nearest.vel * lead_time

    def _wander_target(self, shark: Shark) -> Vector2:
        """A point ahead of the shark with a small random turn (roaming search)."""
        if shark.vel.length_squared() < 1e-6:
            heading = Vector2(1, 0).rotate(self.rng.uniform(0, 360))
        else:
            heading = shark.vel.normalize().rotate(self.rng.uniform(-20, 20))
        return shark.pos + heading * 50

    def _nearest_shark(self, point: Vector2) -> Shark | None:
        if not self.sharks:
            return None
        return min(self.sharks, key=lambda s: point.distance_squared_to(s.pos))

    def _nearest_shark_pos(self, fish: Boid) -> Vector2 | None:
        """Position of the shark closest to ``fish`` (what it flees), or None."""
        nearest = self._nearest_shark(fish.pos)
        return nearest.pos if nearest is not None else None

    # --- Genetic algorithm: death, birth, energy ---------------------------

    def _predate(self) -> None:
        """Each shark catches at most ONE fish per tick — the nearest one within
        ``kill_radius`` — and gains energy for it. Capping intake at one fish (a
        shark can only seize one at a time) stops a single shark from vacuuming a
        whole cluster and breeding explosively; it's the main predator–prey brake."""
        cfg = self.cfg
        kill_sq = cfg.kill_radius * cfg.kill_radius
        eaten: set[int] = set()
        for s in self.sharks:
            catch: Boid | None = None
            catch_d = kill_sq
            for f in self.fish:
                if id(f) in eaten:
                    continue
                d = s.pos.distance_squared_to(f.pos)
                if d <= catch_d:
                    catch, catch_d = f, d
            if catch is not None:
                eaten.add(id(catch))
                s.energy += cfg.shark_energy_per_fish
        if eaten:
            self.fish = [f for f in self.fish if id(f) not in eaten]
            self.deaths += len(eaten)

    def _shark_metabolise_and_die(self) -> None:
        """Burn each shark's per-tick energy cost; cull any that hit zero."""
        cfg = self.cfg
        survivors: list[Shark] = []
        for s in self.sharks:
            g = s.genome
            cost = (
                cfg.shark_metab_base
                + cfg.shark_metab_speed_k * g.max_speed
                + cfg.shark_metab_perc_k * g.perception
            )
            s.energy -= cost
            s.age += 1
            if s.energy > 0:
                survivors.append(s)
        self.shark_deaths += len(self.sharks) - len(survivors)
        self.sharks = survivors

    def _shark_reproduce(self) -> None:
        """Well-fed sharks spawn a mutated child, splitting their energy with it."""
        cfg = self.cfg
        newborns: list[Shark] = []
        for s in self.sharks:
            if len(self.sharks) + len(newborns) >= cfg.max_sharks:
                break
            if s.energy < cfg.shark_repro_energy:
                continue
            child_energy = s.energy / 2 - cfg.shark_birth_cost
            if child_energy <= 0:
                continue  # can't afford the split
            s.energy /= 2
            child_genome = s.genome.mutate(
                self.rng, cfg.shark_mutation_rate, cfg.shark_mutation_sigma
            )
            offset = Vector2(self.rng.uniform(-10, 10), self.rng.uniform(-10, 10))
            newborns.append(Shark(s.pos + offset, child_genome, child_energy))
        self.sharks.extend(newborns)
        self.shark_births += len(newborns)

    def _reproduce(self) -> None:
        """Mature fish occasionally spawn a mutated child nearby (asexual)."""
        cfg = self.cfg
        newborns: list[Boid] = []
        for f in self.fish:
            if len(self.fish) + len(newborns) >= cfg.max_pop:
                break
            if f.age >= cfg.maturity_age and self.rng.random() < cfg.repro_prob:
                child_genome = f.genome.mutate(
                    self.rng, cfg.mutation_rate, cfg.mutation_sigma
                )
                offset = Vector2(self.rng.uniform(-8, 8), self.rng.uniform(-8, 8))
                newborns.append(self._spawn_fish_at(f.pos + offset, child_genome))
        self.fish.extend(newborns)
        self.births += len(newborns)

    # --- HUD statistics -----------------------------------------------------

    def _update_stats(self) -> None:
        stats: dict[str, float] = {
            "population": len(self.fish),
            "births": self.births,
            "deaths": self.deaths,
            "shark_population": len(self.sharks),
            "shark_births": self.shark_births,
            "shark_deaths": self.shark_deaths,
        }
        self._average_genes(stats, self.fish, GENE_NAMES, prefix="", lineage="generation")
        self._average_genes(
            stats, self.sharks, SHARK_GENE_NAMES, prefix="shark_",
            lineage="shark_generation", energy_key="shark_energy",
        )
        self.stats = stats

    @staticmethod
    def _average_genes(stats, animals, gene_names, prefix, lineage, energy_key=None):
        """Fill ``stats`` with the mean of each gene (+ lineage / energy) over a
        living population, or zeros when it's empty."""
        n = len(animals)
        if n:
            stats[lineage] = sum(a.genome.generation for a in animals) / n
            for name in gene_names:
                stats[prefix + name] = sum(getattr(a.genome, name) for a in animals) / n
            if energy_key is not None:
                stats[energy_key] = sum(a.energy for a in animals) / n
        else:
            stats[lineage] = 0.0
            for name in gene_names:
                stats[prefix + name] = 0.0
            if energy_key is not None:
                stats[energy_key] = 0.0

    # --- World edges --------------------------------------------------------

    def _wrap_shark(self, s: Shark) -> None:
        cfg = self.cfg
        if cfg.wrap_edges:
            s.pos.x %= cfg.width
            s.pos.y %= cfg.height
        else:
            # Keep sharks on-screen by reflecting off the borders.
            if s.pos.x < 0 or s.pos.x > cfg.width:
                s.vel.x = -s.vel.x
                s.pos.x = max(0.0, min(float(cfg.width), s.pos.x))
            if s.pos.y < 0 or s.pos.y > cfg.height:
                s.vel.y = -s.vel.y
                s.pos.y = max(0.0, min(float(cfg.height), s.pos.y))

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
