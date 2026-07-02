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

Populations live in struct-of-arrays form (:mod:`state`), and every per-tick rule
below is a whole-population array operation — see :mod:`neighbors` for the pair
search and :mod:`steering` for the flocking math. Distances are toroidal
(minimum-image) whenever ``wrap_edges`` is on.

Deliberately free of any pygame code so the simulation can be stepped and
unit-tested headlessly.
"""

from __future__ import annotations

import numpy as np

from config import Config
from genome import (
    F_MAXSPD,
    F_PERC,
    FISH_SPEC,
    GENE_NAMES,
    S_LEAD,
    S_MAXSPD,
    S_PERC,
    SHARK_GENE_NAMES,
    SHARK_SPEC,
    mutate_genes,
    random_genes,
)
from neighbors import build_grid, min_image, nearest_points, perception_pairs
from state import FishState, SharkState
from steering import EPS, fish_accelerations, limit_norm
from walls import WallField, avoid_all

# Behaviour constants (fixed by design, not evolved or config-tuned).
WANDER_TURN_DEG = 20.0  # max random heading change per tick while roaming
WANDER_AHEAD = 50.0  # how far ahead of a roaming shark its target sits
SHARK_SPAWN_JITTER = 10.0  # newborn shark's offset from its parent
FISH_SPAWN_JITTER = 8.0  # newborn fish's offset from its parent

# Cells sized to the largest *fish* perception so a fish's neighbours always live
# in the 9 surrounding cells. Sharks see much further, but their prey search is a
# dense shark×fish distance matrix, so it never touches the grid.
CELL_MIN = float(FISH_SPEC.hi[F_PERC])


class Simulation:
    def __init__(self, cfg: Config, seed: int | None = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.world = np.array([float(cfg.width), float(cfg.height)])

        # Static obstacles, built once (they never move) — seeded for reproducibility.
        self.rebuild_walls()

        self.fish = FishState(capacity=max(cfg.max_pop, cfg.num_fish))
        self.sharks = SharkState(capacity=max(cfg.max_sharks, cfg.num_sharks))
        self._spawn_initial_populations()

        # Lifetime tallies + a per-step snapshot for the HUD.
        self.births = 0
        self.deaths = 0
        self.shark_births = 0
        self.shark_deaths = 0
        self.stats: dict[str, float] = {}
        self._update_stats()

    # --- Walls --------------------------------------------------------------

    def rebuild_walls(self) -> None:
        """(Re)scatter a fresh shard layout, or clear it, per ``cfg.walls_enabled``."""
        if self.cfg.walls_enabled:
            self.walls = WallField.scatter(self.cfg, self.rng)
        else:
            self.walls = WallField([])

    # --- Spawning -----------------------------------------------------------

    def _spawn_initial_populations(self) -> None:
        cfg = self.cfg
        n = cfg.num_fish
        genes = random_genes(self.rng, FISH_SPEC, n)
        self.fish.append(
            pos=self.rng.uniform(0, self.world, (n, 2)),
            vel=self._random_headings(genes[:, F_MAXSPD]),
            genes=genes,
            generation=np.zeros(n, dtype=np.int64),
            # Seed a spread of ages so a founding population isn't uniformly
            # immature — some can breed right away instead of all waiting out
            # maturity together.
            age=self.rng.integers(0, cfg.maturity_age + 1, n),
        )

        s = cfg.num_sharks
        self.sharks.append(
            pos=self.rng.uniform(0, self.world, (s, 2)),
            vel=np.zeros((s, 2)),
            genes=random_genes(self.rng, SHARK_SPEC, s),
            generation=np.zeros(s, dtype=np.int64),
            age=np.zeros(s, dtype=np.int64),
            energy=np.full(s, cfg.shark_start_energy),
        )

    def _random_headings(self, max_speeds: np.ndarray) -> np.ndarray:
        """``(k, 2)`` velocities: random directions, speeds between the floor and
        each fish's own top speed."""
        angle = self.rng.uniform(0, 2 * np.pi, len(max_speeds))
        speed = self.rng.uniform(self.cfg.min_speed, max_speeds)
        return np.stack([np.cos(angle), np.sin(angle)], axis=1) * speed[:, None]

    def add_fish(self, pos, genes=None, vel=None, age: int = 0, generation: int = 0) -> None:
        """Insert a single fish (mid-range genes by default) — mainly for tests."""
        genes = FISH_SPEC.mid() if genes is None else np.asarray(genes, dtype=float)
        vel = (
            self._random_headings(np.array([genes[F_MAXSPD]]))
            if vel is None
            else np.asarray(vel, dtype=float)[None]
        )
        self.fish.append(
            pos=np.asarray(pos, dtype=float)[None],
            vel=vel,
            genes=genes[None],
            generation=np.array([generation]),
            age=np.array([age]),
        )

    def add_shark(self, pos, genes=None, energy: float | None = None) -> None:
        """Insert a single shark (mid-range genes by default) — mainly for tests."""
        genes = SHARK_SPEC.mid() if genes is None else np.asarray(genes, dtype=float)
        self.sharks.append(
            pos=np.asarray(pos, dtype=float)[None],
            vel=np.zeros((1, 2)),
            genes=genes[None],
            generation=np.zeros(1, dtype=np.int64),
            age=np.zeros(1, dtype=np.int64),
            energy=np.array([self.cfg.shark_start_energy if energy is None else energy]),
        )

    # --- One simulation tick -----------------------------------------------

    def step(self, manual_target=None) -> None:
        """Advance the world one tick.

        ``manual_target`` (the mouse, as an ``(x, y)`` pair) makes the *nearest*
        shark chase that point; every other shark hunts autonomously.
        """
        cfg = self.cfg

        # Sharks move first so fish react to their new positions.
        if len(self.sharks):
            targets = self._shark_targets(manual_target)
            vel = limit_norm(targets - self.sharks.pos, self.sharks.genes[:, S_MAXSPD])
            self.sharks.vel[:] = vel
            self.sharks.pos[:] += vel
            self._confine_sharks()
            self._apply_walls(self.sharks)

        # Fish pass 1 (read): flock with visible neighbours, flee the nearest shark.
        if len(self.fish):
            fish = self.fish
            grid = build_grid(fish.pos, cfg.width, cfg.height, CELL_MIN, cfg.wrap_edges)
            i, j, rel, d2 = perception_pairs(
                grid, fish.pos, fish.genes[:, F_PERC], self.world, cfg.wrap_edges
            )
            if len(self.sharks):
                _, flee_offset, flee_d2 = nearest_points(
                    fish.pos, self.sharks.pos, self.world, cfg.wrap_edges
                )
            else:
                flee_offset = flee_d2 = None  # no sharks alive -> nothing to flee
            accel = fish_accelerations(
                fish.vel, fish.genes, i, j, rel, d2, flee_offset, flee_d2, cfg
            )

            # Pass 2 (write): integrate, then resolve walls and world edges.
            self._integrate_fish(accel)
            self._apply_walls(self.fish, max_speed=fish.genes[:, F_MAXSPD])
            self._apply_edges_fish()

        # Genetic algorithm: predation feeds sharks, then births and deaths.
        self._predate()
        self._shark_metabolise_and_die()
        self._shark_reproduce()
        self._fish_reproduce()
        self._update_stats()

    # --- Shark targeting ----------------------------------------------------

    def _shark_targets(self, manual_target) -> np.ndarray:
        """Where each shark aims ``(S, 2)``: a predicted intercept of the nearest
        visible fish, a gentle wander when nothing is in sight (still burning
        energy), or the mouse for the manually driven shark."""
        cfg = self.cfg
        sharks, fish = self.sharks, self.fish
        targets = self._wander_targets()

        if len(fish):
            delta = fish.pos[None, :, :] - sharks.pos[:, None, :]  # (S, F, 2)
            if cfg.wrap_edges:
                delta = min_image(delta, self.world)
            d2 = np.einsum("sfc,sfc->sf", delta, delta)
            visible = d2 <= sharks.genes[:, S_PERC, None] ** 2
            prey = np.where(visible, d2, np.inf).argmin(axis=1)
            hunters = np.flatnonzero(visible.any(axis=1))
            if hunters.size:
                caught = prey[hunters]
                dist = np.sqrt(d2[hunters, caught])
                # Lead the target: aim where it will be by the time we could
                # reach it. Built from the shark's own frame (the min-imaged
                # delta) so the intercept also works across the wrap seam.
                lead_time = (
                    sharks.genes[hunters, S_LEAD]
                    * dist
                    / np.maximum(sharks.genes[hunters, S_MAXSPD], EPS)
                )
                targets[hunters] = (
                    sharks.pos[hunters]
                    + delta[hunters, caught]
                    + fish.vel[caught] * lead_time[:, None]
                )

        if manual_target is not None:
            # The mouse drives the shark nearest to it (plain screen distance).
            offset = np.asarray(manual_target, dtype=float) - sharks.pos
            nearest = np.einsum("sc,sc->s", offset, offset).argmin()
            targets[nearest] = manual_target
        return targets

    def _wander_targets(self) -> np.ndarray:
        """A point ahead of each shark with a small random turn (roaming search);
        a standing-still shark picks a fresh random direction."""
        sharks = self.sharks
        s = len(sharks)
        speed2 = np.einsum("sc,sc->s", sharks.vel, sharks.vel)
        turn = np.radians(self.rng.uniform(-WANDER_TURN_DEG, WANDER_TURN_DEG, s))
        current = np.arctan2(sharks.vel[:, 1], sharks.vel[:, 0])
        fresh = self.rng.uniform(0, 2 * np.pi, s)
        angle = np.where(speed2 < EPS, fresh, current + turn)
        heading = np.stack([np.cos(angle), np.sin(angle)], axis=1)
        return sharks.pos + heading * WANDER_AHEAD

    # --- Genetic algorithm: death, birth, energy ---------------------------

    def _predate(self) -> None:
        """Each shark catches at most ONE fish per tick — the nearest one within
        ``kill_radius`` — and gains energy for it. Capping intake at one fish (a
        shark can only seize one at a time) stops a single shark from vacuuming a
        whole cluster and breeding explosively; it's the main predator–prey brake.

        The distance matrix is precomputed, but the per-shark walk stays a
        deliberate sequential loop (≤ ``max_sharks`` rows): an earlier shark's
        claim removes that fish from every later shark's menu.
        """
        cfg = self.cfg
        fish, sharks = self.fish, self.sharks
        if not len(sharks) or not len(fish):
            return
        delta = fish.pos[None, :, :] - sharks.pos[:, None, :]
        if cfg.wrap_edges:
            delta = min_image(delta, self.world)
        d2 = np.einsum("sfc,sfc->sf", delta, delta)

        kill2 = cfg.kill_radius**2
        eaten = np.zeros(len(fish), dtype=bool)
        for s in range(len(sharks)):
            candidates = np.where(eaten, np.inf, d2[s])
            f = candidates.argmin()
            if candidates[f] <= kill2:
                eaten[f] = True
                sharks.energy[s] += cfg.shark_energy_per_fish
        if eaten.any():
            self.deaths += fish.compact(~eaten)

    def _shark_metabolise_and_die(self) -> None:
        """Burn each shark's per-tick energy cost; cull any that hit zero."""
        cfg = self.cfg
        sharks = self.sharks
        if not len(sharks):
            return
        cost = (
            cfg.shark_metab_base
            + cfg.shark_metab_speed_k * sharks.genes[:, S_MAXSPD]
            + cfg.shark_metab_perc_k * sharks.genes[:, S_PERC]
        )
        sharks.energy[:] -= cost
        sharks.age[:] += 1
        self.shark_deaths += sharks.compact(sharks.energy > 0)

    def _shark_reproduce(self) -> None:
        """Well-fed sharks spawn a mutated child, splitting their energy with it.

        Eligible sharks breed in row order until the cap is hit — the same
        first-come, first-served rule the rest of the GA uses.
        """
        cfg = self.cfg
        sharks = self.sharks
        headroom = cfg.max_sharks - len(sharks)
        if headroom <= 0 or not len(sharks):
            return
        eligible = (sharks.energy >= cfg.shark_repro_energy) & (
            sharks.energy / 2 - cfg.shark_birth_cost > 0  # can afford the split
        )
        parents = np.flatnonzero(eligible)[:headroom]
        if not parents.size:
            return
        k = parents.size
        child_energy = sharks.energy[parents] / 2 - cfg.shark_birth_cost
        sharks.energy[parents] /= 2
        jitter = self.rng.uniform(-SHARK_SPAWN_JITTER, SHARK_SPAWN_JITTER, (k, 2))
        self.sharks.append(
            pos=sharks.pos[parents] + jitter,
            vel=np.zeros((k, 2)),
            genes=mutate_genes(
                self.rng, SHARK_SPEC, sharks.genes[parents],
                cfg.shark_mutation_rate, cfg.shark_mutation_sigma,
            ),
            generation=sharks.generation[parents] + 1,
            age=np.zeros(k, dtype=np.int64),
            energy=child_energy,
        )
        self.shark_births += k

    def _fish_reproduce(self) -> None:
        """Mature fish occasionally spawn a mutated child nearby (asexual)."""
        cfg = self.cfg
        fish = self.fish
        headroom = cfg.max_pop - len(fish)
        if headroom <= 0 or not len(fish):
            return
        breeding = (fish.age >= cfg.maturity_age) & (
            self.rng.random(len(fish)) < cfg.repro_prob
        )
        parents = np.flatnonzero(breeding)[:headroom]
        if not parents.size:
            return
        k = parents.size
        child_genes = mutate_genes(
            self.rng, FISH_SPEC, fish.genes[parents], cfg.mutation_rate, cfg.mutation_sigma
        )
        jitter = self.rng.uniform(-FISH_SPAWN_JITTER, FISH_SPAWN_JITTER, (k, 2))
        self.fish.append(
            pos=fish.pos[parents] + jitter,
            vel=self._random_headings(child_genes[:, F_MAXSPD]),
            genes=child_genes,
            generation=fish.generation[parents] + 1,
            age=np.zeros(k, dtype=np.int64),
        )
        self.births += k

    # --- Fish motion ----------------------------------------------------------

    def _integrate_fish(self, accel: np.ndarray) -> None:
        """Apply the tick's accelerations: speed-clamped velocity, then position."""
        cfg = self.cfg
        fish = self.fish
        vel = limit_norm(fish.vel + accel, fish.genes[:, F_MAXSPD])
        # Keep a minimum speed so isolated fish keep drifting instead of freezing.
        speed2 = np.einsum("nc,nc->n", vel, vel)
        slow = (speed2 > 0.0) & (speed2 < cfg.min_speed**2)
        if slow.any():
            vel[slow] *= cfg.min_speed / np.sqrt(speed2[slow])[:, None]
        fish.vel[:] = vel
        fish.pos[:] += vel
        fish.age[:] += 1

    # --- World geometry: walls and edges -------------------------------------

    def _apply_walls(self, state, max_speed: np.ndarray | None = None) -> None:
        """Soft-bounce a population off any nearby shard.

        A velocity nudge along the outward wall normal (stronger up close) steers
        around obstacles; anything that crossed *inside* a shard also gets a
        positional ejection back out. Sharks need that ejection because their
        velocity is rewritten from scratch each tick, so a nudge alone would be
        forgotten — they'd end up sliding along a wall instead. Pass ``max_speed``
        (per animal) to re-clamp nudged rows; fish keep their gene's speed cap
        intact, sharks are left unclamped like the rest of their motion.
        """
        cfg = self.cfg
        if not cfg.walls_enabled or not len(state):
            return
        push, eject = avoid_all(state.pos, self.walls, cfg.wall_margin)
        state.pos[:] += eject
        state.vel[:] += push * cfg.wall_push
        if max_speed is not None:
            pushed = np.flatnonzero(np.einsum("nc,nc->n", push, push) > 0.0)
            if pushed.size:
                state.vel[pushed] = limit_norm(state.vel[pushed], max_speed[pushed])

    def _confine_sharks(self) -> None:
        """Keep sharks in the world: toroidal wrap, or reflect off the borders."""
        pos, vel = self.sharks.pos, self.sharks.vel
        if self.cfg.wrap_edges:
            pos %= self.world
        else:
            for axis, limit in enumerate(self.world):
                out = (pos[:, axis] < 0) | (pos[:, axis] > limit)
                vel[out, axis] *= -1
                np.clip(pos[:, axis], 0.0, limit, out=pos[:, axis])

    def _apply_edges_fish(self) -> None:
        cfg = self.cfg
        pos, vel = self.fish.pos, self.fish.vel
        if cfg.wrap_edges:
            # Toroidal wrap-around: off one edge -> back on the opposite.
            pos %= self.world
        else:
            # Soft bounce: steer back inward near the borders.
            for axis, limit in enumerate(self.world):
                low = pos[:, axis] < cfg.edge_margin
                high = ~low & (pos[:, axis] > limit - cfg.edge_margin)
                vel[low, axis] += cfg.edge_turn
                vel[high, axis] -= cfg.edge_turn

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
    def _average_genes(stats, state, gene_names, prefix, lineage, energy_key=None):
        """Fill ``stats`` with the mean of each gene (+ lineage / energy) over a
        living population, or zeros when it's empty."""
        alive = len(state) > 0
        stats[lineage] = float(state.generation.mean()) if alive else 0.0
        for column, name in enumerate(gene_names):
            stats[prefix + name] = float(state.genes[:, column].mean()) if alive else 0.0
        if energy_key is not None:
            stats[energy_key] = float(state.energy.mean()) if alive else 0.0
