"""The simulation world: a coevolving ecosystem of food, fish, and sharks.

Energy is the currency of life. Each tick:

- **Food** (drifting plant particles) spawns at random, carries an energy payload
  scaled by its size, drifts and weakly clumps, and shies away from the wakes of
  passing fish and sharks.
- **Fish** flock (separation/alignment/cohesion), flee the nearest shark, and
  steer toward nearby food. They eat food to gain energy, pay a metabolic cost
  every tick (bigger/faster fish pay more), grow from juvenile toward a genetic
  ``adult_size``, **starve** if energy hits zero, and **breed** by splitting their
  reserve once well-fed and grown. Body size derives speed (small = fast),
  momentum (heavy = sluggish), storage cap, and metabolism — so life-history
  strategy is itself under selection.
- **Sharks hunt** the nearest fish they can see, aiming at a predicted intercept,
  and live or die by their own energy budget; a bigger fish is a bigger meal.

There are no population floors: carrying capacity emerges from the food supply.
Good genes — foraging and dodging for fish, hunting for sharks — spread on their
own. No generations, no resets.

Populations live in struct-of-arrays form (:mod:`state`); every per-tick rule is
a whole-population array operation — see :mod:`neighbors` for the spatial pair
search, :mod:`steering` for the flocking math, and :mod:`physiology` for the
size-derived body physics. Distances are toroidal (minimum-image) when
``wrap_edges`` is on. No pygame code, so the sim is steppable and unit-testable
headlessly.
"""

from __future__ import annotations

import numpy as np

from config import Config
from genome import (
    F_ADULT,
    F_PERC,
    F_REPROF,
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
from neighbors import build_grid, cross_pairs, min_image, nearest_points, perception_pairs
from physiology import energy_cap_of, mass_rel_of, max_speed_of, metabolism
from state import FishState, FoodState, SharkState
from steering import EPS, fish_accelerations, limit_norm
from walls import WallField, avoid_all

# Behaviour constants (fixed by design, not evolved or config-tuned).
WANDER_TURN_DEG = 20.0  # max random heading change per tick while roaming
WANDER_AHEAD = 50.0  # how far ahead of a roaming shark its target sits
SHARK_SPAWN_JITTER = 10.0  # newborn shark's offset from its parent
FISH_SPAWN_JITTER = 8.0  # newborn fish's offset from its parent
FOUNDER_ENERGY_FRAC = 0.6  # starting energy of founders/inserted fish, as frac of cap

# Cells sized to the largest *fish* perception so a fish's neighbours (and the
# food it can see) always live in the 9 surrounding cells. Sharks see further,
# but their prey search is a dense shark×fish matrix, so it skips the grid.
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
        self.food = FoodState(capacity=cfg.max_food)
        self._spawn_initial_populations()

        # Per-tick foraging pairs (fish->food), stashed between the sense pass and
        # eating so the expensive cross-query runs only once. See step().
        self._forage: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

        # Lifetime tallies + a per-step snapshot for the HUD.
        self.births = 0
        self.deaths = 0
        self.starvations = 0
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
        # Founders start as grown adults at their genetic size, moderately fed, so
        # the sim opens with a viable breeding population rather than helpless fry.
        size = genes[:, F_ADULT].copy()
        self.fish.append(
            pos=self.rng.uniform(0, self.world, (n, 2)),
            vel=self._random_headings(max_speed_of(size, cfg)),
            genes=genes,
            size=size,
            energy=FOUNDER_ENERGY_FRAC * energy_cap_of(size, cfg),
            generation=np.zeros(n, dtype=np.int64),
            age=np.zeros(n, dtype=np.int64),
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
        each fish's own (size-derived) top speed."""
        angle = self.rng.uniform(0, 2 * np.pi, len(max_speeds))
        speed = self.rng.uniform(self.cfg.min_speed, max_speeds)
        return np.stack([np.cos(angle), np.sin(angle)], axis=1) * speed[:, None]

    def add_fish(self, pos, genes=None, vel=None, size=None, energy=None, generation=0) -> None:
        """Insert a single fish — mainly for tests. Defaults: mid-range genes, size
        at the genome's adult target, energy at a fraction of the derived cap."""
        cfg = self.cfg
        genes = FISH_SPEC.mid() if genes is None else np.asarray(genes, dtype=float)
        size = float(genes[F_ADULT]) if size is None else float(size)
        cap = float(energy_cap_of(size, cfg))
        energy = FOUNDER_ENERGY_FRAC * cap if energy is None else float(energy)
        if vel is None:
            vel = self._random_headings(np.array([max_speed_of(size, cfg)]))
        else:
            vel = np.asarray(vel, dtype=float)[None]
        self.fish.append(
            pos=np.asarray(pos, dtype=float)[None],
            vel=vel,
            genes=genes[None],
            size=np.array([size]),
            energy=np.array([energy]),
            generation=np.array([generation]),
            age=np.array([0]),
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

    def add_food(self, pos, size: float, energy: float | None = None) -> None:
        """Insert a single food particle — mainly for tests."""
        energy = self.cfg.food_energy_k * size * size if energy is None else energy
        self.food.append(
            pos=np.asarray(pos, dtype=float)[None],
            vel=np.zeros((1, 2)),
            energy=np.array([float(energy)]),
            size=np.array([float(size)]),
        )

    # --- One simulation tick -----------------------------------------------

    def step(self, manual_target=None) -> None:
        """Advance the world one tick.

        ``manual_target`` (the mouse, as an ``(x, y)`` pair) makes the *nearest*
        shark chase that point; every other shark hunts autonomously.

        Ordering note: pair-index arrays (fish–fish ``i/j``, fish–food in
        ``self._forage``) are invalidated by any ``compact``. So all reads that
        use them happen *before* the compactions (eat food, starve fish, predate)
        that shrink those populations.
        """
        cfg = self.cfg

        # New food appears.
        self._food_spawn()

        # Sharks move first so fish react to their new positions.
        if len(self.sharks):
            targets = self._shark_targets(manual_target)
            vel = limit_norm(targets - self.sharks.pos, self.sharks.genes[:, S_MAXSPD])
            self.sharks.vel[:] = vel
            self.sharks.pos[:] += vel
            self._confine_sharks()
            self._apply_walls(self.sharks)

        # Fish sense (read) then move (write): flock, flee the nearest shark, and
        # steer toward nearby food.
        self._forage = None
        if len(self.fish):
            fish = self.fish
            size = fish.size
            max_speed = max_speed_of(size, cfg)
            mass_rel = mass_rel_of(size, cfg)

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
            food_desire = self._forage_desire(grid_radii=fish.genes[:, F_PERC])

            accel = fish_accelerations(
                fish.vel, fish.genes, max_speed, i, j, rel, d2,
                flee_offset, flee_d2, food_desire, cfg,
            )
            self._integrate_fish(accel, mass_rel, max_speed)
            self._apply_walls(self.fish, max_speed=max_speed)
            self._apply_edges_fish()

        # Energy economy: eat, burn/grow, starve. (Uses self._forage before any
        # fish/food compaction invalidates its indices.)
        self._fish_eat()
        self._fish_metabolise_and_grow()
        self._fish_starve()

        # Predation, then births and deaths for both species.
        self._predate()
        self._shark_metabolise_and_die()
        self._shark_reproduce()
        self._fish_reproduce()

        # Food drifts, clumps, and scatters from wakes (after fish have moved).
        self._food_physics()
        self._update_stats()

    # --- Food ---------------------------------------------------------------

    def _food_spawn(self) -> None:
        """Spawn a Poisson trickle of new food, bounded by the cap's headroom.
        Bigger particles carry proportionally more energy (``k * size²``)."""
        cfg = self.cfg
        headroom = cfg.max_food - len(self.food)
        if headroom <= 0:
            return
        k = min(headroom, int(self.rng.poisson(cfg.food_spawn_rate)))
        if k == 0:
            return
        size = self.rng.uniform(cfg.food_size_lo, cfg.food_size_hi, k)
        self.food.append(
            pos=self.rng.uniform(0, self.world, (k, 2)),
            vel=np.zeros((k, 2)),
            energy=cfg.food_energy_k * size * size,
            size=size,
        )

    def _forage_desire(self, grid_radii: np.ndarray) -> np.ndarray:
        """Per-fish steering desire toward nearby food ``(n, 2)``, and stash the
        fish→food pairs for :meth:`_fish_eat`.

        One `cross_pairs` pass (fish query a grid built over food) serves both the
        attraction here (mean direction to visible food) and eating later. Fish
        with no food in range get a zero desire.
        """
        n = len(self.fish)
        desire = np.zeros((n, 2))
        if not len(self.food):
            return desire
        cfg = self.cfg
        food_grid = build_grid(
            self.food.pos, cfg.width, cfg.height, CELL_MIN, cfg.wrap_edges
        )
        fi, fj, frel, fd2 = cross_pairs(
            food_grid, self.fish.pos, grid_radii, self.food.pos, self.world, cfg.wrap_edges
        )
        self._forage = (fi, fj, fd2)
        if len(fi):
            cnt = np.maximum(np.bincount(fi, minlength=n), 1)
            for c in (0, 1):
                desire[:, c] = np.bincount(fi, weights=frel[:, c], minlength=n) / cnt
        return desire

    def _fish_eat(self) -> None:
        """Fish consume food within their (size-scaled) mouth radius.

        Each food goes to its single nearest in-range fish (no double-counting),
        but a fish may swallow several foods in one tick. Distances are those from
        the sense pass — a tick's swim (≤ a few px) is well inside the mouth band.
        """
        forage = self._forage
        self._forage = None
        if forage is None or not len(self.food) or not len(self.fish):
            return
        cfg = self.cfg
        fish, food = self.fish, self.food
        fi, fj, fd2 = forage
        within = fd2 <= (cfg.eat_radius_k * fish.size[fi]) ** 2
        if not within.any():
            return
        ci, cj, cd2 = fi[within], fj[within], fd2[within]
        # For each food (cj), keep only its nearest fish: sort by food then
        # distance, take the first row of each food's run.
        order = np.lexsort((cd2, cj))
        cj_s, ci_s = cj[order], ci[order]
        first = np.empty(len(cj_s), dtype=bool)
        first[0] = True
        first[1:] = cj_s[1:] != cj_s[:-1]
        win_food, win_fish = cj_s[first], ci_s[first]

        energy = fish.energy
        np.add.at(energy, win_fish, food.energy[win_food])
        np.minimum(energy, energy_cap_of(fish.size, cfg), out=energy)
        eaten = np.zeros(len(food), dtype=bool)
        eaten[win_food] = True
        food.compact(~eaten)

    def _food_physics(self) -> None:
        """Drift + weak clumping + wake avoidance, all speed-capped below the fish
        minimum speed so food can never outrun a fish."""
        cfg = self.cfg
        food = self.food
        m = len(food)
        if m == 0:
            return
        vel = food.vel  # view; accumulate contributions in place
        wrap = cfg.wrap_edges

        # Weak clumping: nudge toward the local food centroid.
        if cfg.food_clump_k > 0.0:
            grid = build_grid(food.pos, cfg.width, cfg.height, cfg.food_clump_radius, wrap)
            radii = np.full(m, cfg.food_clump_radius)
            ci, _cj, crel, _cd2 = perception_pairs(grid, food.pos, radii, self.world, wrap)
            if len(ci):
                cnt = np.maximum(np.bincount(ci, minlength=m), 1)
                coh = np.zeros((m, 2))
                for c in (0, 1):
                    coh[:, c] = np.bincount(ci, weights=crel[:, c], minlength=m) / cnt
                vel += cfg.food_clump_k * limit_norm(coh, 1.0)

        # Wakes: nearby fish and sharks shove food away.
        self._food_wake(vel, self.fish, cfg.food_wake_radius, cfg.food_wake_k)
        self._food_wake(vel, self.sharks, cfg.shark_wake_radius, cfg.food_wake_k)

        # Brownian jitter, damping, hard speed cap, then move + confine.
        vel += self.rng.normal(0.0, cfg.food_wander_sigma, (m, 2))
        food.vel[:] = limit_norm(vel * cfg.food_damp, cfg.food_max_speed)
        pos = food.pos
        pos += food.vel
        if wrap:
            np.mod(pos, self.world, out=pos)
        else:
            np.clip(pos, (0.0, 0.0), self.world, out=pos)

    def _food_wake(self, vel: np.ndarray, movers, radius: float, strength: float) -> None:
        """Push food away from nearby ``movers`` (fish or sharks), stronger up
        close. The final speed cap keeps this gentle enough to stay catchable."""
        if strength <= 0.0 or not len(movers) or not len(self.food):
            return
        cfg = self.cfg
        grid = build_grid(movers.pos, cfg.width, cfg.height, radius, cfg.wrap_edges)
        radii = np.full(len(self.food), radius)
        fi, _mj, rel, d2 = cross_pairs(
            grid, self.food.pos, radii, movers.pos, self.world, cfg.wrap_edges
        )
        if not len(fi):
            return
        away = -rel / np.maximum(d2, EPS)[:, None]  # rel points mover-ward
        np.add.at(vel, fi, strength * away)

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
        ``kill_radius`` — and gains energy scaled by that fish's mass (a bigger
        fish is a bigger meal). Capping intake at one fish stops a single shark
        from vacuuming a whole cluster; it's the main predator–prey brake.

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
                sharks.energy[s] += cfg.shark_meal_energy_k * fish.size[f] ** 2
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
        """A grown, well-fed fish spawns a juvenile, splitting its energy reserve.

        Eligible = adult enough (``size >= maturity_size_frac * adult_size``) AND
        energy at/above its genetic ``repro_frac`` of cap AND able to afford the
        birth cost. Breed in row order until the safety cap is hit.
        """
        cfg = self.cfg
        fish = self.fish
        headroom = cfg.max_pop - len(fish)
        if headroom <= 0 or not len(fish):
            return
        genes, size, energy = fish.genes, fish.size, fish.energy
        cap = energy_cap_of(size, cfg)
        invest = energy * 0.5
        eligible = (
            (size >= cfg.maturity_size_frac * genes[:, F_ADULT])
            & (energy >= genes[:, F_REPROF] * cap)
            & (invest - cfg.fish_birth_cost > 0)
        )
        parents = np.flatnonzero(eligible)[:headroom]
        if not parents.size:
            return
        k = parents.size
        juvenile = np.full(k, cfg.juvenile_size)
        child_cap = energy_cap_of(juvenile, cfg)
        child_energy = np.minimum(invest[parents] - cfg.fish_birth_cost, child_cap)
        energy[parents] *= 0.5
        child_genes = mutate_genes(
            self.rng, FISH_SPEC, genes[parents], cfg.mutation_rate, cfg.mutation_sigma
        )
        jitter = self.rng.uniform(-FISH_SPAWN_JITTER, FISH_SPAWN_JITTER, (k, 2))
        self.fish.append(
            pos=fish.pos[parents] + jitter,
            vel=self._random_headings(max_speed_of(juvenile, cfg)),
            genes=child_genes,
            size=juvenile,
            energy=child_energy,
            generation=fish.generation[parents] + 1,
            age=np.zeros(k, dtype=np.int64),
        )
        self.births += k

    # --- Fish motion & energy -------------------------------------------------

    def _integrate_fish(self, accel: np.ndarray, mass_rel: np.ndarray, max_speed: np.ndarray) -> None:
        """Apply the tick's accelerations with momentum (``a = force / mass``), a
        size-derived speed clamp, then integrate position."""
        cfg = self.cfg
        fish = self.fish
        vel = limit_norm(fish.vel + accel / mass_rel[:, None], max_speed)
        # Keep a minimum speed so isolated fish keep drifting instead of freezing.
        speed2 = np.einsum("nc,nc->n", vel, vel)
        slow = (speed2 > 0.0) & (speed2 < cfg.min_speed**2)
        if slow.any():
            vel[slow] *= cfg.min_speed / np.sqrt(speed2[slow])[:, None]
        fish.vel[:] = vel
        fish.pos[:] += vel
        fish.age[:] += 1

    def _fish_metabolise_and_grow(self) -> None:
        """Burn each fish's metabolic cost, then invest surplus energy into growth
        toward its ``adult_size``. Energy is re-capped to the (possibly larger) cap."""
        cfg = self.cfg
        fish = self.fish
        if not len(fish):
            return
        size, energy, genes = fish.size, fish.energy, fish.genes
        speed = np.sqrt(np.einsum("nc,nc->n", fish.vel, fish.vel))
        energy -= metabolism(size, speed, genes[:, F_PERC], cfg)

        cap = energy_cap_of(size, cfg)
        adult = genes[:, F_ADULT]
        grow = (energy >= cfg.growth_surplus_frac * cap) & (size < adult)
        ds = np.where(grow, np.minimum(cfg.growth_speed, adult - size), 0.0)
        # Growing a body of area size² by ds costs energy per unit mass gained;
        # d(size²) ≈ 2·size·ds.
        energy -= cfg.growth_cost_k * (2.0 * size * ds)
        size += ds
        np.minimum(energy, energy_cap_of(size, cfg), out=energy)
        # Growth lowers a fish's top speed (bigger = slower); re-clamp so it never
        # ends the tick moving faster than its new size allows.
        if grow.any():
            fish.vel[:] = limit_norm(fish.vel, max_speed_of(size, cfg))

    def _fish_starve(self) -> None:
        """Cull fish whose energy has run out."""
        fish = self.fish
        if not len(fish):
            return
        starved = fish.compact(fish.energy > 0)
        self.deaths += starved
        self.starvations += starved

    # --- World geometry: walls and edges -------------------------------------

    def _apply_walls(self, state, max_speed: np.ndarray | None = None) -> None:
        """Soft-bounce a population off any nearby shard.

        A velocity nudge along the outward wall normal (stronger up close) steers
        around obstacles; anything that crossed *inside* a shard also gets a
        positional ejection back out. Sharks need that ejection because their
        velocity is rewritten from scratch each tick, so a nudge alone would be
        forgotten — they'd end up sliding along a wall instead. Pass ``max_speed``
        (per animal) to re-clamp nudged rows; fish keep their speed cap intact,
        sharks are left unclamped like the rest of their motion.
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
        fish = self.fish
        alive = len(fish) > 0
        stats: dict[str, float] = {
            "population": len(fish),
            "births": self.births,
            "deaths": self.deaths,
            "starvations": self.starvations,
            "shark_population": len(self.sharks),
            "shark_births": self.shark_births,
            "shark_deaths": self.shark_deaths,
            "food_count": len(self.food),
            "fish_size": float(fish.size.mean()) if alive else 0.0,
            "fish_energy": float(fish.energy.mean()) if alive else 0.0,
        }
        self._average_genes(stats, fish, GENE_NAMES, prefix="", lineage="generation")
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
