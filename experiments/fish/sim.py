"""The simulation world: rule-based fish prey and an evolving shark predator.

Each tick:

- **Sharks hunt** by brain: each runs its own evolved neural net (:mod:`brain`)
  over a sector retina of the fish it can see (limited by its ``perception``
  gene) and picks a turn and a speed between its ``min_speed`` and ``max_speed``
  genes. Eating a fish restores energy; every tick burns some (sprinting and
  keen sight burn more). Out of energy → the shark starves and dies; well-fed →
  it spawns a mutated child (genes and brain), splitting its energy.
- **Fish** are classic boids — identical, rule-based, non-evolving: the three
  Reynolds rules plus flee-from-the-nearest-shark, all weighted by shared
  config knobs. Eaten fish are gone; replacements trickle in at a steady
  configurable rate (up to ``num_fish``), so the sea slowly refills after a
  feeding frenzy.

Only the sharks evolve. They have no population floor: if they can't catch
enough they go extinct (the hall-of-fame rescue rebreeds them); if they're too
good they thin the sea faster than it refills, and then starve.

Populations live in struct-of-arrays form (:mod:`state`), and every per-tick rule
below is a whole-population array operation — see :mod:`neighbors` for the pair
search and :mod:`steering` for the flocking math. Distances are toroidal
(minimum-image) whenever ``wrap_edges`` is on.

Deliberately free of any pygame code so the simulation can be stepped and
unit-tested headlessly.
"""

from __future__ import annotations

import numpy as np

from brain import (
    N_SECTORS,
    N_SENSES,
    BrainSpec,
    crossover_brains,
    forward,
    mutate_brains,
    random_brains,
    sector_retina,
)
from config import Config
from genome import (
    S_MAXSPD,
    S_MINSPD,
    S_PERC,
    SHARK_GENE_NAMES,
    SHARK_SPEC,
    crossover_genes,
    mutate_genes,
    random_genes,
)
from neighbors import build_grid, min_image, nearest_points, perception_pairs
from state import FishState, SharkState
from steering import EPS, fish_accelerations, limit_norm
from walls import WallField, avoid_all

# Behaviour constants (fixed by design, not evolved or config-tuned).
SHARK_SPAWN_JITTER = 10.0  # newborn shark's offset from its parent


class Simulation:
    def __init__(self, cfg: Config, seed: int | None = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.world = np.array([float(cfg.width), float(cfg.height)])

        # Static obstacles, built once (they never move) — seeded for reproducibility.
        self.rebuild_walls()

        # Small capacity floor so tests can hand-insert fish into an empty sea;
        # the sim itself never spawns past num_fish.
        self.fish = FishState(capacity=max(cfg.num_fish, 8))
        self._respawn_debt = 0.0  # fractional fish owed by the trickle respawn
        # One brain layout per sim, derived from config — it sizes the shark
        # state's weight buffer and the observation every net is run on.
        self.brain_spec = BrainSpec(cfg.shark_memory_frames, cfg.shark_hidden)
        self.sharks = SharkState(
            capacity=max(cfg.max_sharks, cfg.num_sharks),
            weight_count=self.brain_spec.weight_count,
            memory_frames=cfg.shark_memory_frames,
        )
        self._spawn_initial_populations()

        # Lifetime tallies + a per-step snapshot for the HUD.
        self.births = 0
        self.deaths = 0
        self.shark_births = 0
        self.shark_deaths = 0
        # Elite (score, genes, brain, generation) snapshots — the best sharks
        # seen so far, kept so a rescue can breed learned hunters back after an
        # extinction.
        self.hall_of_fame: list[tuple[float, np.ndarray, np.ndarray, int]] = []
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
        self.fish.append(
            pos=self.rng.uniform(0, self.world, (n, 2)),
            vel=self._random_headings(n),
        )

        s = cfg.num_sharks
        shark_genes = random_genes(self.rng, SHARK_SPEC, s)
        self.sharks.append(
            pos=self.rng.uniform(0, self.world, (s, 2)),
            vel=self._shark_spawn_vel(shark_genes),
            genes=shark_genes,
            brain=random_brains(
                self.rng, s, cfg.brain_init_scale, self.brain_spec.weight_count
            ),
            memory=self._blank_memory(s),
            generation=np.zeros(s, dtype=np.int64),
            age=np.zeros(s, dtype=np.int64),
            energy=np.full(s, cfg.shark_start_energy),
        )

    def _blank_memory(self, k: int) -> np.ndarray:
        """``(k, memory_frames, N_SENSES)`` zeroed frame stacks — every newborn
        starts blind to the past ("saw nothing before")."""
        return np.zeros((k, self.cfg.shark_memory_frames, N_SENSES))

    def _random_headings(self, k: int) -> np.ndarray:
        """``(k, 2)`` fish velocities: random directions, speeds between the
        drift floor and the shared top speed."""
        angle = self.rng.uniform(0, 2 * np.pi, k)
        speed = self.rng.uniform(self.cfg.min_speed, self.cfg.fish_max_speed, k)
        return np.stack([np.cos(angle), np.sin(angle)], axis=1) * speed[:, None]

    def _shark_spawn_vel(self, genes: np.ndarray) -> np.ndarray:
        """``(k, 2)`` newborn-shark velocities: random headings at each shark's
        cruise floor, so a fresh shark's heading (its retina frame) is defined
        and diverse from tick one instead of every newborn facing +x."""
        angle = self.rng.uniform(0, 2 * np.pi, len(genes))
        speed = np.minimum(genes[:, S_MINSPD], genes[:, S_MAXSPD])
        return np.stack([np.cos(angle), np.sin(angle)], axis=1) * speed[:, None]

    def add_fish(self, pos, vel=None) -> None:
        """Insert a single fish — mainly for tests."""
        vel = (
            self._random_headings(1)
            if vel is None
            else np.asarray(vel, dtype=float)[None]
        )
        self.fish.append(pos=np.asarray(pos, dtype=float)[None], vel=vel)

    def add_shark(self, pos, genes=None, energy: float | None = None, weights=None) -> None:
        """Insert a single shark (mid-range genes, random brain by default) —
        mainly for tests."""
        genes = SHARK_SPEC.mid() if genes is None else np.asarray(genes, dtype=float)
        weights = (
            random_brains(
                self.rng, 1, self.cfg.brain_init_scale, self.brain_spec.weight_count
            )
            if weights is None
            else np.asarray(weights, dtype=float)[None]
        )
        self.sharks.append(
            pos=np.asarray(pos, dtype=float)[None],
            vel=self._shark_spawn_vel(genes[None]),
            genes=genes[None],
            brain=weights,
            memory=self._blank_memory(1),
            generation=np.zeros(1, dtype=np.int64),
            age=np.zeros(1, dtype=np.int64),
            energy=np.array([self.cfg.shark_start_energy if energy is None else energy]),
        )

    # --- Hall of fame & rescue ---------------------------------------------

    def _update_hall_of_fame(self) -> None:
        """Snapshot this tick's best live sharks into the hall-of-fame.

        Score is ``energy + hof_age_weight * age`` — a shark that's both well-fed
        (it can hunt) and long-lived (it survived). Snapshots are *copies*, since
        the SoA rows get overwritten as the population is compacted and reused.
        Runs every tick so elites are archived while alive, not lost to starvation.
        """
        sharks = self.sharks
        if not len(sharks):
            return
        cfg = self.cfg
        scores = sharks.energy + cfg.hof_age_weight * sharks.age
        k = min(cfg.hof_size, len(sharks))
        top = np.argpartition(scores, -k)[-k:]
        for idx in top:
            self.hall_of_fame.append(
                (
                    float(scores[idx]),
                    sharks.genes[idx].copy(),
                    sharks.brain[idx].copy(),
                    int(sharks.generation[idx]),
                )
            )
        # Keep only the best hof_size overall (best-scoring first).
        self.hall_of_fame.sort(key=lambda e: e[0], reverse=True)
        del self.hall_of_fame[cfg.hof_size:]

    def rescue_sharks(self, n: int) -> None:
        """Spawn up to ``n`` sharks *bred* from the hall-of-fame's elites.

        Each newcomer is a child of two distinct elites (uniform crossover of
        genes and brain), then mutated as usual (scaled by ``rescue_mut_scale``)
        — so a rescue continues evolution from the archive instead of replaying
        identical clones. Its ``generation`` is the older parent's plus one,
        keeping lineage depth honest across extinctions. With a single archived
        elite the child degenerates to a mutated clone. Falls back to random
        sharks only if nothing has been archived yet. Never exceeds
        ``max_sharks``.
        """
        cfg = self.cfg
        n = min(n, cfg.max_sharks - len(self.sharks))
        if n <= 0:
            return
        if self.hall_of_fame:
            hof = self.hall_of_fame
            mothers = self.rng.integers(0, len(hof), n)
            fathers = (  # a distinct second parent, when the archive has one
                mothers if len(hof) == 1
                else (mothers + self.rng.integers(1, len(hof), n)) % len(hof)
            )
            genes = mutate_genes(
                self.rng, SHARK_SPEC,
                crossover_genes(
                    self.rng,
                    np.stack([hof[i][1] for i in mothers]),
                    np.stack([hof[i][1] for i in fathers]),
                ),
                cfg.shark_mutation_rate * cfg.rescue_mut_scale,
                cfg.shark_mutation_sigma * cfg.rescue_mut_scale,
            )
            brain = mutate_brains(
                self.rng,
                crossover_brains(
                    self.rng,
                    np.stack([hof[i][2] for i in mothers]),
                    np.stack([hof[i][2] for i in fathers]),
                ),
                cfg.brain_mutation_rate * cfg.rescue_mut_scale,
                cfg.brain_mutation_sigma * cfg.rescue_mut_scale,
                cfg.brain_weight_cap,
            )
            generation = 1 + np.maximum(
                np.array([hof[i][3] for i in mothers], dtype=np.int64),
                np.array([hof[i][3] for i in fathers], dtype=np.int64),
            )
        else:
            genes = random_genes(self.rng, SHARK_SPEC, n)
            brain = random_brains(
                self.rng, n, cfg.brain_init_scale, self.brain_spec.weight_count
            )
            generation = np.zeros(n, dtype=np.int64)
        self.sharks.append(
            pos=self.rng.uniform(0, self.world, (n, 2)),
            vel=self._shark_spawn_vel(genes),
            genes=genes,
            brain=brain,
            memory=self._blank_memory(n),
            generation=generation,
            age=np.zeros(n, dtype=np.int64),
            energy=np.full(n, cfg.shark_start_energy),
        )
        self.shark_births += n

    # --- One simulation tick -----------------------------------------------

    def step(self, manual_target=None) -> None:
        """Advance the world one tick.

        ``manual_target`` (the mouse, as an ``(x, y)`` pair) makes the *nearest*
        shark chase that point; every other shark hunts autonomously.
        """
        cfg = self.cfg

        # Auto-floor: if the sharks went extinct last tick, reseed a few from the
        # hall-of-fame before the world runs, so their learned brains carry on.
        if not len(self.sharks) and self.hall_of_fame:
            self.rescue_sharks(cfg.min_sharks)

        # Sharks move first so fish react to their new positions.
        if len(self.sharks):
            self._update_sharks(manual_target)

        # Fish pass 1 (read): flock with visible neighbours, flee the nearest shark.
        if len(self.fish):
            fish = self.fish
            grid = build_grid(
                fish.pos, cfg.width, cfg.height, cfg.fish_perception, cfg.wrap_edges
            )
            i, j, rel, d2 = perception_pairs(
                grid,
                fish.pos,
                np.full(len(fish), cfg.fish_perception),
                self.world,
                cfg.wrap_edges,
            )
            if len(self.sharks):
                _, flee_offset, flee_d2 = nearest_points(
                    fish.pos, self.sharks.pos, self.world, cfg.wrap_edges
                )
            else:
                flee_offset = flee_d2 = None  # no sharks alive -> nothing to flee
            accel = fish_accelerations(
                fish.vel, i, j, rel, d2, flee_offset, flee_d2, cfg
            )

            # Pass 2 (write): integrate, then resolve walls and world edges.
            self._integrate_fish(accel)
            self._apply_walls(self.fish, max_speed=cfg.fish_max_speed)
            self._apply_edges_fish()

        # Predation feeds the sharks, then their GA births and deaths run and
        # the sea trickle-refills with fresh fish.
        self._predate()
        self._shark_metabolise_and_die()
        self._shark_random_cull()
        self._shark_reproduce()
        self._fish_respawn()
        self._update_hall_of_fame()
        self._update_stats()

    # --- Shark brains -------------------------------------------------------

    def _update_sharks(self, manual_target) -> None:
        """One motion tick for every shark: sense → think → move.

        Each shark's own neural net turns its observation into a heading change
        (capped at ``shark_max_turn``) and a speed between its ``min_speed`` and
        ``max_speed`` genes. The mouse (``manual_target``) overrides the nearest
        shark's brain for the tick, steering it at the cursor at full speed.
        """
        cfg = self.cfg
        sharks = self.sharks
        headings = self._shark_headings()
        obs = self._shark_observe(headings)
        # Push this frame's senses into the stack for the next tick (newest
        # first). NumPy buffers the overlapping slice assignment, so this is a
        # genuine shift; a no-op when shark_memory_frames = 0.
        if sharks.memory.shape[1]:
            sharks.memory[:, 1:] = sharks.memory[:, :-1]
            sharks.memory[:, 0] = obs[:, :N_SENSES]
        turn, throttle = forward(self.brain_spec, sharks.brain, obs)

        theta = np.arctan2(headings[:, 1], headings[:, 0]) + turn * cfg.shark_max_turn
        min_speed = np.minimum(sharks.genes[:, S_MINSPD], sharks.genes[:, S_MAXSPD])
        speed = min_speed + throttle * (sharks.genes[:, S_MAXSPD] - min_speed)
        vel = np.stack([np.cos(theta), np.sin(theta)], axis=1) * speed[:, None]

        if manual_target is not None:
            # The mouse drives the shark nearest to it (plain screen distance).
            offset = np.asarray(manual_target, dtype=float) - sharks.pos
            nearest = np.einsum("sc,sc->s", offset, offset).argmin()
            vel[nearest] = limit_norm(
                offset[nearest][None], sharks.genes[nearest, S_MAXSPD]
            )[0]

        sharks.vel[:] = vel
        sharks.pos[:] += vel
        self._confine_sharks()
        self._apply_walls(self.sharks)

    def _shark_headings(self) -> np.ndarray:
        """``(S, 2)`` unit facing per shark; a (rare) motionless shark faces +x.
        Newborns spawn already moving, so the fallback only covers edge cases
        like a wall push exactly cancelling a shark's velocity."""
        vel = self.sharks.vel
        speed = np.sqrt(np.einsum("sc,sc->s", vel, vel))
        return np.where(
            speed[:, None] > EPS, vel / np.maximum(speed, EPS)[:, None], (1.0, 0.0)
        )

    def _shark_observe(self, headings: np.ndarray) -> np.ndarray:
        """``(S, n_inputs)`` observations: this frame's sense block — the sector
        retina, then energy (1.0 = ready to breed), speed (fraction of own top
        speed) and obstacle proximity — followed by the stacked sense blocks of
        the last ``shark_memory_frames`` frames (newest first) and a constant
        bias. Everything is bounded so no input can dwarf the rest of an
        evolved network. Read-only: the caller shifts the memory stack."""
        cfg = self.cfg
        sharks, fish = self.sharks, self.fish
        obs = np.zeros((len(sharks), self.brain_spec.n_inputs))
        if len(fish):
            delta = fish.pos[None, :, :] - sharks.pos[:, None, :]  # (S, F, 2)
            if cfg.wrap_edges:
                delta = min_image(delta, self.world)
            d2 = np.einsum("sfc,sfc->sf", delta, delta)
            obs[:, : 2 * N_SECTORS] = sector_retina(
                delta, d2, headings, sharks.genes[:, S_PERC], cfg.retina_count_norm
            )
        speed = np.sqrt(np.einsum("sc,sc->s", sharks.vel, sharks.vel))
        obs[:, 2 * N_SECTORS] = np.clip(sharks.energy / cfg.shark_repro_energy, 0.0, 1.0)
        obs[:, 2 * N_SECTORS + 1] = np.clip(speed / sharks.genes[:, S_MAXSPD], 0.0, 1.0)
        obs[:, 2 * N_SECTORS + 2] = self._shark_obstacle_signal()
        obs[:, N_SENSES:-1] = sharks.memory.reshape(len(sharks), -1)
        obs[:, -1] = 1.0
        return obs

    def _shark_obstacle_signal(self) -> np.ndarray:
        """``(S,)`` in 0–1: how hard the world pushes back where the shark is.

        The slot keeps one meaning across every mode so evolved brains transfer:
        0 everywhere in the default wrap-and-no-walls world, rising near a border
        in bounce mode and near a shard face when walls are on (reusing the same
        ``avoid_all`` response the motion pass applies)."""
        cfg = self.cfg
        pos = self.sharks.pos
        signal = np.zeros(len(pos))
        if not cfg.wrap_edges:
            border = np.minimum(pos, self.world - pos).min(axis=1)
            signal = np.clip(1.0 - border / cfg.edge_margin, 0.0, 1.0)
        if cfg.walls_enabled:
            push, _eject = avoid_all(pos, self.walls, cfg.wall_margin)
            push_norm = np.sqrt(np.einsum("sc,sc->s", push, push))
            signal = np.maximum(signal, np.minimum(push_norm, 1.0))
        return signal

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
        """Burn each shark's per-tick energy cost; cull any that hit zero.

        The speed term is quadratic in the speed *actually swum* this tick, not
        the max_speed gene — so the brain's throttle output faces a real bill:
        cruising is cheap, sprinting costs, and bursting only when closing on
        prey is what evolution should reward."""
        cfg = self.cfg
        sharks = self.sharks
        if not len(sharks):
            return
        cost = (
            cfg.shark_metab_base
            + cfg.shark_metab_speed_k * np.einsum("sc,sc->s", sharks.vel, sharks.vel)
            + cfg.shark_metab_perc_k * sharks.genes[:, S_PERC]
        )
        sharks.energy[:] -= cost
        sharks.age[:] += 1
        self.shark_deaths += sharks.compact(sharks.energy > 0)

    def _shark_random_cull(self) -> None:
        """When the population sits at the cap, kill ``shark_cull_count`` at random.

        A full population of sharks that are good at staying alive blocks every
        birth, freezing shark evolution. The victims are drawn *uniformly* — not
        worst-first — so the cull only opens breeding headroom without adding a
        selection pressure of its own. Runs just before reproduction, so the
        freed slots can be filled by well-fed parents on the very same tick.
        """
        cfg = self.cfg
        sharks = self.sharks
        if cfg.shark_cull_count <= 0 or len(sharks) < cfg.max_sharks:
            return
        doomed = self.rng.choice(
            len(sharks), size=min(cfg.shark_cull_count, len(sharks)), replace=False
        )
        keep = np.ones(len(sharks), dtype=bool)
        keep[doomed] = False
        self.shark_deaths += sharks.compact(keep)

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
        child_genes = mutate_genes(
            self.rng, SHARK_SPEC, sharks.genes[parents],
            cfg.shark_mutation_rate, cfg.shark_mutation_sigma,
        )
        self.sharks.append(
            pos=sharks.pos[parents] + jitter,
            vel=self._shark_spawn_vel(child_genes),
            genes=child_genes,
            brain=mutate_brains(
                self.rng, sharks.brain[parents],
                cfg.brain_mutation_rate, cfg.brain_mutation_sigma,
                cfg.brain_weight_cap,
            ),
            memory=self._blank_memory(k),
            generation=sharks.generation[parents] + 1,
            age=np.zeros(k, dtype=np.int64),
            energy=child_energy,
        )
        self.shark_births += k

    def _fish_respawn(self) -> None:
        """Trickle the sea back up to ``num_fish`` after predation.

        Fish don't breed — eaten ones are simply replaced at a steady
        ``fish_respawn_rate`` per tick (fractional rates accumulate as debt),
        each appearing at a random spot with a random heading."""
        cfg = self.cfg
        headroom = cfg.num_fish - len(self.fish)
        if headroom <= 0:
            return
        self._respawn_debt += cfg.fish_respawn_rate
        k = min(int(self._respawn_debt), headroom)
        if k <= 0:
            return
        self._respawn_debt -= k
        self.fish.append(
            pos=self.rng.uniform(0, self.world, (k, 2)),
            vel=self._random_headings(k),
        )
        self.births += k

    # --- Fish motion ----------------------------------------------------------

    def _integrate_fish(self, accel: np.ndarray) -> None:
        """Apply the tick's accelerations: speed-clamped velocity, then position."""
        cfg = self.cfg
        fish = self.fish
        vel = limit_norm(fish.vel + accel, cfg.fish_max_speed)
        # Keep a minimum speed so isolated fish keep drifting instead of freezing.
        speed2 = np.einsum("nc,nc->n", vel, vel)
        slow = (speed2 > EPS) & (speed2 < cfg.min_speed**2)
        if slow.any():
            vel[slow] *= cfg.min_speed / np.sqrt(speed2[slow])[:, None]
        # A fully cancelled velocity (rules exactly opposing the motion) keeps
        # drifting along the previous heading instead of freezing in place.
        dead = speed2 <= EPS
        if dead.any():
            old = fish.vel[dead]
            old_speed = np.sqrt(np.einsum("nc,nc->n", old, old))
            live = old_speed > EPS
            rows = np.flatnonzero(dead)[live]
            vel[rows] = old[live] / old_speed[live][:, None] * cfg.min_speed
        fish.vel[:] = vel
        fish.pos[:] += vel

    # --- World geometry: walls and edges -------------------------------------

    def _apply_walls(self, state, max_speed: float | None = None) -> None:
        """Soft-bounce a population off any nearby shard.

        A velocity nudge along the outward wall normal (stronger up close) steers
        around obstacles; anything that crossed *inside* a shard also gets a
        positional ejection back out. Sharks need that ejection because their
        velocity is rewritten from scratch each tick, so a nudge alone would be
        forgotten — they'd end up sliding along a wall instead. Pass ``max_speed``
        to re-clamp nudged rows; fish keep their shared speed cap intact, sharks
        are left unclamped like the rest of their motion.
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
                state.vel[pushed] = limit_norm(state.vel[pushed], max_speed)

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
            # Soft bounce: steer back inward near the borders, re-clamping any
            # turned fish to the shared top speed.
            turned = np.zeros(len(self.fish), dtype=bool)
            for axis, limit in enumerate(self.world):
                low = pos[:, axis] < cfg.edge_margin
                high = ~low & (pos[:, axis] > limit - cfg.edge_margin)
                vel[low, axis] += cfg.edge_turn
                vel[high, axis] -= cfg.edge_turn
                turned |= low | high
            rows = np.flatnonzero(turned)
            if rows.size:
                vel[rows] = limit_norm(vel[rows], cfg.fish_max_speed)
            # Hard safety bounce at the actual edge: clamp the position onto the
            # border and reflect the velocity component back inward.
            for axis, limit in enumerate(self.world):
                under = pos[:, axis] < 0.0
                over = pos[:, axis] > limit
                pos[under, axis] = 0.0
                vel[under, axis] = np.abs(vel[under, axis])
                pos[over, axis] = limit
                vel[over, axis] = -np.abs(vel[over, axis])

    # --- HUD statistics -----------------------------------------------------

    def _update_stats(self) -> None:
        """Population counts for both species, plus the shark GA's live means
        (per-gene averages, lineage depth, energy) — fish have nothing evolving
        to report."""
        sharks = self.sharks
        alive = len(sharks) > 0
        stats: dict[str, float] = {
            "population": len(self.fish),
            "births": self.births,
            "deaths": self.deaths,
            "shark_population": len(sharks),
            "shark_births": self.shark_births,
            "shark_deaths": self.shark_deaths,
            "shark_generation": float(sharks.generation.mean()) if alive else 0.0,
            "shark_energy": float(sharks.energy.mean()) if alive else 0.0,
        }
        for column, name in enumerate(SHARK_GENE_NAMES):
            stats["shark_" + name] = float(sharks.genes[:, column].mean()) if alive else 0.0
        self.stats = stats
