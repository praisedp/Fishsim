"""Headless tests for the coevolving fish-vs-shark simulation — no display needed.

Run:  uv run python test_sim.py
"""

from __future__ import annotations

import math

import numpy as np
from pygame.math import Vector2

from brain import (
    N_INPUTS,
    N_SECTORS,
    WEIGHT_COUNT,
    forward,
    mutate_brains,
    random_brains,
    sector_retina,
)
from config import Config
from genome import (
    F_FEAR,
    F_MAXSPD,
    F_PERC,
    F_WALI,
    F_WCOH,
    F_WFLEE,
    F_WSEP,
    FISH_SPEC,
    GENE_RANGES,
    S_MAXSPD,
    S_MINSPD,
    S_PERC,
    SHARK_SPEC,
    gene_fractions,
    mutate_genes,
    random_genes,
)
from neighbors import build_grid, min_image, nearest_points, perception_pairs
from sim import Simulation
from state import FishState
from steering import fish_accelerations, flee_desire
from walls import Shard, WallField, avoid_all

_CELL_MIN = float(FISH_SPEC.hi[F_PERC])


def _fish_genes(**overrides) -> np.ndarray:
    """Fish genes ``(G,)`` with every gene at mid-range, save any overrides."""
    genes = FISH_SPEC.mid()
    for name, value in overrides.items():
        genes[FISH_SPEC.col(name)] = value
    return genes


def _speeds(vel: np.ndarray) -> np.ndarray:
    return np.sqrt(np.einsum("nc,nc->n", vel, vel))


# --- Core physics -----------------------------------------------------------


def test_speed_never_exceeds_each_fishs_max():
    cfg = Config(num_fish=120, width=600, height=400)
    sim = Simulation(cfg, seed=1)
    for _ in range(200):
        sim.step((300.0, 200.0))
    assert np.all(_speeds(sim.fish.vel) <= sim.fish.genes[:, F_MAXSPD] + 1e-6)
    assert np.all(np.isfinite(sim.fish.pos))  # not NaN


def test_per_fish_max_speed_clamp():
    cfg = Config(num_fish=0, num_sharks=0)
    sim = Simulation(cfg, seed=1)
    sim.add_fish((100.0, 100.0), genes=_fish_genes(max_speed=2.0), vel=(0.0, 0.0))
    sim.add_fish((100.0, 100.0), genes=_fish_genes(max_speed=5.0), vel=(0.0, 0.0))
    # An absurd shove; each fish should clamp to its own gene's max.
    sim._integrate_fish(np.array([[99.0, 0.0], [99.0, 0.0]]))
    slow, fast = _speeds(sim.fish.vel)
    assert slow <= 2.0 + 1e-6
    assert fast <= 5.0 + 1e-6
    assert fast > slow  # faster gene -> faster fish


def test_positions_stay_in_bounds_when_wrapping():
    cfg = Config(num_fish=80, wrap_edges=True, width=500, height=500)
    sim = Simulation(cfg, seed=2)
    for _ in range(150):
        sim.step((250.0, 250.0))
    assert np.all(sim.fish.pos >= 0.0)
    assert np.all(sim.fish.pos < (cfg.width, cfg.height))


def test_separation_pushes_close_fish_apart():
    cfg = Config(num_fish=0, num_sharks=0)
    sim = Simulation(cfg, seed=1)
    # Separation-only genomes so the rule under test isn't fought by cohesion.
    genes = _fish_genes(w_sep=2.0, w_ali=0.0, w_coh=0.0, w_flee=0.0)
    sim.add_fish((100.0, 100.0), genes=genes, vel=(0.0, 0.0))
    sim.add_fish((105.0, 100.0), genes=genes, vel=(0.0, 0.0))
    start_gap = 5.0
    for _ in range(5):
        sim.step()
    assert np.linalg.norm(sim.fish.pos[0] - sim.fish.pos[1]) > start_gap


def test_flee_desire_points_away_and_handles_no_shark():
    cfg = Config()
    fear = np.array([FISH_SPEC.mid()[F_FEAR]])
    # Shark just to the right (offset points left): desire pushes left, away.
    desire = flee_desire(np.array([[-10.0, 0.0]]), np.array([100.0]), fear, cfg.flee_radius)
    assert desire[0, 0] < 0
    # A shark beyond the fear-scaled radius exerts no force.
    far = flee_desire(np.array([[9000.0, 0.0]]), np.array([9000.0**2]), fear, cfg.flee_radius)
    assert not far.any()
    # No sharks alive: the flee inputs are None and no force appears at all.
    none_pair = np.empty(0, dtype=np.int64)
    accel = fish_accelerations(
        np.zeros((1, 2)), _fish_genes()[None], none_pair, none_pair,
        np.empty((0, 2)), np.empty(0), None, None, cfg,
    )
    assert not accel.any()


# --- Fish genetic algorithm -------------------------------------------------


def test_mature_fish_reproduce_and_grow_population():
    cfg = Config(
        num_fish=10, num_sharks=0, maturity_age=2, repro_prob=1.0, max_pop=40,
    )
    sim = Simulation(cfg, seed=3)
    start = len(sim.fish)
    for _ in range(8):
        sim.step()
    assert sim.births > 0
    assert start < len(sim.fish) <= cfg.max_pop
    assert sim.stats["generation"] > 0.0  # children carry deeper lineage


# --- Shark genetic algorithm ------------------------------------------------


def test_shark_eats_fish_and_gains_energy():
    cfg = Config(num_fish=0, num_sharks=0)
    sim = Simulation(cfg, seed=4)
    sim.add_shark((100.0, 100.0), energy=100.0)
    sim.add_fish((100.0, 100.0), vel=(0.0, 0.0))  # sitting right on the shark
    sim._predate()
    assert len(sim.fish) == 0  # eaten
    assert sim.sharks.energy[0] == 100.0 + cfg.shark_energy_per_fish
    assert sim.deaths >= 1


def test_predation_claims_are_sequential_and_nearest_first():
    cfg = Config(num_fish=0, num_sharks=0)
    sim = Simulation(cfg, seed=4)
    # Two sharks share one catchable fish: the first row claims it, the second
    # goes hungry — one fish feeds at most one shark.
    sim.add_fish((100.0, 100.0), vel=(0.0, 0.0))
    sim.add_shark((101.0, 100.0), energy=50.0)
    sim.add_shark((100.0, 100.0), energy=50.0)
    sim._predate()
    assert len(sim.fish) == 0
    assert sim.sharks.energy[0] == 50.0 + cfg.shark_energy_per_fish
    assert sim.sharks.energy[1] == 50.0

    # A shark with two fish in range eats the *nearest* one.
    sim2 = Simulation(cfg, seed=4)
    sim2.add_fish((105.0, 100.0), vel=(0.0, 0.0))
    sim2.add_fish((102.0, 100.0), vel=(0.0, 0.0))
    sim2.add_shark((100.0, 100.0), energy=50.0)
    sim2._predate()
    assert len(sim2.fish) == 1
    assert np.allclose(sim2.fish.pos[0], (105.0, 100.0))  # the far one survived


def test_unfed_shark_starves_and_can_go_extinct():
    cfg = Config(num_fish=0, num_sharks=0)
    sim = Simulation(cfg, seed=1)
    sim.add_shark((300.0, 200.0), energy=5.0)
    steps = 0
    while len(sim.sharks) and steps < 2000:
        sim.step()
        steps += 1
    assert len(sim.sharks) == 0  # no food -> starved out (no floor)
    assert sim.shark_deaths >= 1


def test_well_fed_shark_reproduces_and_splits_energy():
    cfg = Config(
        num_fish=0, num_sharks=0, shark_repro_energy=200.0, shark_birth_cost=20.0,
    )
    sim = Simulation(cfg, seed=2)
    sim.add_shark((300.0, 200.0), energy=300.0)
    sim._shark_reproduce()
    assert len(sim.sharks) == 2 and sim.shark_births == 1
    assert abs(sim.sharks.energy[0] - 150.0) < 1e-6  # parent: halved
    assert abs(sim.sharks.energy[1] - 130.0) < 1e-6  # child: half minus birth cost
    assert sim.sharks.generation[1] == sim.sharks.generation[0] + 1


def test_shark_population_respects_cap():
    cfg = Config(num_fish=0, num_sharks=0, max_sharks=5)
    sim = Simulation(cfg, seed=2)
    for k in range(4):
        sim.add_shark((100.0 + k, 100.0), energy=5000.0)  # rich enough to breed
    for _ in range(10):
        sim._shark_reproduce()
    assert len(sim.sharks) <= cfg.max_sharks


def test_hall_of_fame_archives_copies_of_best_sharks():
    cfg = Config(num_fish=0, num_sharks=0)
    sim = Simulation(cfg, seed=1)
    sim.add_shark((100.0, 100.0), energy=150.0)
    sim._update_hall_of_fame()
    assert len(sim.hall_of_fame) == 1
    _score, genes, brain, _generation = sim.hall_of_fame[0]
    # The snapshot is a *copy*: mutating the live shark leaves the archive intact.
    sim.sharks.genes[0] += 1.0
    sim.sharks.brain[0] += 1.0
    assert not np.array_equal(genes, sim.sharks.genes[0])
    assert not np.array_equal(brain, sim.sharks.brain[0])


def test_rescue_reseeds_learned_brains_after_extinction():
    # rescue_mut_scale=0 makes clones exact, so we can prove a rescue restores the
    # archived elite's brain rather than fresh random noise.
    cfg = Config(num_fish=0, num_sharks=0, rescue_mut_scale=0.0)
    sim = Simulation(cfg, seed=2)
    sim.add_shark((100.0, 100.0), energy=300.0)
    sim._update_hall_of_fame()  # archive the well-fed shark's brain
    archived_brain = sim.hall_of_fame[0][2].copy()
    sim.sharks.compact(np.zeros(len(sim.sharks), dtype=bool))  # wipe them out
    assert len(sim.sharks) == 0
    sim.rescue_sharks(5)
    assert len(sim.sharks) == 5 and sim.shark_births == 5
    assert all(np.array_equal(sim.sharks.brain[k], archived_brain) for k in range(5))


def test_rescue_breeds_children_from_two_elites():
    # Two archived elites with recognisably different genes/brains; mutation off
    # so every child gene/weight must equal one parent's or the other's.
    cfg = Config(num_fish=0, num_sharks=0, rescue_mut_scale=0.0)
    sim = Simulation(cfg, seed=5)
    lo, hi = SHARK_SPEC.lo.copy(), SHARK_SPEC.hi.copy()
    sim.add_shark((100.0, 100.0), genes=lo, energy=300.0, weights=np.full(WEIGHT_COUNT, -1.0))
    sim.add_shark((200.0, 200.0), genes=hi, energy=300.0, weights=np.full(WEIGHT_COUNT, 1.0))
    sim._update_hall_of_fame()
    sim.sharks.compact(np.zeros(len(sim.sharks), dtype=bool))  # extinct
    sim.rescue_sharks(6)
    assert len(sim.sharks) == 6
    for k in range(6):
        assert np.all((sim.sharks.genes[k] == lo) | (sim.sharks.genes[k] == hi))
        assert np.all(np.abs(sim.sharks.brain[k]) == 1.0)
    # With enough children and 266 weights, crossover virtually surely mixed
    # both parents into at least one child (a pure clone has all-equal weights).
    assert any(
        len(np.unique(sim.sharks.brain[k])) == 2 for k in range(6)
    ), "expected at least one child mixing both parents' brains"


def test_rescue_children_continue_generation_count():
    cfg = Config(num_fish=0, num_sharks=0, rescue_mut_scale=0.0)
    sim = Simulation(cfg, seed=6)
    sim.add_shark((100.0, 100.0), energy=300.0)
    sim.sharks.generation[0] = 7  # a deep lineage before the crash
    sim._update_hall_of_fame()
    sim.sharks.compact(np.zeros(len(sim.sharks), dtype=bool))  # extinct
    sim.rescue_sharks(3)
    assert np.all(sim.sharks.generation == 8)  # children, not gen-0 copies


def test_extinct_sharks_auto_respawn_from_hall_of_fame():
    cfg = Config(num_fish=0, num_sharks=0, min_sharks=6)
    sim = Simulation(cfg, seed=3)
    sim.add_shark((100.0, 100.0), energy=150.0)
    sim._update_hall_of_fame()  # something to reseed from
    sim.sharks.compact(np.zeros(len(sim.sharks), dtype=bool))  # extinct
    assert len(sim.sharks) == 0
    sim.step()  # auto-floor fires at the top of the tick
    assert len(sim.sharks) == cfg.min_sharks


def test_rescue_on_empty_hall_of_fame_uses_random_and_respects_cap():
    cfg = Config(num_fish=0, num_sharks=0, max_sharks=5)
    sim = Simulation(cfg, seed=4)
    assert not sim.hall_of_fame  # nothing archived yet
    sim.rescue_sharks(100)  # asks for far more than the cap allows
    assert len(sim.sharks) == cfg.max_sharks  # clamped to headroom, no crash
    assert np.all(sim.sharks.genes >= SHARK_SPEC.lo)
    assert np.all(sim.sharks.genes <= SHARK_SPEC.hi)


def test_ecosystem_runs_within_invariants():
    cfg = Config(num_fish=60, num_sharks=5, width=600, height=400)
    sim = Simulation(cfg, seed=5)
    for _ in range(300):
        sim.step()  # fully autonomous
        assert 0 <= len(sim.fish) <= cfg.max_pop
        assert 0 <= len(sim.sharks) <= cfg.max_sharks


# --- Shark brains -------------------------------------------------------------


def test_brain_forward_shapes_bounds_and_determinism():
    rng = np.random.default_rng(0)
    brains = random_brains(rng, 6, 0.8)
    obs = rng.uniform(0.0, 1.0, (6, N_INPUTS))
    turn, throttle = forward(brains, obs)
    assert turn.shape == (6,) and throttle.shape == (6,)
    assert np.all(np.abs(turn) <= 1.0)
    assert np.all((throttle >= 0.0) & (throttle <= 1.0))
    turn2, throttle2 = forward(brains, obs)
    assert np.array_equal(turn, turn2) and np.array_equal(throttle, throttle2)


def _oracle_retina(shark_pos, headings, fish_pos, perception, count_norm):
    """Scalar reference for the sector retina — one shark, one fish at a time."""
    out = np.zeros((len(shark_pos), 2 * N_SECTORS))
    for a in range(len(shark_pos)):
        nearest = [math.inf] * N_SECTORS
        count = [0] * N_SECTORS
        facing = math.atan2(headings[a, 1], headings[a, 0])
        for f in range(len(fish_pos)):
            dx = fish_pos[f, 0] - shark_pos[a, 0]
            dy = fish_pos[f, 1] - shark_pos[a, 1]
            if dx * dx + dy * dy > perception[a] ** 2:
                continue
            rel = math.atan2(dy, dx) - facing
            k = min(
                int(((rel + math.pi) % (2 * math.pi)) / (2 * math.pi) * N_SECTORS),
                N_SECTORS - 1,
            )
            count[k] += 1
            nearest[k] = min(nearest[k], math.hypot(dx, dy))
        for k in range(N_SECTORS):
            if count[k]:
                out[a, k] = max(0.0, 1.0 - nearest[k] / perception[a])
            out[a, N_SECTORS + k] = min(count[k] / count_norm, 1.0)
    return out


def test_sector_retina_matches_scalar_reference():
    rng = np.random.default_rng(11)
    shark_pos = rng.uniform(0, (600.0, 400.0), (9, 2))
    fish_pos = rng.uniform(0, (600.0, 400.0), (120, 2))
    angle = rng.uniform(0, 2 * math.pi, 9)
    headings = np.stack([np.cos(angle), np.sin(angle)], axis=1)
    perception = rng.uniform(80.0, 400.0, 9)
    delta = fish_pos[None, :, :] - shark_pos[:, None, :]
    d2 = np.einsum("sfc,sfc->sf", delta, delta)
    retina = sector_retina(delta, d2, headings, perception, count_norm=8.0)
    oracle = _oracle_retina(shark_pos, headings, fish_pos, perception, 8.0)
    assert np.allclose(retina, oracle, atol=1e-9)


def test_sector_retina_empty_cases():
    # No fish: all zeros. No sharks: an empty but well-shaped result.
    no_fish = sector_retina(
        np.empty((3, 0, 2)), np.empty((3, 0)), np.tile((1.0, 0.0), (3, 1)),
        np.full(3, 200.0), count_norm=8.0,
    )
    assert no_fish.shape == (3, 2 * N_SECTORS) and not no_fish.any()
    no_sharks = sector_retina(
        np.empty((0, 5, 2)), np.empty((0, 5)), np.empty((0, 2)),
        np.empty(0), count_norm=8.0,
    )
    assert no_sharks.shape == (0, 2 * N_SECTORS)


def test_brain_mutation_identity_and_clamp():
    rng = np.random.default_rng(2)
    parents = random_brains(rng, 40, 0.8)
    assert np.array_equal(mutate_brains(rng, parents, 0.0, 0.5, 2.0), parents)
    children = mutate_brains(rng, parents, 1.0, 10.0, 2.0)
    assert np.all(np.abs(children) <= 2.0)
    assert not np.array_equal(children, parents)


def test_shark_speed_stays_between_its_speed_genes():
    cfg = Config(num_fish=40, num_sharks=8, width=600, height=400)
    sim = Simulation(cfg, seed=8)
    for _ in range(100):
        sim.step()
        if not len(sim.sharks):
            break
        genes = sim.sharks.genes
        speeds = _speeds(sim.sharks.vel)
        low = np.minimum(genes[:, S_MINSPD], genes[:, S_MAXSPD])
        assert np.all(speeds >= low - 1e-9)
        assert np.all(speeds <= genes[:, S_MAXSPD] + 1e-9)


def test_speed_band_collapses_when_genes_meet():
    cfg = Config(num_fish=0, num_sharks=0)
    sim = Simulation(cfg, seed=3)
    genes = SHARK_SPEC.mid()
    genes[S_MINSPD] = 3.0  # top of the min range...
    genes[S_MAXSPD] = 3.0  # ...meets the bottom of the max range: a fixed pace
    sim.add_shark((100.0, 100.0), genes=genes)
    for _ in range(5):
        sim.step()
        assert np.allclose(_speeds(sim.sharks.vel), 3.0)


def test_shark_metabolism_charges_for_actual_speed():
    cfg = Config(num_fish=0, num_sharks=0)
    sim = Simulation(cfg, seed=1)
    sim.add_shark((100.0, 100.0), energy=50.0)
    sim.add_shark((300.0, 300.0), energy=50.0)
    sim.sharks.vel[0] = (1.0, 0.0)
    sim.sharks.vel[1] = (4.0, 0.0)
    perc = sim.sharks.genes[:, S_PERC].copy()
    sim._shark_metabolise_and_die()
    expected = (
        cfg.shark_metab_base
        + cfg.shark_metab_speed_k * np.array([1.0, 16.0])
        + cfg.shark_metab_perc_k * perc
    )
    assert np.allclose(sim.sharks.energy, 50.0 - expected)
    assert sim.sharks.energy[1] < sim.sharks.energy[0]  # sprinting costs more


def test_manual_target_overrides_the_nearest_sharks_brain():
    cfg = Config(num_fish=0, num_sharks=0)
    sim = Simulation(cfg, seed=4)
    sim.add_shark((100.0, 100.0))
    start = sim.sharks.pos[0].copy()
    sim.step((500.0, 100.0))
    moved = sim.sharks.pos[0] - start
    assert moved[0] > 0 and abs(moved[1]) < 1e-9  # straight toward the cursor
    assert np.isclose(_speeds(sim.sharks.vel)[0], sim.sharks.genes[0, S_MAXSPD])


def test_random_brained_sharks_still_eventually_eat():
    # Dense fish soup: even founder sharks with random brains must blunder into
    # prey well within the tick budget, or the hunting pipeline is broken.
    cfg = Config(num_fish=400, num_sharks=8, width=600, height=400)
    sim = Simulation(cfg, seed=0)
    for _ in range(600):
        sim.step()
        if sim.deaths:
            break
    assert sim.deaths > 0


# --- Wall obstacles ---------------------------------------------------------


def _square_shard(cx: float, cy: float, half: float = 30.0) -> Shard:
    """An axis-aligned square shard centred at (cx, cy)."""
    return Shard([
        Vector2(cx - half, cy - half),
        Vector2(cx + half, cy - half),
        Vector2(cx + half, cy + half),
        Vector2(cx - half, cy + half),
    ])


def test_wall_avoidance_pushes_outward():
    field = WallField([_square_shard(200, 200, half=30.0)])
    points = np.array(
        [[200.0, 200.0], [165.0, 200.0], [160.0, 160.0], [500.0, 500.0]]
    )
    push, eject = avoid_all(points, field, margin=26.0)
    # A point dead-centre inside is ejected and pushed away from the wall.
    assert eject[0].any() and push[0].any()
    # A point just left of the left face is shoved further left (outward),
    # with no ejection (it's outside).
    assert push[1, 0] < 0 and not eject[1].any()
    # A corner point sits in two faces' bands and is shoved diagonally out.
    assert push[2, 0] < 0 and push[2, 1] < 0
    # A point well clear of every shard feels nothing.
    assert not push[3].any() and not eject[3].any()


def test_fish_does_not_penetrate_wall():
    cfg = Config(num_fish=0, num_sharks=0, walls_enabled=False, width=600, height=400)
    sim = Simulation(cfg, seed=1)
    shard = _square_shard(300, 200, half=30.0)
    sim.walls = WallField([shard])
    cfg.walls_enabled = True
    # A fish to the left, driving straight at the shard at top speed.
    sim.add_fish((220.0, 200.0), genes=_fish_genes(max_speed=5.0), vel=(5.0, 0.0))
    for _ in range(60):
        sim.step()
        # Signed distance to the nearest face: >0 outside, <0 inside. Never let the
        # fish end a tick more than a hair inside the shard.
        pos = Vector2(*sim.fish.pos[0])
        depth = max((pos - a).dot(n) for a, _b, n in shard.edges)
        assert depth > -2.0, (pos, depth)
        assert np.all(np.isfinite(sim.fish.pos))


def test_ecosystem_runs_with_walls():
    cfg = Config(num_fish=60, num_sharks=5, width=600, height=400, walls_enabled=True)
    sim = Simulation(cfg, seed=5)
    assert len(sim.walls.shards) == cfg.num_walls
    for _ in range(300):
        sim.step()
        assert 0 <= len(sim.fish) <= cfg.max_pop
        assert 0 <= len(sim.sharks) <= cfg.max_sharks
    assert np.all(np.isfinite(sim.fish.pos))  # not NaN
    assert np.all(_speeds(sim.fish.vel) <= sim.fish.genes[:, F_MAXSPD] + 1e-6)


# --- Vectorized engine kernels ------------------------------------------------


def test_min_image_folds_deltas_across_the_seam():
    world = np.array([1400.0, 800.0])
    folded = min_image(np.array([[1390.0, 0.0], [-750.0, 790.0]]), world)
    assert np.allclose(folded, [[-10.0, 0.0], [650.0, -10.0]])
    # Deltas already shorter than half the world are untouched.
    assert np.allclose(min_image(np.array([[300.0, -150.0]]), world), [[300.0, -150.0]])


def _pair_set(pos, radii, width, height, wrap):
    world = np.array([float(width), float(height)])
    grid = build_grid(pos, width, height, _CELL_MIN, wrap)
    i, j, _rel, _d2 = perception_pairs(grid, pos, radii, world, wrap)
    return set(zip(i.tolist(), j.tolist()))


def _brute_pair_set(pos, radii, width, height, wrap):
    delta = pos[None, :, :] - pos[:, None, :]
    if wrap:
        delta = min_image(delta, np.array([float(width), float(height)]))
    d2 = np.einsum("abc,abc->ab", delta, delta)
    mask = d2 <= radii[:, None] ** 2
    np.fill_diagonal(mask, False)
    return {(int(a), int(b)) for a, b in np.argwhere(mask)}


def test_pairs_match_brute_force():
    rng = np.random.default_rng(42)

    # Wrap mode, incl. a planted pair straddling the seam (75 apart across it).
    w, h = 1400, 800
    pos = rng.uniform(0, (w, h), (300, 2))
    pos[0], pos[1] = (1330.0, 100.0), (5.0, 100.0)
    radii = rng.uniform(*GENE_RANGES["perception"], 300)
    radii[0] = radii[1] = 80.0
    assert _pair_set(pos, radii, w, h, True) == _brute_pair_set(pos, radii, w, h, True)
    assert (0, 1) in _pair_set(pos, radii, w, h, True)  # the seam pair was found

    # Bounce mode, incl. fish drifted off-screen (soft edges allow it).
    w, h = 800, 600
    pos = rng.uniform(-40, (w + 40, h + 40), (300, 2))
    radii = rng.uniform(*GENE_RANGES["perception"], 300)
    assert _pair_set(pos, radii, w, h, False) == _brute_pair_set(pos, radii, w, h, False)

    # Degenerate tiny worlds: a single cell (150 // 80 = 1) and a 2-cell axis
    # (170 // 80 = 2), where wrap offsets would alias cells if not deduplicated.
    for w in (150, 170):
        pos = rng.uniform(0, w, (60, 2))
        radii = rng.uniform(*GENE_RANGES["perception"], 60)
        for wrap in (True, False):
            assert _pair_set(pos, radii, w, w, wrap) == _brute_pair_set(pos, radii, w, w, wrap)


def test_nearest_shark_matches_loop():
    rng = np.random.default_rng(3)
    world = np.array([1400.0, 800.0])
    fish = rng.uniform(0, world, (200, 2))
    sharks = rng.uniform(0, world, (7, 2))
    for wrap in (True, False):
        idx, offset, d2 = nearest_points(fish, sharks, world, wrap)
        for a in range(len(fish)):
            deltas = fish[a] - sharks
            if wrap:
                deltas = min_image(deltas, world)
            dists = np.einsum("sc,sc->s", deltas, deltas)
            assert dists[idx[a]] == dists.min()
            assert abs(d2[a] - dists.min()) < 1e-9
            assert np.allclose(offset[a], deltas[idx[a]])


def _oracle_accel(a, pos, vel, genes, shark, cfg):
    """Scalar reference for one fish's acceleration — a straight port of the
    original per-object Boid rules (separation/alignment/cohesion/flee +
    Reynolds steering), kept as an oracle for the vectorized pipeline."""

    def steer(desired):
        dx, dy = desired
        n2 = dx * dx + dy * dy
        if n2 == 0.0:
            return (0.0, 0.0)
        norm = math.sqrt(n2)
        ms = genes[a, F_MAXSPD]
        sx, sy = dx / norm * ms - vel[a, 0], dy / norm * ms - vel[a, 1]
        s2 = sx * sx + sy * sy
        if s2 > cfg.max_force**2:
            sn = math.sqrt(s2)
            sx, sy = sx / sn * cfg.max_force, sy / sn * cfg.max_force
        return (sx, sy)

    neighbours = [
        b
        for b in range(len(pos))
        if b != a
        and (pos[b, 0] - pos[a, 0]) ** 2 + (pos[b, 1] - pos[a, 1]) ** 2
        <= genes[a, F_PERC] ** 2
    ]

    sep = [0.0, 0.0]
    close = 0
    for b in neighbours:
        ox, oy = pos[a, 0] - pos[b, 0], pos[a, 1] - pos[b, 1]
        d2 = ox * ox + oy * oy
        if 0.0 < d2 < cfg.separation_radius**2:
            sep[0] += ox / d2
            sep[1] += oy / d2
            close += 1
    sep = (sep[0] / close, sep[1] / close) if close else (0.0, 0.0)

    if neighbours:
        ali = (
            sum(vel[b, 0] for b in neighbours) / len(neighbours),
            sum(vel[b, 1] for b in neighbours) / len(neighbours),
        )
        coh = (
            sum(pos[b, 0] for b in neighbours) / len(neighbours) - pos[a, 0],
            sum(pos[b, 1] for b in neighbours) / len(neighbours) - pos[a, 1],
        )
    else:
        ali = coh = (0.0, 0.0)

    fx, fy = pos[a, 0] - shark[0], pos[a, 1] - shark[1]
    fd2 = fx * fx + fy * fy
    radius = cfg.flee_radius * genes[a, F_FEAR]
    flee = (fx / fd2, fy / fd2) if 0.0 < fd2 <= radius * radius else (0.0, 0.0)

    total = [0.0, 0.0]
    for w_col, desired in zip((F_WSEP, F_WALI, F_WCOH, F_WFLEE), (sep, ali, coh, flee)):
        sx, sy = steer(desired)
        total[0] += genes[a, w_col] * sx
        total[1] += genes[a, w_col] * sy
    return total


def test_steering_matches_scalar_reference():
    cfg = Config()
    rng = np.random.default_rng(9)
    n, w, h = 60, 600, 400
    world = np.array([float(w), float(h)])
    pos = rng.uniform(0, (w, h), (n, 2))
    genes = random_genes(rng, FISH_SPEC, n)
    heading = rng.uniform(0, 2 * math.pi, n)
    speed = rng.uniform(cfg.min_speed, genes[:, F_MAXSPD])
    vel = np.stack([np.cos(heading), np.sin(heading)], axis=1) * speed[:, None]
    shark = np.array([300.0, 200.0])

    grid = build_grid(pos, w, h, _CELL_MIN, wrap=False)
    i, j, rel, d2 = perception_pairs(grid, pos, genes[:, F_PERC], world, wrap=False)
    _idx, flee_offset, flee_d2 = nearest_points(pos, shark[None, :], world, wrap=False)
    accel = fish_accelerations(vel, genes, i, j, rel, d2, flee_offset, flee_d2, cfg)

    for a in range(n):
        assert np.allclose(accel[a], _oracle_accel(a, pos, vel, genes, shark, cfg), atol=1e-9)


def test_state_compact_and_append():
    rng = np.random.default_rng(1)
    fish = FishState(capacity=5)
    pos = rng.uniform(0, 100, (4, 2))
    fish.append(
        pos=pos,
        vel=np.zeros((4, 2)),
        genes=random_genes(rng, FISH_SPEC, 4),
        generation=np.arange(4),
        age=np.arange(4) * 10,
    )
    assert len(fish) == 4

    removed = fish.compact(np.array([True, False, True, False]))
    assert removed == 2 and len(fish) == 2
    assert np.array_equal(fish.age, [0, 20])  # survivors kept their order
    assert np.allclose(fish.pos, pos[[0, 2]])

    # Views write through to the buffers.
    fish.pos[:] += 1.0
    assert np.allclose(fish.pos, pos[[0, 2]] + 1.0)

    # Appending past capacity trips the guard assert.
    row = dict(
        pos=np.zeros((4, 2)),
        vel=np.zeros((4, 2)),
        genes=random_genes(rng, FISH_SPEC, 4),
        generation=np.zeros(4, dtype=np.int64),
        age=np.zeros(4, dtype=np.int64),
    )
    try:
        fish.append(**row)
        assert False, "expected the capacity assert to fire"
    except AssertionError as err:
        assert "cap" in str(err)


def test_mutation_rate_zero_is_identity():
    rng = np.random.default_rng(0)
    parents = random_genes(rng, FISH_SPEC, 50)
    children = mutate_genes(rng, FISH_SPEC, parents, rate=0.0, sigma=0.5)
    assert np.array_equal(children, parents)


def test_mutated_gene_arrays_stay_in_range():
    rng = np.random.default_rng(7)
    for spec in (FISH_SPEC, SHARK_SPEC):
        parents = random_genes(rng, spec, 500)
        children = mutate_genes(rng, spec, parents, rate=1.0, sigma=0.5)
        for genes in (parents, children):
            assert np.all(genes >= spec.lo) and np.all(genes <= spec.hi)


def test_gene_fractions_clipped():
    genes = np.tile(FISH_SPEC.mid(), (3, 1))
    genes[1, F_MAXSPD] = 99.0  # beyond the (2, 5) range
    genes[2, F_MAXSPD] = -99.0
    fracs = gene_fractions(genes, FISH_SPEC, "max_speed")
    assert np.allclose(fracs, [0.5, 1.0, 0.0])


def test_walls_no_spurious_far_push():
    # (165, 500) is far from the shard but within 26 of its left face's infinite
    # line (x = 170): the bbox prefilter must yield zero response, where naive
    # face-line distances would push.
    field = WallField([_square_shard(200, 200, half=30.0)])
    push, eject = avoid_all(np.array([[165.0, 500.0]]), field, margin=26.0)
    assert not push.any() and not eject.any()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
