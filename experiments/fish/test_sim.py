"""Headless tests for the coevolving food/fish/shark simulation — no display.

Run:  uv run python test_sim.py   (or: uv run --with pytest pytest test_sim.py -q)
"""

from __future__ import annotations

import math

import numpy as np
from pygame.math import Vector2

from config import Config
from genome import (
    F_ADULT,
    F_PERC,
    F_WALI,
    F_WCOH,
    F_WFLEE,
    F_WSEP,
    FISH_SPEC,
    GENE_RANGES,
    SHARK_SPEC,
    gene_fractions,
    mutate_genes,
    random_genes,
)
from neighbors import build_grid, cross_pairs, min_image, nearest_points, perception_pairs
from physiology import energy_cap_of, mass_rel_of, max_speed_of
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


def _bare(**overrides) -> Config:
    """A world with no auto-spawned populations, for isolating one mechanic.

    Food capacity is left available (so tests can ``add_food``) but the spawn rate
    is zeroed, so ``step()`` never introduces food of its own.
    """
    base = dict(num_fish=0, num_sharks=0, max_food=50, food_spawn_rate=0.0)
    base.update(overrides)
    return Config(**base)


# --- Core physics -----------------------------------------------------------


def test_speed_never_exceeds_derived_max():
    cfg = Config(num_fish=120, width=600, height=400)
    sim = Simulation(cfg, seed=1)
    for _ in range(200):
        sim.step((300.0, 200.0))
    assert np.all(_speeds(sim.fish.vel) <= max_speed_of(sim.fish.size, cfg) + 1e-6)
    assert np.all(np.isfinite(sim.fish.pos))


def test_per_fish_max_speed_clamp():
    cfg = _bare()
    sim = Simulation(cfg, seed=1)
    sim.add_fish((100.0, 100.0), genes=_fish_genes(adult_size=1.5), size=1.5, vel=(0.0, 0.0))
    sim.add_fish((100.0, 100.0), genes=_fish_genes(adult_size=6.0), size=6.0, vel=(0.0, 0.0))
    size = sim.fish.size.copy()
    ms = max_speed_of(size, cfg)
    # An absurd shove; each fish clamps to its own size-derived top speed.
    sim._integrate_fish(np.array([[99.0, 0.0], [99.0, 0.0]]), mass_rel_of(size, cfg), ms)
    speeds = _speeds(sim.fish.vel)
    assert speeds[0] <= ms[0] + 1e-6 and speeds[1] <= ms[1] + 1e-6
    assert speeds[0] > speeds[1]  # smaller fish is faster


def test_momentum_heavier_fish_turns_slower():
    cfg = _bare()
    sim = Simulation(cfg, seed=1)
    # Sizes/velocity chosen so neither the speed cap nor the min-speed floor
    # engages — isolating the a = force/mass momentum effect.
    sim.add_fish((0.0, 0.0), genes=_fish_genes(adult_size=1.5), size=1.5, vel=(2.5, 0.0))
    sim.add_fish((0.0, 0.0), genes=_fish_genes(adult_size=3.0), size=3.0, vel=(2.5, 0.0))
    size = sim.fish.size.copy()
    accel = np.array([[0.0, 0.4], [0.0, 0.4]])  # equal sideways force
    sim._integrate_fish(accel, mass_rel_of(size, cfg), max_speed_of(size, cfg))
    turn_small, turn_big = sim.fish.vel[0, 1], sim.fish.vel[1, 1]
    assert turn_small > turn_big > 0.0  # the heavier fish deflects less


def test_positions_stay_in_bounds_when_wrapping():
    cfg = Config(num_fish=80, wrap_edges=True, width=500, height=500)
    sim = Simulation(cfg, seed=2)
    for _ in range(150):
        sim.step((250.0, 250.0))
    assert np.all(sim.fish.pos >= 0.0)
    assert np.all(sim.fish.pos < (cfg.width, cfg.height))


def test_separation_pushes_close_fish_apart():
    cfg = _bare()
    sim = Simulation(cfg, seed=1)
    genes = _fish_genes(w_sep=2.0, w_ali=0.0, w_coh=0.0, w_flee=0.0, w_food=0.0)
    sim.add_fish((100.0, 100.0), genes=genes, vel=(0.0, 0.0))
    sim.add_fish((105.0, 100.0), genes=genes, vel=(0.0, 0.0))
    for _ in range(5):
        sim.step()
    assert np.linalg.norm(sim.fish.pos[0] - sim.fish.pos[1]) > 5.0


def test_flee_desire_points_away_and_handles_no_shark():
    cfg = Config()
    # Shark just to the right (offset points left): desire pushes left, away.
    desire = flee_desire(np.array([[-10.0, 0.0]]), np.array([100.0]), cfg.flee_radius)
    assert desire[0, 0] < 0
    # A shark beyond the flee radius exerts no force.
    far = flee_desire(np.array([[9000.0, 0.0]]), np.array([9000.0**2]), cfg.flee_radius)
    assert not far.any()
    # No sharks alive: flee inputs are None and no force appears at all.
    none_pair = np.empty(0, dtype=np.int64)
    accel = fish_accelerations(
        np.zeros((1, 2)), _fish_genes()[None], np.array([3.0]),
        none_pair, none_pair, np.empty((0, 2)), np.empty(0),
        None, None, np.zeros((1, 2)), cfg,
    )
    assert not accel.any()


def test_derived_max_speed_from_size():
    cfg = Config()
    assert max_speed_of(2.0, cfg) > max_speed_of(6.0, cfg) > max_speed_of(12.0, cfg)
    assert max_speed_of(1.0, cfg) == cfg.speed_max_cap  # tiny fish hit the ceiling
    assert max_speed_of(100.0, cfg) == cfg.speed_min_cap  # huge fish hit the floor


# --- Food -------------------------------------------------------------------


def test_food_spawn_and_cap():
    cfg = Config(num_fish=0, num_sharks=0, max_food=50, food_spawn_rate=20.0)
    sim = Simulation(cfg, seed=1)
    for _ in range(20):
        sim.step()
        assert 0 <= len(sim.food) <= cfg.max_food
    assert len(sim.food) == cfg.max_food  # a high rate fills to the cap


def test_food_energy_scales_with_size():
    cfg = _bare()
    sim = Simulation(cfg, seed=1)
    sim.add_food((100.0, 100.0), size=3.0)
    sim.add_food((200.0, 200.0), size=5.0)
    assert sim.food.energy[0] == cfg.food_energy_k * 9.0
    assert sim.food.energy[1] == cfg.food_energy_k * 25.0


def test_fish_eats_food_and_gains_energy():
    cfg = _bare()
    sim = Simulation(cfg, seed=1)
    sim.add_fish((100.0, 100.0), genes=_fish_genes(adult_size=6.0), size=6.0, energy=50.0)
    sim.add_food((100.0, 100.0), size=4.0)
    sim._forage_desire(sim.fish.genes[:, F_PERC])
    sim._fish_eat()
    assert len(sim.food) == 0
    assert sim.fish.energy[0] == min(50.0 + cfg.food_energy_k * 16.0, energy_cap_of(6.0, cfg))


def test_eaten_energy_clamped_to_cap():
    cfg = _bare()
    sim = Simulation(cfg, seed=1)
    cap = float(energy_cap_of(4.0, cfg))
    sim.add_fish((100.0, 100.0), genes=_fish_genes(adult_size=4.0), size=4.0, energy=cap - 1.0)
    sim.add_food((100.0, 100.0), size=5.0)  # huge payload
    sim._forage_desire(sim.fish.genes[:, F_PERC])
    sim._fish_eat()
    assert sim.fish.energy[0] == cap  # capped, not overfilled


def test_food_assigned_to_nearest_fish():
    cfg = _bare()
    sim = Simulation(cfg, seed=1)
    genes = _fish_genes(adult_size=6.0)
    sim.add_fish((102.0, 100.0), genes=genes, size=6.0, energy=50.0)  # near
    sim.add_fish((109.0, 100.0), genes=genes, size=6.0, energy=50.0)  # far (>eat radius 9)
    sim.add_food((100.0, 100.0), size=3.0)
    sim._forage_desire(sim.fish.genes[:, F_PERC])
    sim._fish_eat()
    assert len(sim.food) == 0
    assert sim.fish.energy[0] > 50.0  # nearer fish ate it
    assert sim.fish.energy[1] == 50.0


def test_fish_eats_multiple_food_one_tick():
    cfg = _bare()
    sim = Simulation(cfg, seed=1)
    sim.add_fish((100.0, 100.0), genes=_fish_genes(adult_size=12.0), size=12.0, energy=100.0)
    for dx in (-3.0, 0.0, 3.0):  # three foods within the mouth radius (1.5*12=18)
        sim.add_food((100.0 + dx, 100.0), size=2.0)
    gain = 3 * cfg.food_energy_k * 4.0
    sim._forage_desire(sim.fish.genes[:, F_PERC])
    sim._fish_eat()
    assert len(sim.food) == 0
    assert sim.fish.energy[0] == min(100.0 + gain, energy_cap_of(12.0, cfg))


def test_wake_pushes_food_away():
    cfg = _bare()
    sim = Simulation(cfg, seed=1)
    sim.add_food((100.0, 100.0), size=3.0)
    sim.add_fish((105.0, 100.0), genes=_fish_genes(adult_size=6.0), size=6.0)  # to the right
    vel = np.zeros((1, 2))
    sim._food_wake(vel, sim.fish, cfg.food_wake_radius, cfg.food_wake_k)
    assert vel[0, 0] < 0  # food shoved left, away from the fish


def test_food_speed_bounded():
    cfg = Config(num_fish=40, num_sharks=5, max_food=200, food_spawn_rate=30.0)
    sim = Simulation(cfg, seed=3)
    for _ in range(60):
        sim.step()
    assert np.all(_speeds(sim.food.vel) <= cfg.food_max_speed + 1e-9)
    assert cfg.food_max_speed < cfg.min_speed  # fish can always overtake food


# --- Fish energy economy ----------------------------------------------------


def test_growth_toward_adult_size():
    cfg = _bare()
    sim = Simulation(cfg, seed=1)
    cap = float(energy_cap_of(2.0, cfg))
    sim.add_fish((100.0, 100.0), genes=_fish_genes(adult_size=6.0), size=2.0, energy=cap, vel=(0.0, 0.0))
    sizes = []
    for _ in range(120):
        sim._fish_metabolise_and_grow()
        sizes.append(float(sim.fish.size[0]))
    assert sizes[-1] > 2.0  # grew
    assert np.all(np.diff(sizes) >= -1e-12)  # monotonic
    assert sizes[-1] <= 6.0 + 1e-9  # never past the adult target


def test_growth_clamps_at_adult_size():
    cfg = _bare()
    sim = Simulation(cfg, seed=1)
    cap = float(energy_cap_of(6.0, cfg))
    sim.add_fish((100.0, 100.0), genes=_fish_genes(adult_size=6.0), size=5.99, energy=cap, vel=(0.0, 0.0))
    sim._fish_metabolise_and_grow()
    assert sim.fish.size[0] <= 6.0 + 1e-9


def test_no_growth_when_starving():
    cfg = _bare()
    sim = Simulation(cfg, seed=1)
    cap = float(energy_cap_of(2.0, cfg))
    sim.add_fish((100.0, 100.0), genes=_fish_genes(adult_size=6.0), size=2.0,
                 energy=0.1 * cap, vel=(0.0, 0.0))  # below the growth surplus threshold
    before = float(sim.fish.size[0])
    sim._fish_metabolise_and_grow()
    assert sim.fish.size[0] == before


def test_starvation_death():
    cfg = _bare()
    sim = Simulation(cfg, seed=1)
    sim.add_fish((100.0, 100.0), genes=_fish_genes(adult_size=6.0), size=6.0, energy=1.0)
    steps = 0
    while len(sim.fish) and steps < 5000:
        sim.step()
        steps += 1
    assert len(sim.fish) == 0
    assert sim.starvations >= 1 and sim.deaths >= 1


def test_juvenile_cannot_breed():
    cfg = _bare()
    sim = Simulation(cfg, seed=1)
    cap = float(energy_cap_of(3.0, cfg))
    # Full energy but too small to be an adult (0.5*adult < 0.9*adult).
    sim.add_fish((100.0, 100.0), genes=_fish_genes(adult_size=6.0, repro_frac=0.5),
                 size=3.0, energy=cap)
    sim._fish_reproduce()
    assert len(sim.fish) == 1 and sim.births == 0


def test_energy_based_reproduction():
    cfg = _bare()
    sim = Simulation(cfg, seed=2)
    cap = float(energy_cap_of(6.0, cfg))
    sim.add_fish((300.0, 200.0), genes=_fish_genes(adult_size=6.0, repro_frac=0.5),
                 size=6.0, energy=cap)
    sim._fish_reproduce()
    assert len(sim.fish) == 2 and sim.births == 1
    assert abs(sim.fish.energy[0] - cap * 0.5) < 1e-6  # parent halved
    assert sim.fish.size[1] == cfg.juvenile_size  # child is a juvenile
    child_cap = float(energy_cap_of(cfg.juvenile_size, cfg))
    assert sim.fish.energy[1] == min(cap * 0.5 - cfg.fish_birth_cost, child_cap)
    assert sim.fish.generation[1] == sim.fish.generation[0] + 1


# --- Genome -----------------------------------------------------------------


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
    genes[1, F_ADULT] = 99.0  # beyond the (3, 12) range
    genes[2, F_ADULT] = -99.0
    fracs = gene_fractions(genes, FISH_SPEC, "adult_size")
    assert np.allclose(fracs, [0.5, 1.0, 0.0])


# --- Shark genetic algorithm ------------------------------------------------


def test_shark_eats_fish_and_gains_energy():
    cfg = _bare()
    sim = Simulation(cfg, seed=4)
    sim.add_shark((100.0, 100.0), energy=100.0)
    sim.add_fish((100.0, 100.0), genes=_fish_genes(adult_size=5.0), size=5.0, vel=(0.0, 0.0))
    sim._predate()
    assert len(sim.fish) == 0
    assert sim.sharks.energy[0] == 100.0 + cfg.shark_meal_energy_k * 25.0
    assert sim.deaths >= 1


def test_predation_energy_scales_with_fish_size():
    cfg = _bare()
    gains = []
    for size in (3.0, 9.0):
        sim = Simulation(cfg, seed=4)
        sim.add_shark((100.0, 100.0), energy=100.0)
        sim.add_fish((100.0, 100.0), genes=_fish_genes(adult_size=size), size=size, vel=(0.0, 0.0))
        sim._predate()
        gains.append(sim.sharks.energy[0] - 100.0)
    assert gains[1] > gains[0]  # a bigger fish is a bigger meal


def test_predation_claims_are_sequential_and_nearest_first():
    cfg = _bare()
    sim = Simulation(cfg, seed=4)
    sim.add_fish((100.0, 100.0), size=5.0, vel=(0.0, 0.0))
    sim.add_shark((101.0, 100.0), energy=50.0)
    sim.add_shark((100.0, 100.0), energy=50.0)
    sim._predate()
    assert len(sim.fish) == 0
    assert sim.sharks.energy[0] > 50.0  # first shark claimed the fish
    assert sim.sharks.energy[1] == 50.0  # second went hungry

    sim2 = Simulation(cfg, seed=4)
    sim2.add_fish((105.0, 100.0), size=5.0, vel=(0.0, 0.0))
    sim2.add_fish((102.0, 100.0), size=5.0, vel=(0.0, 0.0))
    sim2.add_shark((100.0, 100.0), energy=50.0)
    sim2._predate()
    assert len(sim2.fish) == 1
    assert np.allclose(sim2.fish.pos[0], (105.0, 100.0))  # the far one survived


def test_unfed_shark_starves_and_can_go_extinct():
    cfg = _bare()
    sim = Simulation(cfg, seed=1)
    sim.add_shark((300.0, 200.0), energy=5.0)
    steps = 0
    while len(sim.sharks) and steps < 2000:
        sim.step()
        steps += 1
    assert len(sim.sharks) == 0
    assert sim.shark_deaths >= 1


def test_well_fed_shark_reproduces_and_splits_energy():
    cfg = _bare(shark_repro_energy=200.0, shark_birth_cost=20.0)
    sim = Simulation(cfg, seed=2)
    sim.add_shark((300.0, 200.0), energy=300.0)
    sim._shark_reproduce()
    assert len(sim.sharks) == 2 and sim.shark_births == 1
    assert abs(sim.sharks.energy[0] - 150.0) < 1e-6
    assert abs(sim.sharks.energy[1] - 130.0) < 1e-6
    assert sim.sharks.generation[1] == sim.sharks.generation[0] + 1


def test_shark_population_respects_cap():
    cfg = _bare(max_sharks=5)
    sim = Simulation(cfg, seed=2)
    for k in range(4):
        sim.add_shark((100.0 + k, 100.0), energy=5000.0)
    for _ in range(10):
        sim._shark_reproduce()
    assert len(sim.sharks) <= cfg.max_sharks


# --- Ecosystem --------------------------------------------------------------


def test_ecosystem_runs_within_invariants():
    cfg = Config(num_fish=60, num_sharks=5, width=600, height=400)
    sim = Simulation(cfg, seed=5)
    for _ in range(400):
        sim.step()
        assert 0 <= len(sim.fish) <= cfg.max_pop
        assert 0 <= len(sim.sharks) <= cfg.max_sharks
        assert 0 <= len(sim.food) <= cfg.max_food
    assert np.all(np.isfinite(sim.fish.pos))
    assert np.all(_speeds(sim.fish.vel) <= max_speed_of(sim.fish.size, cfg) + 1e-6)


def test_ecosystem_runs_with_walls():
    cfg = Config(num_fish=60, num_sharks=5, width=600, height=400, walls_enabled=True)
    sim = Simulation(cfg, seed=5)
    assert len(sim.walls.shards) == cfg.num_walls
    for _ in range(300):
        sim.step()
        assert 0 <= len(sim.fish) <= cfg.max_pop
        assert 0 <= len(sim.sharks) <= cfg.max_sharks
    assert np.all(np.isfinite(sim.fish.pos))


# --- Walls ------------------------------------------------------------------


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
    points = np.array([[200.0, 200.0], [165.0, 200.0], [160.0, 160.0], [500.0, 500.0]])
    push, eject = avoid_all(points, field, margin=26.0)
    assert eject[0].any() and push[0].any()  # inside -> ejected + pushed
    assert push[1, 0] < 0 and not eject[1].any()  # left band -> shoved left, no eject
    assert push[2, 0] < 0 and push[2, 1] < 0  # corner -> diagonal shove
    assert not push[3].any() and not eject[3].any()  # clear -> nothing


def test_fish_does_not_penetrate_wall():
    cfg = Config(num_fish=0, num_sharks=0, max_food=0, walls_enabled=False, width=600, height=400)
    sim = Simulation(cfg, seed=1)
    shard = _square_shard(300, 200, half=30.0)
    sim.walls = WallField([shard])
    cfg.walls_enabled = True
    sim.add_fish((220.0, 200.0), genes=_fish_genes(adult_size=5.0), size=5.0, vel=(5.0, 0.0))
    for _ in range(60):
        sim.step()
        pos = Vector2(*sim.fish.pos[0])
        depth = max((pos - a).dot(n) for a, _b, n in shard.edges)
        assert depth > -2.0, (pos, depth)
        assert np.all(np.isfinite(sim.fish.pos))


def test_walls_no_spurious_far_push():
    field = WallField([_square_shard(200, 200, half=30.0)])
    push, eject = avoid_all(np.array([[165.0, 500.0]]), field, margin=26.0)
    assert not push.any() and not eject.any()


# --- Vectorized engine kernels ----------------------------------------------


def test_min_image_folds_deltas_across_the_seam():
    world = np.array([1400.0, 800.0])
    folded = min_image(np.array([[1390.0, 0.0], [-750.0, 790.0]]), world)
    assert np.allclose(folded, [[-10.0, 0.0], [650.0, -10.0]])
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
    w, h = 1400, 800
    pos = rng.uniform(0, (w, h), (300, 2))
    pos[0], pos[1] = (1330.0, 100.0), (5.0, 100.0)
    radii = rng.uniform(*GENE_RANGES["perception"], 300)
    radii[0] = radii[1] = 80.0
    assert _pair_set(pos, radii, w, h, True) == _brute_pair_set(pos, radii, w, h, True)
    assert (0, 1) in _pair_set(pos, radii, w, h, True)

    w, h = 800, 600
    pos = rng.uniform(-40, (w + 40, h + 40), (300, 2))
    radii = rng.uniform(*GENE_RANGES["perception"], 300)
    assert _pair_set(pos, radii, w, h, False) == _brute_pair_set(pos, radii, w, h, False)

    for w in (150, 170):
        pos = rng.uniform(0, w, (60, 2))
        radii = rng.uniform(*GENE_RANGES["perception"], 60)
        for wrap in (True, False):
            assert _pair_set(pos, radii, w, w, wrap) == _brute_pair_set(pos, radii, w, w, wrap)


def test_cross_pairs_match_brute_force():
    rng = np.random.default_rng(7)
    for wrap in (True, False):
        for w, h in ((1400, 800), (150, 150), (170, 170)):
            world = np.array([float(w), float(h)])
            fish = rng.uniform(0, world, (150, 2))
            food = rng.uniform(0, world, (90, 2))
            radii = rng.uniform(*GENE_RANGES["perception"], 150)
            grid = build_grid(food, w, h, _CELL_MIN, wrap)
            a, b, _rel, _d2 = cross_pairs(grid, fish, radii, food, world, wrap)
            got = set(zip(a.tolist(), b.tolist()))
            delta = food[None, :, :] - fish[:, None, :]
            if wrap:
                delta = min_image(delta, world)
            bd2 = np.einsum("fgc,fgc->fg", delta, delta)
            mask = bd2 <= radii[:, None] ** 2
            brute = {(int(f), int(g)) for f, g in np.argwhere(mask)}
            assert got == brute, (wrap, w, h)


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


def _oracle_accel(a, pos, vel, genes, max_speed, shark, cfg):
    """Scalar reference for one fish's acceleration — a straight port of the
    original per-object Boid rules (sep/ali/coh/flee + Reynolds steering), using
    the size-derived ``max_speed`` and a fixed flee radius. Kept as an oracle."""

    def steer(desired):
        dx, dy = desired
        n2 = dx * dx + dy * dy
        if n2 == 0.0:
            return (0.0, 0.0)
        norm = math.sqrt(n2)
        ms = max_speed[a]
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
    flee = (fx / fd2, fy / fd2) if 0.0 < fd2 <= cfg.flee_radius**2 else (0.0, 0.0)

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
    size = rng.uniform(3.0, 12.0, n)
    max_speed = max_speed_of(size, cfg)
    heading = rng.uniform(0, 2 * math.pi, n)
    speed = rng.uniform(cfg.min_speed, max_speed)
    vel = np.stack([np.cos(heading), np.sin(heading)], axis=1) * speed[:, None]
    shark = np.array([300.0, 200.0])

    grid = build_grid(pos, w, h, _CELL_MIN, wrap=False)
    i, j, rel, d2 = perception_pairs(grid, pos, genes[:, F_PERC], world, wrap=False)
    _idx, flee_offset, flee_d2 = nearest_points(pos, shark[None, :], world, wrap=False)
    # food_desire zero -> the foraging term contributes nothing, so parity holds.
    accel = fish_accelerations(
        vel, genes, max_speed, i, j, rel, d2, flee_offset, flee_d2, np.zeros((n, 2)), cfg
    )
    for a in range(n):
        expected = _oracle_accel(a, pos, vel, genes, max_speed, shark, cfg)
        assert np.allclose(accel[a], expected, atol=1e-9)


def test_state_compact_and_append():
    rng = np.random.default_rng(1)
    fish = FishState(capacity=5)
    pos = rng.uniform(0, 100, (4, 2))
    fish.append(
        pos=pos,
        vel=np.zeros((4, 2)),
        genes=random_genes(rng, FISH_SPEC, 4),
        size=np.full(4, 6.0),
        energy=np.arange(4) * 10.0,
        generation=np.arange(4),
        age=np.arange(4) * 10,
    )
    assert len(fish) == 4

    removed = fish.compact(np.array([True, False, True, False]))
    assert removed == 2 and len(fish) == 2
    assert np.array_equal(fish.age, [0, 20])  # survivors kept their order
    assert np.allclose(fish.pos, pos[[0, 2]])
    assert np.array_equal(fish.energy, [0.0, 20.0])

    fish.pos[:] += 1.0  # views write through
    assert np.allclose(fish.pos, pos[[0, 2]] + 1.0)

    row = dict(
        pos=np.zeros((4, 2)), vel=np.zeros((4, 2)), genes=random_genes(rng, FISH_SPEC, 4),
        size=np.full(4, 6.0), energy=np.zeros(4),
        generation=np.zeros(4, dtype=np.int64), age=np.zeros(4, dtype=np.int64),
    )
    try:
        fish.append(**row)
        assert False, "expected the capacity assert to fire"
    except AssertionError as err:
        assert "cap" in str(err)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
