"""Headless tests for the coevolving fish-vs-shark simulation — no display needed.

Run:  uv run python test_sim.py
"""

from __future__ import annotations

import random

from pygame.math import Vector2

from boid import Boid
from config import Config
from genome import (
    GENE_RANGES,
    SHARK_GENE_NAMES,
    SHARK_GENE_RANGES,
    Genome,
    SharkGenome,
)
from grid import SpatialGrid
from sim import Shark, Simulation
from walls import Shard, WallField, avoid


def _genome(**overrides) -> Genome:
    """A fish genome with every gene at its mid-range value, save any overrides."""
    base = {name: (lo + hi) / 2 for name, (lo, hi) in GENE_RANGES.items()}
    base.update(overrides)
    return Genome(**base)


def _boid(pos: Vector2, vel: Vector2, **gene_overrides) -> Boid:
    return Boid(pos, vel, _genome(**gene_overrides))


def _shark_genome(**overrides) -> SharkGenome:
    base = {name: (lo + hi) / 2 for name, (lo, hi) in SHARK_GENE_RANGES.items()}
    base.update(overrides)
    return SharkGenome(**base)


# --- Core physics -----------------------------------------------------------


def test_speed_never_exceeds_each_fishs_max():
    cfg = Config(num_fish=120, width=600, height=400)
    sim = Simulation(cfg, seed=1)
    for _ in range(200):
        sim.step(Vector2(300, 200))
    for f in sim.fish:
        assert f.vel.length() <= f.genome.max_speed + 1e-6, f.vel.length()
        assert f.pos.x == f.pos.x and f.pos.y == f.pos.y  # not NaN


def test_per_fish_max_speed_clamp():
    cfg = Config()
    slow = _boid(Vector2(100, 100), Vector2(0, 0), max_speed=2.0)
    fast = _boid(Vector2(100, 100), Vector2(0, 0), max_speed=5.0)
    for b in (slow, fast):
        b.accel = Vector2(99, 0)  # absurd shove; should clamp to the gene's max
        b.apply(cfg)
    assert slow.vel.length() <= 2.0 + 1e-6
    assert fast.vel.length() <= 5.0 + 1e-6
    assert fast.vel.length() > slow.vel.length()  # faster gene -> faster fish


def test_positions_stay_in_bounds_when_wrapping():
    cfg = Config(num_fish=80, wrap_edges=True, width=500, height=500)
    sim = Simulation(cfg, seed=2)
    for _ in range(150):
        sim.step(Vector2(250, 250))
    for f in sim.fish:
        assert 0 <= f.pos.x < cfg.width
        assert 0 <= f.pos.y < cfg.height


def test_separation_pushes_close_fish_apart():
    cfg = Config()
    # Separation-only genomes so the rule under test isn't fought by cohesion.
    a = _boid(Vector2(100, 100), Vector2(0, 0), w_sep=2.0, w_ali=0.0, w_coh=0.0, w_flee=0.0)
    b = _boid(Vector2(105, 100), Vector2(0, 0), w_sep=2.0, w_ali=0.0, w_coh=0.0, w_flee=0.0)
    start_gap = a.pos.distance_to(b.pos)
    for _ in range(5):
        a.compute([b], Vector2(9999, 9999), cfg)
        b.compute([a], Vector2(9999, 9999), cfg)
        a.apply(cfg)
        b.apply(cfg)
    assert a.pos.distance_to(b.pos) > start_gap


def test_flee_points_away_and_handles_no_shark():
    cfg = Config()
    f = _boid(Vector2(100, 100), Vector2(0, 0))
    shark = Vector2(110, 100)  # just to the right, inside the (fear-scaled) flee range
    steer = f.flee(shark, cfg)
    assert steer.x < 0  # pushed left, away from the shark
    assert f.flee(Vector2(9999, 9999), cfg).length() == 0  # far shark -> no force
    assert f.flee(None, cfg).length() == 0  # no sharks alive -> no force


# --- Spatial grid -----------------------------------------------------------


def test_grid_matches_brute_force_with_varied_perception():
    cfg = Config()
    rng = random.Random(42)
    # Each fish gets a different perception gene; the grid must still be exact when
    # queried with that per-fish radius (cell size is the max fish perception).
    fish = [
        _boid(
            Vector2(rng.uniform(0, 800), rng.uniform(0, 800)),
            Vector2(0, 0),
            perception=rng.uniform(*GENE_RANGES["perception"]),
        )
        for _ in range(300)
    ]
    grid = SpatialGrid(cfg.cell_size)
    grid.build(fish)
    for f in fish:
        r = f.genome.perception
        grid_set = {id(o) for o in grid.neighbours(f, r)}
        brute_set = {
            id(o) for o in fish if o is not f and f.pos.distance_to(o.pos) <= r
        }
        assert grid_set == brute_set


def test_grid_neighbours_at_matches_brute_force_for_wide_radius():
    cfg = Config()
    rng = random.Random(11)
    fish = [
        _boid(Vector2(rng.uniform(0, 1000), rng.uniform(0, 1000)), Vector2(0, 0))
        for _ in range(300)
    ]
    grid = SpatialGrid(cfg.cell_size)
    grid.build(fish)
    for _ in range(20):
        x, y = rng.uniform(0, 1000), rng.uniform(0, 1000)
        r = rng.uniform(50, 400)  # well past the cell size -> multi-ring scan
        got = {id(o) for o in grid.neighbours_at(x, y, r)}
        brute = {
            id(o) for o in fish if (o.pos.x - x) ** 2 + (o.pos.y - y) ** 2 <= r * r
        }
        assert got == brute


# --- Fish genetic algorithm -------------------------------------------------


def test_random_and_mutated_genomes_stay_in_range():
    rng = random.Random(7)
    for _ in range(500):
        g = Genome.random(rng)
        child = g.mutate(rng, rate=1.0, sigma=0.5)
        for genome in (g, child):
            for name, (lo, hi) in GENE_RANGES.items():
                assert lo <= getattr(genome, name) <= hi, (name, getattr(genome, name))
        assert child.generation == g.generation + 1


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


def test_shark_genomes_stay_in_range():
    rng = random.Random(7)
    for _ in range(500):
        g = SharkGenome.random(rng)
        child = g.mutate(rng, rate=1.0, sigma=0.5)
        for genome in (g, child):
            for name, (lo, hi) in SHARK_GENE_RANGES.items():
                assert lo <= getattr(genome, name) <= hi, (name, getattr(genome, name))
        assert child.generation == g.generation + 1


def test_shark_eats_fish_and_gains_energy():
    cfg = Config(num_fish=0, num_sharks=0)
    sim = Simulation(cfg, seed=4)
    shark = Shark(Vector2(100, 100), _shark_genome(), energy=100.0)
    sim.sharks = [shark]
    sim.fish = [_boid(Vector2(100, 100), Vector2(0, 0))]  # sitting right on the shark
    before = shark.energy
    sim._predate()
    assert sim.fish == []  # eaten
    assert shark.energy == before + cfg.shark_energy_per_fish
    assert sim.deaths >= 1


def test_unfed_shark_starves_and_can_go_extinct():
    cfg = Config(num_fish=0, num_sharks=0)
    sim = Simulation(cfg, seed=1)
    sim.sharks = [Shark(Vector2(300, 200), _shark_genome(), energy=5.0)]
    steps = 0
    while sim.sharks and steps < 2000:
        sim.step()
        steps += 1
    assert sim.sharks == []  # no food -> starved out (no floor)
    assert sim.shark_deaths >= 1


def test_well_fed_shark_reproduces_and_splits_energy():
    cfg = Config(
        num_fish=0, num_sharks=0, shark_repro_energy=200.0, shark_birth_cost=20.0,
    )
    sim = Simulation(cfg, seed=2)
    parent = Shark(Vector2(300, 200), _shark_genome(), energy=300.0)
    sim.sharks = [parent]
    sim._shark_reproduce()
    assert len(sim.sharks) == 2 and sim.shark_births == 1
    child = next(s for s in sim.sharks if s is not parent)
    assert abs(parent.energy - 150.0) < 1e-6  # halved
    assert abs(child.energy - 130.0) < 1e-6  # half minus birth cost
    assert child.genome.generation == parent.genome.generation + 1


def test_shark_population_respects_cap():
    cfg = Config(num_fish=0, num_sharks=0, max_sharks=5)
    sim = Simulation(cfg, seed=2)
    sim.sharks = [
        Shark(Vector2(100 + i, 100), _shark_genome(), energy=1000.0) for i in range(4)
    ]
    for _ in range(10):
        sim._shark_reproduce()
    assert len(sim.sharks) <= cfg.max_sharks


def test_ecosystem_runs_within_invariants():
    cfg = Config(num_fish=60, num_sharks=5, width=600, height=400)
    sim = Simulation(cfg, seed=5)
    for _ in range(300):
        sim.step()  # fully autonomous
        assert 0 <= len(sim.fish) <= cfg.max_pop
        assert 0 <= len(sim.sharks) <= cfg.max_sharks


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
    shard = _square_shard(200, 200, half=30.0)
    # A point dead-centre inside is ejected and pushed away from the wall.
    push, eject = avoid(Vector2(200, 200), [shard], margin=26.0)
    assert eject is not None and eject.length() > 0
    # A point just left of the left face is shoved further left (outward).
    push, eject = avoid(Vector2(165, 200), [shard], margin=26.0)
    assert push.x < 0
    assert eject is None  # outside, no ejection
    # A point well clear of every shard feels nothing.
    push, eject = avoid(Vector2(500, 500), [shard], margin=26.0)
    assert push.length() == 0 and eject is None


def test_wall_index_finds_shard():
    cfg = Config()
    field = WallField([_square_shard(200, 200, half=30.0)], cfg.cell_size, cfg.wall_margin)
    assert len(field.nearby(200, 200)) == 1  # point over the shard
    assert len(field.nearby(1000, 700)) == 0  # far away cell


def test_fish_does_not_penetrate_wall():
    cfg = Config(num_fish=0, num_sharks=0, walls_enabled=False, width=600, height=400)
    sim = Simulation(cfg, seed=1)
    shard = _square_shard(300, 200, half=30.0)
    sim.walls = WallField([shard], cfg.cell_size, cfg.wall_margin)
    cfg.walls_enabled = True
    # A fish to the left, driving straight at the shard at top speed.
    f = _boid(Vector2(220, 200), Vector2(5, 0), max_speed=5.0)
    sim.fish = [f]
    for _ in range(60):
        sim.step()
        # Signed distance to the nearest face: >0 outside, <0 inside. Never let the
        # fish end a tick more than a hair inside the shard.
        depth = max((f.pos - a).dot(n) for a, _b, n in shard.edges)
        assert depth > -2.0, (f.pos, depth)
        assert f.pos.x == f.pos.x  # not NaN


def test_ecosystem_runs_with_walls():
    cfg = Config(num_fish=60, num_sharks=5, width=600, height=400, walls_enabled=True)
    sim = Simulation(cfg, seed=5)
    assert len(sim.walls.shards) == cfg.num_walls
    for _ in range(300):
        sim.step()
        assert 0 <= len(sim.fish) <= cfg.max_pop
        assert 0 <= len(sim.sharks) <= cfg.max_sharks
    for f in sim.fish:
        assert f.pos.x == f.pos.x and f.pos.y == f.pos.y  # not NaN
        assert f.vel.length() <= f.genome.max_speed + 1e-6


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
