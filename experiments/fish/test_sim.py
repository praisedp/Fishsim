"""Headless tests for the Boids simulation logic — no pygame display needed.

Run:  uv run python test_sim.py
"""

from __future__ import annotations

from pygame.math import Vector2

from boid import Boid
from config import Config
from grid import SpatialGrid
from sim import Simulation


def test_speed_never_exceeds_max():
    cfg = Config(num_fish=120, width=600, height=400)
    sim = Simulation(cfg, seed=1)
    for _ in range(200):
        sim.step(Vector2(300, 200))
    for f in sim.fish:
        assert f.vel.length() <= cfg.max_speed + 1e-6, f.vel.length()
        assert f.pos.x == f.pos.x and f.pos.y == f.pos.y  # not NaN


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
    a = Boid(Vector2(100, 100), Vector2(0, 0))
    b = Boid(Vector2(105, 100), Vector2(0, 0))  # well inside separation_radius
    start_gap = a.pos.distance_to(b.pos)
    for _ in range(5):
        a.compute([b], Vector2(9999, 9999), cfg)
        b.compute([a], Vector2(9999, 9999), cfg)
        a.apply(cfg)
        b.apply(cfg)
    assert a.pos.distance_to(b.pos) > start_gap


def test_flee_points_away_from_shark():
    cfg = Config()
    f = Boid(Vector2(100, 100), Vector2(0, 0))
    shark = Vector2(110, 100)  # just to the right, inside flee_radius
    steer = f.flee(shark, cfg)
    assert steer.x < 0  # pushed left, away from the shark
    # And no flee force when the shark is far away.
    assert f.flee(Vector2(9999, 9999), cfg).length() == 0


def test_grid_matches_brute_force():
    cfg = Config()
    import random

    rng = random.Random(42)
    fish = [
        Boid(Vector2(rng.uniform(0, 800), rng.uniform(0, 800)), Vector2(0, 0))
        for _ in range(300)
    ]
    grid = SpatialGrid(cfg.cell_size)
    grid.build(fish)
    r = cfg.perception_radius
    for f in fish:
        grid_set = {id(o) for o in grid.neighbours(f, r)}
        brute_set = {
            id(o) for o in fish if o is not f and f.pos.distance_to(o.pos) <= r
        }
        assert grid_set == brute_set


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
