"""Headless benchmark for the simulation engine — no display needed.

Run:  uv run python bench.py

Times ``Simulation.step()`` under two scenarios:

- **pinned**: populations pinned at the caps (no births, deaths, or kills), so the
  number is a like-for-like measure of the per-tick physics across engine rewrites.
- **ecosystem**: the default live world, births and deaths included — the realistic
  number that determines whether the game holds its target fps.
"""

from __future__ import annotations

import time

from config import Config
from sim import Simulation

WARMUP_STEPS = 20
TIMED_STEPS = 200
SEED = 12345


def bench(name: str, cfg: Config) -> None:
    sim = Simulation(cfg, seed=SEED)
    for _ in range(WARMUP_STEPS):
        sim.step()
    start = time.perf_counter()
    for _ in range(TIMED_STEPS):
        sim.step()
    elapsed = time.perf_counter() - start
    ms = elapsed / TIMED_STEPS * 1000
    print(
        f"{name:>10}: {ms:7.3f} ms/step   "
        f"({len(sim.fish)} fish, {len(sim.sharks)} sharks after run)"
    )


def main() -> None:
    # Max-load world frozen in place: start at both population caps and disable
    # every birth/death channel so old and new engines time identical work.
    pinned = Config(
        num_fish=1000,
        num_sharks=50,
        max_sharks=50,
        kill_radius=0.0,
        shark_cull_count=0,
        shark_metab_base=0.0,
        shark_metab_speed_k=0.0,
        shark_metab_perc_k=0.0,
        shark_repro_energy=float("inf"),
    )
    bench("pinned", pinned)
    bench("ecosystem", Config())


if __name__ == "__main__":
    main()
