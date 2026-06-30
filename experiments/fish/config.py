"""Tunable parameters for the coevolving fish-vs-shark simulation.

The behavioural knobs that used to live here (fish rule weights, perception, speed;
the shark's speed) are now *per-animal genes* — see ``genome.py``. What remains is
the world setup, shared physics, and the genetic-algorithm dials for both species:
how fish and sharks are born, fed, and culled.
"""

from dataclasses import dataclass

from genome import GENE_RANGES


@dataclass
class Config:
    # --- World ---
    width: int = 1400
    height: int = 800
    fps: int = 45
    num_fish: int = 440  # starting fish population
    num_sharks: int = 0  # starting shark population
    wrap_edges: bool = True  # True = toroidal wrap-around, False = soft bounce

    # --- Shared physics (NOT per-animal) ---
    separation_radius: float = 20.0  # fish personal-space distance
    min_speed: float = 1.0  # keeps fish drifting so they never freeze
    max_force: float = 0.18  # fish steering clamp (Reynolds)

    # --- Predation ---
    flee_radius: float = 130.0  # base range; each fish scales it by its ``fear`` gene
    kill_radius: float = 14.0  # fish within this of a shark get eaten

    # --- Fish genetic algorithm ---
    maturity_age: int = 150  # ticks before a fish can reproduce
    repro_prob: float = 0.004  # per-tick chance a mature fish spawns a child
    mutation_rate: float = 0.5  # per-gene chance of a nudge when breeding
    mutation_sigma: float = 0.12  # nudge size, as a fraction of each gene's span
    max_pop: int = 600  # hard cap on living fish (no floor: true ecosystem)

    # --- Shark genetic algorithm (energy / hunger driven) ---
    max_sharks: int = 30  # hard cap on living sharks (no floor: they can go extinct)
    shark_start_energy: float = 120.0  # energy a fresh/immigrant shark begins with
    shark_energy_per_fish: float = 9.0  # energy gained per fish eaten
    shark_repro_energy: float = 250.0  # energy needed to spawn a child
    shark_birth_cost: float = 500.0  # energy lost from the split at birth
    shark_metab_base: float = 0.05  # baseline energy burned per tick
    shark_metab_speed_k: float = 0.020  # extra burn per unit of max_speed gene
    shark_metab_perc_k: float = 0.0006  # extra burn per unit of perception gene
    shark_mutation_rate: float = 0.5  # per-gene mutation chance when breeding
    shark_mutation_sigma: float = 0.12  # nudge size, fraction of each gene's span
    shark_color_gene: str = "max_speed"  # which shark gene the colour maps to ([K])

    # --- Fish colour ---
    color_gene: str = "fear"  # which fish gene the colour maps to (cycled with [C])

    # --- Edge handling (soft-bounce mode only) ---
    edge_margin: float = 80.0
    edge_turn: float = 0.3

    # --- Wall obstacles (static geometric shards; both species soft-bounce) ---
    walls_enabled: bool = (
        False  # True = shards scatter across the sea, False = no walls
    )
    num_walls: int = 28  # how many shards scatter across the sea
    wall_min_radius: float = 22.0  # smallest shard size
    wall_max_radius: float = 50.0  # largest shard size
    wall_margin: float = 26.0  # soft-bounce band; > max speed (5–6) so no tunneling
    wall_push: float = 0.5  # velocity-nudge strength near a wall face
    wall_border_margin: float = 60.0  # keep shards off the screen edges

    @property
    def cell_size(self) -> float:
        # Sized to the largest *fish* perception so a fish's neighbours always live
        # in the 9 surrounding cells (the tight, common case). Sharks see much
        # further, so their prey query (grid.neighbours_at) scans as many cell-rings
        # as its own perception needs rather than relying on the 3x3 guarantee.
        return GENE_RANGES["perception"][1]
