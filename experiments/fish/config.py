"""Tunable parameters for the coevolving fish-vs-shark simulation.

The behavioural knobs that used to live here (fish rule weights, perception, speed;
the shark's speed) are now *per-animal genes* — see ``genome.py``. What remains is
the world setup, shared physics, and the genetic-algorithm dials for both species:
how fish and sharks are born, fed, and culled.
"""

from dataclasses import dataclass


@dataclass
class Config:
    # --- World ---
    width: int = 1400
    height: int = 800
    fps: int = 45
    num_fish: int = 440  # starting fish population
    num_sharks: int = 10  # starting shark population
    wrap_edges: bool = True  # True = toroidal wrap-around, False = soft bounce

    # --- Shared physics (NOT per-animal) ---
    separation_radius: float = 20.0  # fish personal-space distance
    min_speed: float = 1.0  # keeps fish drifting so they never freeze
    max_force: float = 0.18  # fish steering clamp (Reynolds)

    # --- Predation ---
    flee_radius: float = 130.0  # base range; each fish scales it by its ``fear`` gene
    kill_radius: float = 14.0  # fish within this of a shark get eaten

    # --- Fish genetic algorithm ---
    maturity_age: int = 100  # ticks before a fish can reproduce
    repro_prob: float = 0.004  # per-tick chance a mature fish spawns a child
    mutation_rate: float = 0.5  # per-gene chance of a nudge when breeding
    mutation_sigma: float = 0.12  # nudge size, as a fraction of each gene's span
    max_pop: int = 1000  # hard cap on living fish (no floor: true ecosystem)

    # --- Shark genetic algorithm (energy / hunger driven) ---
    max_sharks: int = 15  # hard cap on living sharks (no floor: they can go extinct)
    shark_start_energy: float = 100.0  # energy a fresh/immigrant shark begins with
    shark_energy_per_fish: float = 40.0  # energy gained per fish eaten
    shark_repro_energy: float = 250.0  # energy needed to spawn a child
    shark_birth_cost: float = 100.0  # energy lost from the split at birth
    shark_metab_base: float = 0.04  # baseline energy burned per tick
    shark_metab_speed_k: float = 0.010  # extra burn per (current speed)² per tick
    shark_metab_perc_k: float = 0.0003  # extra burn per unit of perception gene
    shark_mutation_rate: float = 0.5  # per-gene mutation chance when breeding
    shark_mutation_sigma: float = 0.12  # nudge size, fraction of each gene's span
    shark_color_gene: str = "max_speed"  # which shark gene the colour maps to ([K])

    # --- Shark rescue (hall-of-fame reseeding, so learning survives extinction) ---
    # Sharks can starve to extinction before their brains learn to hunt. The sim
    # keeps a hall-of-fame of the best shark (genes, brain) snapshots seen so far;
    # a rescue clones sharks back from it — on demand ([B] key) and automatically
    # whenever the population hits zero — so evolved policies aren't lost.
    rescue_count: int = 8  # sharks spawned by a manual ([B]) rescue
    min_sharks: int = 8  # auto-floor: repopulate to this when sharks go extinct
    hof_size: int = 12  # how many elite (genes, brain) snapshots to retain
    hof_age_weight: float = 0.05  # age's weight in score = energy + w*age
    rescue_mut_scale: float = (
        1.0  # ×shark/brain mutation on clones (diversity, not copies)
    )

    # --- Shark brain (evolved neural net; architecture lives in brain.py) ---
    shark_max_turn: float = 0.6  # max heading change per tick, radians (~34°)
    retina_count_norm: float = 8.0  # fish per retina sector that reads as "full"
    brain_init_scale: float = 0.8  # founder weights drawn uniform in ± this
    brain_weight_cap: float = 2.0  # hard clip on every weight
    brain_mutation_rate: float = 0.10  # per-weight nudge chance (266 weights —
    # much lower than the gene rate, or every birth would scramble the policy)
    brain_mutation_sigma: float = 0.15  # nudge size, absolute weight units

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
