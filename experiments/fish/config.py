"""Tunable parameters for the fish-vs-shark simulation.

Fish are plain rule-based boids: every fish shares the flocking weights below —
no genes, no evolution. Sharks are the evolving species; their *body* knobs are
per-animal genes (see ``genome.py``) and their behaviour an evolved neural net
(``brain.py``). What remains here is the world setup, shared physics, the fish
rule weights, and the shark genetic-algorithm dials.
"""

from dataclasses import dataclass


@dataclass
class Config:
    # --- World ---
    width: int = 1400
    height: int = 800
    tps: int = 60  # simulation ticks per second — world speed ([T] cycles it)
    fps: int = 120  # render frames per second — smoothness only ([F] cycles it)
    num_fish: int = 440  # starting fish population
    num_sharks: int = 20  # starting shark population
    wrap_edges: bool = True  # True = toroidal wrap-around, False = soft bounce

    # --- Shared physics (NOT per-animal) ---
    separation_radius: float = 18.0  # fish personal-space distance
    min_speed: float = 0.2  # keeps fish drifting so they never freeze
    max_force: float = 0.14  # fish steering clamp (Reynolds)

    # --- Predation ---
    flee_radius: float = 130.0  # range at which a fish starts fleeing a shark
    kill_radius: float = 14.0  # fish within this of a shark get eaten

    # --- Fish boids (rule-based, no evolution; every fish is identical) ---
    fish_perception: float = 40.0  # neighbour visibility radius
    fish_max_speed: float = 5.2  # top speed
    fish_w_sep: float = 1.7  # separation weight (avoid crowding)
    fish_w_ali: float = 1.0  # alignment weight (match direction)
    fish_w_coh: float = 0.9  # cohesion weight (move toward group)
    fish_w_flee: float = 1.0  # flee-from-shark weight
    fish_respawn_rate: float = 0.5  # new fish per tick (fractional), up to num_fish
    fish_clone_radius: float = 20.0  # a "born" fish appears within this of its parent
    fish_clone_min_pop: int = 25  # below this many fish, respawns scatter randomly

    # --- Shark genetic algorithm (energy / hunger driven) ---
    max_sharks: int = 50  # hard cap on living sharks (no floor: they can go extinct)
    shark_cull_count: int = (
        15  # random sharks removed whenever the population hits the cap (0 = off)
    )
    shark_start_energy: float = 100.0  # energy a fresh/immigrant shark begins with
    shark_energy_per_fish: float = 20.0  # energy gained per fish eaten
    shark_repro_energy: float = 250.0  # energy needed to spawn a child
    shark_birth_cost: float = 150.0  # energy lost from the split at birth
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
    rescue_count: int = 20  # sharks spawned by a manual ([B]) rescue
    min_sharks: int = 20  # auto-floor: repopulate to this when sharks go extinct
    hof_size: int = 22  # how many elite (genes, brain) snapshots to retain
    hof_age_weight: float = 0.5  # age's weight in score = energy + w*age
    rescue_mut_scale: float = (
        1.0  # ×shark/brain mutation on clones (diversity, not copies)
    )

    # --- Shark brain (evolved neural net; architecture derived in brain.py) ---
    shark_memory_frames: int = (
        2  # past frames stacked into the observation (0 = no memory)
    )
    shark_hidden: tuple[int, ...] = (24, 12)  # hidden layer sizes of the shark MLP
    shark_max_turn: float = 0.6  # max heading change per tick, radians (~34°)
    retina_count_norm: float = 8.0  # fish per retina sector that reads as "full"
    brain_init_scale: float = 0.8  # founder weights drawn uniform in ± this
    brain_weight_cap: float = 2.0  # hard clip on every weight
    brain_mutation_rate: float = 0.03  # per-weight nudge chance (~2200 weights with
    # the defaults above — far lower than the gene rate, or every birth would
    # scramble the policy; scale it down further if you grow the brain)
    brain_mutation_sigma: float = 0.15  # nudge size, absolute weight units

    # --- Edge handling (soft-bounce mode only) ---
    edge_margin: float = 40.0  # start turning inward this far from a border
    edge_turn: float = 0.25  # inward velocity nudge per tick inside the margin

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
