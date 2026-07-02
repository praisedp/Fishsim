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
    num_sharks: int = 5  # starting shark population
    wrap_edges: bool = True  # True = toroidal wrap-around, False = soft bounce

    # --- Shared physics (NOT per-animal) ---
    separation_radius: float = 20.0  # fish personal-space distance
    min_speed: float = 1.0  # keeps fish drifting so they never freeze
    max_force: float = 0.18  # fish steering clamp (Reynolds)

    # --- Predation ---
    flee_radius: float = 130.0  # range at which a fish starts fleeing a shark
    kill_radius: float = 14.0  # fish within this of a shark get eaten

    # --- Fish bioenergetics (size-driven; derived physics live in physiology.py) ---
    # The size scale is deliberately small: fish bodies run ~1.5–6 units. All the
    # size-derived coefficients below are set so the energy economy is the same as
    # it would be at any other scale — shrinking the sizes doesn't unbalance it.
    juvenile_size: float = 1.0  # size a newborn fish starts at
    size_ref: float = 3.0  # reference body size for speed/mass scaling
    speed_ref: float = 3.5  # max speed at size_ref
    speed_alpha: float = (
        0.5  # speed = speed_ref * (size_ref/size)**alpha (small = fast)
    )
    speed_min_cap: float = 2.0  # floor on derived max speed
    speed_max_cap: float = 5.5  # ceiling; < shark top (6) so small fish stay catchable
    energy_cap_k: float = 32.0  # energy storage cap = energy_cap_k * size**2
    metab_basal_k: float = (
        0.017  # baseline burn = metab_basal_k * mass**metab_basal_exp
    )
    metab_basal_exp: float = (
        0.75  # sub-quadratic (Kleiber): big bodies cost less per mass
    )
    metab_move_k: float = 0.0032  # burn per (mass * speed**2)
    metab_perc_k: float = 0.0006  # burn per unit perception gene
    growth_speed: float = 0.01  # size grown per tick when in surplus
    growth_surplus_frac: float = 0.5  # only grow when energy >= frac * cap
    growth_cost_k: float = 8.0  # energy spent per unit mass gained
    maturity_size_frac: float = 0.9  # can breed only when size >= frac * adult_size
    fish_birth_cost: float = (
        12.0  # energy overhead lost at each spawn (favours fitter parents)
    )
    eat_radius_k: float = 3.0  # mouth radius = eat_radius_k * size

    # --- Fish genetic algorithm ---
    mutation_rate: float = 0.5  # per-gene chance of a nudge when breeding
    mutation_sigma: float = 0.12  # nudge size, as a fraction of each gene's span
    max_pop: int = 1000  # hard safety cap on living fish (real limit is food supply)

    # --- Food (drifting plant particles the fish eat) ---
    max_food: int = 300  # hard cap on living food particles
    food_spawn_rate: float = (
        3.0  # Poisson mean of new food per tick (sets carrying capacity)
    )
    food_size_lo: float = 1.0  # smallest food radius
    food_size_hi: float = 5.0  # largest food radius
    food_energy_k: float = 1.0  # food energy = food_energy_k * size**2
    food_wander_sigma: float = 0.15  # Brownian velocity kick per tick
    food_damp: float = 0.85  # per-tick velocity damping
    food_max_speed: float = 0.6  # < min_speed (1.0) so fish always overtake food
    food_clump_radius: float = 40.0  # range of weak food-food cohesion
    food_clump_k: float = 0.03  # clumping nudge strength
    food_wake_radius: float = 25.0  # range at which a fish's wake shoves food
    food_wake_k: float = 0.4  # wake push strength
    shark_wake_radius: float = 30.0  # range at which a shark's wake shoves food

    # --- Shark genetic algorithm (energy / hunger driven) ---
    max_sharks: int = 18  # hard cap on living sharks (no floor: they can go extinct)
    shark_start_energy: float = 100.0  # energy a fresh/immigrant shark begins with
    shark_meal_energy_k: float = 2.8  # energy gained = k * eaten fish's mass (size**2)
    shark_repro_energy: float = 200.0  # energy needed to spawn a child
    shark_birth_cost: float = (
        70.0  # energy lost from the split at birth (< repro/2, so breeding is viable)
    )
    shark_metab_base: float = 0.03  # baseline energy burned per tick
    shark_metab_speed_k: float = 0.015  # extra burn per unit of max_speed gene
    shark_metab_perc_k: float = 0.00022  # extra burn per unit of perception gene
    shark_mutation_rate: float = 0.5  # per-gene mutation chance when breeding
    shark_mutation_sigma: float = 0.12  # nudge size, fraction of each gene's span
    shark_color_gene: str = "max_speed"  # which shark gene the colour maps to ([K])

    # --- Fish colour ---
    color_gene: str = (
        "adult_size"  # which fish gene the colour maps to (cycled with [C])
    )

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
