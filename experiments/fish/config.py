"""Tunable parameters for the Boids fish-school simulation.

Everything that affects how the school *feels* lives here so it can be tweaked
in one place. The defaults are tuned for a few hundred fish at 60 fps.
"""

from dataclasses import dataclass


@dataclass
class Config:
    # --- World ---
    width: int = 1420
    height: int = 1000
    fps: int = 90
    num_fish: int = 440
    wrap_edges: bool = True  # True = toroidal wrap-around, False = soft bounce

    # --- Perception ---
    perception_radius: float = 40.0  # how far a fish sees neighbours defult 50
    separation_radius: float = 20.0  # personal-space distance defult 20

    # --- Motion limits ---
    max_speed: float = 3.0
    min_speed: float = 1.0  # keeps fish drifting so they never freeze defult 2
    max_force: float = 0.18  # steering clamp (Reynolds)

    # --- Rule weights ---
    w_sep: float = 1.6
    w_ali: float = 1.0
    w_coh: float = 1.0
    w_flee: float = 3.0

    # --- Predator (shark) ---
    flee_radius: float = 130.0
    shark_max_speed: float = 10.0  # slightly slower than fish -> chase & scatter

    # --- Edge handling (soft-bounce mode only) ---
    edge_margin: float = 80.0
    edge_turn: float = 0.3

    @property
    def cell_size(self) -> float:
        # Grid cells must be at least the perception radius so neighbours can
        # only ever live in the 9 surrounding cells.
        return self.perception_radius
