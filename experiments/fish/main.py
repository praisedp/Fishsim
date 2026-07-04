"""Boids vs. evolving sharks: rule-based fish prey and a GA-bred predator.

Fish are classic boids — identical, config-tuned, non-evolving. Sharks carry
body genes (speed range, sight) plus a small evolved neural net that steers
them, and live or die by an energy budget. Watch the sharks learn to hunt in
the HUD over a few minutes.

Run it:  uv run python main.py   (from experiments/fish/)

Controls:
  - M: toggle MANUAL shark — the mouse drives the nearest shark; others stay auto.
  - K: cycle which gene the SHARK colour maps to.
  - O: toggle the geometric wall obstacles (re-scatters a fresh layout when on).
  - R: reset / reseed the world (handy if the sharks have gone extinct).
  - W: toggle wrap-around vs soft-bounce edges.
  - Space: pause/resume.   Esc / window close: quit.
"""

from __future__ import annotations

import numpy as np
import pygame

from config import Config
from genome import SHARK_GENE_NAMES, SHARK_SPEC, gene_fractions
from sim import Simulation

# Colours
OCEAN = (12, 28, 54)
KILL_RING = (220, 70, 70, 60)
HUD = (180, 200, 220)
WALL = (110, 118, 126)  # muted stony grey
WALL_EDGE = (70, 76, 82)  # darker facet outline
HUE_BLUE = 210.0  # base hue of both palettes (blue)

# Body geometry, in pixels along/across the heading.
FISH_NOSE = 7.0
FISH_TAIL = 4.0
FISH_HALF_W = 3.0
SHARK_NOSE = 14.0  # long front point
SHARK_CROSS_FWD = 2.0  # crossbar sits this far forward of centre
SHARK_HALF_W = 6.0
SHARK_TAIL = 8.0  # back point
SHARK_BACK_SHADE = 0.25  # darker back triangle for depth

# HUD layout
HUD_X = 10
HUD_Y = 10
HUD_LINE_H = 18

# Every fish is identical (rule-based boids, no genes), so one calm blue.
FISH_COLOR = pygame.Color(0)
FISH_COLOR.hsva = (HUE_BLUE, 85, 100, 100)


def shark_color(fraction: float, energy_frac: float) -> pygame.Color:
    """Map a shark-gene fraction to white→blue; stays bright, slight energy dim."""
    color = pygame.Color(0)
    sat = 100.0 * fraction  # 0=white(low) -> 100=blue(high)
    value = 80.0 + 20.0 * min(1.0, max(0.0, energy_frac))  # a touch brighter when fed
    color.hsva = (HUE_BLUE, sat, value, 100)
    return color


def _headings(vel: np.ndarray) -> np.ndarray:
    """Unit heading per row; motionless rows point along +x."""
    norm = np.sqrt(np.einsum("nc,nc->n", vel, vel))
    return np.where(norm[:, None] > 0.0, vel / np.maximum(norm, 1e-12)[:, None], (1.0, 0.0))


def draw_walls(surface: pygame.Surface, walls) -> None:
    """Fill each shard as a flat grey polygon with a slightly darker facet outline."""
    for shard in walls.shards:
        verts = [(v.x, v.y) for v in shard.verts]
        # Deterministic per-shard shade so facets read distinctly without flicker.
        jitter = int(shard.centroid.x + shard.centroid.y) % 24 - 12
        shade = tuple(max(0, min(255, c + jitter)) for c in WALL)
        pygame.draw.polygon(surface, shade, verts)
        pygame.draw.polygon(surface, WALL_EDGE, verts, 1)


def draw_fish_all(surface: pygame.Surface, sim: Simulation) -> None:
    """Every fish as a small triangle pointing along its velocity.

    The triangle geometry is computed for the whole school in a few array ops;
    only the unavoidable per-polygon draw call remains a Python loop.
    """
    fish = sim.fish
    if not len(fish):
        return
    heading = _headings(fish.vel)
    side = np.stack([-heading[:, 1], heading[:, 0]], axis=1)
    tips = (fish.pos + heading * FISH_NOSE).tolist()
    lefts = (fish.pos - heading * FISH_TAIL + side * FISH_HALF_W).tolist()
    rights = (fish.pos - heading * FISH_TAIL - side * FISH_HALF_W).tolist()
    for tip, left, right in zip(tips, lefts, rights):
        pygame.draw.polygon(surface, FISH_COLOR, (tip, left, right))


def draw_sharks(
    surface: pygame.Surface, overlay: pygame.Surface, sim: Simulation, gene: str
) -> None:
    """Every shark as a kite — two triangles meeting at the crossbar — pointing
    along its velocity, gene-tinted, with its kill radius ringed on the overlay."""
    sharks = sim.sharks
    if not len(sharks):
        return
    cfg = sim.cfg
    heading = _headings(sharks.vel)
    side = np.stack([-heading[:, 1], heading[:, 0]], axis=1)
    noses = (sharks.pos + heading * SHARK_NOSE).tolist()
    lefts = (sharks.pos + heading * SHARK_CROSS_FWD + side * SHARK_HALF_W).tolist()
    rights = (sharks.pos + heading * SHARK_CROSS_FWD - side * SHARK_HALF_W).tolist()
    tails = (sharks.pos - heading * SHARK_TAIL).tolist()
    fractions = gene_fractions(sharks.genes, SHARK_SPEC, gene)
    energy_fracs = sharks.energy / cfg.shark_repro_energy
    centers = sharks.pos.tolist()
    for k in range(len(sharks)):
        color = shark_color(fractions[k], energy_fracs[k])
        shade = color.lerp(pygame.Color(0, 0, 0), SHARK_BACK_SHADE)
        pygame.draw.polygon(surface, color, (noses[k], lefts[k], rights[k]))
        pygame.draw.polygon(surface, shade, (tails[k], lefts[k], rights[k]))
        pygame.draw.circle(overlay, KILL_RING, centers[k], int(cfg.kill_radius), 1)


def main() -> None:
    cfg = Config()
    pygame.init()
    screen = pygame.display.set_mode((cfg.width, cfg.height))
    pygame.display.set_caption("Coevolving Boids — fish vs. sharks")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)
    # One reusable transparent overlay (kill rings); cleared, not reallocated.
    overlay = pygame.Surface((cfg.width, cfg.height), pygame.SRCALPHA)

    sim = Simulation(cfg)
    paused = False
    manual = False
    shark_gene_idx = SHARK_GENE_NAMES.index(cfg.shark_color_gene)
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_w:
                    cfg.wrap_edges = not cfg.wrap_edges
                elif event.key == pygame.K_m:
                    manual = not manual
                elif event.key == pygame.K_k:
                    shark_gene_idx = (shark_gene_idx + 1) % len(SHARK_GENE_NAMES)
                elif event.key == pygame.K_o:
                    cfg.walls_enabled = not cfg.walls_enabled
                    sim.rebuild_walls()  # (re)scatter or clear shards
                elif event.key == pygame.K_b:
                    sim.rescue_sharks(cfg.rescue_count)  # reseed best sharks
                elif event.key == pygame.K_r:
                    sim = Simulation(cfg)

        shark_gene = SHARK_GENE_NAMES[shark_gene_idx]

        if not paused:
            target = pygame.mouse.get_pos() if manual else None
            sim.step(target)

        # --- Render ---
        screen.fill(OCEAN)
        overlay.fill((0, 0, 0, 0))

        if cfg.walls_enabled:
            draw_walls(screen, sim.walls)
        draw_fish_all(screen, sim)
        draw_sharks(screen, overlay, sim, shark_gene)

        screen.blit(overlay, (0, 0))
        _draw_hud(screen, font, sim, cfg, clock, manual, shark_gene)

        pygame.display.flip()
        clock.tick(cfg.fps)

    pygame.quit()


def _draw_hud(screen, font, sim, cfg, clock, manual, shark_gene) -> None:
    """Live stats: a fish block, a shark block, then a controls/legend line."""
    s = sim.stats
    mode = "mouse" if manual else "auto"
    edges = "wrap" if cfg.wrap_edges else "bounce"

    fish_line = (
        f"FISH   pop:{int(s['population']):3d}  respawned:{int(s['births'])}  "
        f"eaten:{int(s['deaths'])}"
    )
    shark_line = (
        f"SHARK  pop:{int(s['shark_population']):3d}  births:{int(s['shark_births'])}  "
        f"deaths:{int(s['shark_deaths'])}  gen:{s['shark_generation']:4.1f}  "
        f"energy:{s['shark_energy']:5.1f}   "
        + "  ".join(f"{n}:{s['shark_' + n]:.1f}" for n in SHARK_GENE_NAMES)
    )
    status_line = (
        f"fps:{clock.get_fps():3.0f}  shark:{mode}  edges:{edges}   "
        f"shark=[{shark_gene}]   "
        f"[M]shark [K]sharkcol [O]walls [B]rescue [R]reset [W]edges [Space]pause"
    )

    for i, text in enumerate((fish_line, shark_line, status_line)):
        screen.blit(font.render(text, True, HUD), (HUD_X, HUD_Y + i * HUD_LINE_H))


if __name__ == "__main__":
    main()
