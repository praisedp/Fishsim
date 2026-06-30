"""Coevolving Boids: a sea of fish and a population of sharks, both evolving.

Fish carry genes for flocking/fleeing; sharks carry genes for hunting (speed, sight,
predictive lead) and live or die by an energy budget. Watch the predator–prey arms
race play out in the HUD over a few minutes.

Run it:  uv run python main.py   (from experiments/fish/)

Controls:
  - M: toggle MANUAL shark — the mouse drives the nearest shark; others stay auto.
  - C: cycle which gene the FISH colour maps to.
  - K: cycle which gene the SHARK colour maps to.
  - O: toggle the geometric wall obstacles (re-scatters a fresh layout when on).
  - R: reset / reseed the world (handy if a species has gone extinct).
  - W: toggle wrap-around vs soft-bounce edges.
  - Space: pause/resume.   Esc / window close: quit.
"""

from __future__ import annotations

import pygame
from pygame.math import Vector2

from config import Config
from genome import GENE_NAMES, SHARK_GENE_NAMES
from sim import Simulation

# Colours
OCEAN = (12, 28, 54)
KILL_RING = (220, 70, 70, 60)
HUD = (180, 200, 220)
WALL = (110, 118, 126)  # muted stony grey
WALL_EDGE = (70, 76, 82)  # darker facet outline


def fish_color(fraction: float) -> pygame.Color:
    """Map a 0–1 fish-gene fraction to a hue: low = calm blue, high = bold red."""
    color = pygame.Color(0)
    hue = 210.0 * (1.0 - _clamp01(fraction))  # 210=blue -> 0=red
    color.hsva = (hue, 85, 100, 100)
    return color


def shark_color(fraction: float, energy_frac: float) -> pygame.Color:
    """Map a shark-gene fraction to white→blue; stays bright, slight energy dim."""
    color = pygame.Color(0)
    sat = 100.0 * _clamp01(fraction)  # 0=white(low) -> 100=blue(high)
    value = 80.0 + 20.0 * _clamp01(energy_frac)  # always bright, a touch brighter when fed
    color.hsva = (210, sat, value, 100)
    return color


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def draw_walls(surface: pygame.Surface, walls) -> None:
    """Fill each shard as a flat grey polygon with a slightly darker facet outline."""
    for shard in walls.shards:
        verts = [(v.x, v.y) for v in shard.verts]
        # Deterministic per-shard shade so facets read distinctly without flicker.
        jitter = int(shard.centroid.x + shard.centroid.y) % 24 - 12
        shade = tuple(max(0, min(255, c + jitter)) for c in WALL)
        pygame.draw.polygon(surface, shade, verts)
        pygame.draw.polygon(surface, WALL_EDGE, verts, 1)


def draw_fish(surface: pygame.Surface, boid, color: pygame.Color) -> None:
    """A small triangle pointing along the velocity, tinted by a gene."""
    vel = boid.vel
    heading = vel if vel.length_squared() > 0 else Vector2(1, 0)
    heading = heading.normalize()
    side = Vector2(-heading.y, heading.x)
    tip = boid.pos + heading * 7
    left = boid.pos - heading * 4 + side * 3
    right = boid.pos - heading * 4 - side * 3
    pygame.draw.polygon(surface, color, (tip, left, right))


def draw_shark(surface, overlay, shark, cfg, color_gene) -> None:
    """A kite — two triangles meeting at the crossbar — pointing along velocity, gene-tinted."""
    energy_frac = shark.energy / cfg.shark_repro_energy
    color = shark_color(shark.genome.gene_fraction(color_gene), energy_frac)
    vel = shark.vel
    heading = vel if vel.length_squared() > 0 else Vector2(1, 0)
    heading = heading.normalize()
    side = Vector2(-heading.y, heading.x)
    pos = shark.pos
    nose = pos + heading * 14.0  # long front point
    left = pos + heading * 2.0 + side * 6.0  # crossbar, forward of center
    right = pos + heading * 2.0 - side * 6.0
    tail = pos - heading * 8.0  # back point
    shade = color.lerp(pygame.Color(0, 0, 0), 0.25)  # darker back triangle for depth
    pygame.draw.polygon(surface, color, (nose, left, right))  # front triangle
    pygame.draw.polygon(surface, shade, (tail, left, right))  # back triangle
    pygame.draw.circle(overlay, KILL_RING, pos, int(cfg.kill_radius), 1)


def main() -> None:
    cfg = Config()
    pygame.init()
    screen = pygame.display.set_mode((cfg.width, cfg.height))
    pygame.display.set_caption("Coevolving Boids — fish vs. sharks")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    sim = Simulation(cfg)
    paused = False
    manual = False
    fish_gene_idx = GENE_NAMES.index(cfg.color_gene)
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
                elif event.key == pygame.K_c:
                    fish_gene_idx = (fish_gene_idx + 1) % len(GENE_NAMES)
                elif event.key == pygame.K_k:
                    shark_gene_idx = (shark_gene_idx + 1) % len(SHARK_GENE_NAMES)
                elif event.key == pygame.K_o:
                    cfg.walls_enabled = not cfg.walls_enabled
                    sim.walls = sim._build_walls()  # (re)scatter or clear shards
                elif event.key == pygame.K_r:
                    sim = Simulation(cfg)

        fish_gene = GENE_NAMES[fish_gene_idx]
        shark_gene = SHARK_GENE_NAMES[shark_gene_idx]

        if not paused:
            target = Vector2(pygame.mouse.get_pos()) if manual else None
            sim.step(target)

        # --- Render ---
        screen.fill(OCEAN)
        overlay = pygame.Surface((cfg.width, cfg.height), pygame.SRCALPHA)

        if cfg.walls_enabled:
            draw_walls(screen, sim.walls)
        for f in sim.fish:
            draw_fish(screen, f, fish_color(f.genome.gene_fraction(fish_gene)))
        for s in sim.sharks:
            draw_shark(screen, overlay, s, cfg, shark_gene)

        screen.blit(overlay, (0, 0))
        _draw_hud(screen, font, sim, cfg, clock, manual, fish_gene, shark_gene)

        pygame.display.flip()
        clock.tick(cfg.fps)

    pygame.quit()


def _draw_hud(screen, font, sim, cfg, clock, manual, fish_gene, shark_gene) -> None:
    """Live stats: a fish block, a shark block, then a controls/legend line."""
    s = sim.stats
    mode = "mouse" if manual else "auto"
    edges = "wrap" if cfg.wrap_edges else "bounce"

    fish_line = (
        f"FISH   pop:{int(s['population']):3d}  births:{int(s['births'])}  "
        f"deaths:{int(s['deaths'])}  gen:{s['generation']:4.1f}   "
        + "  ".join(f"{n}:{s[n]:.2f}" for n in GENE_NAMES)
    )
    shark_line = (
        f"SHARK  pop:{int(s['shark_population']):3d}  births:{int(s['shark_births'])}  "
        f"deaths:{int(s['shark_deaths'])}  gen:{s['shark_generation']:4.1f}  "
        f"energy:{s['shark_energy']:5.1f}   "
        + "  ".join(f"{n}:{s['shark_' + n]:.1f}" for n in SHARK_GENE_NAMES)
    )
    status_line = (
        f"fps:{clock.get_fps():3.0f}  shark:{mode}  edges:{edges}   "
        f"fish=[{fish_gene}] shark=[{shark_gene}]   "
        f"[M]shark [C]fishcol [K]sharkcol [O]walls [R]reset [W]edges [Space]pause"
    )

    for i, text in enumerate((fish_line, shark_line, status_line)):
        screen.blit(font.render(text, True, HUD), (10, 10 + i * 18))


if __name__ == "__main__":
    main()
