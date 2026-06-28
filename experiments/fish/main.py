"""Boids fish-school simulation with a mouse-controlled shark.

Run it:  uv run python main.py   (from experiments/fish/)

Controls:
  - Move the mouse: the shark follows; nearby fish scatter.
  - W: toggle wrap-around vs soft-bounce edges.
  - Space: pause/resume.   Esc / window close: quit.
"""

from __future__ import annotations

import pygame
from pygame.math import Vector2

from config import Config
from sim import Simulation

# Colours
OCEAN = (12, 28, 54)
FISH = (120, 200, 255)
SHARK = (220, 70, 70)
SHARK_RANGE = (220, 70, 70, 40)


def draw_fish(surface: pygame.Surface, pos: Vector2, vel: Vector2) -> None:
    """A small triangle pointing along the velocity."""
    heading = vel if vel.length_squared() > 0 else Vector2(1, 0)
    heading = heading.normalize()
    side = Vector2(-heading.y, heading.x)
    tip = pos + heading * 7
    left = pos - heading * 4 + side * 3
    right = pos - heading * 4 - side * 3
    pygame.draw.polygon(surface, FISH, (tip, left, right))


def main() -> None:
    cfg = Config()
    pygame.init()
    screen = pygame.display.set_mode((cfg.width, cfg.height))
    pygame.display.set_caption("Boids — fish school & shark")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    sim = Simulation(cfg)
    paused = False
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

        shark_target = Vector2(pygame.mouse.get_pos())

        if not paused:
            sim.step(shark_target)

        # --- Render ---
        screen.fill(OCEAN)

        # Shark danger radius (drawn on a per-pixel-alpha overlay).
        overlay = pygame.Surface((cfg.width, cfg.height), pygame.SRCALPHA)
        pygame.draw.circle(overlay, SHARK_RANGE, sim.shark.pos, int(cfg.flee_radius))
        screen.blit(overlay, (0, 0))

        for f in sim.fish:
            draw_fish(screen, f.pos, f.vel)

        pygame.draw.circle(screen, SHARK, sim.shark.pos, 12)

        edge_mode = "wrap" if cfg.wrap_edges else "bounce"
        hud = font.render(
            f"fish:{len(sim.fish)}  fps:{clock.get_fps():4.0f}  "
            f"edges:{edge_mode}  [W]edges [Space]pause",
            True,
            (180, 200, 220),
        )
        screen.blit(hud, (10, 10))

        pygame.display.flip()
        clock.tick(cfg.fps)

    pygame.quit()


if __name__ == "__main__":
    main()
