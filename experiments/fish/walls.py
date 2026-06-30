"""Static geometric wall obstacles and soft-bounce avoidance.

A handful of small convex polygon *shards* litter the sea. Fish and sharks steer
away from them with a soft push (no hard stops), the same flavour of nudge the world
edges use in ``sim._handle_edges``.

Pure vector logic — no pygame display, no global state — so it stays headless-testable
like ``boid.py``. The shards never move, so the spatial index (:class:`WallField`) is
built **once** and queried per entity, rather than rebuilt every tick like the fish
:class:`~grid.SpatialGrid`.
"""

from __future__ import annotations

import math

from pygame.math import Vector2

from config import Config


def closest_point_on_segment(p: Vector2, a: Vector2, b: Vector2) -> Vector2:
    """The point on segment ``a``–``b`` nearest to ``p`` (projection, clamped to ends)."""
    ab = b - a
    len_sq = ab.length_squared()
    if len_sq == 0:
        return Vector2(a)
    t = (p - a).dot(ab) / len_sq
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    return a + ab * t


class Shard:
    """A small convex polygon obstacle.

    Stored as its vertices plus, per edge, an *outward* normal (pointing away from the
    polygon centre). A point's signed distance to an edge — ``(p - a)·normal`` — is
    positive outside that face and negative inside, which is all the avoidance maths
    below needs.
    """

    __slots__ = ("verts", "edges", "centroid", "min_x", "min_y", "max_x", "max_y")

    def __init__(self, verts: list[Vector2]):
        self.verts = verts
        cx = sum(v.x for v in verts) / len(verts)
        cy = sum(v.y for v in verts) / len(verts)
        self.centroid = Vector2(cx, cy)

        # Per edge: (a, b, outward_normal). Flip the perpendicular so it faces away
        # from the centroid, giving a consistent "outside is positive" sign.
        self.edges: list[tuple[Vector2, Vector2, Vector2]] = []
        for i, a in enumerate(verts):
            b = verts[(i + 1) % len(verts)]
            edge = b - a
            normal = Vector2(-edge.y, edge.x)
            if normal.length_squared() == 0:
                continue
            normal = normal.normalize()
            mid = (a + b) * 0.5
            if (mid - self.centroid).dot(normal) < 0:
                normal = -normal
            self.edges.append((a, b, normal))

        xs = [v.x for v in verts]
        ys = [v.y for v in verts]
        self.min_x, self.max_x = min(xs), max(xs)
        self.min_y, self.max_y = min(ys), max(ys)


class WallField:
    """A static set of shards with a build-once cell index for fast lookups."""

    def __init__(self, shards: list[Shard], cell_size: float, margin: float):
        self.shards = shards
        self.cell_size = cell_size
        # Register each shard in every cell its (bbox + margin) overlaps, so a query at
        # any point within ``margin`` of the shard finds it in its own cell.
        self.cells: dict[tuple[int, int], list[Shard]] = {}
        for shard in shards:
            c0 = int((shard.min_x - margin) // cell_size)
            c1 = int((shard.max_x + margin) // cell_size)
            r0 = int((shard.min_y - margin) // cell_size)
            r1 = int((shard.max_y + margin) // cell_size)
            for col in range(c0, c1 + 1):
                for row in range(r0, r1 + 1):
                    self.cells.setdefault((col, row), []).append(shard)

    def nearby(self, x: float, y: float) -> list[Shard]:
        """Shards registered in the cell containing ``(x, y)`` (possibly empty)."""
        key = (int(x // self.cell_size), int(y // self.cell_size))
        return self.cells.get(key, ())

    @classmethod
    def scatter(cls, cfg: Config, rng) -> "WallField":
        """Generate ``cfg.num_walls`` random shards kept off the screen borders."""
        shards: list[Shard] = []
        for _ in range(cfg.num_walls):
            cx = rng.uniform(cfg.wall_border_margin, cfg.width - cfg.wall_border_margin)
            cy = rng.uniform(cfg.wall_border_margin, cfg.height - cfg.wall_border_margin)
            radius = rng.uniform(cfg.wall_min_radius, cfg.wall_max_radius)
            sides = rng.choice((3, 3, 4))  # triangles mostly, some quads
            base = rng.uniform(0, 2 * math.pi)
            verts: list[Vector2] = []
            for i in range(sides):
                # Even angular spread + jitter keeps every shard convex (no self-cross).
                angle = base + (i + rng.uniform(-0.18, 0.18)) * (2 * math.pi / sides)
                r = radius * rng.uniform(0.6, 1.0)
                verts.append(Vector2(cx + math.cos(angle) * r, cy + math.sin(angle) * r))
            shards.append(Shard(verts))
        return cls(shards, cfg.cell_size, cfg.wall_margin)


def avoid(
    pos: Vector2, shards, margin: float
) -> tuple[Vector2, Vector2 | None]:
    """Soft-bounce response for ``pos`` against ``shards``.

    Returns ``(push, eject)``:

    - ``push`` — a steering direction away from any wall face within ``margin``,
      stronger the closer (or deeper) the point is. Zero when clear.
    - ``eject`` — a positional correction that lifts ``pos`` back out when it has
      actually crossed *inside* a shard, or ``None``. Velocity nudges alone can't
      rescue a point already inside (and sharks rewrite their velocity each tick), so
      the ejection is the hard backstop.
    """
    push = Vector2(0, 0)
    eject: Vector2 | None = None
    for shard in shards:
        # Signed distance to each edge's outward face; convex => inside iff all <= 0.
        dists = [(pos - a).dot(n) for a, _b, n in shard.edges]
        if not dists:
            continue
        max_sd = max(dists)
        if max_sd <= 0.0:
            # Inside the shard: push out through the nearest face (least-negative edge).
            idx = dists.index(max_sd)
            normal = shard.edges[idx][2]
            depth = -max_sd
            push += normal
            correction = normal * depth
            eject = correction if eject is None else eject + correction
        else:
            # Outside: any face we're within the band of contributes an outward shove.
            for (a, _b, normal), sd in zip(shard.edges, dists):
                if 0.0 <= sd < margin:
                    push += normal * ((margin - sd) / margin)
    return push, eject
