"""Static geometric wall obstacles and soft-bounce avoidance.

A handful of small convex polygon *shards* litter the sea. Fish and sharks steer
away from them with a soft push (no hard stops), the same flavour of nudge the
world edges use in ``sim._apply_edges_fish``.

The shards never move, so :class:`WallField` packs their geometry into numpy
arrays **once**, and :func:`avoid_all` answers for a whole population per tick
with a single ``(points, shards, edges)`` signed-distance tensor. No pygame
display code — headless-testable like the rest of the engine."""

from __future__ import annotations

import math

import numpy as np
from pygame.math import Vector2

from config import Config

# Shards are triangles or quads (see WallField.scatter), so the packed edge
# arrays are padded to four edges with `edge_valid` marking the real ones.
MAX_EDGES = 4


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
    """A static set of shards, packed once into arrays for :func:`avoid_all`."""

    def __init__(self, shards: list[Shard]):
        self.shards = shards
        # Packed geometry for the vectorized avoid_all(): per shard, each edge's
        # anchor vertex and outward normal, padded to MAX_EDGES, plus the bbox
        # as (min_x, min_y, max_x, max_y).
        count = len(shards)
        self.edge_a = np.zeros((count, MAX_EDGES, 2))
        self.edge_n = np.zeros((count, MAX_EDGES, 2))
        self.edge_valid = np.zeros((count, MAX_EDGES), dtype=bool)
        self.bbox = np.zeros((count, 4))
        for k, shard in enumerate(shards):
            for e, (a, _b, normal) in enumerate(shard.edges):
                self.edge_a[k, e] = (a.x, a.y)
                self.edge_n[k, e] = (normal.x, normal.y)
                self.edge_valid[k, e] = True
            self.bbox[k] = (shard.min_x, shard.min_y, shard.max_x, shard.max_y)

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
        return cls(shards)


def avoid_all(
    pos: np.ndarray, field: WallField, margin: float
) -> tuple[np.ndarray, np.ndarray]:
    """Soft-bounce response for every position at once.

    Returns ``(push, eject)``, both ``(n, 2)``:

    - ``push`` — a steering direction away from any wall face within ``margin``,
      stronger the closer (or deeper) the point is. Zero when clear.
    - ``eject`` — a positional correction that lifts a point back out when it has
      actually crossed *inside* a shard, else zero. Velocity nudges alone can't
      rescue a point already inside (and sharks rewrite their velocity each
      tick), so the ejection is the hard backstop.

    The bbox+margin prefilter is for correctness as much as speed: the signed
    distance below is to each face's *infinite line*, so without a spatial
    filter a point far away but near a face's extension would get a spurious
    push.
    """
    n = len(pos)
    push = np.zeros((n, 2))
    eject = np.zeros((n, 2))
    if n == 0 or not field.shards:
        return push, eject

    bbox = field.bbox
    near = (
        (pos[:, 0:1] >= bbox[:, 0] - margin)
        & (pos[:, 0:1] <= bbox[:, 2] + margin)
        & (pos[:, 1:2] >= bbox[:, 1] - margin)
        & (pos[:, 1:2] <= bbox[:, 3] + margin)
    )  # (n, shards)
    if not near.any():
        return push, eject

    # Signed distance from every point to every face line: (point - anchor)·normal,
    # shaped (n, shards, edges). Positive outside that face; a convex shard
    # contains a point iff every face's distance is <= 0. Padded edges read as
    # -inf so they never win the "least-negative face" argmax nor enter the band.
    sd = np.einsum("nsec,sec->nse", pos[:, None, None, :] - field.edge_a, field.edge_n)
    sd = np.where(field.edge_valid, sd, -np.inf)
    max_sd = sd.max(axis=2)  # (n, shards)

    inside = near & (max_sd <= 0.0)
    n_idx, s_idx = np.nonzero(inside)  # the handful of point-in-shard incidents
    if n_idx.size:
        # Eject through the nearest (least-negative) face, push along its normal.
        face = sd[n_idx, s_idx].argmax(axis=1)
        normals = field.edge_n[s_idx, face]
        depth = -max_sd[n_idx, s_idx]
        np.add.at(push, n_idx, normals)
        np.add.at(eject, n_idx, normals * depth[:, None])

    # Outside a shard: every face whose band the point sits in shoves outward,
    # scaled from 1 at the face down to 0 at the margin's edge.
    band = (
        near[:, :, None]
        & ~inside[:, :, None]
        & field.edge_valid
        & (sd >= 0.0)
        & (sd < margin)
    )
    if band.any():
        weight = np.where(band, (margin - sd) / margin, 0.0)
        push += np.einsum("nse,sec->nc", weight, field.edge_n)
    return push, eject
