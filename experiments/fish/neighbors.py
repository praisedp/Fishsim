"""Vectorized neighbour search: cell binning + candidate-pair construction.

Naively, every fish checking every other fish is O(n²). We bin fish into cells at
least as wide as the largest perception radius, so a fish's neighbours can only
live in its own cell plus the 8 adjacent ones. The classic spatial hash — but
built and queried with array ops instead of dicts and per-fish loops:

1. :func:`build_grid` sorts fish by cell id and records, per cell, where its run
   of fish starts in that sorted order (a CSR-style layout).
2. :func:`perception_pairs` walks the ≤9 cell offsets (a Python loop of nine,
   each iteration fully vectorized) and emits every candidate ``(i, j)`` pair,
   then filters by each fish's own perception radius.

Cells **evenly divide** the world (``cell_w = width / nx``) rather than using a
fixed size with a remainder cell. In wrap mode this is load-bearing: with a
narrow leftover cell at the seam, two fish within perception of each other across
the wrap could land in cells that aren't adjacent mod nx and be missed.

Distances honour the torus when ``wrap`` is on: :func:`min_image` folds any
delta to the shortest of its wrapped images.
"""

from __future__ import annotations

import numpy as np


def min_image(delta: np.ndarray, world: np.ndarray) -> np.ndarray:
    """Fold position deltas to their shortest toroidal image, per axis.

    ``delta`` is ``(..., 2)``; ``world`` is ``(2,)`` (width, height). Rounding
    ``delta / world`` to the nearest integer counts how many worlds away the
    naive delta is, and subtracting them leaves a delta in ``[-world/2, world/2]``.

    Runs on ~50k-element arrays several times per tick, so it works axis by
    axis with scalar operands: broadcasting the ``(2,)`` divisor directly would
    trap numpy in length-2 inner loops (~2.6x slower). Returns a new array,
    leaving ``delta`` untouched.
    """
    folded = np.empty_like(delta)
    for axis in range(2):
        naive = delta[..., axis]
        out = folded[..., axis]
        np.divide(naive, world[axis], out=out)
        np.rint(out, out=out)
        out *= world[axis]
        np.subtract(naive, out, out=out)
    return folded


class Grid:
    """Cell binning of one frame's positions, in sorted-runs (CSR) form.

    ``order`` lists fish indices sorted by cell id; cell ``c``'s fish are
    ``order[starts[c] : starts[c] + counts[c]]``.
    """

    __slots__ = ("nx", "ny", "cell_w", "cell_h", "cell_x", "cell_y",
                 "order", "starts", "counts")

    def __init__(self, nx, ny, cell_w, cell_h, cell_x, cell_y, order, starts, counts):
        self.nx = nx
        self.ny = ny
        self.cell_w = cell_w
        self.cell_h = cell_h
        self.cell_x = cell_x  # (n,) each fish's cell column
        self.cell_y = cell_y  # (n,) each fish's cell row
        self.order = order
        self.starts = starts
        self.counts = counts


def grid_dims(width: float, height: float, cell_min: float) -> tuple[int, int, float, float]:
    """Cell counts and sizes that evenly divide the world with cells ≥ ``cell_min``."""
    nx = max(1, int(width // cell_min))
    ny = max(1, int(height // cell_min))
    return nx, ny, width / nx, height / ny


def build_grid(
    pos: np.ndarray, width: float, height: float, cell_min: float, wrap: bool
) -> Grid:
    """Bin ``pos`` ``(n, 2)`` into cells of side ≥ ``cell_min``.

    In wrap mode cell coordinates wrap with the world. In bounce mode the soft
    edge-turn lets fish drift slightly off-screen; clamping their cell into range
    merges them into the border cells — adjacency is preserved and any extra
    candidates are removed by the radius filter anyway.
    """
    nx, ny, cell_w, cell_h = grid_dims(width, height, cell_min)
    cell_x = np.floor(pos[:, 0] / cell_w).astype(np.int64)
    cell_y = np.floor(pos[:, 1] / cell_h).astype(np.int64)
    if wrap:
        cell_x %= nx
        cell_y %= ny
    else:
        np.clip(cell_x, 0, nx - 1, out=cell_x)
        np.clip(cell_y, 0, ny - 1, out=cell_y)
    cell_id = cell_y * nx + cell_x
    order = np.argsort(cell_id, kind="stable")
    counts = np.bincount(cell_id, minlength=nx * ny)
    starts = np.concatenate(([0], np.cumsum(counts)))
    return Grid(nx, ny, cell_w, cell_h, cell_x, cell_y, order, starts, counts)


def _axis_offsets(ncells: int, wrap: bool) -> tuple[int, ...]:
    """Neighbour-cell offsets along one axis, deduplicated for tiny wrap grids.

    On a wrapped axis of 2 cells, offsets -1 and +1 are the *same* cell mod 2
    (and on 1 cell, so is 0) — visiting both would emit every pair twice. In
    bounce mode there's no modular aliasing, so the full (-1, 0, 1) always
    applies (out-of-range cells are masked out by the caller).
    """
    if not wrap or ncells >= 3:
        return (-1, 0, 1)
    return (0, 1) if ncells == 2 else (0,)


def _query_cells(pos: np.ndarray, grid: Grid, wrap: bool) -> tuple[np.ndarray, np.ndarray]:
    """Which of ``grid``'s cells each of ``pos`` falls in (for cross-grid queries)."""
    cx = np.floor(pos[:, 0] / grid.cell_w).astype(np.int64)
    cy = np.floor(pos[:, 1] / grid.cell_h).astype(np.int64)
    if wrap:
        cx %= grid.nx
        cy %= grid.ny
    else:
        np.clip(cx, 0, grid.nx - 1, out=cx)
        np.clip(cy, 0, grid.ny - 1, out=cy)
    return cx, cy


def _gather_pairs(
    grid: Grid,
    q_cell_x: np.ndarray,
    q_cell_y: np.ndarray,
    q_pos: np.ndarray,
    q_radii: np.ndarray,
    t_pos: np.ndarray,
    world: np.ndarray,
    wrap: bool,
    exclude_self: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Directed pairs ``(a, b)`` where target ``b`` (in ``grid``) lies within
    query ``a``'s radius. ``a`` indexes ``q_pos``; ``b`` indexes ``t_pos`` (the
    points ``grid`` was built over). Returns ``(a, b, rel, d2)`` with
    ``rel = t_pos[b] - q_pos[a]`` (minimum-image when wrapping).

    The shared core of :func:`perception_pairs` (query == target, self excluded)
    and :func:`cross_pairs` (two different populations, no self concept).
    """
    nq = len(q_pos)
    empty_i = np.empty(0, dtype=np.int64)
    if nq == 0:
        return empty_i, empty_i.copy(), np.empty((0, 2)), np.empty(0)

    # Every (query, neighbour-cell) combination as flat arrays: K ≤ 9 offsets
    # stacked into one (K, nq) block, so the ragged gather below runs once over
    # all of them instead of once per offset.
    xoffs = _axis_offsets(grid.nx, wrap)
    yoffs = _axis_offsets(grid.ny, wrap)
    dxs = np.array([dx for dx in xoffs for _dy in yoffs])
    dys = np.array([dy for _dx in xoffs for dy in yoffs])
    ncx = q_cell_x + dxs[:, None]
    ncy = q_cell_y + dys[:, None]
    src = np.tile(np.arange(nq), len(dxs))
    if wrap:
        ncx %= grid.nx
        ncy %= grid.ny
        ncell = (ncy * grid.nx + ncx).ravel()
    else:
        ok = ((ncx >= 0) & (ncx < grid.nx) & (ncy >= 0) & (ncy < grid.ny)).ravel()
        src = src[ok]
        ncell = (ncy * grid.nx + ncx).ravel()[ok]

    # Ragged gather: query src[k] must be paired with every target in its
    # neighbour cell's run of length reps[k]. Expand each query reps[k] times,
    # then build each copy's position *within its own run* (0, 1, ..., reps[k]-1)
    # by subtracting the run's global start from a flat arange. Adding the cell's
    # start in `order` turns those run positions back into target indices.
    reps = grid.counts[ncell]
    total = int(reps.sum())
    if total == 0:
        return empty_i, empty_i.copy(), np.empty((0, 2)), np.empty(0)
    a = np.repeat(src, reps)
    run_first = np.repeat(np.cumsum(reps) - reps, reps)
    within_run = np.arange(total) - run_first
    b = grid.order[np.repeat(grid.starts[ncell], reps) + within_run]
    rel = t_pos[b] - q_pos[a]
    if wrap:
        rel = min_image(rel, world)
    d2 = np.einsum("pc,pc->p", rel, rel)
    keep = d2 <= q_radii[a] ** 2
    if exclude_self:
        keep &= a != b
    return a[keep], b[keep], rel[keep], d2[keep]


def perception_pairs(
    grid: Grid,
    pos: np.ndarray,
    radii: np.ndarray,
    world: np.ndarray,
    wrap: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Every directed pair ``(i, j)`` with fish ``j`` inside fish ``i``'s radius.

    Returns ``(i, j, rel, d2)`` flat arrays over pairs, where ``rel = pos[j] -
    pos[i]`` (minimum-image when wrapping) and ``d2 = |rel|²``. Self-pairs are
    excluded by index (``i != j``), *not* by distance — two fish at the exact
    same spot still see each other, which alignment/cohesion rely on.
    """
    return _gather_pairs(
        grid, grid.cell_x, grid.cell_y, pos, radii, pos, world, wrap, exclude_self=True
    )


def cross_pairs(
    grid_b: Grid,
    pos_a: np.ndarray,
    radii_a: np.ndarray,
    pos_b: np.ndarray,
    world: np.ndarray,
    wrap: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Directed cross-population pairs ``(a, b)`` with B-point ``b`` inside
    A-point ``a``'s radius. ``grid_b`` is a :class:`Grid` built over ``pos_b``;
    A-points are re-binned into its cells. Returns ``(a, b, rel, d2)`` with
    ``rel = pos_b[b] - pos_a[a]`` — no self-exclusion (the populations differ).

    One pass over fish→food serves foraging (steer toward food), eating (filter
    by mouth radius), and wakes (by symmetry, food pushed by nearby fish)."""
    cx, cy = _query_cells(pos_a, grid_b, wrap)
    return _gather_pairs(
        grid_b, cx, cy, pos_a, radii_a, pos_b, world, wrap, exclude_self=False
    )


def nearest_points(
    query: np.ndarray, points: np.ndarray, world: np.ndarray, wrap: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each ``query`` row, the nearest of ``points`` (which must be non-empty).

    A dense ``(Q, P)`` distance matrix — with ≤1000 fish querying ≤50 sharks
    that's tiny, and anything cleverer would cost more than it saves. Returns
    ``(index, offset, d2)`` where ``offset = query - nearest_point``
    (minimum-image when wrapping), i.e. the direction *away* from it.
    """
    delta = query[:, None, :] - points[None, :, :]
    if wrap:
        delta = min_image(delta, world)
    d2 = np.einsum("qpc,qpc->qp", delta, delta)
    k = d2.argmin(axis=1)
    rows = np.arange(len(query))
    return k, delta[rows, k], d2[rows, k]
