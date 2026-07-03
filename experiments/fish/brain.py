"""Evolvable shark brains: one tiny MLP per shark, weights bred by the GA.

Each shark carries a flat weight vector (a row of the population's ``(S,
WEIGHT_COUNT)`` array, stored beside its genes in :mod:`state`). Every tick the
sim builds an egocentric observation per shark — a sector retina of the fish it
can see plus a few body senses — and :func:`forward` maps all of them to a
``(turn, throttle)`` pair in one batched pass. There is no learning rule:
weights change only at birth (:func:`mutate_brains`), and hunting quality is
selected the same way as any other gene — sharks that steer badly starve.

Like :mod:`steering`, this module is pure array math (no config, no rng state,
no sim imports) so every piece can be oracle-tested headlessly.

The architecture is fixed at import time — ``N_SECTORS``/``N_HIDDEN`` size both
the ``SharkState`` weight buffer and the reshape layout in :func:`forward`, so
they are constants here rather than runtime config knobs.
"""

from __future__ import annotations

import numpy as np

# --- Architecture (compile-time constants, see module docstring) ---------------

N_SECTORS = 8  # egocentric retina resolution around the shark's heading
# Retina (closeness + count per sector) then body senses: energy, speed,
# obstacle proximity, and a constant bias input. See sim._shark_observe for the
# exact slot layout and normalisations.
N_INPUTS = 2 * N_SECTORS + 4
N_HIDDEN = 12
N_OUTPUTS = 2  # turn in [-1, 1], throttle in [0, 1]

# Flat weight layout: input->hidden matrix first, then hidden->output. The
# hidden layer's bias rides on a constant 1.0 appended to its activations (the
# input layer's bias is the constant input slot), hence the +1 row in W2.
_W1_LEN = N_INPUTS * N_HIDDEN
WEIGHT_COUNT = _W1_LEN + (N_HIDDEN + 1) * N_OUTPUTS


# --- Breeding weights (the GA operations, mirroring genome.py) ------------------


def random_brains(rng: np.random.Generator, n: int, init_scale: float) -> np.ndarray:
    """``(n, WEIGHT_COUNT)`` founder brains, weights uniform in ``±init_scale``."""
    return rng.uniform(-init_scale, init_scale, (n, WEIGHT_COUNT))


def mutate_brains(
    rng: np.random.Generator,
    parents: np.ndarray,
    rate: float,
    sigma: float,
    cap: float,
) -> np.ndarray:
    """Child copies of ``parents`` ``(k, WEIGHT_COUNT)``: each weight has a
    ``rate`` chance of a Gaussian nudge, clipped to ``±cap``.

    Unlike :func:`genome.mutate_genes`, ``sigma`` is in absolute weight units —
    weights share one uniform scale, so a span-relative sigma would just couple
    the nudge size to the clip cap for no reason.
    """
    mutated = rng.random(parents.shape) < rate
    noise = rng.normal(0.0, sigma, parents.shape)
    return np.clip(parents + mutated * noise, -cap, cap)


# --- Perception and thinking, batched over the whole population ----------------


def sector_retina(
    delta: np.ndarray,
    d2: np.ndarray,
    headings: np.ndarray,
    perception: np.ndarray,
    count_norm: float,
) -> np.ndarray:
    """The retina block of every shark's observation, ``(S, 2 * N_SECTORS)``.

    The full circle around each shark's heading is cut into ``N_SECTORS`` equal
    wedges. For every fish within that shark's ``perception`` gene, its wedge
    records how close the *nearest* such fish is (``1 - dist/perception``, so 0
    = barely visible, ~1 = touching) and how *many* sit there (saturating at
    ``count_norm``). Sharks that see nothing return all zeros.

    ``delta`` ``(S, F, 2)`` points shark→fish (already minimum-imaged when the
    world wraps) and ``d2`` ``(S, F)`` is its squared length, both exactly as the
    sim's predation pass builds them.
    """
    s = len(headings)
    retina = np.zeros((s, 2 * N_SECTORS))
    visible = d2 <= perception[:, None] ** 2
    if not visible.any():
        return retina

    # Bearing of each fish relative to the shark's facing, folded to [0, 2π)
    # and bucketed. The scale puts dead-ahead on the seam between the two
    # front sectors; the minimum() guards the exact-2π float edge.
    facing = np.arctan2(headings[:, 1], headings[:, 0])
    bearing = np.arctan2(delta[:, :, 1], delta[:, :, 0]) - facing[:, None]
    sector = np.mod(bearing + np.pi, 2 * np.pi) * (N_SECTORS / (2 * np.pi))
    sector = np.minimum(sector.astype(np.int64), N_SECTORS - 1)

    # One flat bucket per (shark, sector); reduce the visible pairs into them.
    flat = (np.arange(s)[:, None] * N_SECTORS + sector)[visible]
    nearest = np.full(s * N_SECTORS, np.inf)
    np.minimum.at(nearest, flat, np.sqrt(d2[visible]))
    count = np.bincount(flat, minlength=s * N_SECTORS).astype(np.float64)

    closeness = 1.0 - nearest / np.repeat(perception, N_SECTORS)
    closeness = np.where(np.isfinite(nearest), np.clip(closeness, 0.0, 1.0), 0.0)
    retina[:, :N_SECTORS] = closeness.reshape(s, N_SECTORS)
    retina[:, N_SECTORS:] = np.minimum(count / count_norm, 1.0).reshape(s, N_SECTORS)
    return retina


def forward(brains: np.ndarray, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run every shark's own network on its own observation, in one batch.

    ``brains`` ``(S, WEIGHT_COUNT)``, ``obs`` ``(S, N_INPUTS)`` →
    ``(turn, throttle)``, each ``(S,)``: ``turn`` in ``[-1, 1]`` (tanh) is a
    fraction of the max turn rate, ``throttle`` in ``[0, 1]`` (sigmoid) picks a
    speed between the shark's min- and max-speed genes. Pure and deterministic;
    the caller applies both scalings.
    """
    s = len(brains)
    w1 = brains[:, :_W1_LEN].reshape(s, N_INPUTS, N_HIDDEN)
    w2 = brains[:, _W1_LEN:].reshape(s, N_HIDDEN + 1, N_OUTPUTS)
    hidden = np.tanh(np.einsum("si,sih->sh", obs, w1))
    hidden = np.concatenate([hidden, np.ones((s, 1))], axis=1)
    out = np.einsum("sk,sko->so", hidden, w2)
    turn = np.tanh(out[:, 0])
    throttle = 1.0 / (1.0 + np.exp(-out[:, 1]))
    return turn, throttle
