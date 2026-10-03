"""Small test shapes. A 'shape' is a dict  {part_name: (N, 2) array of points}.

Shapes are stored as plain polygons (vertices only); the plotting code densifies them only when a
projective map needs it (affine maps keep straight edges straight, so vertices are enough).

The house is deliberately ASYMMETRIC (chimney on the right, door on the left)
so rotations and reflections are easy to tell apart.
"""
from __future__ import annotations

import numpy as np


def rect(x0, y0, x1, y1) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], dtype=float)


def densify(poly: np.ndarray, per_edge: int = 12) -> np.ndarray:
    """Insert points along each edge (needed when a projective map sends part of an edge to infinity)."""
    poly = np.asarray(poly, float)
    t = np.linspace(0, 1, per_edge, endpoint=False)[None, :, None]           # (1, k, 1)
    a, b = poly[:-1, None, :], poly[1:, None, :]                              # (E, 1, 2)
    pts = ((1 - t) * a + t * b).reshape(-1, 2)
    return np.vstack([pts, poly[-1:]])


def house(scale: float = 1.0, offset=(0.0, 0.0)) -> dict:
    outline = np.array([[0, 0], [2, 0], [2, 1.5], [1.6, 1.9], [1.6, 2.6], [1.3, 2.6],
                        [1.3, 2.2], [1.0, 2.5], [0, 1.5], [0, 0]], dtype=float)
    parts = {
        "outline": outline,
        "door": rect(0.3, 0.0, 0.7, 0.85),
        "window": rect(1.15, 0.65, 1.65, 1.1),
    }
    off = np.asarray(offset, dtype=float)
    return {k: v * scale + off for k, v in parts.items()}


def unit_square() -> dict:
    return {"outline": rect(0, 0, 1, 1)}


def square(x0=-1.0, y0=-1.0, x1=1.0, y1=1.0) -> dict:
    return {"outline": rect(x0, y0, x1, y1)}


def letter_f(scale: float = 1.0, offset=(0.0, 0.0)) -> dict:
    f = np.array([[0, 0], [0.4, 0], [0.4, 1.0], [1.0, 1.0], [1.0, 1.35], [0.4, 1.35], [0.4, 1.65],
                  [1.2, 1.65], [1.2, 2.0], [0, 2.0], [0, 0]], dtype=float)
    return {"outline": f * scale + np.asarray(offset, float)}


def circle(r: float = 1.0, c=(0.0, 0.0), n: int = 120) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n)
    return np.c_[c[0] + r * np.cos(t), c[1] + r * np.sin(t)]


def triangle() -> np.ndarray:
    return np.array([[0.0, 0.0], [2.0, 0.0], [0.6, 1.6], [0.0, 0.0]])


def grid_lines(extent: float = 4.0, step: float = 1.0, n: int = 60) -> list[np.ndarray]:
    """Horizontal + vertical lines covering [-extent, extent]^2, each densely sampled."""
    vals = np.arange(-extent, extent + 1e-9, step)
    s = np.linspace(-extent, extent, n)
    lines = [np.c_[s, np.full_like(s, v)] for v in vals]
    lines += [np.c_[np.full_like(s, v), s] for v in vals]
    return lines
