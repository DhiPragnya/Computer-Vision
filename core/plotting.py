"""Plotly helpers so every lesson's figures look and behave the same.

The central helper is `transform_figure`: give it a shape and a matrix (or a
function t -> matrix) and it draws
    * the original grid + shape (grey, dashed),
    * the transformed grid + shape (blue),
    * where the basis vectors e1, e2 land (orange / green arrows),
and, optionally, a Play button + scrubber that morphs identity -> transform.
"""
from __future__ import annotations

import functools
from typing import Callable, Iterable

import numpy as np
import plotly.graph_objects as go

from . import shapes as shp
from . import theme
from . import transforms as tf
from .fastplotly import fast

FAR = 1e3   # anything further than this is "practically at infinity" - don't draw it


class TraceDict(dict):
    """A plain-dict Plotly trace. Creating go.Scatter objects validates every property (slow when an animation
    needs hundreds of them); a dict with type="scatter" is understood by Plotly just the same.
    .update() returns the trace so calls can be chained like with go objects."""

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        return self


def _strip_none(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if v is None:
            continue
        out[k] = _strip_none(v) if isinstance(v, dict) else v
    return out


def _scatter(**kw) -> TraceDict:
    return TraceDict(type="scatter", **_strip_none(kw))


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------

def _extent(extent):
    if np.isscalar(extent):
        return (-extent, extent, -extent, extent)
    return tuple(extent)


def base_figure(extent=4.0, height: int = 480, title: str | None = None, legend: bool = True) -> go.Figure:
    x0, x1, y0, y1 = _extent(extent)
    axis = dict(zeroline=True, zerolinewidth=1.5, zerolinecolor=theme.AXIS, gridcolor=theme.GRID, showline=False,
                ticks="", tickfont=dict(size=11), constrain="domain")
    layout = dict(
        height=height,
        margin=dict(l=8, r=8, t=40 if title else 28, b=8),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                    font=dict(size=12), bgcolor="rgba(0,0,0,0)"),
        hovermode="closest",
        dragmode="pan",
        modebar=dict(orientation="v", bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(range=[x0, x1], **axis),
        yaxis=dict(range=[y0, y1], scaleanchor="x", scaleratio=1, **axis),
    )
    if title:
        layout["title"] = dict(text=title, x=0.01, font=dict(size=15))
    return go.Figure(layout=layout)        # one validation pass (separate update_* calls cost ~10x more)


def mapped(H, pts) -> np.ndarray:
    """Apply H for DRAWING: points that go behind the camera (w <= 0) or to infinity become gaps."""
    H = np.asarray(H, float)
    pts = np.asarray(pts, float).reshape(-1, 2)
    q = pts @ H[:, :2].T + H[:, 2]                       # = [x y 1] @ H.T without building [x y 1]
    w = q[:, 2]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out = q[:, :2] / w[:, None]
    bad = (w <= 1e-6) | ~np.isfinite(out).all(axis=1) | (np.abs(out) > FAR).any(axis=1)
    out[bad] = np.nan
    return out


@functools.lru_cache(maxsize=64)
def _grid_points(extent: float, step: float, n: int) -> np.ndarray:
    """All grid lines stacked into one array, each line followed by a NaN row (a gap in the drawing)."""
    gap = np.array([[np.nan, np.nan]])
    pts = np.vstack([np.vstack([ln, gap]) for ln in shp.grid_lines(extent, step, n)])
    pts.flags.writeable = False
    return pts


def _grid_xy(H, extent, step, n):
    """The grid mapped by H as two float32 arrays with NaN gaps (one matrix product for the whole grid)."""
    q = mapped(H, _grid_points(float(extent), float(step), int(n))).astype(np.float32)
    return q[:, 0], q[:, 1]


def _join(polylines: Iterable[np.ndarray]):
    """Concatenate polylines with NaN gaps -> two float32 arrays (sent to the browser as compact typed arrays)."""
    parts = []
    gap = np.array([[np.nan, np.nan]])
    for p in polylines:
        parts += [np.asarray(p, float), gap]
    if not parts:
        return np.array([], np.float32), np.array([], np.float32)
    allp = np.vstack(parts).astype(np.float32)
    return allp[:, 0], allp[:, 1]


def is_affine(H) -> bool:
    H = np.asarray(H, float)
    return abs(H[2, 0]) < 1e-12 and abs(H[2, 1]) < 1e-12


def grid_trace(H=None, extent=4.0, step=1.0, color=theme.GRID_TRANSFORMED, name="grid",
               width=1.0, showlegend=False, n: int | None = None) -> go.Scatter:
    H = np.eye(3) if H is None else H
    # straight lines stay straight under affine maps: 2 points per line are enough (much lighter figures);
    # projective maps need dense sampling so that parts behind the camera can be cut out.
    if n is None:
        n = 2 if is_affine(H) else 60
    xs, ys = _grid_xy(H, extent, step, n)
    return _scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=width),
                      name=name, hoverinfo="skip", showlegend=showlegend)


def shape_traces(parts: dict, H=None, color=theme.TRANSFORMED, name="transformed",
                 dash: str | None = None, fill_alpha: float = 0.22, width: float = 2.4,
                 showlegend: bool = True, hover: bool = True) -> list[go.Scatter]:
    H = np.eye(3) if H is None else H
    traces = []
    dense = not is_affine(H)
    for i, (part, pts) in enumerate(parts.items()):
        pts = np.asarray(pts, float)
        if dense and len(pts) < 60:
            pts = shp.densify(pts, 12)
        q = mapped(H, pts).astype(np.float32)
        traces.append(_scatter(
            x=q[:, 0], y=q[:, 1], mode="lines",
            fill="toself" if fill_alpha > 0 else None,
            fillcolor=theme.rgba(color, fill_alpha if part == "outline" else fill_alpha * 1.6) if fill_alpha > 0 else None,
            line=dict(color=color, width=width if part == "outline" else width * 0.7, dash=dash),
            name=name, legendgroup=name, showlegend=showlegend and i == 0,
            hovertemplate=f"{name}<br>(%{{x:.2f}}, %{{y:.2f}})<extra></extra>" if hover else None,
            hoverinfo=None if hover else "skip",
        ))
    return traces


def arrow_trace(p0, p1, color, name="", width: float = 3.0, showlegend: bool = True,
                dash: str | None = None, size: int = 13, hover: str | None = None) -> go.Scatter:
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    return _scatter(
        x=[p0[0], p1[0]], y=[p0[1], p1[1]], mode="lines+markers",
        line=dict(color=color, width=width, dash=dash),
        marker=dict(symbol=["circle", "arrow"], size=[0, size], angleref="previous", color=color),
        name=name, legendgroup=name, showlegend=showlegend,
        hovertemplate=(hover or f"{name}<br>tip (%{{x:.2f}}, %{{y:.2f}})") + "<extra></extra>",
    )


def arrows_trace(starts, ends, color, name="", width: float = 2.2, size: int = 10, showlegend: bool = False,
                 dash: str | None = None) -> go.Scatter:
    """Many arrows in ONE trace (NaN-separated), so a step animation can grow / move them together."""
    starts, ends = np.asarray(starts, float).reshape(-1, 2), np.asarray(ends, float).reshape(-1, 2)
    xs, ys, sz = [], [], []
    for a, b in zip(starts, ends):
        tiny = np.hypot(*(b - a)) < 1e-6
        xs += [a[0], b[0], np.nan]
        ys += [a[1], b[1], np.nan]
        sz += [0, 0 if tiny else size, 0]
    return _scatter(x=np.array(xs, np.float32), y=np.array(ys, np.float32), mode="lines+markers",
                      line=dict(color=color, width=width, dash=dash),
                      marker=dict(symbol="arrow", size=sz, angleref="previous", color=color),
                      name=name, showlegend=showlegend, hoverinfo="skip")


def basis_traces(H=None, showlegend=True, faint=False) -> list[go.Scatter]:
    """Arrows from H(0) to H(e1) and H(e2): for an affine map these are exactly the matrix columns."""
    H = np.eye(3) if H is None else H
    o, a, b = mapped(H, np.array([[0, 0], [1, 0], [0, 1]]))
    if not (np.isfinite(o).all() and np.isfinite(a).all() and np.isfinite(b).all()):
        return []
    alpha = 0.45 if faint else 1.0
    w = 2.0 if faint else 3.2
    return [
        arrow_trace(o, a, theme.rgba(theme.E1, alpha), name="e₁ → 1st column" if not faint else "e₁",
                    width=w, showlegend=showlegend and not faint),
        arrow_trace(o, b, theme.rgba(theme.E2, alpha), name="e₂ → 2nd column" if not faint else "e₂",
                    width=w, showlegend=showlegend and not faint),
    ]


def points_trace(pts, color, name, labels=None, size: int = 11, symbol: str = "circle",
                 showlegend: bool = True, textpos="top right", line_color="white") -> go.Scatter:
    pts = np.asarray(pts, float).reshape(-1, 2)
    return _scatter(
        x=pts[:, 0], y=pts[:, 1], mode="markers+text" if labels is not None else "markers",
        marker=dict(color=color, size=size, symbol=symbol, line=dict(color=line_color, width=1.5)),
        text=labels, textposition=textpos, textfont=dict(size=13),
        name=name, showlegend=showlegend,
        hovertemplate=f"{name}<br>(%{{x:.2f}}, %{{y:.2f}})<extra></extra>",
    )


def line_trace(pts, color, name, width=2.0, dash=None, showlegend=True, hover=False) -> go.Scatter:
    pts = np.asarray(pts, float)
    return _scatter(x=pts[:, 0], y=pts[:, 1], mode="lines", line=dict(color=color, width=width, dash=dash),
                      name=name, showlegend=showlegend, hoverinfo=None if hover else "skip")


# --------------------------------------------------------------------------
# the workhorse
# --------------------------------------------------------------------------

def transform_figure(
    parts: dict,
    H: np.ndarray | Callable[[float], np.ndarray],
    extent=4.0,
    grid: bool = True,
    basis: bool = True,
    animate: bool = True,
    n_frames: int = 24,
    height: int = 500,
    grid_step: float = 1.0,
    grid_extent: float | None = None,
    extra_static: list | None = None,
    extra_dynamic: Callable[[np.ndarray, float], list] | None = None,
    title: str | None = None,
    legend: bool = True,
) -> go.Figure:
    """Draw original vs transformed. `H` may be a matrix or a function t -> matrix (t in [0, 1]).

    Passing a function lets each lesson morph in a meaningful way (e.g. rotate by t*theta)
    instead of a straight blend of matrix entries.
    """
    H_at = H if callable(H) else (lambda t, _H=np.asarray(H, float): tf.lerp_from_identity(_H, t))
    g_ext = grid_extent if grid_extent is not None else (max(np.abs(_extent(extent))) + 2)

    from .anim import AnimFigure, _trace_dicts

    fig = AnimFigure(base_figure(extent, height=height, title=title, legend=legend))

    # ---- static layers (never animated)
    static = []
    if grid:
        static.append(grid_trace(None, g_ext, grid_step, color=theme.GRID_ORIGINAL))
    static += shape_traces(parts, None, theme.ORIGINAL, name="original", dash="dash",
                           fill_alpha=0.06, width=1.8)
    if basis:
        static += basis_traces(None, faint=True)
    static += list(extra_static or [])
    for tr in static:
        fig.add_trace(tr)

    # ---- dynamic layers (rebuilt per frame)
    def dynamic(t: float) -> list:
        Ht = H_at(t)
        layers = []
        if grid:
            layers.append(grid_trace(Ht, g_ext, grid_step, color=theme.GRID_TRANSFORMED))
        layers += shape_traces(parts, Ht, theme.TRANSFORMED, name="transformed")
        if basis:
            b = basis_traces(Ht)
            layers += b if b else [_scatter(x=[None], y=[None], showlegend=False)] * 2
        if extra_dynamic is not None:
            layers += extra_dynamic(Ht, t)
        return layers

    final = dynamic(1.0)
    k0 = len(fig.data)
    for tr in final:
        fig.add_trace(tr)

    if animate:
        from .live import dragging
        ts = np.linspace(0.0, 1.0, 2 if dragging() else n_frames)   # mid-drag: Play frames come on release
        idx = list(range(k0, k0 + len(final)))
        with fast():            # trusted traces: skip Plotly's slow per-property validation
            fig._raw_frames = [dict(data=_trace_dicts(dynamic(float(t))), traces=idx, name=f"{t:.2f}") for t in ts]
        fig.update_layout(margin=dict(b=70))
        fig._raw_layout = dict(     # plain dicts (see anim.AnimFigure)
            updatemenus=[dict(
                type="buttons", direction="left", x=0.0, y=-0.035, xanchor="left", yanchor="top",
                pad=dict(r=6, t=6), showactive=False,
                buttons=[dict(label="▶ Play", method="animate",
                              args=[None, dict(frame=dict(duration=45, redraw=False),
                                               transition=dict(duration=0), fromcurrent=False,
                                               mode="immediate")])],
            )],
            sliders=[dict(
                active=len(ts) - 1, x=0.16, y=-0.02, len=0.84, xanchor="left", yanchor="top",
                pad=dict(t=6), currentvalue=dict(prefix="t = ", font=dict(size=12), xanchor="right", offset=6),
                ticklen=0, minorticklen=0, font=dict(size=1),
                steps=[dict(method="animate", label=f"{t:.2f}",
                            args=[[f"{t:.2f}"], dict(mode="immediate", frame=dict(duration=0, redraw=False),
                                                     transition=dict(duration=0))]) for t in ts],
            )],
        )
    return fig


def small_multiples(panels: list[tuple[str, dict, np.ndarray]], extent=3.0, height=300, cols=None) -> go.Figure:
    """Several (title, parts, H) panels side by side, same axes - used for the hierarchy overview."""
    from plotly.subplots import make_subplots

    n = len(panels)
    cols = cols or n
    rows = int(np.ceil(n / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[p[0] for p in panels],
                        horizontal_spacing=0.025, vertical_spacing=0.08)
    x0, x1, y0, y1 = _extent(extent)
    for i, (_, parts, H) in enumerate(panels):
        r, c = i // cols + 1, i % cols + 1
        fig.add_trace(grid_trace(None, 6, 1, theme.GRID_ORIGINAL), row=r, col=c)
        fig.add_trace(grid_trace(H, 6, 1, theme.GRID_TRANSFORMED), row=r, col=c)
        for tr in shape_traces(parts, None, theme.ORIGINAL, "original", dash="dash", fill_alpha=0.05,
                               width=1.4, showlegend=(i == 0), hover=False):
            fig.add_trace(tr, row=r, col=c)
        for tr in shape_traces(parts, H, theme.TRANSFORMED, "transformed", showlegend=(i == 0), width=2.0):
            fig.add_trace(tr, row=r, col=c)
    fig.update_xaxes(range=[x0, x1], showticklabels=False, zeroline=True, zerolinecolor=theme.AXIS,
                     gridcolor=theme.GRID, constrain="domain")
    for i in range(1, n + 1):
        fig.update_yaxes(scaleanchor=f"x{i if i > 1 else ''}", scaleratio=1, row=(i - 1) // cols + 1,
                         col=(i - 1) % cols + 1)
    fig.update_yaxes(range=[y0, y1], showticklabels=False, zeroline=True, zerolinecolor=theme.AXIS,
                     gridcolor=theme.GRID, constrain="domain")
    fig.update_layout(height=height * rows, margin=dict(l=4, r=4, t=48, b=4),
                      legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0), hovermode="closest")
    fig.update_annotations(font_size=13)
    return fig


# --------------------------------------------------------------------------
# fixed-structure layers for step animations (same number of points in every state)
# --------------------------------------------------------------------------

def layer_traces(parts: dict, H=None, grid_extent: float = 3.0, grid_step: float = 1.0, n: int = 24,
                 color: str = theme.TRANSFORMED, basis: bool = True, grid: bool = True,
                 name: str = "transformed", fill_alpha: float = 0.22, per_edge: int = 8) -> list:
    """Grid + shape (+ basis arrows) mapped by H, ALWAYS with the same number of points,
    so consecutive states can be morphed into each other."""
    H = np.eye(3) if H is None else np.asarray(H, float)
    out = []
    if grid:
        xs, ys = _grid_xy(H, grid_extent, grid_step, n)
        out.append(_scatter(x=xs, y=ys, mode="lines", line=dict(color=theme.GRID_TRANSFORMED, width=1),
                              hoverinfo="skip", showlegend=False))
    for i, (part, pts) in enumerate(parts.items()):
        pts = np.asarray(pts, float)
        q = mapped(H, shp.densify(pts, per_edge) if per_edge > 1 else pts).astype(np.float32)
        out.append(_scatter(x=q[:, 0], y=q[:, 1], mode="lines", fill="toself",
                              fillcolor=theme.rgba(color, fill_alpha if part == "outline" else fill_alpha * 1.6),
                              line=dict(color=color, width=2.4 if part == "outline" else 1.7),
                              name=name, showlegend=False, hoverinfo="skip"))
    if basis:
        o, a, b = mapped(H, np.array([[0, 0], [1, 0], [0, 1.0]]))
        o, a, b = [np.nan_to_num(v, nan=0.0) for v in (o, a, b)]
        out.append(arrow_trace(o, a, theme.E1, "e₁ → 1st column", showlegend=False))
        out.append(arrow_trace(o, b, theme.E2, "e₂ → 2nd column", showlegend=False))
    return out


def static_layers(parts: dict, grid_extent: float = 3.0, grid_step: float = 1.0, basis: bool = True) -> list:
    """The faint 'before' picture used as a fixed background in step animations."""
    out = [grid_trace(None, grid_extent, grid_step, color=theme.GRID_ORIGINAL)]
    out += shape_traces(parts, None, theme.ORIGINAL, name="original", dash="dash", fill_alpha=0.05, width=1.6,
                        showlegend=False, hover=False)
    if basis:
        out += basis_traces(None, faint=True)
    return out
