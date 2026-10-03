"""UI pieces shared by the contour / polygon / Hough lessons (topic 3): pixel-grid figures, accumulator heat-maps,
image-space figures and the scene picker."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from . import contours as K
from . import live, theme
from .plotting import TraceDict

GRID_BG = "#2b2b2a"
GRID_FG = "#d9d7d0"
BINARY = [[0.0, GRID_BG], [1.0, GRID_FG]]


def pick_scene(prefix: str, default: str = "Shapes (synthetic)", t_default: float = 0.45, invert: bool = True,
               label: str = "Image"):
    """Scene selector + live threshold slider. Returns (binary B, grey image, scene name, threshold)."""
    names = list(K.SCENES)
    name = st.selectbox(label, names, index=names.index(default), key=f"{prefix}_scene")
    img = K.SCENES[name]()
    t = live.slider("threshold t", 0.05, 0.95, t_default, 0.01, key=f"{prefix}_t",
                    help="B(x, y) = 1 where the image is brighter than t")
    inv = st.toggle("dark objects (invert)", False, key=f"{prefix}_inv",
                    help="Use this when the objects are DARKER than the background") if invert else False
    return K.binarize(img, t, inv), img, name, t


# --------------------------------------------------------------------------
# pixel grids
# --------------------------------------------------------------------------


def grid_heatmap(Z, colorscale=None, zmin=0.0, zmax=1.0, gap: int = 2, text=None, name: str = "") -> TraceDict:
    """A tiny image as a heat-map trace: column = x, row = y (drawn downwards by grid_layout)."""
    Z = np.asarray(Z, float)
    d = TraceDict(type="heatmap", z=Z, colorscale=colorscale or BINARY, zmin=zmin, zmax=zmax, showscale=False,
                  xgap=gap, ygap=gap, hoverinfo="skip", name=name)
    if text is not None:
        d.update(text=text, texttemplate="%{text}", textfont=dict(size=11, color="#1a1a19"))
    return d


def grid_layout(shape, height: int = 420, xaxis: str = "xaxis", yaxis: str = "yaxis", legend: bool = False) -> dict:
    """Axes for a pixel grid: y downwards, square pixels, no ticks."""
    h, w = shape
    ax = dict(visible=False, constrain="domain", range=[-0.5, w - 0.5])
    ay = dict(visible=False, range=[h - 0.5, -0.5], scaleanchor=xaxis.replace("axis", ""), scaleratio=1,
              constrain="domain")
    return {xaxis: ax, yaxis: ay, "height": height, "margin": dict(l=0, r=0, t=10, b=0), "showlegend": legend,
            "legend": dict(orientation="h", x=0, y=1.0, yanchor="bottom", bgcolor="rgba(0,0,0,0)")}


def grid_figure(Z, height: int = 420, colorscale=None, zmin=0.0, zmax=1.0, gap: int = 2, legend: bool = False):
    return go.Figure(data=[grid_heatmap(Z, colorscale, zmin, zmax, gap)], layout=grid_layout(np.shape(Z), height,
                                                                                         legend=legend))


def markers(rc, color, size=14, symbol="square", name="", showlegend=False, line_color="white", opacity=1.0,
            text=None, textcolor="#1a1a19") -> TraceDict:
    """Markers on a pixel grid at (row, col) positions; `size` may be an array (0 = hidden)."""
    rc = np.asarray(rc, float).reshape(-1, 2)
    d = TraceDict(type="scatter", x=rc[:, 1].astype(np.float32), y=rc[:, 0].astype(np.float32),
                  mode="markers+text" if text is not None else "markers",
                  marker=dict(size=size, color=color, symbol=symbol, line=dict(color=line_color, width=1),
                              opacity=opacity),
                  name=name, showlegend=showlegend, hoverinfo="skip")
    if text is not None:
        d.update(text=text, textfont=dict(size=10, color=textcolor), textposition="middle center")
    return d


def path(rc, color=theme.TRANSFORMED, width=3.0, name="", showlegend=False, dash=None) -> TraceDict:
    """A polyline through pixel centres given as (row, col); NaN rows are gaps."""
    rc = np.asarray(rc, float).reshape(-1, 2)
    return TraceDict(type="scatter", x=rc[:, 1].astype(np.float32), y=rc[:, 0].astype(np.float32), mode="lines",
                     line=dict(color=color, width=width, dash=dash), name=name, showlegend=showlegend,
                     hoverinfo="skip")


def grow(points: np.ndarray, f: float) -> np.ndarray:
    """The first fraction f of a polyline, ALWAYS with the same number of points (the rest sit on the tip), so a
    step animation can grow it smoothly."""
    P = np.asarray(points, float)
    n = len(P)
    if n == 0:
        return P
    pos = np.clip(f, 0, 1) * (n - 1)
    k = int(np.floor(pos))
    tip = P[k] if k >= n - 1 else P[k] + (pos - k) * (P[k + 1] - P[k])
    out = P.copy()
    out[k + 1:] = tip
    return out


# --------------------------------------------------------------------------
# parameter space
# --------------------------------------------------------------------------

def heat_trace(A, x, y, name="votes", zmax=None, xaxis=None, yaxis=None, hover="θ=%{x:.0f}°, ρ=%{y:.0f}: %{z} votes",
               colorscale=None, showscale=False) -> TraceDict:
    from .hough import HEAT

    A = np.asarray(A, np.float32)
    d = TraceDict(type="heatmap", z=A, x=np.asarray(x, np.float32), y=np.asarray(y, np.float32),
                  colorscale=colorscale or HEAT, zmin=0, zmax=float(zmax if zmax is not None else max(A.max(), 1)),
                  showscale=showscale, name=name, hovertemplate=hover + "<extra></extra>")
    if xaxis:
        d.update(xaxis=xaxis, yaxis=yaxis)
    return d


def scatter(x, y, color, size=7, name="", mode="markers", width=2.0, dash=None, showlegend=False, symbol="circle",
            xaxis=None, yaxis=None, opacity=None, hover=False, line_color=None) -> TraceDict:
    d = TraceDict(type="scatter", x=np.asarray(x, np.float32), y=np.asarray(y, np.float32), mode=mode,
                  name=name, showlegend=showlegend)
    if "markers" in mode:
        d["marker"] = dict(color=color, size=size, symbol=symbol)
        if line_color:
            d["marker"]["line"] = dict(color=line_color, width=1)
    if "lines" in mode:
        d["line"] = dict(color=color, width=width, dash=dash)
    if opacity is not None:
        d["opacity"] = opacity
    if not hover:
        d["hoverinfo"] = "skip"
    if xaxis:
        d.update(xaxis=xaxis, yaxis=yaxis)
    return d


def two_panel(left_title: str, right_title: str, height: int = 460, widths=(0.45, 0.55), image_left=None,
              left_range=None, right_range=None, left_title_x="x", left_title_y="y", right_title_x="",
              right_title_y="", left_equal: bool = True, left_ydown: bool = False, legend: bool = False) -> go.Figure:
    """Image space (left) | parameter space (right) - the figure the whole Hough idea lives in.
    left_range / right_range: (x0, x1, y0, y1)."""
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2, column_widths=list(widths), horizontal_spacing=0.11,
                        subplot_titles=[left_title, right_title])
    ax = dict(gridcolor=theme.GRID, zeroline=False, tickfont=dict(size=10))
    lay = dict(height=height, margin=dict(l=8, r=8, t=36, b=8), showlegend=legend, hovermode="closest",
               legend=dict(orientation="h", x=0, y=1.08, yanchor="bottom", bgcolor="rgba(0,0,0,0)"),
               xaxis=dict(title=dict(text=left_title_x, font=dict(size=12)), **ax),
               yaxis=dict(title=dict(text=left_title_y, font=dict(size=12)), **ax),
               xaxis2=dict(title=dict(text=right_title_x, font=dict(size=12)), **ax),
               yaxis2=dict(title=dict(text=right_title_y, font=dict(size=12)), **ax))
    if left_range is not None:
        x0, x1, y0, y1 = left_range
        lay["xaxis"]["range"] = [x0, x1]
        lay["yaxis"]["range"] = [y1, y0] if left_ydown else [y0, y1]
    if left_equal:
        lay["yaxis"].update(scaleanchor="x", scaleratio=1, constrain="domain")
        lay["xaxis"]["constrain"] = "domain"
    if right_range is not None:
        x0, x1, y0, y1 = right_range
        lay["xaxis2"]["range"] = [x0, x1]
        lay["yaxis2"]["range"] = [y0, y1]
    fig.update_layout(**lay)
    fig.update_annotations(font_size=13)
    return fig


def image_trace(gray01: np.ndarray, xaxis=None, yaxis=None, dim: float = 1.0) -> TraceDict:
    """A grey image as a heat-map trace in image coordinates (pixel (x, y) at column x, row y)."""
    d = TraceDict(type="heatmap", z=np.asarray(gray01, np.float32) * dim,
                  colorscale=[[0, "#111111"], [1, "#f5f5f5"]], zmin=0, zmax=1, showscale=False, hoverinfo="skip")
    if xaxis:
        d.update(xaxis=xaxis, yaxis=yaxis)
    return d


def image_axes(w: int, h: int, height: int = 430, pad: float = 5, legend: bool = False, title_x: str = "x",
               title_y: str = "y (down)") -> go.Figure:
    """An empty figure in image coordinates: origin top-left, y DOWN, square pixels."""
    return go.Figure(layout=dict(height=height, margin=dict(l=8, r=8, t=10, b=8), showlegend=legend,
                                 legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom", bgcolor="rgba(0,0,0,0)"),
                                 xaxis=dict(range=[-pad, w + pad], gridcolor=theme.GRID, zeroline=False,
                                            constrain="domain", title=title_x),
                                 yaxis=dict(range=[h + pad, -pad], gridcolor=theme.GRID, zeroline=False,
                                            scaleanchor="x", constrain="domain", title=title_y)))
