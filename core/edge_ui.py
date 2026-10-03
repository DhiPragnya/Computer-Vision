"""UI pieces shared by the edge-detection lessons (image picker, kernel grids, 1D signal plots)."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from . import edges as E
from . import live, theme, ui

DIVERGING = [[0.0, "#2a78d6"], [0.5, "#f0efec"], [1.0, "#e34948"]]
GREYS = [[0.0, "#111111"], [1.0, "#f5f5f5"]]
LIGHT_BLUES = [[0.0, "#eef4fc"], [1.0, "#86b6ef"]]      # readable behind dark numbers


def pick_image(prefix: str, default: str = "Cameraman (photo)", noise: bool = True, noise_default: float = 0.0,
               upload: bool = True, label: str = "Image"):
    """Image selector (+ optional upload and noise). Returns (noisy image, clean image, name)."""
    options = list(E.SAMPLES) + (["Upload my own"] if upload else [])
    name = st.selectbox(label, options, index=options.index(default), key=f"{prefix}_img")
    if name == "Upload my own":
        up = st.file_uploader("PNG / JPG", type=["png", "jpg", "jpeg"], key=f"{prefix}_up")
        clean = None
        if up is not None:
            try:
                clean = E.load_upload(up)
            except Exception:
                st.error("Could not read that file - using the sample image instead.", icon=":material/error:")
        if clean is None:
            clean = E.SAMPLES[default]()
    else:
        clean = E.SAMPLES[name]()
    sigma_n = 0.0
    if noise:
        sigma_n = live.slider("added noise σₙ", 0.0, 0.2, noise_default, 0.01, key=f"{prefix}_noise",
                              help="Gaussian noise with this standard deviation (image range is 0…1)")
    return E.add_noise(clean, sigma_n, seed=1), clean, name


def show_row(items: list[tuple[np.ndarray, str]], cols=None) -> None:
    cols = st.columns(cols or len(items))
    for c, (img, cap) in zip(cols, items):
        if img.ndim == 2:
            img = E.to_u8(img, 0, 1) if img.min() >= -1e-9 and img.max() <= 1 + 1e-9 else E.to_u8(img)
        c.image(ui.datauri(img), caption=cap, width="stretch")


def kernel_figure(K, title: str = "", fmt: str = "{:g}", height: int = 210, diverging: bool = True,
                  zmax: float | None = None) -> go.Figure:
    K = np.atleast_2d(np.asarray(K, float))
    zmax = zmax or (np.abs(K).max() or 1.0)
    text = [[fmt.format(v) for v in row] for row in K]
    fsize = int(np.clip(17 - 1.2 * max(K.shape) - (2 if len(fmt.format(0.5)) > 4 else 0), 8, 15))
    fig = go.Figure(go.Heatmap(z=K, text=text, texttemplate="%{text}", textfont=dict(size=fsize, color="#1a1a19"),
                               colorscale=DIVERGING if diverging else LIGHT_BLUES,
                               zmin=-zmax if diverging else min(K.min(), 0), zmax=zmax if diverging else K.max(),
                               showscale=False, xgap=3, ygap=3,
                               hovertemplate="row %{y}, col %{x}: %{z:.3g}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", visible=False, scaleanchor="x", scaleratio=1)
    fig.update_xaxes(visible=False)
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=28 if title else 4, b=0),
                      title=dict(text=title, x=0.5, xanchor="center", font=dict(size=13)) if title else None)
    return fig


def patch_figure(P, title: str = "", fmt: str = "{:.2f}", height: int = 300, colorscale=None, zmin=None, zmax=None,
                 show_text: bool = True) -> go.Figure:
    P = np.asarray(P, float)
    fig = go.Figure(go.Heatmap(z=P, colorscale=colorscale or GREYS, zmin=zmin, zmax=zmax, showscale=False,
                               text=[[fmt.format(v) for v in r] for r in P] if show_text else None,
                               texttemplate="%{text}" if show_text else None, textfont=dict(size=11),
                               xgap=1, ygap=1, hovertemplate="x=%{x}, y=%{y}: %{z:.3f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=28 if title else 4, b=0),
                      title=dict(text=title, x=0.01, font=dict(size=13)) if title else None)
    return fig


def signal_figure(series: list[dict], height: int = 260, title: str | None = None, x_title: str = "x (pixels)",
                  y_title: str | None = None, vlines: list[tuple[float, str]] | None = None,
                  hlines: list[tuple[float, str]] | None = None, legend: bool = True) -> go.Figure:
    """series: dicts with x, y, name, color, (mode, dash, width)."""
    fig = go.Figure()
    for s in series:
        fig.add_trace(go.Scatter(x=s["x"], y=s["y"], name=s.get("name", ""), mode=s.get("mode", "lines"),
                                 line=dict(color=s.get("color", theme.TRANSFORMED), width=s.get("width", 2.2),
                                           dash=s.get("dash")),
                                 marker=dict(size=s.get("size", 7), color=s.get("color", theme.TRANSFORMED)),
                                 showlegend=s.get("showlegend", legend and bool(s.get("name"))),
                                 hovertemplate=f"{s.get('name', '')}<br>x=%{{x:.2f}}, y=%{{y:.3f}}<extra></extra>"))
    for xv, lab in vlines or []:
        fig.add_vline(x=xv, line=dict(color=theme.HIGHLIGHT, width=1.5, dash="dot"),
                      annotation_text=lab, annotation_position="top")
    for yv, lab in hlines or []:
        fig.add_hline(y=yv, line=dict(color=theme.MAGENTA, width=1.5, dash="dash"),
                      annotation_text=lab, annotation_position="right")
    fig.update_layout(height=height + (30 if title else 0), margin=dict(l=8, r=8, t=74 if title else 20, b=8),
                      hovermode="x unified",
                      title=dict(text=title, x=0.01, y=0.985, yanchor="top", yref="container", font=dict(size=14))
                      if title else None,
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(title=x_title, gridcolor=theme.GRID, zeroline=False)
    fig.update_yaxes(title=y_title, gridcolor=theme.GRID, zerolinecolor=theme.AXIS)
    return fig


def gray_heatmap(img: np.ndarray, height: int = 420, title: str | None = None, zmin=0.0, zmax=1.0) -> go.Figure:
    """An image as a Plotly heatmap in image coordinates (y down) so arrows/points can be drawn on it."""
    fig = go.Figure(go.Heatmap(z=img, colorscale=GREYS, zmin=zmin, zmax=zmax, showscale=False,
                               hovertemplate="x=%{x}, y=%{y}: I=%{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False,
                     constrain="domain")
    fig.update_xaxes(showgrid=False, zeroline=False, constrain="domain")
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=30 if title else 6, b=0),
                      title=dict(text=title, x=0.01, font=dict(size=13)) if title else None,
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, bgcolor="rgba(0,0,0,0)"))
    return fig


def arrows_trace(x0, y0, dx, dy, color=theme.E1, name="gradient ∇I", width=2.0, showlegend=True) -> go.Scatter:
    """Many arrows in one trace (shaft lines + arrow-head markers)."""
    xs, ys, mx, my, ang = [], [], [], [], []
    for a, b, u, v in zip(np.ravel(x0), np.ravel(y0), np.ravel(dx), np.ravel(dy)):
        if not np.isfinite(u) or (abs(u) + abs(v)) < 1e-9:
            continue
        xs += [a, a + u, None]
        ys += [b, b + v, None]
    return go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=width), name=name,
                      showlegend=showlegend, hoverinfo="skip")


def arrow_heads(x0, y0, dx, dy, color=theme.E1, size=8) -> go.Scatter:
    tips_x = np.ravel(x0) + np.ravel(dx)
    tips_y = np.ravel(y0) + np.ravel(dy)
    # plotly marker angle: degrees clockwise from "up"; our y axis is reversed (down), so up on screen = −y
    ang = np.degrees(np.arctan2(np.ravel(dx), -np.ravel(dy)))
    keep = (np.abs(np.ravel(dx)) + np.abs(np.ravel(dy))) > 1e-9
    return go.Scatter(x=tips_x[keep], y=tips_y[keep], mode="markers",
                      marker=dict(symbol="triangle-up", size=size, color=color, angle=ang[keep]),
                      showlegend=False, hoverinfo="skip")
