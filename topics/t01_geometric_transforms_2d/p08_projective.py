import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core import anim, images, live, plotting, shapes, theme, ui
from core import transforms as tf

ui.lesson_header(
    __file__,
    "Free up the bottom row of the matrix and you get a <b>homography</b> - the map between two pictures of "
    "the same plane taken from different viewpoints. Parallel lines may now meet, but straight lines stay "
    "straight. 9 numbers, defined up to scale → 8 degrees of freedom.",
)

st.latex(r"\begin{bmatrix} \tilde x' \\ \tilde y' \\ \tilde w' \end{bmatrix} = "
         r"\begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ "
         r"\color{" + theme.MAGENTA + r"}{h_{31}} & \color{" + theme.MAGENTA + r"}{h_{32}} & h_{33} \end{bmatrix}"
         r"\begin{bmatrix} x \\ y \\ 1 \end{bmatrix}"
         r"\quad\Longrightarrow\quad x' = \frac{h_{11}x + h_{12}y + h_{13}}{\color{" + theme.MAGENTA +
         r"}{h_{31}x + h_{32}y} + h_{33}},\quad y' = \frac{h_{21}x + h_{22}y + h_{23}}{\color{" + theme.MAGENTA +
         r"}{h_{31}x + h_{32}y} + h_{33}}")

TAB_NAMES = ["1 · Feel the perspective terms", "2 · Why four points?", "3 · Four points define H (DLT)",
             "4 · Undo perspective (rectify)", "5 · What survives?"]
# switching tab re-runs the page and only the open tab is computed (much faster)
tabs = st.tabs(TAB_NAMES, key="pj_tabs", on_change="rerun")

HOUSE1 = shapes.house(offset=(-1, -1.2))
SRC4 = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
NAMES4 = ["P₁ (bottom-left)", "P₂ (bottom-right)", "P₃ (top-right)", "P₄ (top-left)"]
DEFAULTS4 = [(-1.6, -1.3), (1.7, -1.1), (0.8, 0.9), (-0.7, 1.0)]


def current_dst4():
    """The four targets of tab 3 - read from session state so that other tabs can use them too."""
    ss = st.session_state
    return np.array([[ss.get(f"pj_d{i}x", dx), ss.get(f"pj_d{i}y", dy)] for i, (dx, dy) in enumerate(DEFAULTS4)])


def vanishing_traces(Ht, lim=60):
    out = []
    for d, col, name in [((1, 0, 0), theme.E1, "vanishing point of horizontal lines"),
                         ((0, 1, 0), theme.E2, "vanishing point of vertical lines")]:
        v = Ht @ np.array(d, float)
        if abs(v[2]) > 1e-6 and np.all(np.abs(v[:2] / v[2]) < lim):
            out.append(plotting.points_trace([v[:2] / v[2]], col, name, size=13, symbol="star"))
        else:
            out.append(go.Scatter(x=[None], y=[None], showlegend=False, name=name))
    lv = np.linalg.inv(Ht).T @ np.array([0, 0, 1.0])            # image of the line at infinity
    if np.abs(lv[:2]).max() > 1e-6:
        xs = np.linspace(-lim, lim, 2)
        if abs(lv[1]) > abs(lv[0]):
            pts = np.c_[xs, -(lv[0] * xs + lv[2]) / lv[1]]
        else:
            pts = np.c_[-(lv[1] * xs + lv[2]) / lv[0], xs]
        out.append(plotting.line_trace(pts, theme.MAGENTA, "horizon = image of the line at infinity", width=2,
                                       dash="dash"))
    else:
        out.append(go.Scatter(x=[None], y=[None], showlegend=False, name="horizon"))
    return out


# =====================================================================================
# TAB 1
# =====================================================================================
def divide_steps(H1):
    """x̃′ = H x̃ in 3D: the plane w = 1 tilts, then every point slides along its ray back to w = 1."""
    lines = [np.asarray(v, float) for v in shapes.grid_lines(3.0, 1.0, 25)]
    outline = shapes.densify(HOUSE1["outline"], 10)

    def lift(P, t_mul, t_div):
        P3 = np.c_[P, np.ones(len(P))]
        Q = P3 @ (((1 - t_mul) * np.eye(3) + t_mul * H1).T)
        w = Q[:, 2:3]
        with np.errstate(divide="ignore", invalid="ignore"):
            R = Q * ((1 - t_div) + t_div / w)
        bad = (t_div > 0) & ((w[:, 0] <= 0.04) | (np.abs(R) > 9).any(axis=1))
        R[bad] = np.nan
        return R

    def build(p):
        segs = []
        for ln in lines:
            segs += [lift(ln, p["m"], p["d"]), np.full((1, 3), np.nan)]
        G = np.vstack(segs)
        Hs = lift(outline, p["m"], p["d"])
        return [go.Scatter3d(x=G[:, 0], y=G[:, 1], z=G[:, 2], mode="lines",
                             line=dict(color=theme.TRANSFORMED, width=2.5), opacity=0.6,
                             showlegend=False, hoverinfo="skip"),
                go.Scatter3d(x=Hs[:, 0], y=Hs[:, 1], z=Hs[:, 2], mode="lines",
                             line=dict(color=theme.TRANSFORMED, width=7), showlegend=False, hoverinfo="skip")]

    P = [dict(m=0.0, d=0.0), dict(m=1.0, d=0.0), dict(m=1.0, d=1.0)]
    h31, h32 = H1[2, 0], H1[2, 1]
    f, H = ui.num, anim.hl
    info = [
        ("Points of the plane, lifted to w = 1", "x̃ = (x, y, 1)",
         "The grid and the house sit on the flat plane w = 1."),
        ("Multiply by H", f"w̃′ = {H(f(h31))}·x + {H(f(h32))}·y + 1   -   different for every point",
         "The first two rows act like an affine map, but the third row gives every point its own height w̃′: "
         "the plane TILTS. Where w̃′ < 0 it has gone below the floor - behind the camera."),
        ("Divide by w̃′", "(x′, y′) = (x̃′ / w̃′, ỹ′ / w̃′)",
         "Each point slides along its ray through the origin back to w = 1. High points (w̃′ > 1) come back "
         "SHRUNK towards the centre - far away; low points (w̃′ < 1) come back blown up - close by. That uneven "
         "shrinking IS perspective: the grid lines now converge."),
    ]
    return [anim.Step(t_, e_, n_, build(P[i]), between=None if i == 0 else anim.morph(build, P[i - 1], P[i]))
            for i, (t_, e_, n_) in enumerate(info)]


@live.section
def perspective_section():
    c_ctrl, c_fig, c_txt = st.columns([0.75, 1.9, 1.05], gap="large")
    with c_ctrl:
        st.markdown(f"<span style='color:{theme.MAGENTA}'><b>Perspective row</b></span>", unsafe_allow_html=True)
        h31 = live.slider("h₃₁", -0.5, 0.5, 0.22, 0.01, key="pj_h20")
        h32 = live.slider("h₃₂", -0.5, 0.5, 0.15, 0.01, key="pj_h21")
        st.markdown("**Affine part** (optional)")
        shx = live.slider("shear h₁₂", -1.0, 1.0, 0.0, 0.05, key="pj_h01")
        show_vl = st.toggle("Vanishing points", value=True, key="pj_vl")
    H1 = np.array([[1.0, shx, 0.0], [0.0, 1.0, 0.0], [h31, h32, 1.0]])

    with c_fig:
        fig = plotting.transform_figure(HOUSE1, H1, extent=8, grid_extent=4, grid_step=0.5, height=580,
                                        extra_dynamic=(lambda Ht, t: vanishing_traces(Ht)) if show_vl else None)
        st.plotly_chart(fig, key="pj_fig1")
    with c_txt:
        ui.show_matrix("H", H1, 2, {(2, 0): theme.MAGENTA, (2, 1): theme.MAGENTA})
        ui.callout("intuition",
                   "With h₃₁ = h₃₂ = 0 the denominator is always 1 - an ordinary affine map. Make them non-zero "
                   "and **w' changes from place to place**: points where w' is large get shrunk (far away), points "
                   "where w' is small get blown up (close). That uneven division *is* perspective foreshortening.")
        xs = np.linspace(-4, 4, 41)
        X, Y = np.meshgrid(xs, xs)
        W = h31 * X + h32 * Y + 1.0
        hm = go.Figure(go.Contour(x=xs, y=xs, z=W, colorscale=[[0, "#cde2fb"], [0.5, "#5598e7"], [1, "#0d366b"]],
                                  contours=dict(showlabels=True, labelfont=dict(size=10, color="white")),
                                  colorbar=dict(title="w'", thickness=10), hovertemplate="w' = %{z:.2f}<extra></extra>"))
        if abs(h31) + abs(h32) > 1e-6 and W.min() < 0 < W.max():
            hm.add_trace(go.Contour(x=xs, y=xs, z=W, contours=dict(start=0, end=0, size=1, coloring="none"),
                                    line=dict(color=theme.MAGENTA, width=3, dash="dash"), showscale=False,
                                    hoverinfo="skip", name="w' = 0"))
        hm.update_layout(height=250, margin=dict(l=0, r=0, t=26, b=0),
                         title=dict(text="w' over the ORIGINAL plane", font=dict(size=13), x=0.01))
        hm.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(hm, key="pj_w")
        st.caption("Source points on the dashed line (w' = 0) are sent **to infinity**. Their images form no "
                   "finite line - instead the line at infinity of the source comes back as the finite pink "
                   "**horizon** in the output, where parallel grid lines now meet.")

    # --------------------------------------------------------------- step by step in 3D
    ui.anim_intro("Step by step: multiply, then divide by w̃′ (in 3D)",
                  "Uses your h₃₁, h₃₂ and shear. Drag the 3D view to look from the side - you will see the tilted plane "
                  "in step 2 and the points sliding along their rays in step 3.")
    base = go.Figure()
    ext = 3.2
    for w_, col, a_ in [(1.0, theme.TRANSFORMED, 0.06), (0.0, theme.ORIGINAL, 0.05)]:
        base.add_trace(go.Mesh3d(x=[-ext, ext, ext, -ext], y=[-ext, -ext, ext, ext], z=[w_] * 4, i=[0, 0], j=[1, 2],
                                 k=[2, 3], color=col, opacity=a_, hoverinfo="skip", showscale=False))
    corners = HOUSE1["outline"][[0, 1, 4, 7]]
    for c in corners:                                   # a few rays through the origin
        q = H1 @ np.array([c[0], c[1], 1.0])
        if q[2] > 0.05:
            ray = np.vstack([[0, 0, 0], q * min(1.25, 2.0 / q[2])])
            base.add_trace(go.Scatter3d(x=ray[:, 0], y=ray[:, 1], z=ray[:, 2], mode="lines",
                                        line=dict(color=theme.ORIGINAL, width=2, dash="dot"), showlegend=False,
                                        hoverinfo="skip"))
    base.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", marker=dict(size=5, color=theme.HIGHLIGHT),
                                showlegend=False, hoverinfo="skip"))
    base.update_layout(height=680, showlegend=False, uirevision="keep",
                       scene=dict(xaxis=dict(range=[-4.5, 4.5], title="x"), yaxis=dict(range=[-4.5, 4.5], title="y"),
                                  zaxis=dict(range=[-0.6, 2.2], title="w"), aspectmode="manual",
                                  aspectratio=dict(x=1, y=1, z=0.55), camera=dict(eye=dict(x=0.9, y=-1.35, z=0.55))))
    st.plotly_chart(anim.step_player(base, divide_steps(H1), n_sub=16, wrap_at=110), key="pj_divide")

    # --------------------------------------------------------------- live h31 sweep
    ui.anim_intro("Live: sweep h₃₁ and watch the vanishing point and the horizon",
                  f"Drag the slider under the picture (h₃₂ = {ui.num(h32)} and the shear stay at your values). "
                  "At h₃₁ = 0 the horizontal lines are parallel; the bigger |h₃₁|, the closer their vanishing point.")
    base2 = plotting.base_figure(8, height=560, legend=False)
    base2.add_trace(plotting.grid_trace(None, 3, 1.0, theme.GRID_ORIGINAL))
    for tr in plotting.shape_traces(HOUSE1, None, theme.ORIGINAL, "original", dash="dash", fill_alpha=0.05,
                                    width=1.6, showlegend=False, hover=False):
        base2.add_trace(tr)
    vals = np.round(np.arange(-0.5, 0.5001, 0.05), 2)

    def make(v):
        Hv = np.array([[1.0, shx, 0.0], [0.0, 1.0, 0.0], [v, h32, 1.0]])
        trs = [plotting.grid_trace(Hv, 3, 1.0, theme.GRID_TRANSFORMED, n=40)]
        trs += plotting.shape_traces(HOUSE1, Hv, theme.TRANSFORMED, "transformed", showlegend=False, hover=False)
        trs += vanishing_traces(Hv)
        vx = Hv @ np.array([1.0, 0, 0])
        where = (f"horizontal lines meet at <b>({vx[0] / vx[2]:.1f}, {vx[1] / vx[2]:.1f})</b>" if abs(vx[2]) > 1e-9
                 else "<b>horizontal lines stay parallel</b> (their meeting point is at infinity)")
        cap = f"h₃₁ = <b>{v:+.2f}</b>   →   w′ = {v:+.2f}·x {h32:+.2f}·y + 1   ·   {where}"
        return trs, cap

    st.plotly_chart(anim.scrubber(base2, vals, make, active=int(np.argmin(np.abs(vals - h31))), label="h₃₁",
                                  fmt="{:+.2f}", frame_ms=90), key="pj_h31_scrub")


# =====================================================================================
# TAB 2 - why four points
# =====================================================================================
DEMO_SRC = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.2, 0.3]])
H_DEMO = np.array([[1.1, 0.2, 0.1], [0.05, 0.95, 0.1], [0.18, 0.12, 1.0]])
DEMO_DST = tf.apply(H_DEMO, DEMO_SRC)


def null_basis(n):
    A = tf.dlt_rows(DEMO_SRC[:n], DEMO_DST[:n])
    _, _, Vt = np.linalg.svd(A)
    r = np.linalg.matrix_rank(A, tol=1e-9)
    return Vt[r:]


@st.cache_resource(show_spinner=False)
@live.full_quality()          # cached: never store a coarse mid-drag version
def collapse_figure(seed: int):
    """Twelve homographies that fit 1 pair; add pairs and project each onto the smaller solution set."""
    rng = np.random.default_rng(seed)
    h_true = H_DEMO.ravel() / np.linalg.norm(H_DEMO)
    sq = shapes.densify(np.vstack([DEMO_SRC[:4], DEMO_SRC[:1]]), 16)
    B1 = null_basis(1)
    fam = []
    while len(fam) < 12:
        c = rng.normal(size=len(B1))
        d = c @ B1
        d -= (d @ h_true) * h_true
        h = h_true + 0.55 * d / np.linalg.norm(d)
        q = plotting.mapped(h.reshape(3, 3), sq)
        if not np.isnan(q).any() and np.abs(q).max() < 3.6:
            fam.append(h)
    fam = np.array(fam)

    def family_at(n):
        B = null_basis(min(n, 4))
        out = []
        for h in fam:
            hp = B.T @ (B @ h)
            hp *= np.sign(hp @ h_true) or 1.0
            out.append(hp / (hp[8] if abs(hp[8]) > 1e-9 else 1.0))
        return out

    def state(n):
        quads = []
        for h in family_at(n):
            quads += [plotting.mapped(h.reshape(3, 3), sq), np.array([[np.nan, np.nan]])]
        Q = np.vstack(quads)
        tgt = np.full((5, 2), np.nan)
        tgt[:n] = DEMO_DST[:n]
        return [go.Scatter(x=Q[:, 0], y=Q[:, 1], mode="lines", line=dict(color=theme.TRANSFORMED, width=1.5),
                           showlegend=False, hoverinfo="skip"),
                go.Scatter(x=tgt[:, 0], y=tgt[:, 1], mode="markers+text", text=["1′", "2′", "3′", "4′", "5′"],
                           textposition="top right", marker=dict(size=14, color=theme.HIGHLIGHT, symbol="diamond",
                                                                  line=dict(color="white", width=1.5)),
                           showlegend=False, hoverinfo="skip")]

    H = anim.hl
    info = [
        (1, "1 pair fixed", f"2 equations, 8 unknowns → a {H('6')}-parameter family",
         "Twelve different homographies, all sending corner 1 to 1′ - and wildly different everywhere else."),
        (2, "2 pairs fixed", f"4 equations → a {H('4')}-parameter family",
         "Every candidate is pulled onto the new constraint: the quadrilaterals now agree at 1′ and 2′."),
        (3, "3 pairs fixed", f"6 equations → a {H('2')}-parameter family",
         "Still a whole family - they pivot about three fixed corners."),
        (4, "4 pairs fixed", f"8 equations, 8 unknowns → {H('exactly one')} homography",
         "All twelve candidates collapse onto the same quadrilateral. This is why FOUR points determine H."),
        (5, "5 pairs fixed", "10 equations for 8 unknowns → over-determined",
         "With perfect data the 5th pair agrees with the solution. With noisy matches it would not, and least "
         "squares (the SVD) picks the best compromise."),
    ]
    steps = [anim.Step(t_, e_, n_, state(n)) for n, t_, e_, n_ in info]
    base = plotting.base_figure(2.6, height=560, legend=False)
    base.add_trace(plotting.line_trace(sq, theme.ORIGINAL, "source square", dash="dash", showlegend=False))
    base.add_trace(plotting.points_trace(DEMO_SRC, theme.ORIGINAL, "source points", labels=["1", "2", "3", "4", "5"],
                                         size=8, showlegend=False, textpos="bottom left"))
    return anim.step_player(base, steps, n_sub=16, wrap_at=80)


@live.section
def why_four_section():
    st.markdown("Count first: H has **9** entries, but H and λH describe the same map, so only **9 − 1 = 8** are "
                "free. Every correspondence (x, y) → (x′, y′) gives **2** equations. Add correspondences one at a "
                "time and watch the set of possible homographies shrink:")
    c_ctrl, c_fig, c_txt = st.columns([0.8, 1.5, 1.1], gap="large")
    with c_ctrl:
        n_pts = live.slider("number of correspondences", 1, 5, 3, key="pj_n4")
        st.session_state.setdefault("pj_fam_seed", 0)
        if st.button("Show other possible H", icon=":material/shuffle:", key="pj_fam_btn"):
            st.session_state.pj_fam_seed += 1
        eqs, free = 2 * n_pts, max(8 - 2 * n_pts, 0)
        st.metric("equations / unknowns", f"{eqs} / 8", border=True)
        st.metric("free parameters left", free, border=True)
    basis = null_basis(n_pts)
    null_dim = len(basis)
    if null_dim == 0:
        basis = np.linalg.svd(tf.dlt_rows(DEMO_SRC[:n_pts], DEMO_DST[:n_pts]))[2][-1:]
    rng = np.random.default_rng(st.session_state.pj_fam_seed)
    with c_fig:
        fig = plotting.base_figure(2.6, height=470)
        sq = shapes.densify(np.vstack([DEMO_SRC[:4], DEMO_SRC[:1]]), 20)
        fig.add_trace(plotting.line_trace(sq, theme.ORIGINAL, "source square", dash="dash"))
        drawn = 0
        for _ in range(60):
            if drawn >= (1 if null_dim <= 1 else 7):
                break
            coef = rng.normal(size=len(basis))
            Hc = (coef @ basis).reshape(3, 3)
            Hc = Hc if null_dim > 1 else basis[0].reshape(3, 3)
            q = plotting.mapped(Hc, sq)
            if np.isnan(q).any() or np.abs(q).max() > 2.6 * 1.6:
                Hc = -Hc
                q = plotting.mapped(Hc, sq)
                if np.isnan(q).any() or np.abs(q).max() > 2.6 * 1.6:
                    continue
            fig.add_trace(plotting.line_trace(q, theme.TRANSFORMED, "homographies that fit" if drawn == 0 else "",
                                              width=2.2 if null_dim <= 1 else 1.4, showlegend=drawn == 0))
            drawn += 1
        fig.add_trace(plotting.points_trace(DEMO_DST[:n_pts], theme.HIGHLIGHT, "the targets you fixed",
                                            labels=[f"{i + 1}′" for i in range(n_pts)], size=13, symbol="diamond"))
        st.plotly_chart(fig, key="pj_fam")
    with c_txt:
        if n_pts < 4:
            st.warning(f"**Under-determined.** {eqs} equations, 8 unknowns → a {free}-parameter family of "
                       "homographies passes through your targets. All blue quadrilaterals agree at the amber "
                       "points but disagree elsewhere.", icon=":material/help:")
        elif n_pts == 4:
            st.success("**Exactly determined.** 8 equations, 8 unknowns → one homography (up to scale). "
                       "Every 'random' solution is the same quadrilateral.", icon=":material/check_circle:")
        else:
            st.info("**Over-determined.** 10 equations, 8 unknowns. With perfect data they are consistent; "
                    "with real (noisy) matches no H fits all of them exactly, so we minimise ‖Ah‖ - "
                    "least squares through the SVD.", icon=":material/functions:")
        st.latex(r"\underbrace{9}_{\text{entries}} - \underbrace{1}_{\text{scale}} = 8 = "
                 r"\underbrace{4}_{\text{points}} \times \underbrace{2}_{\text{each}}")
        st.caption(f"Null space of the {2 * n_pts}×9 matrix A has dimension {max(null_dim, 1)}: one dimension is "
                   "always the harmless overall scale, the rest are genuinely different maps.")

    ui.anim_intro("Step by step: twelve candidate homographies, squeezed by each new pair",
                  "Each step adds one correspondence; every candidate is replaced by the closest homography that also "
                  "satisfies the new pair. Watch the family shrink until it is a single quadrilateral.")
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        st.plotly_chart(collapse_figure(int(st.session_state.pj_fam_seed)), key="pj_collapse")
    with right:
        ui.callout("intuition",
                   "Think of each pair as a pin. One pin lets the quadrilateral swing about wildly; with four pins "
                   "(no three in a line) it cannot move any more. A fifth pin adds nothing new - unless the data "
                   "is noisy, in which case the pins disagree slightly and we need least squares.")


# =====================================================================================
# TAB 3 - DLT
# =====================================================================================
def dlt_steps(dst4, H2):
    A = tf.dlt_rows(SRC4, dst4)
    m = float(np.abs(A).max())
    cols = ["h₁₁", "h₁₂", "h₁₃", "h₂₁", "h₂₂", "h₂₃", "h₃₁", "h₃₂", "h₃₃"]
    rows = [f"pair {i // 2 + 1} · {'x′' if i % 2 == 0 else 'y′'}" for i in range(8)]
    poly = shapes.densify(np.vstack([SRC4, SRC4[:1]]), 12)

    def build(p):
        rw = np.repeat(np.asarray(p["rows"], float), 2)[:, None]
        Z = A * rw
        Hw = tf.lerp_from_identity(H2, float(p["warp"]))
        q = plotting.mapped(Hw, poly)
        grow = np.asarray(p["arrows"], float)[:, None]
        tips = SRC4 + grow * (dst4 - SRC4)
        return [go.Heatmap(z=Z, x=cols, y=rows, zmin=-m, zmax=m, colorscale=[[0, "#2a78d6"], [0.5, "#f2f1ee"],
                                                                            [1, "#e34948"]],
                           showscale=False, xaxis="x", yaxis="y", texttemplate="%{z:.1f}", textfont=dict(size=10),
                           hovertemplate="%{y}, %{x}: %{z:.2f}<extra></extra>"),
                plotting.arrows_trace(SRC4, tips, theme.HIGHLIGHT, width=2.2, size=11).update(xaxis="x2", yaxis="y2"),
                go.Scatter(x=q[:, 0], y=q[:, 1], mode="lines", fill="toself", fillcolor=theme.rgba(theme.TRANSFORMED, 0.18),
                           line=dict(color=theme.TRANSFORMED, width=2.5), showlegend=False, hoverinfo="skip",
                           xaxis="x2", yaxis="y2")]

    P = [dict(rows=[0, 0, 0, 0], arrows=[0, 0, 0, 0], warp=0.0)]
    for k in range(4):
        r = list(P[-1]["rows"])
        a = list(P[-1]["arrows"])
        r[k], a[k] = 1, 1
        P.append(dict(rows=r, arrows=a, warp=0.0))
    P.append(dict(rows=[1, 1, 1, 1], arrows=[1, 1, 1, 1], warp=1.0))
    H = anim.hl
    info = [("The unknown h = (h₁₁, …, h₃₃)", "A h = 0,   A is still empty",
             "Each correspondence will add two rows to A - one from the x′ equation, one from the y′ equation.")]
    for k in range(4):
        (x, y), (u, v) = SRC4[k], dst4[k]
        info.append((f"Pair {k + 1}: ({x:g}, {y:g}) → ({u:.2f}, {v:.2f})",
                     f"[{H(f'{x:g}')} {H(f'{y:g}')} 1  0 0 0  {-u * x:.2f} {-u * y:.2f} {-u:.2f}] · h = 0   and   "
                     f"[0 0 0  {H(f'{x:g}')} {H(f'{y:g}')} 1  {-v * x:.2f} {-v * y:.2f} {-v:.2f}] · h = 0",
                     "Cross-multiplying x′ = (…)/(…) made the equation LINEAR in h: known numbers × unknowns = 0. "
                     f"{'Rows so far: ' + str(2 * (k + 1)) + ' of 8.' if k < 3 else 'A is complete: 8 equations, 9 unknowns.'}"))
    info.append(("Solve with the SVD, reshape to H",
                 "h = last right singular vector of A  →  H = reshape(h) / h₃₃",
                 "The square is mapped by the H we just solved for - and its corners land exactly on the four targets."))
    return [anim.Step(t_, e_, n_, build(P[i]), between=None if i == 0 else anim.morph(build, P[i - 1], P[i]))
            for i, (t_, e_, n_) in enumerate(info)]


@live.section
def dlt_section():
    st.markdown("A homography has 8 DOF and each point correspondence gives 2 equations, so **4 points** "
                "(no 3 collinear) pin it down. Move where the corners of the square land:")
    c_ctrl, c_fig, c_img = st.columns([0.9, 1.25, 1.25], gap="large")
    with c_ctrl:
        dst4 = []
        for i, (dx, dy) in enumerate(DEFAULTS4):
            st.markdown(f"<small><b>{NAMES4[i]}</b> → (x′, y′)</small>", unsafe_allow_html=True)
            r = st.columns(2)
            with r[0]:
                px = live.slider(f"x′{i + 1}", -2.4, 2.4, dx, 0.05, key=f"pj_d{i}x", label_visibility="collapsed")
            with r[1]:
                py = live.slider(f"y′{i + 1}", -2.4, 2.4, dy, 0.05, key=f"pj_d{i}y", label_visibility="collapsed")
            dst4.append([px, py])
        dst4 = np.array(dst4)
    H2 = tf.homography_dlt(SRC4, dst4)
    info2 = tf.classify(H2)
    with c_fig:
        poly = {"outline": shapes.densify(np.vstack([SRC4, SRC4[:1]]))}
        poly.update({k: v * 0.55 + np.array([-0.55, -0.7]) for k, v in shapes.house().items()})
        fig = plotting.transform_figure(
            poly, H2, extent=2.6, grid_extent=1, grid_step=0.25, height=470, basis=False,
            extra_static=[plotting.points_trace(SRC4, theme.ORIGINAL, "source corners", labels=["1", "2", "3", "4"],
                                                size=9, showlegend=False)],
            extra_dynamic=lambda Ht, t: [plotting.points_trace(tf.apply(Ht, SRC4), theme.TRANSFORMED, "targets",
                                                               labels=["1'", "2'", "3'", "4'"], size=12)])
        st.plotly_chart(fig, key="pj_fig2")
    with c_img:
        img = images.poster(240)
        Wc = 360
        to_px = lambda P: np.c_[(P[:, 0] + 2.5) / 5 * (Wc - 1), (2.5 - P[:, 1]) / 5 * (Wc - 1)]
        h, w = img.shape[:2]
        img_corners = np.array([[0, h - 1], [w - 1, h - 1], [w - 1, 0], [0, 0]], np.float32)  # BL BR TR TL
        Hpx = cv2.getPerspectiveTransform(img_corners, to_px(dst4).astype(np.float32))
        warped = cv2.warpPerspective(img, Hpx, (Wc, Wc), flags=cv2.INTER_LINEAR, borderValue=(245, 245, 245))
        st.image(ui.datauri(warped), caption="The same H applied to an image (cv2.warpPerspective)", width="stretch")
    a, b = st.columns([1, 1.2], gap="large")
    with a:
        ui.show_matrix("H", H2, 3, {(2, 0): theme.MAGENTA, (2, 1): theme.MAGENTA})
        st.markdown(f"Detected class: **{info2['name']}**")
        st.caption("Make the 4 targets a parallelogram (e.g. P₃ = P₂ + P₄ − P₁) and the pink entries drop to 0: "
                   "the homography becomes affine.")
    with b:
        ui.callout("intuition",
                   "Four corners of a square can be sent to **any** four points (no three on a line) - that is the "
                   "freedom of 8 DOF. A parallelogram target gives an affine map; a general quadrilateral needs the "
                   "perspective row.")

    st.subheader("The Direct Linear Transform, step by step", divider="gray")
    ui.anim_intro("Animated: build A two rows at a time, then solve",
                  "Uses your four target points. Left: the 8×9 matrix A filling up (blue = negative, red = positive). "
                  "Right: the correspondence each step adds; the last step maps the square with the solved H.")
    base = make_subplots(rows=1, cols=2, column_widths=[0.56, 0.44], horizontal_spacing=0.08)
    base.add_trace(plotting.grid_trace(None, 2.5, 0.5, theme.GRID_ORIGINAL), row=1, col=2)
    sq_line = np.vstack([SRC4, SRC4[:1]])
    base.add_trace(plotting.line_trace(sq_line, theme.ORIGINAL, "source", dash="dash", showlegend=False), row=1, col=2)
    base.add_trace(plotting.points_trace(SRC4, theme.ORIGINAL, "src", labels=["1", "2", "3", "4"], size=9,
                                         showlegend=False, textpos="bottom left"), row=1, col=2)
    base.add_trace(plotting.points_trace(dst4, theme.TRANSFORMED, "tgt", labels=["1′", "2′", "3′", "4′"], size=11,
                                         showlegend=False, symbol="diamond"), row=1, col=2)
    base.update_layout(height=600, showlegend=False)
    base.update_yaxes(autorange="reversed", row=1, col=1)
    base.update_xaxes(side="bottom", tickfont=dict(size=11), row=1, col=1)
    base.update_xaxes(range=[-2.6, 2.6], zeroline=True, zerolinecolor=theme.AXIS, gridcolor=theme.GRID, row=1, col=2)
    base.update_yaxes(range=[-2.6, 2.6], scaleanchor="x2", scaleratio=1, zeroline=True, zerolinecolor=theme.AXIS,
                      gridcolor=theme.GRID, row=1, col=2)
    st.plotly_chart(anim.step_player(base, dlt_steps(dst4, H2), n_sub=12, wrap_at=120), key="pj_dlt_steps")

    s1, s2 = st.columns(2, gap="large")
    with s1:
        st.markdown("**Step 1 · cross-multiply** the division for one pair (xᵢ, yᵢ) → (x′ᵢ, y′ᵢ):")
        st.latex(r"x'_i = \frac{h_{11}x_i + h_{12}y_i + h_{13}}{h_{31}x_i + h_{32}y_i + h_{33}} \;\Rightarrow\; "
                 r"x'_i\,(h_{31}x_i + h_{32}y_i + h_{33}) = h_{11}x_i + h_{12}y_i + h_{13}")
        st.markdown("**Step 2 · rearrange** - now it is *linear* in the unknown h's:")
        st.latex(r"\begin{aligned} h_{11}x_i + h_{12}y_i + h_{13} - h_{31}x_ix'_i - h_{32}y_ix'_i - h_{33}x'_i &= 0 \\ "
                 r"h_{21}x_i + h_{22}y_i + h_{23} - h_{31}x_iy'_i - h_{32}y_iy'_i - h_{33}y'_i &= 0 \end{aligned}")
        st.markdown("**Step 3 · write as rows** acting on h = (h₁₁, h₁₂, …, h₃₃)ᵀ and stack all four pairs: "
                    "**A h = 0** with A of size 8 × 9.")
        st.latex(r"\begin{bmatrix} x_i & y_i & 1 & 0 & 0 & 0 & -x_ix'_i & -y_ix'_i & -x'_i \\ "
                 r"0 & 0 & 0 & x_i & y_i & 1 & -x_iy'_i & -y_iy'_i & -y'_i \end{bmatrix}\mathbf{h} = \mathbf{0}")
    with s2:
        A_raw = tf.dlt_rows(SRC4, dst4)
        _, S_raw, Vt_raw = np.linalg.svd(A_raw)          # full_matrices: Vt is 9x9 even though A is 8x9
        S_raw = np.r_[S_raw, np.zeros(9 - len(S_raw))]    # an 8x9 matrix has a 9th singular value of 0
        h_raw = Vt_raw[-1]
        cols = ["h11", "h12", "h13", "h21", "h22", "h23", "h31", "h32", "h33"]
        st.markdown("Your A, with the current target points:")
        st.dataframe(pd.DataFrame(np.round(A_raw, 2), columns=cols,
                                  index=[f"pair {i // 2 + 1}, {'x′' if i % 2 == 0 else 'y′'}" for i in range(8)]),
                     width="stretch")
        st.markdown("**Step 4 · SVD.** A = UΣVᵀ. The solution is the last column of V (last row of Vᵀ) - the "
                    "direction that A squashes the most:")
        sv = go.Figure(go.Bar(x=[f"σ{i + 1}" for i in range(len(S_raw))], y=S_raw,
                              marker_color=[theme.TRANSFORMED] * (len(S_raw) - 1) + [theme.HIGHLIGHT],
                              hovertemplate="%{x} = %{y:.3g}<extra></extra>"))
        sv.update_layout(height=200, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="singular value")
        st.plotly_chart(sv, key="pj_sv")
        st.caption(f"σ₈ = {S_raw[-2]:.3g} > 0 but σ₉ = 0: an 8×9 matrix always has one direction it sends "
                   "exactly to zero - that direction is h. With more than 4 (noisy) points σ₉ becomes small but "
                   "not zero, and the same column gives the least-squares answer.")
        st.markdown("**Step 5 · reshape and normalise** (h₃₃ = 1):")
        hh = h_raw.reshape(3, 3)
        st.latex(r"H = \frac{1}{h_{33}}\begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ "
                 r"h_{31} & h_{32} & h_{33} \end{bmatrix} = " +
                 ui.tex_matrix(tf.normalize_h(hh) if abs(hh[2, 2]) > 1e-12 else hh, 3))


# =====================================================================================
# TAB 4 - rectify
# =====================================================================================
@live.section
def rectify_section():
    st.markdown("A photo of a flat page is a homography of the page. Click-free version: tell the app where the "
                "4 page corners are, and it maps them back to a rectangle.")
    up = st.file_uploader("Use your own photo of a document / poster / floor tile (optional)",
                          type=["png", "jpg", "jpeg"], key="pj_up")
    true_frac = [(0.22, 0.12), (0.83, 0.2), (0.9, 0.86), (0.08, 0.75)]
    photo = None
    if up is not None:
        try:
            photo = images.load_upload(up, 520)
        except Exception:
            st.error("Could not read that image file - showing the sample photo instead.", icon=":material/error:")
    if photo is not None:
        init = [(0.15, 0.15), (0.85, 0.15), (0.85, 0.85), (0.15, 0.85)]
    else:
        photo, _ = images.photo_of_document(images.document(320), true_frac, canvas=(380, 500))
        init = true_frac
    own = up is not None and photo is not None and init[0] == (0.15, 0.15)
    ph, pw = photo.shape[:2]
    c_ctrl, c_in, c_out = st.columns([0.9, 1.3, 1.0], gap="large")
    with c_ctrl:
        lbl = ["TL", "TR", "BR", "BL"]
        corners = []
        for i, (fx, fy) in enumerate(init):
            r = st.columns(2)
            with r[0]:
                cx = live.slider(f"{lbl[i]} x", 0.0, 1.0, fx, 0.005, key=f"pj_r{i}x{own}")
            with r[1]:
                cy = live.slider(f"{lbl[i]} y", 0.0, 1.0, fy, 0.005, key=f"pj_r{i}y{own}")
            corners.append([cx, cy])
        corners = np.array(corners) * np.array([pw - 1, ph - 1])
        aspect = live.slider("output height / width", 0.5, 2.0, 1.25, 0.05, key="pj_aspect",
                             help="A4 paper is about 1.41; the sample page is 1.25.")
    out_w = 300
    out_h = int(out_w * aspect)
    rect = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], float)
    Hr = tf.homography_dlt(corners, rect)
    ok = np.all(np.isfinite(Hr))
    with c_in:
        st.image(ui.datauri(images.draw_polygon(photo, corners, labels=lbl)), caption="Photo with the 4 chosen corners",
                 width="stretch")
    with c_out:
        if ok:
            rectified = cv2.warpPerspective(photo, Hr, (out_w, out_h), flags=cv2.INTER_LINEAR)
            st.image(ui.datauri(rectified), caption="Rectified: H maps the corners to a rectangle", width="stretch")
        else:
            st.warning("These corners do not define a homography (three of them are on one line).",
                       icon=":material/error:")
    ui.callout("practice",
               "This is how phone **document scanners** work (corners come from an edge/quad detector), how a "
               "**bird's-eye view** of a road or factory floor is made for planning (inverse perspective mapping), "
               "and how you calibrate a camera to a **table plane** so a robot arm can turn pixel clicks into "
               "(x, y) positions on the table.")


# =====================================================================================
# TAB 5 - invariants
# =====================================================================================
def invariants_tab():
    st.markdown("Using the homography of tab 3 (change the targets there and come back):")
    H2 = tf.homography_dlt(SRC4, current_dst4())
    line_pts = np.array([[-2.0, -1.2], [-0.8, -0.6], [0.4, 0.0], [2.0, 0.8]])      # 4 collinear points
    q = tf.apply(H2, line_pts)
    left, right = st.columns([1.3, 1], gap="large")
    mid = line_pts[[0, -1]].mean(axis=0)
    with left:
        fig = plotting.base_figure(3.2, height=430)
        fig.add_trace(plotting.grid_trace(None, 3, 0.5, theme.GRID_ORIGINAL))
        fig.add_trace(plotting.grid_trace(H2, 3, 0.5, theme.GRID_TRANSFORMED))
        fig.add_trace(plotting.line_trace(line_pts[[0, -1]], theme.ORIGINAL, "line before", dash="dash"))
        fig.add_trace(plotting.points_trace(line_pts, theme.ORIGINAL, "A B C D", labels=list("ABCD"), size=9))
        fig.add_trace(plotting.line_trace(q[[0, -1]], theme.TRANSFORMED, "line after", width=2.5))
        fig.add_trace(plotting.points_trace(q, theme.TRANSFORMED, "A' B' C' D'", labels=["A'", "B'", "C'", "D'"]))
        fig.add_trace(plotting.points_trace(tf.apply(H2, [mid]), theme.HIGHLIGHT, "image of midpoint of AD",
                                            size=12, symbol="diamond"))
        fig.add_trace(plotting.points_trace([q[[0, -1]].mean(axis=0)], theme.MAGENTA, "midpoint of A'D'",
                                            size=10, symbol="x"))
        st.plotly_chart(fig, key="pj_inv")
    with right:
        cross2 = lambda u, v: u[0] * v[1] - u[1] * v[0]
        finite = np.isfinite(q).all()
        coll = abs(cross2(q[1] - q[0], q[3] - q[0])) + abs(cross2(q[2] - q[0], q[3] - q[0])) if finite else np.nan
        cr0 = tf.cross_ratio(*line_pts)
        cr1 = tf.cross_ratio(*q) if finite else np.nan
        ratio0 = np.linalg.norm(line_pts[1] - line_pts[0]) / np.linalg.norm(line_pts[2] - line_pts[1])
        ratio1 = np.linalg.norm(q[1] - q[0]) / np.linalg.norm(q[2] - q[1]) if finite else np.nan
        mid_err = np.linalg.norm(tf.apply(H2, [mid])[0] - q[[0, -1]].mean(axis=0)) if finite else np.nan
        st.dataframe({
            "quantity": ["A', B', C', D' still collinear?", "cross-ratio (A,B;C,D)", "length ratio AB / BC",
                         "midpoint maps to midpoint?"],
            "before": ["yes", f"{cr0:.4f}", f"{ratio0:.4f}", "-"],
            "after": ["yes" if coll < 1e-6 else f"no ({coll:.1e})", f"{cr1:.4f}", f"{ratio1:.4f}",
                      f"off by {mid_err:.3f}"],
        }, hide_index=True, width="stretch")
        st.latex(r"(A,B;C,D) = \frac{|AC|\,|BD|}{|BC|\,|AD|}")
        ui.callout("intuition",
                   "Railway sleepers in a photo are equally spaced in the world but bunch up towards the horizon - "
                   "so simple length ratios (and midpoints!) are **not** kept. The cross-ratio, a 'ratio of ratios' "
                   "of four collinear points, is the one number a homography cannot change.")


with tabs[0]:
    if tabs[0].open:
        perspective_section()
with tabs[1]:
    if tabs[1].open:
        why_four_section()
with tabs[2]:
    if tabs[2].open:
        dlt_section()
with tabs[2]:
    if tabs[2].open:
        ui.whys([
            ("Why is H only defined up to scale?",
             "Multiply every entry by λ: numerator and denominator of x′ = (h₁₁x + h₁₂y + h₁₃)/(h₃₁x + h₃₂y + h₃₃) "
             "both get λ, which cancels. So H and 2H (or −5H) move every point to the same place - one of the 9 "
             "numbers is redundant."),
            ("Why take the singular vector of the smallest singular value?",
             "We want A h = 0, but h = 0 is a useless answer, so we ask for the h of length 1 that makes ‖A h‖ as "
             "small as possible. ‖A v‖ for a unit vector v is largest along the first right singular vector and "
             "smallest along the last one - so the last column of V is the best h. With exact data ‖A h‖ is 0; "
             "with noisy data it is the least-squares answer."),
            ("Why not just set h₃₃ = 1 first and solve an 8×8 system?",
             "You can, and it often works - but if the true homography has h₃₃ = 0 (it happens when the origin is "
             "mapped to infinity) the method breaks. The SVD route treats all 9 entries equally and never fails "
             "that way; we normalise only at the end."),
            ("Why does cross-multiplying make the problem linear?",
             "The unknowns appear in both numerator and denominator, which is non-linear. Multiplying both sides by "
             "the denominator moves them into a sum of terms 'known number × one unknown' - the definition of a "
             "linear equation."),
            ("Why must no three of the four points be collinear?",
             "Three collinear source points only say how one line moves; the equations they give become dependent, "
             "A loses rank and the null space grows beyond one dimension - many homographies fit again."),
            ("Why normalise the points before the DLT (Hartley)?",
             "Pixel coordinates like 1000 make the entries of A range from 1 to 10⁶ (x·x′). Tiny rounding errors in the "
             "big entries swamp the small ones. Centring the points and scaling them to an average distance of √2 "
             "keeps all entries similar in size - a much more accurate SVD. The answer is then 'un-normalised'."),
        ], title="Questions students ask about the DLT")
with tabs[3]:
    if tabs[3].open:
        rectify_section()
with tabs[4]:
    if tabs[4].open:
        invariants_tab()

st.subheader("What changes and what stays", divider="gray")
a, b = st.columns([1.1, 1], gap="large")
with a:
    ui.preserved_badges("projective")
    st.write("")
    ui.callout("takeaway",
               "A homography is exactly the class of maps that **send lines to lines**. Every picture of a plane "
               "(a page, a wall, a floor, a marker), and every pair of pictures taken by a camera that only "
               "*rotates* about its centre (panoramas), is related by one.")
with b:
    ui.callout("pitfall",
               "Forgetting to **divide by w'** is the classic bug: H x̃ gives (x̃', ỹ', w̃'), not (x', y'). "
               "Also, H is only defined up to scale - compare two homographies only after normalising them "
               "(e.g. h₃₃ = 1).")

st.subheader("Affine vs projective at a glance", divider="gray")
st.markdown(
    "| | Affine | Projective |\n|---|---|---|\n"
    "| Matrix | 3×3 with last row (0, 0, 1) | full 3×3, up to scale |\n"
    "| Degrees of freedom | 6 | 8 |\n"
    "| Constraints per point | 2 | 2 |\n"
    "| Minimum correspondences | 3 (non-collinear) | 4 (no three collinear) |\n"
    "| Parallel lines | preserved | not necessarily preserved |\n"
    "| Straight lines | preserved | preserved |\n"
    "| Perspective effects | cannot model general perspective | models perspective |\n"
    "| Typical estimation | linear system (M⁻¹ or least squares) | DLT + SVD |\n"
    "| Relationship | special case h₃₁ = h₃₂ = 0, h₃₃ = 1 | contains affine |")

ui.whys([
    ("Why can't an affine map model a photo of a tilted rectangle?",
     "An affine map keeps parallel lines parallel, so a rectangle can only become a parallelogram. In a "
     "tilted photo the far edge looks shorter than the near edge - a general quadrilateral. Only the "
     "position-dependent division by w′ can shrink far parts more than near parts."),
    ("Why do parallel lines meet in a photo but not in the world?",
     "Parallel lines meet at a point at infinity, (x, y, 0). An affine map keeps w = 0, so the meeting point "
     "stays at infinity. A homography's bottom row can turn w = 0 into w′ ≠ 0, i.e. a finite **vanishing "
     "point** you can see in the picture."),
    ("Why are straight lines still straight?",
     "A line is the set of points with l·x̃ = 0. After x̃′ = Hx̃ the same points satisfy (H⁻ᵀl)·x̃′ = 0 - again "
     "a linear equation, i.e. a line (with coefficients l′ = H⁻ᵀl). Lines map to lines, only their spacing "
     "and directions change."),
    ("Why is affine a special case of projective?",
     "Set h₃₁ = h₃₂ = 0 and h₃₃ = 1. Then w′ = 1 for every point, the division does nothing, and the formula "
     "is exactly x′ = ax + by + tₓ, y′ = cx + dy + t_y. Every affine map is a homography, not the other way round."),
])

st.divider()
ui.mcq("pj1", "In a photo of a straight road, the two edges meet at a point on the horizon. Which "
              "transformation can produce this?",
       ["Similarity", "Affine", "Projective (homography)", "Any of them"], 2,
       "Only the perspective row (h₃₁, h₃₂) can bring parallel lines' point at infinity (w = 0) to a finite "
       "vanishing point. Affine maps keep w = 0 → 0, so parallel lines stay parallel.")
ui.mcq("pj2", "How many point correspondences are needed to estimate a homography?",
       ["2", "3", "4", "8"], 2,
       "8 DOF, 2 equations per correspondence → 4 points, with no 3 of them collinear.")

ui.lesson_nav(__file__)
