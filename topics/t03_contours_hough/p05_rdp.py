import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import anim, live, plotting, theme, ui
from core import contour_ui as cui
from core import contours as K

ui.lesson_header(
    __file__,
    "A contour may hold a thousand points while the shape is just a rectangle. <b>Polygonal approximation</b> keeps "
    "only the points that matter. The Ramer-Douglas-Peucker algorithm does it with one idea, used recursively: "
    "<i>draw a straight line, find the point that strays furthest, keep it only if it strays more than ε.</i>",
)

# ------------------------------------------------------------------ the objective
st.subheader("The goal: fewer points, bounded error", divider="gray")
g1, g2 = st.columns([1, 1.2], gap="large")
with g1:
    st.latex(r"C = \{p_0, \dots, p_{N-1}\} \;\longrightarrow\; P = \{q_0, \dots, q_{M-1}\}, \qquad M \ll N")
    st.latex(r"\text{minimise } M \quad \text{subject to} \quad \text{approximation error} \le \epsilon")
with g2:
    st.markdown("**RDP in five lines** - for a stretch p₀ … pₙ of the contour:\n"
                "1. connect the **first and last** point with a straight line;\n"
                "2. compute the **perpendicular distance** of every point in between;\n"
                "3. find the point with the **largest** distance d_max;\n"
                "4. if d_max ≤ ε: the line is good enough - **drop** all points in between;\n"
                "5. otherwise **keep** that point and do the same on both halves.")

# ------------------------------------------------------------------ derivation 1: the line through two points
P0, PN = np.array([1.0, 1.0]), np.array([9.0, 4.0])
A_, B_, C_ = K.line_coeffs(P0, PN)
EXT = (-0.5, 10.5, -1.0, 6.5)


def _line_pts(f=1.0):
    d = PN - P0
    a, b = P0 - 0.12 * d, PN + 0.12 * d
    mid = (a + b) / 2
    return np.array([mid + f * (a - mid), mid + f * (b - mid)])


@st.cache_resource(show_spinner=False)
@live.full_quality()
def line_figure():
    d = PN - P0
    n = np.array([A_, B_])
    ang_d = np.arctan2(d[1], d[0])

    def rot(v, a):
        c, s = np.cos(a), np.sin(a)
        return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])

    def state(line_f=0.0, d_on=0.0, n_vec=None, p_t=None):
        L = _line_pts(line_f) if line_f > 0 else np.array([P0, P0])
        nv = d * 0 if n_vec is None else n_vec
        if p_t is None:
            p, size = P0, 0
        else:
            p, size = P0 + p_t * d, 12
        return [cui.scatter(L[:, 0], L[:, 1], theme.TRANSFORMED, mode="lines", width=2.5),
                plotting.arrow_trace(P0, P0 + d_on * d, theme.E1, "d", showlegend=False, size=13 if d_on else 0),
                plotting.arrow_trace(P0, P0 + nv, theme.E2, "n", showlegend=False, size=13 if n_vec is not None else 0),
                cui.scatter([P0[0], p[0]], [P0[1], p[1]], theme.HIGHLIGHT, mode="lines", width=2, dash="dot"),
                cui.scatter([p[0]], [p[1]], theme.HIGHLIGHT, size=size)]

    nv = 0.45 * n
    steps = [
        anim.Step("Two end points of the stretch", f"p₀ = ({P0[0]:g}, {P0[1]:g}),  pₙ = ({PN[0]:g}, {PN[1]:g})",
                  "RDP's first approximation is the straight line through them.", state()),
        anim.Step("Direction of the line", f"d = pₙ − p₀ = ({d[0]:g}, {d[1]:g})",
                  "Walking along d from p₀ stays on the line.", state(d_on=1)),
        anim.Step("Turn d by 90° to get a normal", f"n = (−d<sub>y</sub>, d<sub>x</sub>) = (y₀ − yₙ, xₙ − x₀) = "
                  f"{anim.hl(f'({A_:g}, {B_:g}) = (A, B)')}",
                  "A vector at right angles to the line. (Drawn at 45 % length.)", state(d_on=1, n_vec=nv),
                  between=lambda t: state(d_on=1, n_vec=0.45 * np.hypot(*n) / np.hypot(*d) *
                                          rot(d, t * (np.arctan2(n[1], n[0]) - ang_d)))),
        anim.Step("Every point p of the line: p − p₀ is perpendicular to n",
                  "n · (p − p₀) = 0   ⇔   A x + B y − (A x₀ + B y₀) = 0",
                  "Slide p along the line: the dotted vector p − p₀ always stays at 90° to n.",
                  state(1, 1, nv, 0.75), between=lambda t: state(t, 1, nv, 0.75 * t)),
        anim.Step("Name the constant", f"C = −(A x₀ + B y₀) = {anim.hl('x₀ yₙ − xₙ y₀')} = {C_:g}",
                  f"So the line is ({A_:g}) x + ({B_:g}) y + ({C_:g}) = 0 - exactly the notes' (34)-(37).",
                  state(1, 1, nv, 0.3)),
        anim.Step("Check: pₙ is on it too - and it is a cross product",
                  f"A xₙ + B yₙ + C = {A_ * PN[0] + B_ * PN[1] + C_:g} ✓     (x₀, y₀, 1) × (xₙ, yₙ, 1) = (A, B, C)",
                  "The homogeneous line through two points from topic 1: l = p × q.", state(1, 1, nv, 1.0)),
    ]
    base = plotting.base_figure(EXT, height=430, legend=False)
    base.add_trace(cui.scatter([P0[0], PN[0]], [P0[1], PN[1]], "#7d7b76", size=10, mode="markers+text").update(
        text=["p₀", "pₙ"], textposition="top left", textfont=dict(size=14)))
    return anim.step_player(base, steps, n_sub=14, hold=16, wrap_at=58)


@st.cache_resource(show_spinner=False)
@live.full_quality()
def distance_figure():
    pi = np.array([4.0, 5.2])
    n = np.array([A_, B_])
    nh = n / np.hypot(*n)
    v = pi - P0
    s = float(v @ nh)                       # signed distance
    foot = pi - s * nh
    yl = -(A_ * pi[0] + C_) / B_            # the line's y at x = x_i (for the vertical distance)

    def state(v_f=0.0, n_f=0.0, stem_f=0.0, vert_f=0.0):
        stem = np.array([pi, pi + stem_f * (foot - pi)])
        proj = np.array([P0, P0 + stem_f * s * nh])
        vert = np.array([pi, pi + vert_f * (np.array([pi[0], yl]) - pi)])
        return [plotting.arrow_trace(P0, P0 + v_f * v, theme.ORIGINAL, "v", showlegend=False, size=12 if v_f else 0),
                plotting.arrow_trace(P0, P0 + n_f * 1.4 * nh, theme.E2, "n̂", showlegend=False, size=12 if n_f else 0),
                cui.scatter(stem[:, 0], stem[:, 1], theme.HIGHLIGHT, mode="lines", width=4),
                cui.scatter(proj[:, 0], proj[:, 1], theme.HIGHLIGHT, mode="lines", width=7, opacity=0.55),
                cui.scatter(vert[:, 0], vert[:, 1], theme.OUTLIER, mode="lines", width=2.5, dash="dash")]

    d = abs(A_ * pi[0] + B_ * pi[1] + C_) / np.hypot(A_, B_)
    dv = abs(pi[1] - yl)
    steps = [
        anim.Step("A point p_i between the end points", f"p<sub>i</sub> = ({pi[0]:g}, {pi[1]:g})",
                  "How far is it from the line? 'Far' means the SHORTEST distance: straight across.", state()),
        anim.Step("Vector from p₀ to p_i", "v = p<sub>i</sub> − p₀", "Any point of the line works as an anchor.",
                  state(v_f=1)),
        anim.Step("Unit normal", "n̂ = (A, B) / √(A² + B²)",
                  "Dividing by the length makes the normal 1 unit long - otherwise we would measure in 'units of |n|'.",
                  state(1, 1)),
        anim.Step("Project v onto n̂: that is the distance", f"d<sub>i</sub> = |v · n̂| = {anim.hl(f'{d:.3f}')}",
                  "The thick amber bar on n̂ and the amber gap from p_i to the line have the same length.",
                  state(1, 1, 1)),
        anim.Step("Write it out", "v·(A, B) = A x<sub>i</sub> + B y<sub>i</sub> − (A x₀ + B y₀) = "
                  f"A x<sub>i</sub> + B y<sub>i</sub> + C   ⇒   d<sub>i</sub> = {anim.hl('|A x<sub>i</sub> + B y<sub>i</sub> + C| / √(A² + B²)')}",
                  "The notes' equation (40). No angles, no square roots per point except one per line.", state(1, 1, 1)),
        anim.Step("Why not the vertical distance?", f"vertical gap = {anim.hl(f'{dv:.3f}', theme.OUTLIER)}  >  "
                  f"perpendicular d = {d:.3f}",
                  "The vertical gap depends on the line's slope (it explodes for steep lines). The perpendicular one "
                  "does not: rotate the picture and d stays the same.", state(1, 1, 1, 1)),
    ]
    base = plotting.base_figure(EXT, height=430, legend=False)
    L = _line_pts()
    base.add_trace(cui.scatter(L[:, 0], L[:, 1], theme.TRANSFORMED, mode="lines", width=2.5))
    base.add_trace(cui.scatter([P0[0], PN[0], pi[0]], [P0[1], PN[1], pi[1]], "#7d7b76", size=10,
                               mode="markers+text").update(text=["p₀", "pₙ", "p_i"], textposition="top left",
                                                           textfont=dict(size=14)))
    return anim.step_player(base, steps, n_sub=14, hold=16, wrap_at=58)


st.subheader("Step by step: the two formulas RDP needs", divider="gray")
c1, c2 = st.columns(2, gap="large")
with c1:
    ui.anim_intro("Step 1: the line A x + B y + C = 0 through p₀ and pₙ")
    st.plotly_chart(line_figure(), key="rd_line")
    st.latex(r"A = y_0 - y_n,\quad B = x_n - x_0,\quad C = x_0 y_n - x_n y_0")
with c2:
    ui.anim_intro("Step 2: the perpendicular distance of p_i")
    st.plotly_chart(distance_figure(), key="rd_dist")
    st.latex(r"d_i = \frac{|(y_0 - y_n)x_i + (x_n - x_0)y_i + x_0 y_n - x_n y_0|}{\sqrt{(y_0 - y_n)^2 + (x_n - x_0)^2}}")

# ------------------------------------------------------------------ curves
CURVES = ["Wavy line", "Coastline (random walk)", "Noisy corner"]


@st.cache_resource(show_spinner=False, max_entries=8)
def curve(name: str, n: int = 70) -> np.ndarray:
    rng = np.random.default_rng(5)
    if name == "Wavy line":
        x = np.linspace(0, 12, n)
        return np.c_[x, 3 + 1.6 * np.sin(0.9 * x) + 0.5 * np.sin(2.7 * x) + rng.normal(0, 0.04, n)]
    if name == "Coastline (random walk)":
        y = np.cumsum(rng.normal(0, 0.35, n))
        y = np.convolve(y, np.ones(3) / 3, mode="same")
        y[0], y[-1] = y[1], y[-2]
        return np.c_[np.linspace(0, 12, n), 3 + (y - y.mean()) * 0.8]
    pts = np.array([[0.3, 1.0], [7.0, 1.2], [11.5, 5.6]])
    seg = np.r_[np.linspace(0, 1, n // 2, endpoint=False), np.linspace(1, 2, n - n // 2)]
    P = np.array([pts[int(min(s_, 1.999))] + (s_ - int(min(s_, 1.999))) * (pts[int(min(s_, 1.999)) + 1] -
                                                                            pts[int(min(s_, 1.999))]) for s_ in seg])
    return P + rng.normal(0, 0.08, P.shape)


@st.cache_resource(show_spinner=False, max_entries=64)
def optimal(name: str, eps: float) -> list[int]:
    return K.optimal_vertices(curve(name), eps)


def tubes(P, idx, eps):
    """The ε-wide band around every kept segment, as one NaN-separated filled trace."""
    xs, ys = [], []
    for a, b in zip(idx, idx[1:]):
        p, q = P[a], P[b]
        d = q - p
        L = np.hypot(*d)
        if L < 1e-9:
            continue
        nn = np.array([-d[1], d[0]]) / L * eps
        quad = np.array([p + nn, q + nn, q - nn, p - nn, p + nn])
        xs += list(quad[:, 0]) + [np.nan]
        ys += list(quad[:, 1]) + [np.nan]
    return cui.scatter(xs, ys, theme.HIGHLIGHT, mode="lines", width=0.5, opacity=0.9).update(
        fill="toself", fillcolor=theme.rgba(theme.HIGHLIGHT, 0.13))


@live.section
def explore_section():
    st.subheader("RDP live: drag ε", divider="gray")
    c_ctrl, c_fig, c_stats = st.columns([0.8, 1.9, 0.9], gap="large")
    with c_ctrl:
        name = st.selectbox("Curve", CURVES, key="rd_curve")
        eps = live.slider("tolerance ε", 0.02, 2.0, 0.3, 0.02, key="rd_eps")
        show_opt = st.toggle("also show the OPTIMAL polygon", False, key="rd_opt")
    P = curve(name)
    idx, events, ops = K.rdp(P, eps, trace=True)
    cv_n = len(cv2.approxPolyDP(P.astype(np.float32).reshape(-1, 1, 2), eps, False))
    opt = optimal(name, round(eps, 2))
    with c_fig:
        fig = plotting.base_figure((-0.5, 12.5, -0.5, 7.0), height=430, legend=True)
        Q = P[idx]
        O = P[opt]
        fig.add_traces([
            tubes(P, idx, eps).update(name="ε-band", showlegend=True),
            cui.scatter(P[:, 0], P[:, 1], theme.ORIGINAL, size=5, mode="lines+markers", width=1, name="contour points",
                        showlegend=True),
            cui.scatter(O[:, 0], O[:, 1], theme.E2 if show_opt else "rgba(0,0,0,0)", mode="lines+markers", size=8,
                        width=2.5, dash="dash", name="optimal" if show_opt else " ", showlegend=show_opt),
            cui.scatter(Q[:, 0], Q[:, 1], theme.TRANSFORMED, mode="lines+markers", size=10, width=3, name="RDP",
                        showlegend=True),
        ])
        st.plotly_chart(fig, key="rd_fig")
    with c_stats:
        st.metric("points", f"{len(P)} → {len(idx)}", border=True)
        st.metric("cv2.approxPolyDP", f"{cv_n} vertices", border=True, help="OpenCV's implementation of the same "
                  "algorithm, on the same points")
        st.metric("optimal (fewest possible)", f"{len(opt)} vertices", border=True,
                  help="Shortest path through all 'allowed' shortcuts - see below")
        st.metric("largest error", f"{K.max_deviation(P, idx):.3f} ≤ ε", border=True)
        st.caption(f"{ops} distance evaluations")


explore_section()


# ------------------------------------------------------------------ recursion replay
def recursion_tree(events, n_points):
    ev = [e for e in events if e["i1"] - e["i0"] >= 2]
    par = []
    for e in ev:
        best = -1
        for j, f in enumerate(ev):
            if f is not e and f["split"] and f["i0"] <= e["i0"] and e["i1"] <= f["i1"] and \
                    (f["i1"] - f["i0"]) > (e["i1"] - e["i0"]):
                if best < 0 or (ev[best]["i1"] - ev[best]["i0"]) > (f["i1"] - f["i0"]):
                    best = j
        par.append(best)
    depth = []
    for i in range(len(ev)):
        d, p = 0, par[i]
        while p >= 0:
            d, p = d + 1, par[p]
        depth.append(d)
    xs = [(e["i0"] + e["i1"]) / 2 / max(n_points - 1, 1) for e in ev]
    return ev, par, depth, xs


@st.cache_resource(show_spinner=False, max_entries=24)
@live.full_quality()
def replay_figure(name: str, eps: float):
    P = curve(name)
    n = len(P)
    idx, events, _ = K.rdp(P, eps, trace=True)
    ev = [e for e in events if e["i1"] - e["i0"] >= 2]
    MAXS = 16
    shown = ev[:MAXS]

    def approx(kept):
        ks = sorted(kept)
        return np.c_[np.interp(np.arange(n), ks, P[ks, 0]), np.interp(np.arange(n), ks, P[ks, 1])]

    def state(e, kept, show_stems=True, chord_on=True):
        sx, sy = [], []
        if e is not None:
            p, q = P[e["i0"]], P[e["i1"]]
            dvec = q - p
            L2 = float(dvec @ dvec)
        for i in range(n):
            if e is not None and e["i0"] < i < e["i1"] and show_stems:
                t = float((P[i] - p) @ dvec) / L2
                f = p + t * dvec
                sx += [P[i][0], f[0], np.nan]
                sy += [P[i][1], f[1], np.nan]
            else:
                sx += [P[i][0], P[i][0], np.nan]
                sy += [P[i][1], P[i][1], np.nan]
        if e is not None:
            p, q = P[e["i0"]], P[e["i1"]]
            dd = q - p
            nn = np.array([-dd[1], dd[0]]) / max(np.hypot(*dd), 1e-9) * eps
            band = np.array([p + nn, q + nn, q - nn, p - nn, p + nn])
            chord = np.array([p, q])
            k = e["k"]
            km = P[k] if e["split"] else (p + q) / 2
            ksz = 16 if e["split"] else 0
        else:
            band = np.array([P[0]] * 5)
            chord = np.array([P[0], P[-1]]) if chord_on else np.array([P[0], P[0]])
            km, ksz = P[0], 0
        A = approx(kept)
        vs = np.array([11.0 if i in kept else 0.0 for i in range(n)])
        return [cui.scatter(band[:, 0], band[:, 1], theme.HIGHLIGHT, mode="lines", width=0.5).update(
                    fill="toself", fillcolor=theme.rgba(theme.HIGHLIGHT, 0.15)),
                cui.scatter(sx, sy, theme.OUTLIER, mode="lines", width=1.5),
                cui.scatter(chord[:, 0], chord[:, 1], theme.HIGHLIGHT, mode="lines", width=2, dash="dash"),
                cui.scatter(A[:, 0], A[:, 1], theme.TRANSFORMED, mode="lines", width=3),
                cui.scatter(P[:, 0], P[:, 1], theme.TRANSFORMED, mode="markers").update(
                    marker=dict(size=vs, color=theme.TRANSFORMED, line=dict(color="white", width=1))),
                cui.scatter([km[0]], [km[1]], theme.HIGHLIGHT, size=ksz, symbol="star")]

    kept = {0, n - 1}
    steps = [anim.Step("Start: keep the two end points", f"N = {n} points, ε = {eps:.2f}",
                       "The first approximation is ONE straight line.", state(None, set(kept)))]
    for j, e in enumerate(shown):
        depth = 0
        for f in ev:
            if f["split"] and f["i0"] <= e["i0"] and e["i1"] <= f["i1"] and (f["i1"] - f["i0"]) > (e["i1"] - e["i0"]):
                depth += 1
        tag = f"segment [{e['i0']}, {e['i1']}] (recursion depth {depth})"
        if e["split"]:
            kept.add(e["k"])
            steps.append(anim.Step(tag, f"d<sub>max</sub> = d<sub>{e['k']}</sub> = {e['dmax']:.2f} "
                                        f"{anim.hl('> ε')}  ⇒  keep p<sub>{e['k']}</sub>, split into "
                                        f"[{e['i0']}, {e['k']}] and [{e['k']}, {e['i1']}]",
                                   "The farthest point (★) leaves the ε-band, so the straight line is not good enough.",
                                   state(e, set(kept))))
        else:
            steps.append(anim.Step(tag, f"d<sub>max</sub> = {e['dmax']:.2f} {anim.hl('≤ ε', theme.E2)}  ⇒  drop the "
                                        f"{e['i1'] - e['i0'] - 1} points in between",
                                   "Every point lies inside the ε-band: the straight line is accurate enough.",
                                   state(e, set(kept))))
    rest = len(ev) - len(shown)
    steps.append(anim.Step("Result", f"{n} → {anim.hl(f'{len(idx)} vertices')}" +
                           (f"   ({rest} more segments processed off-screen)" if rest > 0 else ""),
                           "Every original point is within ε of the polygon.", state(None, set(idx), False, False)))
    base = plotting.base_figure((-0.5, 12.5, -0.5, 7.0), height=470, legend=False)
    base.add_trace(cui.scatter(P[:, 0], P[:, 1], theme.ORIGINAL, size=5, mode="lines+markers", width=1))
    fig = anim.step_player(base, steps, n_sub=10, hold=16, wrap_at=95)

    tev, par, depth, xs = recursion_tree(events, n)
    ex, ey = [], []
    for i, p in enumerate(par):
        if p >= 0:
            ex += [xs[p], xs[i], np.nan]
            ey += [-depth[p], -depth[i], np.nan]
    tree = go.Figure(data=[cui.scatter(ex, ey, theme.ORIGINAL, mode="lines", width=1.5),
                           dict(type="scatter", x=xs, y=[-d for d in depth], mode="markers",
                                marker=dict(size=13, color=[theme.HIGHLIGHT if e["split"] else theme.E2 for e in tev],
                                            line=dict(color="white", width=1)),
                                text=[f"[{e['i0']}, {e['i1']}] d_max = {e['dmax']:.2f} → "
                                      f"{'split at ' + str(e['k']) if e['split'] else 'straight'}" for e in tev],
                                hovertemplate="%{text}<extra></extra>", showlegend=False)],
                     layout=dict(height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                                 xaxis=dict(visible=False, range=[-0.05, 1.05]),
                                 yaxis=dict(title="depth", tickvals=[-d for d in range(max(depth or [0]) + 1)],
                                            ticktext=[str(d) for d in range(max(depth or [0]) + 1)],
                                            gridcolor=theme.GRID)))
    return fig, tree


@live.section
def replay_section():
    st.subheader("Step by step: the recursion, segment by segment", divider="gray")
    c_ctrl, c_fig = st.columns([0.8, 3.2], gap="large")
    with c_ctrl:
        name = st.selectbox("Curve", CURVES, key="rd_curve2")
        eps = live.slider("tolerance ε", 0.25, 2.0, 0.6, 0.05, key="rd_eps2")
        st.markdown("**Recursion tree**")
        st.caption("amber = split (point kept), green = straight enough (points dropped). Hover for details.")
    fig, tree = replay_figure(name, round(eps, 2))
    with c_ctrl:
        st.plotly_chart(tree, key="rd_tree")
    with c_fig:
        ui.anim_intro("Step by step: RDP(P, ε)")
        st.plotly_chart(fig, key="rd_replay")
    st.latex(r"\mathrm{RDP}(P, \epsilon) = \begin{cases} \{p_0, p_n\}, & d_{\max} \le \epsilon \\ "
             r"\mathrm{RDP}(P_1, \epsilon) \cup \mathrm{RDP}(P_2, \epsilon), & d_{\max} > \epsilon \end{cases}"
             r"\qquad P_1 = \{p_0, \dots, p_k\},\; P_2 = \{p_k, \dots, p_n\}")


replay_section()


# ------------------------------------------------------------------ greedy vs optimal, and cost
@st.cache_resource(show_spinner=False)
def measured():
    rows = []
    for name in CURVES:
        for eps in (0.15, 0.3, 0.6):
            rows.append({"curve": name, "ε": eps, "RDP vertices": len(K.rdp(curve(name), eps)),
                         "optimal vertices": len(optimal(name, eps))})
    ns = [50, 100, 200, 400, 800, 1600]
    typ, worst = [], []
    for n in ns:
        x = np.linspace(0, 12, n)
        P = np.c_[x, 3 + 1.6 * np.sin(0.9 * x) + 0.5 * np.sin(2.7 * x) + np.random.default_rng(0).normal(0, .03, n)]
        typ.append(K.rdp(P, 0.05, trace=True)[2])
        i = np.arange(n)
        W = np.c_[i, (-1.0) ** i * 0.995 ** i * 10]
        worst.append(K.rdp(W, 1e-6, trace=True)[2])
    return pd.DataFrame(rows), ns, typ, worst


st.subheader("Is RDP the best possible? How fast is it?", divider="gray")
df, ns, typ, worst = measured()
m1, m2 = st.columns([1, 1.2], gap="large")
with m1:
    st.markdown("**Greedy, not optimal.** The fewest vertices with every point within ε is a *shortest path* through "
                "all allowed shortcuts p_i → p_j (O(n³)). Measured:")
    st.dataframe(df, hide_index=True, width="stretch")
    st.caption("RDP commits to the farthest point at every split. That point is not always a vertex of the best "
               "polygon, so RDP sometimes needs extra vertices - but it is fast and never worse than the tolerance.")
with m2:
    fig = go.Figure(data=[cui.scatter(ns, typ, theme.TRANSFORMED, mode="lines+markers", size=7, width=2.5,
                                      name="typical curve", showlegend=True, hover=True),
                          cui.scatter(ns, worst, theme.OUTLIER, mode="lines+markers", size=7, width=2.5,
                                      name="worst case: shrinking zigzag", showlegend=True, hover=True),
                          cui.scatter(ns, [n * np.log2(n) for n in ns], theme.ORIGINAL, mode="lines", dash="dot",
                                      name="n log₂ n", showlegend=True),
                          cui.scatter(ns, [n * n / 2 for n in ns], theme.ORIGINAL, mode="lines", dash="dash",
                                      name="n² / 2", showlegend=True)],
                    layout=dict(height=330, margin=dict(l=8, r=8, t=30, b=8), xaxis=dict(type="log", title="points n"),
                                yaxis=dict(type="log", title="distance evaluations"),
                                legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0)))
    st.plotly_chart(fig, key="rd_cost")
    st.caption("Measured: balanced splits cost about n log n distance computations; a zigzag whose farthest point is "
               "always next to an end peels off one point per level and costs n²/2.")

ui.callout("intuition", "RDP is a sculptor working from rough to fine: one straight cut first, then add a corner exactly "
                        "where the stone sticks out most - and only if it sticks out more than ε.")
ui.callout("pitfall", "ε is in **pixels** (the units of the points). ε = 2 is tiny on a 1000-pixel contour and huge on "
                      "a 20-pixel one. In practice set it relative to the size: ε = 0.01-0.05 × perimeter (lesson 6).")
ui.callout("takeaway", "Keep a point only if it is more than ε from the current line; recurse on both halves. Two "
                       "formulas do all the work: the line A x + B y + C = 0 and d = |A x + B y + C| / √(A² + B²).")

ui.whys([
    ("Why perpendicular distance and not vertical distance?",
     "The vertical distance depends on the slope: for a steep line a point 1 pixel away sideways can be 50 pixels "
     "away vertically, and for a vertical line it is undefined. The perpendicular distance is the true closest "
     "distance and does not change when you rotate the shape (step 6 of the second animation)."),
    ("Why divide by √(A² + B²)?",
     "A x + B y + C is the dot product with the normal (A, B), which is not 1 unit long. Multiply A, B, C by 10 and "
     "it describes the same line but the value grows 10×. Dividing by the normal's length makes the number an "
     "actual distance in pixels."),
    ("Is RDP optimal?",
     "No - it is greedy. The table above measures it against the true minimum (a shortest-path search): RDP often "
     "needs a few more vertices. It is popular because it is simple, fast and guarantees the ε bound."),
    ("Why can RDP keep a 'useless' vertex?",
     "Because it always splits at the farthest point. On a closed contour that point depends on where you start, "
     "and on a noisy corner it may be a noise spike rather than the corner itself. Its two halves then need extra "
     "vertices to stay within ε."),
    ("How is a CLOSED contour handled?",
     "There are no end points. cv2.approxPolyDP first picks two far-apart points (here: the point farthest from p₀ "
     "and the point farthest from that), splits the loop there and runs RDP on both halves (core/contours.py, "
     "rdp_closed)."),
    ("Why not just keep every 10th point?",
     "Uniform subsampling keeps points on long straight runs (useless) and drops corners (essential). RDP spends "
     "points where the shape bends, which is exactly where they carry information."),
])

st.divider()
ui.mcq("rd1", "RDP keeps the farthest point p_k of a segment when …",
       ["d_max ≤ ε", "d_max > ε", "the segment has more than 10 points", "p_k is a pixel corner"], 1,
       "d_max > ε means the straight line is not accurate enough there; p_k is kept and both halves are processed.")
ui.mcq("rd2", "For the line through p₀ = (1, 1) and pₙ = (4, 5), what is (A, B)?",
       ["(4, 3)", "(−4, 3)", "(3, 4)", "(1, 1)"], 1,
       "A = y₀ − yₙ = 1 − 5 = −4, B = xₙ − x₀ = 4 − 1 = 3.")

ui.lesson_nav(__file__)
