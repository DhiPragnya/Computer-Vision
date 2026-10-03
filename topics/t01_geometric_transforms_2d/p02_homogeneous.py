import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core import anim, live, plotting, shapes, theme, ui
from core import transforms as tf

ui.lesson_header(
    __file__,
    "One small trick - writing the point (x, y) as (x, y, 1) - turns <b>every</b> transformation in this "
    "topic, including translation and perspective, into a single 3×3 matrix multiplication. "
    "It also gives us a way to talk about points <i>at infinity</i>.",
)

# switching tab re-runs the page and only the open tab is computed
tab1, tab2, tab3 = st.tabs(["1 · Why a third coordinate?", "2 · Points are rays", "3 · Lines & cross products"],
                           key="h_tabs", on_change="rerun")


def plane_mesh(w=1.0, ext=3.0, color=theme.TRANSFORMED, alpha=0.08, name="plane"):
    return go.Mesh3d(x=[-ext, ext, ext, -ext], y=[-ext, -ext, ext, ext], z=[w] * 4, i=[0, 0], j=[1, 2], k=[2, 3],
                     color=color, opacity=alpha, name=name, hoverinfo="skip", showscale=False)


def scene_layout(fig, height=520, zrange=(-0.2, 1.6), ext=3.2, eye=(1.25, -1.35, 0.75)):
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=10, b=0), showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0),
        scene=dict(
            xaxis=dict(range=[-ext, ext], title="x"), yaxis=dict(range=[-ext, ext], title="y"),
            zaxis=dict(range=list(zrange), title="w"),
            aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.75),
            camera=dict(eye=dict(x=eye[0], y=eye[1], z=eye[2])),
        ),
        uirevision="keep",
    )
    return fig


def line3d(P, color, width=5, dash=None, name="", opacity=1.0, showlegend=False):
    P = np.asarray(P, float)
    return go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="lines", line=dict(color=color, width=width, dash=dash),
                        name=name, showlegend=showlegend, opacity=opacity, hoverinfo="skip")


# =====================================================================================
# TAB 1
# =====================================================================================
@live.section
def origin_is_pinned():
    st.subheader("The problem: a 2×2 matrix can never move the origin", divider="gray")
    c_ctrl, c_fig, c_txt = st.columns([0.8, 1.4, 1.1], gap="large")
    with c_ctrl:
        st.markdown("Pick **any** 2×2 matrix:")
        a = live.slider("a", -2.0, 2.0, 1.2, 0.1, key="h_a")
        b = live.slider("b", -2.0, 2.0, 0.4, 0.1, key="h_b")
        c = live.slider("c", -2.0, 2.0, -0.3, 0.1, key="h_c")
        d = live.slider("d", -2.0, 2.0, 0.9, 0.1, key="h_d")
    M = tf.affine(a, b, c, d, 0, 0)
    with c_fig:
        fig = plotting.transform_figure(shapes.house(), M, extent=3.5, height=430, animate=False)
        fig.add_trace(plotting.points_trace([[0, 0]], theme.HIGHLIGHT, "origin - always stays at (0,0)", size=14))
        st.plotly_chart(fig, key="h_fig1")
    with c_txt:
        st.latex(r"\begin{bmatrix} a & b \\ c & d \end{bmatrix}\begin{bmatrix} 0 \\ 0 \end{bmatrix} = "
                 r"\begin{bmatrix} 0 \\ 0 \end{bmatrix}")
        ui.callout("intuition",
                   "A 2×2 matrix can rotate, scale, shear and mirror - but the origin is pinned. "
                   "Translation $\\mathbf{x}' = \\mathbf{x} + \\mathbf{t}$ moves the origin to $\\mathbf{t}$, so it is "
                   "**not** a 2×2 matrix product. It is *affine*, not *linear*.")
        st.markdown("So we would have to write every rigid motion as $\\mathbf{x}' = R\\mathbf{x} + \\mathbf{t}$ - "
                    "and chaining three of them becomes a mess of nested brackets.")


def lift_steps(tx, ty):
    """The derivation 'translation = one 3x3 product' as five pictures in 3D."""
    outline = shapes.densify(shapes.house(0.9, offset=(-1.4, -1.2))["outline"], 6)
    ks = [0.25, 0.5, 0.75]

    def at(P2, w):                      # 2D points -> 3D points at height w
        return np.c_[P2, np.full(len(P2), w)]

    moved = outline + [tx, ty]

    def state(main, deck, arrow_len, cols=(theme.TRANSFORMED, theme.TRANSFORMED)):
        trs = [line3d(main, cols[0], 8)]
        trs += [line3d(Pk, cols[1], 3, opacity=0.55) for Pk in deck]
        trs.append(go.Scatter3d(x=[0, arrow_len * tx], y=[0, arrow_len * ty], z=[1, 1], mode="lines",
                                line=dict(color=theme.E1, width=7), showlegend=False, hoverinfo="skip"))
        return trs

    flat = at(outline, 0.0)
    lifted = at(outline, 1.0)
    shifted = at(moved, 1.0)
    rays = [k * lifted for k in ks]
    rays_shifted = [k * shifted for k in ks]
    T = r"[1 0 t<sub>x</sub> ; 0 1 t<sub>y</sub> ; 0 0 1]"
    return [
        anim.Step("A 2D point", "<b>x</b> = (x, y)",
                  "A 2×2 matrix cannot add a constant to it - so we give it one more coordinate.",
                  state(flat, [flat] * 3, 0.0, (theme.ORIGINAL, theme.ORIGINAL))),
        anim.Step("Lift it to height w = 1", "(x, y)  →  (x, y, " + anim.hl("1") + ")   and every λ·(x, y, 1) is the same point",
                  "The point becomes a RAY through the origin; its copy at height 1 is the official one "
                  "(the faint copies are λ = ¼, ½, ¾).",
                  state(lifted, rays, 0.0)),
        anim.Step("Multiply by the 3×3 matrix",
                  f"{T} · (x, y, 1) = (x + {anim.hl('t<sub>x</sub>·1')}, y + {anim.hl('t<sub>y</sub>·1')}, 1)",
                  "The 1 picks up the last column. A layer at height w slides by w·t (a 3D SHEAR): "
                  "the floor w = 0 never moves, so the map is LINEAR in 3D.",
                  state(shifted, rays_shifted, 1.0)),
        anim.Step("Divide by w", "λ(x + t<sub>x</sub>, y + t<sub>y</sub>, 1) " + anim.hl("/ λ") + "  →  (x + t<sub>x</sub>, y + t<sub>y</sub>, 1)",
                  "Every copy on the ray lands on the same point - dividing by w finds where the ray meets w = 1.",
                  state(shifted, [shifted] * 3, 1.0)),
        anim.Step("Drop the 1", "(x′, y′) = (x + " + anim.hl("t<sub>x</sub>") + ", y + " + anim.hl("t<sub>y</sub>") + ")",
                  "Back on the page: a translation, produced by ONE matrix product.",
                  state(at(moved, 0.0), [at(moved, 0.0)] * 3, 1.0)),
    ]


@live.section
def lift_section():
    st.subheader("The trick: lift the plane into 3D at height w = 1", divider="gray")
    c_ctrl, c_fig, c_txt = st.columns([0.7, 2.0, 1.0], gap="large")
    with c_ctrl:
        tx = live.slider("translation tₓ", -2.0, 2.0, 1.5, 0.1, key="h_tx")
        ty = live.slider("translation t_y", -2.0, 2.0, 0.6, 0.1, key="h_ty")
        show_deck = st.toggle("Layers at other w", value=True, key="h_deck",
                              help="Show copies of the house at heights w = 0.25, 0.5, 0.75 - like a deck of cards")
    with c_fig:
        fig = go.Figure()
        outline = shapes.house(0.9, offset=(-1.4, -1.2))["outline"]
        levels = [0.25, 0.5, 0.75, 1.0] if show_deck else [1.0]
        for w in levels:
            P = np.c_[outline * w, np.full(len(outline), w)]          # points k*(x, y, 1)
            Q = P @ np.array([[1, 0, 0], [0, 1, 0], [tx, ty, 1]])         # 3D shear  = T applied to rows
            main = w == 1.0
            fig.add_trace(line3d(P, theme.ORIGINAL, 5 if main else 3, "dash", "original (w = 1)" if main else f"w = {w}",
                                 1 if main else 0.5, main))
            fig.add_trace(line3d(Q, theme.TRANSFORMED, 7 if main else 3, None,
                                 "translated (w = 1)" if main else f"w = {w}", 1 if main else 0.5, main))
        fig.add_trace(plane_mesh(1.0, 3.2, theme.TRANSFORMED, 0.07, "plane w = 1"))
        fig.add_trace(plane_mesh(0.0, 3.2, theme.ORIGINAL, 0.05, "plane w = 0"))
        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", marker=dict(size=6, color=theme.HIGHLIGHT),
                                   name="origin (0,0,0) - fixed"))
        fig.add_trace(go.Scatter3d(x=[0, tx], y=[0, ty], z=[1, 1], mode="lines",
                                   line=dict(color=theme.E1, width=6), name="shift t at w = 1"))
        scene_layout(fig, height=560)
        st.plotly_chart(fig, key="h_fig3d")
    with c_txt:
        ui.show_matrix(r"\begin{bmatrix}x'\\y'\\1\end{bmatrix}",
                       np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]]),
                       suffix=r"\begin{bmatrix}x\\y\\1\end{bmatrix}")
        st.latex(r"= \begin{bmatrix} x + t_x \\ y + t_y \\ 1 \end{bmatrix}")
        ui.callout("intuition",
                   "In 3D this matrix is a **shear**: every layer at height *w* slides by *w·t*. "
                   "The floor (w = 0) does not move - so the 3D origin stays fixed and the map is linear "
                   "again - while our plane at w = 1 slides by exactly **t**. "
                   "*A 2D translation is a 3D shear.*")

    ui.anim_intro("Step by step: how one matrix product performs a translation",
                  "Each step morphs into the next so you can see what the algebra does. ▶ Play runs all "
                  "steps; the numbered buttons replay one step; drag the slider to scrub back and forth. "
                  "The tₓ, t_y sliders above change the numbers used here too.")
    base = go.Figure()
    for tr in [plane_mesh(1.0, 3.2, theme.TRANSFORMED, 0.07, "plane w = 1"),
               plane_mesh(0.0, 3.2, theme.ORIGINAL, 0.06, "plane w = 0 (the page)"),
               line3d(np.c_[shapes.house(0.9, offset=(-1.4, -1.2))["outline"], np.zeros(10)], theme.ORIGINAL, 3, "dash",
                      "original (on the page)", 0.8, True),
               go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", marker=dict(size=5, color=theme.HIGHLIGHT),
                            name="origin", hoverinfo="skip")]:
        base.add_trace(tr)
    scene_layout(base, height=700)
    base.update_layout(scene_camera=dict(eye=dict(x=1.0, y=-1.12, z=0.62)))
    st.plotly_chart(anim.step_player(base, lift_steps(tx, ty), n_sub=12), key="h_lift_steps")


with tab1:
    if tab1.open:
        origin_is_pinned()
        lift_section()


# =====================================================================================
# TAB 2
# =====================================================================================
def ray_steps(hx, hy, hw):
    p = np.array([hx, hy, hw])
    on_plane = p / hw

    def state(pt, show_plane_pt, label):
        return [
            go.Scatter3d(x=[0, pt[0]], y=[0, pt[1]], z=[0, pt[2]], mode="lines",
                         line=dict(color=theme.E1, width=6), showlegend=False, hoverinfo="skip"),
            go.Scatter3d(x=[pt[0]], y=[pt[1]], z=[pt[2]], mode="markers+text", text=[label],
                         textposition="top center", marker=dict(size=7, color=theme.E1), showlegend=False,
                         hoverinfo="skip"),
            go.Scatter3d(x=[show_plane_pt[0]], y=[show_plane_pt[1]], z=[show_plane_pt[2]], mode="markers",
                         marker=dict(size=8, color=theme.TRANSFORMED), showlegend=False, hoverinfo="skip"),
        ]

    hidden = [on_plane[0], on_plane[1], on_plane[2]]
    big = 1.8 * p if hw * 1.8 < 1.7 else (1.6 / hw) * p
    small = 0.45 * p
    flat = np.array([on_plane[0], on_plane[1], 0.0])
    f = ui.num
    return [
        anim.Step("A homogeneous vector", f"x̃ = ({f(hx)}, {f(hy)}, {f(hw)})",
                  "Three numbers. The 2D point is NOT simply its first two entries.", state(p, p, "x̃")),
        anim.Step("Scale it", "λ·x̃ for λ = " + anim.hl("1.8") + " …",
                  "The vector slides along its ray through the origin …", state(big, p, "λx̃")),
        anim.Step("Scale it again", "… and λ = " + anim.hl("0.45"),
                  "… every point of this ray means the SAME 2D point.", state(small, p, "λx̃")),
        anim.Step("Divide by w̃", f"x̃ / {anim.hl(f(hw))} = ({f(on_plane[0])}, {f(on_plane[1])}, " + anim.hl("1") + ")",
                  "Dividing by the last entry jumps to where the ray pierces the plane w = 1.",
                  state(on_plane, hidden, "x̃ / w̃")),
        anim.Step("Read off the 2D point", f"(x, y) = ({anim.hl(f(on_plane[0]))}, {anim.hl(f(on_plane[1]))})",
                  "Drop the 1: that is the point on the page.", state(flat, hidden, "(x, y)")),
    ]


@live.section
def rays_section():
    st.subheader("Every point of the plane is a ray through the origin", divider="gray")
    c_ctrl, c_fig, c_fig2 = st.columns([0.8, 1.5, 1.1], gap="large")
    with c_ctrl:
        st.markdown("Homogeneous vector  $\\tilde{\\mathbf{x}} = (\\tilde x, \\tilde y, \\tilde w)$")
        hx = live.slider("x̃", -3.0, 3.0, 1.6, 0.1, key="h_hx")
        hy = live.slider("ỹ", -3.0, 3.0, 1.0, 0.1, key="h_hy")
        hw = live.slider("w̃", 0.05, 2.0, 1.4, 0.05, key="h_hw",
                         help="Push w̃ towards 0 and watch the 2D point run away.")
        k = live.slider("scale the whole vector by k", 0.2, 2.0, 1.0, 0.1, key="h_k")
    p = np.array([hx, hy, hw])
    on_plane = p / p[2]
    with c_fig:
        fig = go.Figure()
        s = np.linspace(0, 1.55 / hw, 2)
        fig.add_trace(go.Scatter3d(x=s * hx, y=s * hy, z=s * hw, mode="lines",
                                   line=dict(color=theme.ORIGINAL, width=4), name="the ray  λ·x̃"))
        fig.add_trace(plane_mesh(1.0, 3.2, theme.TRANSFORMED, 0.08, "plane w = 1"))
        fig.add_trace(go.Scatter3d(x=[k * hx], y=[k * hy], z=[k * hw], mode="markers",
                                   marker=dict(size=7, color=theme.E1), name=f"k·x̃ = ({k*hx:.2f}, {k*hy:.2f}, {k*hw:.2f})"))
        fig.add_trace(go.Scatter3d(x=[on_plane[0]], y=[on_plane[1]], z=[1], mode="markers",
                                   marker=dict(size=8, color=theme.TRANSFORMED),
                                   name=f"hits w = 1 at ({on_plane[0]:.2f}, {on_plane[1]:.2f}, 1)"))
        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", marker=dict(size=5, color=theme.HIGHLIGHT),
                                   name="origin"))
        scene_layout(fig, height=480, zrange=(-0.1, 1.7), ext=max(3.2, float(np.abs(on_plane[:2]).max()) + 0.5))
        fig.update_layout(uirevision=None)
        st.plotly_chart(fig, key="h_ray")
    with c_fig2:
        ws = np.linspace(hw, 0.05, 40)
        trail = np.c_[hx / ws, hy / ws]
        ext = 8
        fig2 = plotting.base_figure(ext, height=330, title="The 2D point (x̃/w̃, ỹ/w̃)")
        fig2.add_trace(go.Scatter(x=trail[:, 0], y=trail[:, 1], mode="lines", line=dict(color=theme.ORIGINAL, dash="dot"),
                                  name="if w̃ shrinks to 0 …", hoverinfo="skip"))
        fig2.add_trace(plotting.points_trace([on_plane[:2]], theme.TRANSFORMED, "current point", size=12))
        fig2.add_trace(plotting.arrow_trace([0, 0], np.array([hx, hy]) / max(np.hypot(hx, hy), 1e-9) * 2, theme.MAGENTA,
                                            "direction (x̃, ỹ) = point at infinity"))
        st.plotly_chart(fig2, key="h_trail")
        st.latex(r"(\tilde x, \tilde y, \tilde w) \;\mapsto\; \left(\frac{\tilde x}{\tilde w},\ \frac{\tilde y}{\tilde w}\right)"
                 rf"= ({ui.num(on_plane[0])},\ {ui.num(on_plane[1])})")

    ui.anim_intro("Step by step: from a homogeneous vector to a 2D point")
    base = go.Figure()
    s_ = np.linspace(-0.1, 1.75 / hw, 2)
    for tr in [plane_mesh(1.0, 3.2, theme.TRANSFORMED, 0.07, "plane w = 1"),
               plane_mesh(0.0, 3.2, theme.ORIGINAL, 0.05, "plane w = 0"),
               line3d(np.c_[s_ * hx, s_ * hy, s_ * hw], theme.ORIGINAL, 3, "dash", "the ray λ·x̃", 0.9, True)]:
        base.add_trace(tr)
    scene_layout(base, height=640, zrange=(-0.1, 1.8), ext=max(3.2, float(np.abs(on_plane[:2]).max()) + 0.5))
    base.update_layout(scene_camera=dict(eye=dict(x=1.05, y=-1.15, z=0.62)))
    st.plotly_chart(anim.step_player(base, ray_steps(hx, hy, hw), n_sub=12), key="h_ray_steps")

    ui.anim_intro("Live: drag w̃ towards 0",
                  "Drag the slider under the picture: the point (x̃/w̃, ỹ/w̃) runs off along the magenta direction. "
                  "At w̃ = 0 it would be a point at infinity.")
    ws_vals = np.round(np.r_[np.arange(2.0, 0.3, -0.1), np.arange(0.3, 0.019, -0.02)], 3)
    d = np.array([hx, hy]) / max(np.hypot(hx, hy), 1e-9)
    base2 = plotting.base_figure(10, height=430, legend=False)
    base2.add_trace(plotting.arrow_trace([0, 0], d * 9, theme.rgba(theme.MAGENTA, 0.6), "direction (x̃, ỹ)",
                                         dash="dot", showlegend=False))

    def make(wv):
        q = np.array([hx, hy]) / wv
        inside = np.all(np.abs(q) < 9.6)
        shown = q if inside else q / np.abs(q).max() * 9.6
        trs = [plotting.points_trace([shown], theme.TRANSFORMED if inside else theme.MAGENTA, "point",
                                     size=13 if inside else 11, symbol="circle" if inside else "triangle-right",
                                     showlegend=False)]
        cap = (f"w̃ = <b>{wv:.2f}</b>   →   (x̃/w̃, ỹ/w̃) = ({hx:.1f}/{wv:.2f}, {hy:.1f}/{wv:.2f}) = "
               f"<b>({q[0]:.1f}, {q[1]:.1f})</b>" + ("" if inside else "   <i>(off the chart - still running!)</i>"))
        return trs, cap

    st.plotly_chart(anim.scrubber(base2, ws_vals, make, active=0, label="w̃", fmt="{:.2f}", frame_ms=70),
                    key="h_w_scrub")
    a, b = st.columns(2, gap="large")
    with a:
        ui.callout("intuition",
                   "Move **k**: the orange point slides along the ray, but the blue point where the ray pierces "
                   "the plane w = 1 never moves. So $(x, y, w)$ and $k\\,(x, y, w)$ are **the same 2D point**. "
                   "Homogeneous vectors are only defined *up to scale* - that is why a 3×3 homography has "
                   "9 numbers but only **8** degrees of freedom.")
    with b:
        ui.callout("note",
                   "When $\\tilde w \\to 0$ the ray becomes parallel to the plane and never hits it: the 2D point runs "
                   "off to infinity along the direction $(\\tilde x, \\tilde y)$. Vectors $(x, y, 0)$ are called "
                   "**points at infinity** (ideal points). They are perfectly ordinary in homogeneous coordinates - "
                   "and projective maps can bring them back to finite positions as **vanishing points**.",
                   title="Points at infinity")


with tab2:
    if tab2.open:
        rays_section()


# =====================================================================================
# TAB 3
# =====================================================================================
DEFAULTS = dict(l_p1x=-2.5, l_p1y=-1.0, l_p2x=2.0, l_p2y=1.5, l_p3x=-2.0, l_p3y=2.0, l_p4x=2.5, l_p4y=-1.5)


def make_parallel():
    """Move line 2 (P₃, P₄) so it has exactly the direction of line 1, staying inside the slider range."""
    ss = st.session_state
    d = np.round([ss.l_p2x - ss.l_p1x, ss.l_p2y - ss.l_p1y], 1)
    if not d.any():
        return
    p3 = np.array([ss.l_p3x, ss.l_p3y])
    for sign in (1, -1):
        if np.all(np.abs(p3 + sign * d) <= 3 + 1e-9):
            p4 = p3 + sign * d
            break
    else:                       # slide P₃ just enough that P₃ + d fits
        lo = np.maximum(-3, -3 - d)
        hi = np.minimum(3, 3 - d)
        p3 = np.round(np.clip(p3, lo, hi), 1)
        p4 = p3 + d
    ss.l_p3x, ss.l_p3y = float(round(p3[0], 1)), float(round(p3[1], 1))
    ss.l_p4x, ss.l_p4y = float(round(p4[0], 1)), float(round(p4[1], 1))


def reset_lines():
    for kk, vv in DEFAULTS.items():
        st.session_state[kk] = vv


def line_pts(l, ext=6.0):
    a_, b_, c_ = l
    if abs(b_) > abs(a_):
        xs = np.array([-ext, ext])
        return np.c_[xs, -(a_ * xs + c_) / b_]
    ys = np.array([-ext, ext])
    return np.c_[-(b_ * ys + c_) / a_, ys]


@live.section
def lines_section():
    st.subheader("Lines are vectors too - and the cross product does the geometry", divider="gray")
    for kk, vv in DEFAULTS.items():
        st.session_state.setdefault(kk, vv)

    c_ctrl, c_fig, c_txt = st.columns([0.95, 1.45, 1.1], gap="large")
    with c_ctrl:
        st.markdown(f"<span style='color:{theme.E1}'><b>Line 1</b></span> through P₁, P₂", unsafe_allow_html=True)
        pts = []
        for i in range(4):
            if i == 2:
                st.markdown(f"<span style='color:{theme.E2}'><b>Line 2</b></span> through P₃, P₄",
                            unsafe_allow_html=True)
            r = st.columns(2)
            sub = "₁₂₃₄"[i]
            with r[0]:
                x_ = live.slider(f"P{sub} x", -3.0, 3.0, DEFAULTS[f"l_p{i + 1}x"], 0.1, key=f"l_p{i + 1}x")
            with r[1]:
                y_ = live.slider(f"P{sub} y", -3.0, 3.0, DEFAULTS[f"l_p{i + 1}y"], 0.1, key=f"l_p{i + 1}y")
            pts.append(np.array([x_, y_]))
        p1, p2, p3, p4 = pts
        b1, b2 = st.columns(2)
        b1.button("Parallel", on_click=make_parallel, icon=":material/drag_handle:", width="stretch",
                  help="Move P₄ so that line 2 becomes parallel to line 1", key="l_par")
        b2.button("Reset", on_click=reset_lines, icon=":material/restart_alt:", width="stretch", key="l_reset")

    l1 = tf.line_through(p1, p2)
    l2 = tf.line_through(p3, p4)
    X = tf.intersect(l1, l2)
    degenerate = np.abs(l1[:2]).sum() < 1e-9 or np.abs(l2[:2]).sum() < 1e-9
    parallel = (not degenerate) and abs(X[2]) < 1e-9 * max(1.0, np.abs(X[:2]).max())

    with c_fig:
        fig = plotting.base_figure(4.0, height=500)
        if np.abs(l1[:2]).sum() > 1e-9:
            fig.add_trace(plotting.line_trace(line_pts(l1), theme.E1, "line 1  (l₁ = P₁ × P₂)", width=2.6))
        if np.abs(l2[:2]).sum() > 1e-9:
            fig.add_trace(plotting.line_trace(line_pts(l2), theme.E2, "line 2  (l₂ = P₃ × P₄)", width=2.6))
        fig.add_trace(plotting.points_trace([p1, p2], theme.E1, "P₁, P₂", labels=["P₁", "P₂"], showlegend=False))
        fig.add_trace(plotting.points_trace([p3, p4], theme.E2, "P₃, P₄", labels=["P₃", "P₄"], showlegend=False))
        if degenerate:
            pass
        elif not parallel:
            xi = X[:2] / X[2]
            fig.add_trace(plotting.points_trace([xi], theme.HIGHLIGHT, "intersection  x = l₁ × l₂", size=15,
                                                symbol="diamond"))
        else:
            dirv = X[:2] / (np.linalg.norm(X[:2]) + 1e-12)
            fig.add_trace(plotting.arrow_trace(-dirv * 3.6, dirv * 3.6, theme.MAGENTA,
                                               "they meet at infinity in this direction", dash="dot"))
        st.plotly_chart(fig, key="h_lines")
    with c_txt:
        st.markdown("A line $ax + by + c = 0$ is the vector $\\mathbf{l} = (a, b, c)$. "
                    "A point lies on it exactly when $\\mathbf{l}\\cdot\\tilde{\\mathbf{x}} = 0$.")
        st.latex(rf"\mathbf{{l}}_1 = \tilde P_1 \times \tilde P_2 = {ui.tex_vec(l1)}^{{T}}")
        st.latex(rf"\mathbf{{l}}_2 = \tilde P_3 \times \tilde P_4 = {ui.tex_vec(l2)}^{{T}}")
        st.latex(rf"\tilde{{\mathbf{{x}}}} = \mathbf{{l}}_1 \times \mathbf{{l}}_2 = {ui.tex_vec(X)}^{{T}}")
        if degenerate:
            st.warning("Two of the points coincide, so one of the 'lines' is not defined (its vector is 0). "
                       "Move the points apart.", icon=":material/error:")
        elif parallel:
            st.error("**w = 0** → the lines are parallel. Their 'intersection' is the point at infinity in "
                     f"direction ({ui.num(X[0])}, {ui.num(X[1])}).", icon=":material/all_inclusive:")
        else:
            xi = X[:2] / X[2]
            st.success(f"Divide by w: the lines cross at **({xi[0]:.2f}, {xi[1]:.2f})**.", icon=":material/place:")
        ui.callout("intuition",
                   "The cross product of two vectors is perpendicular to both. "
                   "$\\mathbf{l} = \\tilde P_1 \\times \\tilde P_2$ is 'perpendicular' to both points, i.e. "
                   "$\\mathbf{l}\\cdot\\tilde P_1 = \\mathbf{l}\\cdot\\tilde P_2 = 0$ - so both points lie on $\\mathbf{l}$. "
                   "Points and lines play **symmetric roles** (duality): the same formula gives the intersection.")

    # ------------------------------------------------------------- live: turn line 2 until it is parallel
    if degenerate:
        return
    ui.anim_intro("Live: turn line 2 until it is parallel to line 1",
                  "Drag the slider under the picture. Line 2 turns about P₃; watch the third entry w of "
                  "x̃ = l₁ × l₂ shrink to 0 as the crossing point runs off to infinity - and come back from the "
                  "OTHER side. (In projective geometry both ends of a line meet at the same point at infinity.)")
    d1 = (p2 - p1) / np.linalg.norm(p2 - p1)
    base = plotting.base_figure(4.0, height=470, legend=False)
    base.add_trace(plotting.line_trace(line_pts(l1), theme.E1, "line 1", width=2.6, showlegend=False))
    base.add_trace(plotting.points_trace([p1, p2], theme.E1, "P₁, P₂", labels=["P₁", "P₂"], showlegend=False))
    base.add_trace(plotting.points_trace([p3], theme.E2, "P₃", labels=["P₃"], showlegend=False))
    angles = list(range(-40, 41, 1))

    def make(a_deg):
        r = np.deg2rad(a_deg)
        d2 = np.array([d1[0] * np.cos(r) - d1[1] * np.sin(r), d1[0] * np.sin(r) + d1[1] * np.cos(r)])
        q = p3 + 2.0 * d2
        l2_ = tf.line_through(p3, q)
        X_ = tf.intersect(l1, l2_)
        n = np.abs(X_).max()
        Xn = X_ / n if n > 0 else X_
        trs = [plotting.line_trace(line_pts(l2_, 12), theme.E2, "line 2", width=2.6, showlegend=False)]
        if abs(X_[2]) > 1e-9 * n:
            xi = X_[:2] / X_[2]
            if np.all(np.abs(xi) < 3.9):
                trs.append(plotting.points_trace([xi], theme.HIGHLIGHT, "crossing", size=15, symbol="diamond",
                                                 showlegend=False))
                where = f"crossing at <b>({xi[0]:.2f}, {xi[1]:.2f})</b>"
            else:
                edge = xi / np.abs(xi).max() * 3.85
                trs.append(plotting.points_trace([edge], theme.MAGENTA, "off chart", size=12, symbol="diamond-open",
                                                 showlegend=False))
                where = f"crossing at <b>({xi[0]:.1f}, {xi[1]:.1f})</b> - off the chart"
        else:
            trs.append(plotting.points_trace([[np.nan, np.nan]], theme.MAGENTA, "inf", showlegend=False))
            where = "<b>parallel: they meet at infinity</b>"
        cap = (f"angle between the lines = {a_deg:+d}°   →   x̃ = l₁ × l₂ ∝ ({Xn[0]:.2f}, {Xn[1]:.2f}, "
               f"<span style='color:{theme.MAGENTA}'><b>w = {Xn[2]:.3f}</b></span>)<br>{where}")
        return trs, cap

    st.plotly_chart(anim.scrubber(base, angles, make, active=len(angles) - 1, label="angle", fmt="{:+d}°",
                                  frame_ms=60), key="h_par_scrub")


with tab3:
    if tab3.open:
        lines_section()
    ui.callout("takeaway",
               "Homogeneous coordinates buy us three things: (1) **every** 2D transform is one 3×3 matrix, so "
               "chaining transforms = multiplying matrices; (2) points and lines are 3-vectors linked by dot and "
               "cross products; (3) parallel lines meet at a perfectly usable point with w = 0.")

ui.whys([
    ('Why add a 1 as the third coordinate and not a 0?',
     "With w = 1 the translation column gets multiplied by 1 and the point moves. With w = 0 the translation is multiplied by 0 and nothing moves - which is exactly right for a **direction** (a vector, or a point at infinity): shifting the world does not change which way 'north' points. So (x, y, 1) is a point, (x, y, 0) is a direction."),
    ('Why are (x, y, 1) and (2x, 2y, 2) the same point?',
     'Both lie on the same ray through the origin, and the ray hits the plane w = 1 at only one place. Dividing by w is the act of finding that place: (2x/2, 2y/2) = (x, y).'),
    ('What happens if I divide by w = 0?',
     "You can't - and that is the point. The ray is parallel to the plane w = 1 and never meets it: the 2D point is infinitely far away in the direction (x, y). Code must check for w ≈ 0 before dividing."),
    ("Why is it called 'homogeneous'?",
     'Because equations written in these coordinates become *homogeneous polynomials* - every term has the same degree. The line ax + by + c = 0 becomes ax + by + cw = 0, and multiplying (x, y, w) by any λ leaves it true.'),
    ('Is the third dimension real?',
     "No, it is bookkeeping. We never move 'up' into 3D; the extra coordinate is a trick that makes translation (and perspective) linear. Geometrically the set of rays through the origin is called the *projective plane*."),
    ('Why does the crossing point come back from the other side?',
     "In the projective plane a line is a closed loop: its two 'ends' meet at ONE point at infinity, the direction (a, b, 0) - and (−a, −b, 0) is the same point because vectors are only defined up to scale. As the angle passes through 0 the sign of w flips, so the crossing jumps from far away on one side to far away on the other."),
])

st.divider()
ui.mcq("hom1", "Which homogeneous vector represents the same 2D point as (2, 3)?",
       ["(2, 3, 0)", "(4, 6, 2)", "(2, 3, 2)", "(6, 9, 1)"], 1,
       "(4, 6, 2) / 2 = (2, 3, 1). (2, 3, 0) is a point at infinity (a direction), (2, 3, 2) is (1, 1.5), "
       "and (6, 9, 1) is (6, 9).")
ui.mcq("hom2", "Why does a 3×3 homography have only 8 degrees of freedom, not 9?",
       ["Because the last row is always (0, 0, 1)", "Because H and k·H give the same mapping",
        "Because det(H) must be 1", "Because one entry must be zero"], 1,
       "Homogeneous vectors are defined up to scale: (k H) x̃ = k (H x̃), which is the same 2D point after "
       "dividing by w. So one of the 9 numbers is redundant - we often fix h₃₃ = 1 or ‖H‖ = 1.")

ui.lesson_nav(__file__)
