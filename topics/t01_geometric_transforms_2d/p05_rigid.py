import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core import anim, live, plotting, shapes, theme, ui
from core import transforms as tf

ui.lesson_header(
    __file__,
    "Turn the object, then slide it. A <b>rigid</b> (Euclidean) motion is what happens to a part you pick up "
    "and put down somewhere else on a table: nothing bends, nothing grows. Three numbers describe it - an angle "
    "θ and a shift (tₓ, t_y).",
)


def arc(radius, a0, a1, n=40):
    t = np.linspace(a0, a1, n)
    return np.c_[radius * np.cos(t), radius * np.sin(t)]


def rot2(v, ang):
    c, s = np.cos(ang), np.sin(ang)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


# ------------------------------------------------------------------ derivation 1: polar coordinates
def polar_steps(rr, phi_deg, th_deg):
    """p′ = R p derived twice: as 'add θ to the angle', and as 'x·Re₁ + y·Re₂'."""
    ph, th = np.deg2rad(phi_deg), np.deg2rad(th_deg)
    P = rr * np.array([np.cos(ph), np.sin(ph)])
    x, y = P

    def build(p):
        a = p["a"]                       # how much of θ has been applied to the point
        b = p["b"]                       # how much of θ has been applied to the x / y pieces
        pt = rr * np.array([np.cos(ph + a * th), np.sin(ph + a * th)])
        ex = rot2(np.array([x, 0.0]), b * th)
        ey = rot2(np.array([0.0, y]), b * th)
        corner = ex
        tip = ex + ey
        k = p["pieces"]                  # 0 = hidden, 1 = visible
        trs = [
            plotting.line_trace(np.array([[0, 0], pt]), theme.TRANSFORMED, "r", width=2, showlegend=False),
            plotting.line_trace(arc(0.55, ph, ph + a * th + 1e-6), theme.HIGHLIGHT, "θ", width=3, showlegend=False),
            plotting.arrows_trace([[0, 0]], [k * corner], theme.E1, width=3.2, size=13 if k > 0.5 else 0),
            plotting.arrows_trace([k * corner], [k * corner + k * (tip - corner)], theme.E2, width=3.2,
                                  size=13 if k > 0.5 else 0),
            plotting.line_trace(np.array([[pt[0], 0], pt, [np.nan, np.nan], [0, pt[1]], pt]), theme.TRANSFORMED,
                                "proj", width=1.3 * p["proj"] + 1e-3, dash="dash", showlegend=False),
            go.Scatter(x=[pt[0]], y=[pt[1]], mode="markers+text", text=["p′" if a > 0.5 else "p"],
                       textposition="top right", textfont=dict(size=14),
                       marker=dict(size=13, color=theme.TRANSFORMED if a > 0.5 else theme.ORIGINAL,
                                   line=dict(color="white", width=1.5)), showlegend=False, hoverinfo="skip"),
        ]
        return trs

    S = [dict(a=0.0, b=0.0, pieces=0.0, proj=0.0),
         dict(a=1.0, b=0.0, pieces=0.0, proj=0.0),
         dict(a=0.0, b=0.0, pieces=1.0, proj=0.0),
         dict(a=1.0, b=1.0, pieces=1.0, proj=0.0),
         dict(a=1.0, b=1.0, pieces=1.0, proj=1.0)]
    f, H = ui.num, anim.hl
    c, s = np.cos(th), np.sin(th)
    info = [
        ("A point in polar form", f"p = (r cos φ, r sin φ) = ({f(x)}, {f(y)})   with r = {f(rr)}, φ = {phi_deg}°",
         "Every point is 'a distance r at an angle φ'."),
        ("Rotating = adding θ to the angle", f"p′ = (r cos(φ {H('+ θ')}), r sin(φ {H('+ θ')}))",
         "The distance r does not change - the point slides along its circle by θ (amber arc)."),
        ("Split p into an x-piece and a y-piece",
         f"p = {H('x', theme.E1)}·e₁ + {H('y', theme.E2)}·e₂ = {f(x)}·(1, 0) + {f(y)}·(0, 1)",
         "The angle-addition identity does exactly this: r cos φ becomes x and r sin φ becomes y."),
        ("Rotate both pieces by θ",
         f"p′ = {H('x', theme.E1)}·(cos θ, sin θ) + {H('y', theme.E2)}·(−sin θ, cos θ)",
         "Rotating the two arrows and adding them again lands on exactly the same p′ as before - because rotation "
         "is LINEAR. The arrows are x·Re₁ and y·Re₂."),
        ("Read off x′ and y′",
         f"x′ = x cos θ − y sin θ = {f(x * c - y * s)},   y′ = x sin θ + y cos θ = {f(x * s + y * c)}",
         "Collect the numbers in front of x and y: [cos θ  −sin θ ; sin θ  cos θ]. No r and no φ left - the same "
         "matrix works for every point."),
    ]
    steps = []
    for i, (t_, e_, n_) in enumerate(info):
        steps.append(anim.Step(t_, e_, n_, build(S[i]), between=None if i == 0 else anim.morph(build, S[i - 1], S[i])))
    return steps


@live.section
def polar_section():
    st.subheader("Derivation 1 · polar coordinates and the angle-addition identities", divider="gray")
    c_ctrl, c_fig, c_txt = st.columns([0.75, 1.35, 1.25], gap="large")
    with c_ctrl:
        rr = live.slider("distance r from the origin", 0.5, 2.0, 1.6, 0.05, key="rg_r")
        phi = live.slider("starting angle φ (degrees)", -180, 180, 25, 1, key="rg_phi", format="%d°")
        thp = live.slider("rotate by θ (degrees)", -180, 180, 40, 1, key="rg_thp", format="%d°")
    ph, tp = np.deg2rad(phi), np.deg2rad(thp)
    P = rr * np.array([np.cos(ph), np.sin(ph)])
    Pp = rr * np.array([np.cos(ph + tp), np.sin(ph + tp)])
    with c_fig:
        fig = plotting.base_figure(2.3, height=430)
        fig.add_trace(plotting.line_trace(shapes.circle(rr), theme.ORIGINAL, f"circle of radius r = {rr:.2f}",
                                          width=1.2, dash="dot"))
        fig.add_trace(plotting.line_trace(arc(0.35, 0, ph), theme.ORIGINAL, "φ", width=2.5, showlegend=False))
        fig.add_trace(plotting.line_trace(arc(0.55, ph, ph + tp), theme.HIGHLIGHT, "θ", width=3, showlegend=False))
        fig.add_trace(plotting.line_trace(np.array([[0, 0], P]), theme.ORIGINAL, "r", width=2, showlegend=False))
        fig.add_trace(plotting.line_trace(np.array([[0, 0], Pp]), theme.TRANSFORMED, "r", width=2, showlegend=False))
        fig.add_trace(plotting.line_trace(np.array([[Pp[0], 0], Pp]), theme.TRANSFORMED, "y'", width=1.2, dash="dash",
                                          showlegend=False))
        fig.add_trace(plotting.line_trace(np.array([[0, Pp[1]], Pp]), theme.TRANSFORMED, "x'", width=1.2, dash="dash",
                                          showlegend=False))
        fig.add_trace(plotting.points_trace([P], theme.ORIGINAL, "p = (r cos φ, r sin φ)", labels=["p"], size=12))
        fig.add_trace(plotting.points_trace([Pp], theme.TRANSFORMED, "p' = (r cos(φ+θ), r sin(φ+θ))", labels=["p'"],
                                            size=13))
        fig.add_annotation(x=0.45 * np.cos(ph / 2), y=0.45 * np.sin(ph / 2), text="φ", showarrow=False,
                           font=dict(size=14))
        fig.add_annotation(x=0.7 * np.cos(ph + tp / 2), y=0.7 * np.sin(ph + tp / 2), text="θ", showarrow=False,
                           font=dict(size=14, color=theme.HIGHLIGHT))
        st.plotly_chart(fig, key="rg_polar")
    with c_txt:
        x, y = P
        c, s = np.cos(tp), np.sin(tp)
        st.markdown("Rotating keeps the distance **r** and adds θ to the angle:")
        st.latex(r"x' = r\cos(\varphi+\theta) = \underbrace{r\cos\varphi}_{x}\cos\theta - "
                 r"\underbrace{r\sin\varphi}_{y}\sin\theta")
        st.latex(r"y' = r\sin(\varphi+\theta) = \underbrace{r\sin\varphi}_{y}\cos\theta + "
                 r"\underbrace{r\cos\varphi}_{x}\sin\theta")
        st.latex(rf"x' = {x:.2f}\cdot{c:.2f} - ({y:.2f})\cdot({s:.2f}) = \mathbf{{{x * c - y * s:.2f}}}")
        st.latex(rf"y' = {x:.2f}\cdot{s:.2f} + ({y:.2f})\cdot({c:.2f}) = \mathbf{{{x * s + y * c:.2f}}}")
        st.caption(f"Check on the plot: p' = ({Pp[0]:.2f}, {Pp[1]:.2f}). Collect the coefficients of x and y and the "
                   "matrix appears - with no r and no φ left in it.")
        st.latex(r"R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ "
                 r"0 & 0 & 1 \end{bmatrix}")

    ui.anim_intro("Step by step: two ways to see the same rotation",
                  "Uses your r, φ and θ. Steps 1-2 rotate the point along its circle; steps 3-5 rotate its x- and "
                  "y-pieces instead - and land on the very same p′. ▶ Play runs everything; the numbers replay one step; "
                  "the slider under the picture scrubs back and forth.")
    base = plotting.base_figure(2.3, height=600, legend=False)
    base.add_trace(plotting.line_trace(shapes.circle(rr), theme.ORIGINAL, "circle", width=1.2, dash="dot",
                                       showlegend=False))
    base.add_trace(plotting.line_trace(arc(0.35, 0, ph), theme.ORIGINAL, "φ", width=2.5, showlegend=False))
    base.add_trace(plotting.points_trace([P], theme.rgba(theme.ORIGINAL, 0.5), "p", size=10, showlegend=False))
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        st.plotly_chart(anim.step_player(base, polar_steps(rr, phi, thp), n_sub=14, wrap_at=80), key="rg_polar_steps")
    with right:
        ui.callout("intuition",
                   "The angle-addition identities do the heavy lifting: they split 'the angle of the point plus θ' "
                   "into a part that only depends on the **point** (r cos φ = x, r sin φ = y) and a part that only "
                   "depends on the **rotation** (cos θ, sin θ). That separation is exactly what makes rotation a "
                   "matrix: the same four numbers work for every point.")
        ui.callout("note",
                   "Step 4 is the deep one: rotate the pieces, then add - or add, then rotate - same answer. "
                   "That property, R(a + b) = R a + R b, is what *linear* means, and it is why a matrix can do it.",
                   title="Why the two routes agree")


polar_section()


# ------------------------------------------------------------------ the rotation matrix
def column_steps(th_deg):
    r = np.deg2rad(th_deg)

    def build(p):
        a1, a2, sh = p["a1"], p["a2"], p["proj"]
        e1 = np.array([np.cos(a1 * r), np.sin(a1 * r)])
        e2 = np.array([-np.sin(a2 * r), np.cos(a2 * r)])
        e1_full = np.array([np.cos(r), np.sin(r)])
        return [
            plotting.arrows_trace([[0, 0]], [e1], theme.E1, width=3.4, size=14),
            plotting.arrows_trace([[0, 0]], [e2], theme.E2, width=3.4, size=14),
            plotting.line_trace(arc(0.3, 0, a1 * r + 1e-6), theme.HIGHLIGHT, "θ", width=2.5, showlegend=False),
            plotting.line_trace(arc(0.22, np.pi / 2, np.pi / 2 + a2 * r + 1e-6), theme.HIGHLIGHT, "θ", width=2.5,
                                showlegend=False),
            plotting.line_trace(np.array([[0, 0], [e1[0], 0], e1]), theme.E1, "cos/sin", width=1.3 * sh + 1e-3,
                                dash="dash", showlegend=False),
            plotting.line_trace(np.array([[0, 0], [e2[0], 0], e2]), theme.E2, "−sin/cos", width=1.3 * sh + 1e-3,
                                dash="dash", showlegend=False),
            plotting.line_trace(arc(1.0, r, np.pi / 2 + r) * p["perp"] + (1 - p["perp"]) * e1_full,
                                theme.rgba(theme.HIGHLIGHT, 0.7), "90°", width=2, dash="dot", showlegend=False),
        ]

    S = [dict(a1=0.0, a2=0.0, proj=0.0, perp=0.0), dict(a1=1.0, a2=0.0, proj=1.0, perp=0.0),
         dict(a1=1.0, a2=1.0, proj=1.0, perp=1.0), dict(a1=1.0, a2=1.0, proj=0.0, perp=0.0)]
    c, s = np.cos(r), np.sin(r)
    f, H = ui.num, anim.hl
    info = [
        ("Start with the basis arrows", "e₁ = (1, 0),  e₂ = (0, 1)",
         "Whatever a matrix does to these two arrows, it does to every point (every point is x·e₁ + y·e₂)."),
        ("Rotate e₁ by θ", f"R e₁ = ({H('cos θ', theme.E1)}, {H('sin θ', theme.E1)}) = ({f(c, 3)}, {f(s, 3)})",
         "On the unit circle the dashed legs ARE cos θ (along x) and sin θ (up). This is column 1."),
        ("Rotate e₂ by θ", f"R e₂ = ({H('−sin θ', theme.E2)}, {H('cos θ', theme.E2)}) = ({f(-s, 3)}, {f(c, 3)})",
         "e₂ is e₁ turned 90°, so its image is R e₁ turned another 90° (dotted arc): swap the entries and "
         "negate the first. That is where the minus sign comes from. This is column 2."),
        ("Put the two images in the columns",
         f"R(θ) = [{H('cos θ', theme.E1)}  {H('−sin θ', theme.E2)} ; {H('sin θ', theme.E1)}  {H('cos θ', theme.E2)}]",
         "The columns of ANY matrix are the images of the basis vectors - read them off the picture."),
    ]
    return [anim.Step(t_, e_, n_, build(S[i]), between=None if i == 0 else anim.morph(build, S[i - 1], S[i]))
            for i, (t_, e_, n_) in enumerate(info)]


@live.section
def circle_section():
    st.subheader("Derivation 2 · a rotation matrix is just two rotated arrows", divider="gray")
    c_ctrl, c_fig, c_txt = st.columns([0.75, 1.35, 1.25], gap="large")
    with c_ctrl:
        th = live.slider("θ (degrees)", -180, 180, 35, 1, key="rg_th0", format="%d°")
        st.caption("Positive θ = counter-clockwise (in maths axes, y up). Drag it and watch the numbers.")
    r = np.deg2rad(th)
    e1r = np.array([np.cos(r), np.sin(r)])
    e2r = np.array([-np.sin(r), np.cos(r)])
    with c_fig:
        fig = plotting.base_figure(1.5, height=430)
        fig.add_trace(plotting.line_trace(shapes.circle(1.0), theme.ORIGINAL, "unit circle", width=1.2, dash="dot"))
        fig.add_trace(plotting.arrow_trace([0, 0], [1, 0], theme.rgba(theme.E1, 0.35), "e₁", width=2, showlegend=False))
        fig.add_trace(plotting.arrow_trace([0, 0], [0, 1], theme.rgba(theme.E2, 0.35), "e₂", width=2, showlegend=False))
        fig.add_trace(plotting.line_trace(arc(0.3, 0, r), theme.HIGHLIGHT, "θ", width=2.5, showlegend=False))
        fig.add_trace(plotting.line_trace(arc(0.22, np.pi / 2, np.pi / 2 + r), theme.HIGHLIGHT, "θ", width=2.5,
                                          showlegend=False))
        fig.add_trace(plotting.line_trace(np.array([[e1r[0], 0], e1r]), theme.E1, "sin θ", width=1.2, dash="dash",
                                          showlegend=False))
        fig.add_trace(plotting.line_trace(np.array([[0, 0], [e1r[0], 0]]), theme.E1, "cos θ", width=5,
                                          showlegend=False))
        fig.add_trace(plotting.arrow_trace([0, 0], e1r, theme.E1, "R e₁ = (cos θ, sin θ)  → 1st column"))
        fig.add_trace(plotting.arrow_trace([0, 0], e2r, theme.E2, "R e₂ = (−sin θ, cos θ) → 2nd column"))
        fig.add_annotation(x=e1r[0] / 2, y=-0.09 if e1r[1] >= 0 else 0.09, text="cos θ", showarrow=False,
                           font=dict(color=theme.E1, size=13))
        fig.add_annotation(x=e1r[0] + (0.17 if e1r[0] >= 0 else -0.17), y=e1r[1] / 2, text="sin θ", showarrow=False,
                           font=dict(color=theme.E1, size=13))
        st.plotly_chart(fig, key="rg_circle")
    with c_txt:
        R = tf.rotation(th)
        st.latex(r"R(\theta) = \begin{bmatrix} \color{" + theme.E1 + r"}{\cos\theta} & \color{" + theme.E2 +
                 r"}{-\sin\theta} \\ \color{" + theme.E1 + r"}{\sin\theta} & \color{" + theme.E2 + r"}{\cos\theta} "
                 r"\end{bmatrix}")
        st.latex(r"R(" + str(th) + r"^\circ) = " + ui.tex_matrix(R[:2, :2], 3, ui.column_highlight()))
        ui.callout("intuition",
                   "Don't memorise where the minus sign goes - **derive it**. Column 1 is where e₁ = (1, 0) lands: "
                   "rotate it by θ and you get (cos θ, sin θ). Column 2 is where e₂ = (0, 1) lands: it is e₁'s image "
                   "turned another 90°, i.e. (−sin θ, cos θ). *The columns of any matrix are the images of the basis "
                   "vectors.*")
        st.markdown("Two facts that define a rotation, checked live:")
        RtR = R[:2, :2].T @ R[:2, :2]
        st.latex(r"R^TR = " + ui.tex_matrix(RtR, 3) + r" = I, \qquad \det R = " + ui.num(np.linalg.det(R[:2, :2]), 3))
        st.caption("RᵀR = I: the columns stay unit length and perpendicular → lengths and angles are kept. "
                   "det R = +1: no mirror flip. And the inverse is free: R⁻¹ = Rᵀ = R(−θ).")

    ui.anim_intro("Step by step: read the matrix off two rotated arrows")
    base = plotting.base_figure(1.5, height=560, legend=False)
    base.add_trace(plotting.line_trace(shapes.circle(1.0), theme.ORIGINAL, "unit circle", width=1.2, dash="dot",
                                       showlegend=False))
    base.add_trace(plotting.arrow_trace([0, 0], [1, 0], theme.rgba(theme.E1, 0.3), "e₁", width=2, showlegend=False))
    base.add_trace(plotting.arrow_trace([0, 0], [0, 1], theme.rgba(theme.E2, 0.3), "e₂", width=2, showlegend=False))
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        st.plotly_chart(anim.step_player(base, column_steps(th), n_sub=14, wrap_at=78), key="rg_col_steps")
    with right:
        ui.callout("try", "Set θ to 90° with the slider above and replay: e₁ lands exactly where e₂ was, and e₂ lands "
                          "on −e₁ = (−1, 0). The matrix becomes [0 −1 ; 1 0].")


circle_section()


# ------------------------------------------------------------------ rigid motion
HOUSE = shapes.house(offset=(-0.5, -0.6))


def pivot_steps(th, pcx, pcy):
    """T(c)·R(θ)·T(−c): move the pivot to the origin, rotate, move back."""
    c = np.array([pcx, pcy])

    def build(p):
        M = tf.translation(*(p["back"] * c)) @ tf.rotation(p["ang"]) @ tf.translation(*(-p["go"] * c))
        trs = plotting.layer_traces(HOUSE, M, grid_extent=4.0, n=2)
        pv = tf.apply(M, [c])[0]
        trs.append(go.Scatter(x=[pv[0]], y=[pv[1]], mode="markers", showlegend=False, hoverinfo="skip",
                              marker=dict(size=15, color=theme.HIGHLIGHT, symbol="x",
                                          line=dict(color=theme.HIGHLIGHT, width=2))))
        return trs

    S = [dict(go=0.0, ang=0.0, back=0.0), dict(go=1.0, ang=0.0, back=0.0), dict(go=1.0, ang=th, back=0.0),
         dict(go=1.0, ang=th, back=1.0)]
    f, H = ui.num, anim.hl
    info = [
        ("Start", f"pivot c = ({f(pcx)}, {f(pcy)})  (amber ×)",
         "Goal: turn the house by θ about c. But a matrix R(θ) only knows how to turn about the ORIGIN."),
        ("T(−c): slide the pivot to the origin", f"p₁ = {H('T(−c)')} p",
         "Everything moves by −c. Now the pivot sits exactly on the origin."),
        ("R(θ): rotate about the origin", f"p₂ = {H('R(θ)')} T(−c) p",
         "This is the only step that turns anything. The pivot is at the origin, so it does not move."),
        ("T(c): slide back", f"p′ = {H('T(c)')} R(θ) T(−c) p   (read right to left)",
         "The pivot returns to c and the house has turned about it. The three matrices multiply into ONE."),
    ]
    return [anim.Step(t_, e_, n_, build(S[i]), between=None if i == 0 else anim.morph(build, S[i - 1], S[i]))
            for i, (t_, e_, n_) in enumerate(info)]


@live.section
def rigid_section():
    st.subheader("Rigid motion = rotate, then translate", divider="gray")
    c_ctrl, c_fig, c_txt = st.columns([0.8, 1.6, 1.05], gap="large")
    with c_ctrl:
        th = live.slider("rotation θ (degrees)", -180, 180, 60, 1, key="rg_th", format="%d°")
        tx = live.slider("tₓ", -3.0, 3.0, 0.5, 0.1, key="rg_tx")
        ty = live.slider("t_y", -3.0, 3.0, 0.2, 0.1, key="rg_ty")
        pivot_mode = st.segmented_control("Rotate about", ["origin", "a pivot point"], default="origin",
                                          key="rg_pivot", required=True)
        cx = cy = 0.0
        if pivot_mode == "a pivot point":
            cx = live.slider("pivot cₓ", -3.0, 3.0, 1.0, 0.1, key="rg_cx")
            cy = live.slider("pivot c_y", -3.0, 3.0, 1.2, 0.1, key="rg_cy")
        mirror = st.toggle("Mirror flip (det = −1)", value=False, key="rg_mirror")

    F = tf.reflection_x() if mirror else np.eye(3)

    def H_at(t):
        flip = (1 - t) * np.eye(3) + t * F
        return tf.translation(t * tx, t * ty) @ tf.rotation_about(t * th, cx, cy) @ flip

    H = H_at(1.0)
    pivot_static = [plotting.points_trace([[cx, cy]], theme.HIGHLIGHT, "pivot", size=13, symbol="x")] \
        if pivot_mode == "a pivot point" else []
    with c_fig:
        fig = plotting.transform_figure(HOUSE, H_at, extent=4.0, height=540, extra_static=pivot_static)
        st.plotly_chart(fig, key="rg_fig")

    with c_txt:
        if pivot_mode == "a pivot point":
            st.latex(r"H = T(\mathbf{t})\;T(\mathbf{c})\,R(\theta)\,T(-\mathbf{c})" + (r"\,F" if mirror else ""))
        else:
            st.latex(r"H = T(\mathbf{t})\,R(\theta)" + (r"\,F" if mirror else "") +
                     r" = \begin{bmatrix} R & \mathbf{t} \\ \mathbf{0}^T & 1\end{bmatrix}")
        ui.show_matrix("H", H, 3, ui.column_highlight())
        info = tf.classify(H)
        st.markdown(f"Class detected: **{info['name']}**")
        # invariance check - edge lengths and an angle
        pts = np.array([[0, 0], [2, 0], [2, 1.5], [1.0, 2.5]]) + np.array([-0.5, -0.6])
        q = tf.apply(H, pts)
        L0 = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        L1 = np.linalg.norm(np.diff(q, axis=0), axis=1)
        st.dataframe({"edge": ["bottom", "right wall", "roof (right)"], "length before": L0.round(3),
                      "length after": L1.round(3)}, hide_index=True, width="stretch")
        if mirror:
            st.warning("With the flip the chimney jumps to the other side: **orientation is reversed**. "
                       "Lengths are still kept, but no sliding/turning in the plane can produce this - during "
                       "Play the house must collapse to a line half-way (det passes through 0).",
                       icon=":material/flip:")

    st.markdown("#### The pivot trick, step by step: T(c) · R(θ) · T(−c)")
    pcx, pcy = (cx, cy) if pivot_mode == "a pivot point" else (1.0, 1.2)
    ui.anim_intro("Rotate about a pivot in three moves",
                  f"Uses your θ = {th}° and pivot c = ({ui.num(pcx)}, {ui.num(pcy)})"
                  + ("" if pivot_mode == "a pivot point" else " (pick 'a pivot point' above to move it)")
                  + ". Read matrix products right to left: the rightmost matrix acts first.")
    base = plotting.base_figure(4.0, height=560, legend=False)
    for tr in plotting.static_layers(HOUSE, grid_extent=4.0):
        base.add_trace(tr)
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        st.plotly_chart(anim.step_player(base, pivot_steps(th, pcx, pcy), n_sub=14, wrap_at=78), key="rg_steps")
    with right:
        ui.callout("intuition",
                   "The pivot (amber ×) is the only point that ends where it started. In step 3 the rotation happens "
                   "about the origin - that is all a rotation matrix can do - and the two translations around it "
                   "make the origin 'stand in' for the pivot.")


rigid_section()

st.subheader("What changes and what stays", divider="gray")
a, b = st.columns([1.1, 1], gap="large")
with a:
    ui.preserved_badges("euclidean")
    st.write("")
    ui.callout("practice",
               "- A robot's planar pose (x, y, θ) **is** an element of SE(2) - this matrix.\n"
               "- Pick-and-place: the pose of a part on a conveyor seen by an overhead camera.\n"
               "- Rotation augmentation in training data; aligning two scans of the same object.")
with b:
    ui.callout("pitfall",
               "Rotating an image 'by 30°' with a library usually rotates about the **image centre**, but the "
               "raw matrix R rotates about the **origin** (the top-left pixel!). Use T(c)·R·T(−c). "
               "Also: with image axes (y down) a positive angle *looks* clockwise, and many APIs want "
               "**radians**, not degrees.")

ui.whys([
    ("Why did r and φ disappear from the rotation matrix?",
     "Because r cos φ is just x and r sin φ is just y. After the angle-addition step everything that depends on "
     "the point is collected into x and y, and what is left (cos θ, sin θ) depends only on the rotation. "
     "If the matrix had kept an r or a φ, every point would need its own matrix - and rotation would not be "
     "linear."),
    ("Why is the inverse of a rotation simply its transpose?",
     "The columns of R are unit-length and perpendicular (orthonormal). Then RᵀR multiplies each column with "
     "every column: 1 with itself, 0 with the other - the identity. So Rᵀ undoes R. Geometrically Rᵀ = R(−θ): "
     "transposing flips the sign of sin θ, which is turning back by the same angle."),
    ("Why is a positive angle counter-clockwise?",
     "It is a convention inherited from the maths axes (x right, y **up**): angles are measured from the +x axis "
     "towards the +y axis, which is counter-clockwise. In image coordinates y points **down**, so the very same "
     "matrix turns pictures *clockwise* on the screen."),
    ("Why does a rigid motion have 3 degrees of freedom?",
     "One number for the angle θ and two for the shift (tₓ, t_y). The 2×2 block has four entries but they are "
     "tied together by cos²θ + sin²θ = 1 and the pattern [[c, −s], [s, c]] - all four come from one angle."),
    ("Why do lengths stay the same?",
     "|Rp|² = (Rp)ᵀ(Rp) = pᵀRᵀRp = pᵀp = |p|². Because RᵀR = I the length is untouched, and adding t moves both "
     "ends of a segment equally, so their distance cannot change."),
    ("Why rotate about the origin and not about the object's centre?",
     "The matrix has no idea where your object is; the only special point it knows is (0, 0), which every linear "
     "map leaves fixed. Rotating about any other pivot c needs the 'move-rotate-move back' sandwich "
     "T(c)·R·T(−c) shown above."),
])

st.divider()
ui.mcq("rg1", "Which of these 2×2 matrices is a rotation?",
       ["[[0.8, 0.6], [0.6, 0.8]]", "[[0.6, −0.8], [0.8, 0.6]]", "[[1, 1], [0, 1]]", "[[0.6, 0.8], [0.8, −0.6]]"], 1,
       "Columns (0.6, 0.8) and (−0.8, 0.6) are unit length and perpendicular, and det = 0.36 + 0.64 = +1. "
       "The last option is orthonormal too, but det = −1: it is a reflection.")
ui.mcq("rg2", "Rotating about a point c (instead of the origin) is written as …",
       ["R(θ) · T(c)", "T(c) · R(θ) · T(−c)", "T(−c) · R(θ) · T(c)", "R(θ) + c"], 1,
       "Right to left: first move c to the origin (T(−c)), then rotate, then move it back (T(c)).")

ui.lesson_nav(__file__)
