import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core import anim, live, plotting, shapes, theme, ui
from core import transforms as tf

ui.lesson_header(
    __file__,
    "Let the 2×2 block be <b>any</b> invertible matrix and you get an affine map: "
    "<i>x' = a x + b y + tₓ</i>, <i>y' = c x + d y + t_y</i>. Squares can become parallelograms and circles "
    "become ellipses - but parallel lines stay parallel. Six unknowns: a, b, c, d, tₓ, t_y.",
)

MODES = ["Matrix entries", "Building blocks", "Three point pairs"]
SRC_TRI = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
HOUSE = shapes.house(0.9, offset=(-0.2, -0.2))
CIRCLE = {"outline": shapes.circle(1.0, n=73), "radius": np.array([[0.0, 0.0], [1.0, 0.0]])}
SEG_A = np.array([[-2.5, -1.5], [1.5, -0.5]])
SEG_B = SEG_A + np.array([0.0, 2.0])               # parallel to seg_a
PERP = np.array([[-2.5, -1.5], [-3.0, 0.5]])       # meets seg_a at a right angle
MID = SEG_A.mean(axis=0)


def rot_flip(angle_deg, flip):
    """R(angle) · diag(1, flip): a rotation (flip = 1) or a reflection (flip = −1) - morphable."""
    return tf.rotation(angle_deg) @ np.diag([1.0, flip, 1.0])


def as_rot_flip(Q2):
    Q2 = np.asarray(Q2, float)
    if np.linalg.det(Q2) < 0:
        return tf.angle_of(Q2 @ np.diag([1.0, -1.0])), -1.0
    return tf.angle_of(Q2), 1.0


def blocks_steps(sx, sy, k, th, tx, ty):
    """A = R(θ) · Shear(k) · Scale(sₓ, s_y), then + t: one factor at a time (right to left)."""
    def M(p):
        return (tf.translation(p["tx"], p["ty"]) @ tf.rotation(p["th"]) @ tf.shear(p["k"])
                @ tf.scaling(p["sx"], p["sy"]))

    def build(p):
        return plotting.layer_traces(HOUSE, M(p), grid_extent=4.5, n=2)

    P = [dict(sx=1.0, sy=1.0, k=0.0, th=0.0, tx=0.0, ty=0.0)]
    P.append(dict(P[-1], sx=sx, sy=sy))
    P.append(dict(P[-1], k=k))
    P.append(dict(P[-1], th=th))
    P.append(dict(P[-1], tx=tx, ty=ty))
    f, H = ui.num, anim.hl
    info = [
        ("Start", "H = I", "Watch the grid squares: each step changes them in exactly one way."),
        ("Scale(sₓ, s_y)", f"x′ = {H(f(sx))}·x,  y′ = {H(f(sy))}·y",
         "Axis-aligned stretch: squares become rectangles. Right angles survive (for now)."),
        ("Shear(k)", f"x′ = x + {H(f(k))}·y",
         "Each row slides sideways in proportion to its height: rectangles become parallelograms. This is the step "
         "that destroys right angles - but parallel lines stay parallel."),
        ("R(θ)", f"rotate by {H(f(th) + '°')}", "Turning changes no shape - the parallelograms just turn."),
        ("+ t", f"slide by ({H(f(tx))}, {H(f(ty))})",
         "H = [R·Shear·Scale | t]. Four 2×2 numbers + two shifts = 6 degrees of freedom."),
    ]
    return [anim.Step(t_, e_, n_, build(P[i]), between=None if i == 0 else anim.morph(build, P[i - 1], P[i]))
            for i, (t_, e_, n_) in enumerate(info)]


def svd_steps(H):
    """A = U Σ Vᵀ: rotate (or flip), stretch along the axes, rotate (or flip) - then translate."""
    U, S, Vt = tf.svd_2x2(H[:2, :2])
    va, vf = as_rot_flip(Vt)
    ua, uf = as_rot_flip(U)

    def M(p):
        return (tf.translation(p["tx"], p["ty"]) @ rot_flip(p["ua"], p["uf"]) @ tf.scaling(p["s1"], p["s2"])
                @ rot_flip(p["va"], p["vf"]))

    def build(p):
        return plotting.layer_traces(CIRCLE, M(p), grid_extent=4.0, n=2, per_edge=1, fill_alpha=0.18)

    P = [dict(va=0.0, vf=1.0, s1=1.0, s2=1.0, ua=0.0, uf=1.0, tx=0.0, ty=0.0)]
    P.append(dict(P[-1], va=va, vf=vf))
    P.append(dict(P[-1], s1=float(S[0]), s2=float(S[1])))
    P.append(dict(P[-1], ua=ua, uf=uf))
    P.append(dict(P[-1], tx=float(H[0, 2]), ty=float(H[1, 2])))
    f, Hh = ui.num, anim.hl
    flip_v = " (a reflection: det Vᵀ = −1)" if vf < 0 else ""
    flip_u = " (a reflection: det U = −1)" if uf < 0 else ""
    info = [
        ("The unit circle", "x = (cos t, sin t)",
         "Every affine map turns this circle into an ellipse. The SVD tells us exactly which one, in three moves."),
        ("Vᵀ: rotate", f"turn by {Hh(f(va, 1) + '°')}{flip_v}",
         "A rotation does nothing to a circle's shape - it only turns the radius line and the arrows. It lines up "
         "the two directions that A will stretch most and least with the x and y axes."),
        ("Σ: stretch along the axes", f"σ₁ = {Hh(f(S[0]))} along x,  σ₂ = {Hh(f(S[1]))} along y",
         "The ONLY step that changes shape: the circle becomes an axis-aligned ellipse with semi-axes σ₁, σ₂. "
         "Area × σ₁σ₂ = |det A|."),
        ("U: rotate again", f"turn by {Hh(f(ua, 1) + '°')}{flip_u}   →   U Σ Vᵀ = A",
         "The ellipse turns into its final direction. The arrows now point to the columns of A."),
        ("+ t: slide", f"t = ({f(H[0, 2])}, {f(H[1, 2])})", "The translation only moves the ellipse."),
    ]
    return [anim.Step(t_, e_, n_, build(P[i]), between=None if i == 0 else anim.morph(build, P[i - 1], P[i]))
            for i, (t_, e_, n_) in enumerate(info)]


@live.section
def affine_section():
    mode = st.segmented_control("How do you want to control A?", MODES, default=MODES[1], key="af_mode",
                                required=True)
    c_ctrl, c_fig, c_txt = st.columns([0.85, 1.6, 1.05], gap="large")
    blocks = None
    with c_ctrl:
        if mode == "Matrix entries":
            st.markdown("Type the matrix directly")
            r1 = st.columns(2)
            with r1[0]:
                a00 = live.slider("a", -2.0, 2.0, 1.2, 0.05, key="af_a00")
            with r1[1]:
                a01 = live.slider("b", -2.0, 2.0, 0.6, 0.05, key="af_a01")
            r2 = st.columns(2)
            with r2[0]:
                a10 = live.slider("c", -2.0, 2.0, 0.2, 0.05, key="af_a10")
            with r2[1]:
                a11 = live.slider("d", -2.0, 2.0, 0.9, 0.05, key="af_a11")
            r3 = st.columns(2)
            with r3[0]:
                tx = live.slider("tₓ", -3.0, 3.0, 0.0, 0.1, key="af_tx1")
            with r3[1]:
                ty = live.slider("t_y", -3.0, 3.0, 0.0, 0.1, key="af_ty1")
            H = tf.affine(a00, a01, a10, a11, tx, ty)
            H_at = H
        elif mode == "Building blocks":
            st.markdown("A = R(θ) · Shear(k) · Scale(sₓ, s_y)")
            sx = live.slider("scale sₓ", -2.0, 2.0, 1.4, 0.05, key="af_sx", help="Negative = mirror")
            sy = live.slider("scale s_y", 0.1, 2.0, 0.8, 0.05, key="af_sy")
            k = live.slider("shear k", -1.5, 1.5, 0.6, 0.05, key="af_k")
            th = live.slider("rotation θ (degrees)", -180, 180, 15, 1, key="af_th", format="%d°")
            r3 = st.columns(2)
            with r3[0]:
                tx = live.slider("tₓ", -3.0, 3.0, 0.0, 0.1, key="af_tx2")
            with r3[1]:
                ty = live.slider("t_y", -3.0, 3.0, 0.0, 0.1, key="af_ty2")
            blocks = (sx, sy, k, th, tx, ty)

            def H_at(t):
                return (tf.translation(t * tx, t * ty) @ tf.rotation(t * th) @ tf.shear(t * k)
                        @ tf.scaling(1 + t * (sx - 1), 1 + t * (sy - 1)))

            H = H_at(1.0)
        else:
            st.markdown("Move the 3 corners of the triangle - A and **t** are solved for you.")
            defaults = [(0.5, -0.5), (3.0, 0.5), (1.2, 2.5)]
            dst = []
            for i, (dx, dy) in enumerate(defaults):
                r = st.columns(2)
                with r[0]:
                    px = live.slider(f"P{i + 1}' x", -4.0, 4.0, dx, 0.1, key=f"af_p{i}x")
                with r[1]:
                    py = live.slider(f"P{i + 1}' y", -4.0, 4.0, dy, 0.1, key=f"af_p{i}y")
                dst.append([px, py])
            dst = np.array(dst)
            H = tf.fit_affine(SRC_TRI, dst)
            H_at = H
        st.divider()
        show_sq = st.toggle("Unit square → area", value=True, key="af_sq")
        show_circ = st.toggle("Unit circle → ellipse", value=False, key="af_circ")

    A = H[:2, :2]
    detA = float(np.linalg.det(A))
    U, S, Vt = tf.svd_2x2(A)

    def extras(Ht, t):
        out = []
        if show_sq:
            sq = plotting.mapped(Ht, shapes.densify(shapes.rect(0, 0, 1, 1)))
            out.append(go.Scatter(x=sq[:, 0], y=sq[:, 1], mode="lines", fill="toself",
                                  fillcolor=theme.rgba(theme.HIGHLIGHT, 0.28),
                                  line=dict(color=theme.HIGHLIGHT, width=1.5),
                                  name="unit square → area |det A|", hoverinfo="skip"))
        if show_circ:
            c = plotting.mapped(Ht, shapes.circle(1.0))
            out.append(go.Scatter(x=c[:, 0], y=c[:, 1], mode="lines", line=dict(color=theme.MAGENTA, width=2),
                                  name="unit circle → ellipse", hoverinfo="skip"))
        if mode == "Three point pairs":
            out.append(plotting.points_trace(tf.apply(Ht, SRC_TRI), theme.TRANSFORMED, "P₁' P₂' P₃'",
                                             labels=["P₁'", "P₂'", "P₃'"], size=12))
        return out

    static = []
    if show_circ:
        static.append(go.Scatter(x=shapes.circle(1.0)[:, 0], y=shapes.circle(1.0)[:, 1], mode="lines",
                                 line=dict(color=theme.ORIGINAL, width=1.2, dash="dot"), name="unit circle",
                                 hoverinfo="skip", showlegend=False))
        t_ = H[:2, 2]
        for i in range(2):
            static.append(plotting.arrow_trace(t_, t_ + S[i] * U[:, i], theme.MAGENTA, f"σ{i + 1} u{i + 1}", width=2,
                                               showlegend=False, dash="dot"))
    part = HOUSE if mode != "Three point pairs" else {"outline": shapes.densify(np.vstack([SRC_TRI, SRC_TRI[:1]]))}
    if mode == "Three point pairs":
        static.append(plotting.points_trace(SRC_TRI, theme.ORIGINAL, "P₁ P₂ P₃", labels=["P₁", "P₂", "P₃"], size=10,
                                            showlegend=False))

    with c_fig:
        fig = plotting.transform_figure(part, H_at, extent=4.5, height=560, extra_static=static, extra_dynamic=extras)
        st.plotly_chart(fig, key="af_fig")

    with c_txt:
        ui.show_matrix("H", H, 2, ui.column_highlight())
        st.latex(rf"\det A = {ui.num(detA, 3)}")
        if abs(detA) < 1e-3:
            st.error("det A ≈ 0: the plane is squashed onto a line. Not invertible - information is lost.",
                     icon=":material/block:")
        elif detA < 0:
            st.warning(f"det A < 0: areas scale by |det A| = {abs(detA):.2f} **and** orientation flips (mirror "
                       "image).", icon=":material/flip:")
        else:
            st.info(f"Every area is multiplied by det A = **{detA:.2f}** - look at the amber unit square.",
                    icon=":material/square_foot:")
        st.markdown("**Singular value decomposition** of A")
        st.latex(r"A = U\,\Sigma\,V^T")
        st.latex(rf"\Sigma = \mathrm{{diag}}({ui.num(S[0], 2)},\ {ui.num(S[1], 2)})")
        st.caption(f"The unit circle becomes an ellipse with semi-axes σ₁ = {S[0]:.2f} and σ₂ = {S[1]:.2f}. "
                   f"Anisotropy σ₁/σ₂ = {S[0] / max(S[1], 1e-9):.2f} - equal to 1 only for a similarity.")
        if mode == "Three point pairs":
            st.caption("3 pairs × 2 equations = 6 equations for the 6 unknowns - an affine map is fixed by "
                       "three non-collinear points (cv2.getAffineTransform does exactly this).")

    if blocks is not None:
        ui.anim_intro("Step by step: H = T(t) · R(θ) · Shear(k) · Scale(sₓ, s_y)",
                      "Uses your sliders. Read the product right to left: scale, then shear, then rotate, then slide.")
        base = plotting.base_figure(4.5, height=560, legend=False)
        for tr in plotting.static_layers(HOUSE, grid_extent=4.5):
            base.add_trace(tr)
        left, right = st.columns([1.5, 1], gap="large")
        with left:
            st.plotly_chart(anim.step_player(base, blocks_steps(*blocks), n_sub=14, wrap_at=80), key="af_blocks")
        with right:
            ui.callout("intuition",
                       "Only ONE of the four building blocks bends right angles: the shear. Scale stretches along "
                       "the axes, rotation and translation are rigid. Yet all four keep parallel lines parallel - "
                       "so their product does too.")

    # ------------------------------------------------------------------ invariants
    st.subheader("Test the invariants yourself", divider="gray")
    left, right = st.columns([1.1, 1.2], gap="large")
    with left:
        fig = plotting.base_figure(5.0, height=420)
        for P, name, col in [(SEG_A, "segment 1", theme.E1), (SEG_B, "segment 2 (parallel to 1)", theme.E2),
                             (PERP, "segment 3 (⟂ to 1)", theme.MAGENTA)]:
            fig.add_trace(plotting.line_trace(P, theme.ORIGINAL, name, width=1.5, dash="dash", showlegend=False))
            fig.add_trace(plotting.line_trace(tf.apply(H, P), col, name, width=3))
        fig.add_trace(plotting.points_trace([MID], theme.ORIGINAL, "midpoint before", size=9, showlegend=False))
        fig.add_trace(plotting.points_trace(tf.apply(H, [MID]), theme.HIGHLIGHT, "image of the midpoint", size=13,
                                            symbol="diamond"))
        st.plotly_chart(fig, key="af_inv")
    with right:
        st.dataframe(invariant_table(H), hide_index=True, width="stretch")
        ui.callout("intuition",
                   "An affine map treats **every place in the plane the same way** (A does not depend on x). "
                   "So two parallel segments get stretched identically and stay parallel, and 'halfway along' stays "
                   "'halfway along'. Angles, however, are at the mercy of the shear.")

    # ------------------------------------------------------------------ SVD story
    st.subheader("Every affine map is: rotate → stretch along the axes → rotate", divider="gray")
    ui.anim_intro("Step by step: A = U Σ Vᵀ on the unit circle (uses the matrix above)",
                  "Change the matrix with the controls above and replay. The radius line shows how the circle turns; "
                  "only the Σ step changes its shape.")
    base = plotting.base_figure(3.5, height=560, legend=False)
    for tr in plotting.static_layers(CIRCLE, grid_extent=4.0):
        base.add_trace(tr)
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        st.plotly_chart(anim.step_player(base, svd_steps(H), n_sub=14, wrap_at=80), key="af_svd")
    with right:
        st.latex(r"A = \underbrace{U}_{\text{rotate}}\;\underbrace{\Sigma}_{\text{stretch}}\;"
                 r"\underbrace{V^T}_{\text{rotate}}")
        st.caption(f"Here: Vᵀ turns by {tf.angle_of(Vt):.1f}°, Σ stretches by σ₁ = {S[0]:.2f} along x and "
                   f"σ₂ = {S[1]:.2f} along y, U turns by {tf.angle_of(U):.1f}°. If det A < 0 one of the two turns is a "
                   "reflection - you will see the circle flip through a line.")
        ui.callout("takeaway",
                   "However complicated A looks, it only ever does three simple things: turn, stretch along two "
                   "perpendicular directions, turn. σ₁ and σ₂ measure how much it distorts; their product is |det A|.")


def invariant_table(H):
    qa, qb, qp = tf.apply(H, SEG_A), tf.apply(H, SEG_B), tf.apply(H, PERP)
    da, db, dp = np.diff(qa, axis=0)[0], np.diff(qb, axis=0)[0], np.diff(qp, axis=0)[0]
    na, nb, npp = (max(np.linalg.norm(v), 1e-12) for v in (da, db, dp))
    ang_par = np.degrees(np.arccos(np.clip(abs(da @ db) / na / nb, -1, 1)))
    ang_perp = np.degrees(np.arccos(np.clip(abs(da @ dp) / na / npp, -1, 1)))
    mid_err = np.linalg.norm(tf.apply(H, [MID])[0] - qa.mean(axis=0))
    r_before = np.linalg.norm(np.diff(SEG_B, axis=0)) / np.linalg.norm(np.diff(SEG_A, axis=0))
    r_after = nb / na
    return {
        "check": ["angle between segment 1 & 2", "angle between segment 1 & 3", "midpoint of 1 → midpoint of image?",
                  "length ratio  seg 2 / seg 1"],
        "before": ["0° (parallel)", "90°", "-", f"{r_before:.3f}"],
        "after": [f"{ang_par:.2f}°", f"{ang_perp:.1f}°", "exact" if mid_err < 1e-9 else f"off by {mid_err:.2g}",
                  f"{r_after:.3f}"],
        "preserved?": ["✓ yes", "✗ no" if abs(ang_perp - 90) > 0.05 else "✓ (this A happens to keep it)",
                       "✓ yes", "✓ yes"],
    }


affine_section()


# ------------------------------------------------------------------ estimating from 3 correspondences
def solve_steps(S3, D3):
    """Row 1 of the affine matrix fixes every x′, row 2 fixes every y′ - watch them act one after the other."""
    A3 = tf.fit_affine(S3, D3)
    row1, row2 = A3[0], A3[1]

    def M(p):
        r1 = (1 - p["r1"]) * np.array([1.0, 0.0, 0.0]) + p["r1"] * row1
        r2 = (1 - p["r2"]) * np.array([0.0, 1.0, 0.0]) + p["r2"] * row2
        return np.array([r1, r2, [0.0, 0.0, 1.0]])

    def build(p):
        Mp = M(p)
        q = tf.apply(Mp, S3)
        tri = np.vstack([q, q[:1]])
        return [plotting.grid_trace(Mp, 4, 1, theme.GRID_TRANSFORMED),
                go.Scatter(x=tri[:, 0], y=tri[:, 1], mode="lines", fill="toself",
                           fillcolor=theme.rgba(theme.HIGHLIGHT, 0.25), line=dict(color=theme.HIGHLIGHT, width=2.5),
                           showlegend=False, hoverinfo="skip"),
                go.Scatter(x=q[:, 0], y=q[:, 1], mode="markers+text", text=["1", "2", "3"], textposition="top left",
                           marker=dict(size=12, color=theme.HIGHLIGHT, line=dict(color="white", width=1.5)),
                           showlegend=False, hoverinfo="skip")]

    P = [dict(r1=0.0, r2=0.0), dict(r1=1.0, r2=0.0), dict(r1=1.0, r2=1.0)]
    f, H = ui.num, anim.hl
    info = [
        ("Source triangle (amber) and target triangle (blue)", "unknowns: a, b, tₓ (row 1) and c, d, t_y (row 2)",
         "Each corner must land on its target: 3 corners × 2 coordinates = 6 equations."),
        ("Solve row 1 → every x′ is fixed",
         f"[a, b, tₓ] = M⁻¹ x′ = [{H(f(row1[0]))}, {H(f(row1[1]))}, {H(f(row1[2]))}]",
         "Row 1 decides only x′ = a x + b y + tₓ. The corners slide HORIZONTALLY until each sits exactly above or "
         "below its target - and the whole grid slides with them."),
        ("Solve row 2 → every y′ is fixed",
         f"[c, d, t_y] = M⁻¹ y′ = [{H(f(row2[0]))}, {H(f(row2[1]))}, {H(f(row2[2]))}]",
         "Row 2 decides only y′ = c x + d y + t_y: the corners slide VERTICALLY onto their targets. Same matrix M - "
         "two small 3×3 solves instead of one 6×6."),
    ]
    return [anim.Step(t_, e_, n_, build(P[i]), between=None if i == 0 else anim.morph(build, P[i - 1], P[i]))
            for i, (t_, e_, n_) in enumerate(info)]


@live.section
def three_points_section():
    st.subheader("Estimating an affine map from 3 point pairs - by hand", divider="gray")
    st.markdown("Each pair (xᵢ, yᵢ) → (x′ᵢ, y′ᵢ) gives **two** equations, so 3 pairs give 3 × 2 = 6 equations for "
                "the 6 unknowns. Move the source triangle **and** the target triangle:")
    c_src, c_dst, c_fig2 = st.columns([0.8, 0.8, 1.6], gap="large")
    src_def = [(-2.0, -1.0), (1.0, -1.5), (-0.5, 1.5)]
    dst_def = [(0.5, -2.0), (3.0, -0.5), (0.8, 2.0)]
    with c_src:
        st.markdown(f"<span style='color:{theme.ORIGINAL}'><b>Source points</b></span>", unsafe_allow_html=True)
        S3 = np.array([[live.slider(f"x{i + 1}", -3.0, 3.0, sx_, 0.1, key=f"af3_sx{i}"),
                        live.slider(f"y{i + 1}", -3.0, 3.0, sy_, 0.1, key=f"af3_sy{i}")]
                       for i, (sx_, sy_) in enumerate(src_def)])
    with c_dst:
        st.markdown(f"<span style='color:{theme.TRANSFORMED}'><b>Target points</b></span>", unsafe_allow_html=True)
        D3 = np.array([[live.slider(f"x′{i + 1}", -3.5, 3.5, dx_, 0.1, key=f"af3_dx{i}"),
                        live.slider(f"y′{i + 1}", -3.5, 3.5, dy_, 0.1, key=f"af3_dy{i}")]
                       for i, (dx_, dy_) in enumerate(dst_def)])
    M3 = np.c_[S3, np.ones(3)]
    detM = float(np.linalg.det(M3))
    signed_area = 0.5 * detM
    ok = abs(detM) > 1e-6
    with c_fig2:
        fig = plotting.base_figure(3.6, height=470)
        tri = np.vstack([S3, S3[:1]])
        fig.add_trace(go.Scatter(x=tri[:, 0], y=tri[:, 1], mode="lines", fill="toself",
                                 fillcolor=theme.rgba(theme.HIGHLIGHT, 0.25), line=dict(color=theme.ORIGINAL, dash="dash"),
                                 name=f"source triangle (area = det M / 2 = {signed_area:.2f})", hoverinfo="skip"))
        fig.add_trace(plotting.points_trace(S3, theme.ORIGINAL, "source", labels=["1", "2", "3"], size=10,
                                            showlegend=False))
        tri2 = np.vstack([D3, D3[:1]])
        fig.add_trace(go.Scatter(x=tri2[:, 0], y=tri2[:, 1], mode="lines", line=dict(color=theme.TRANSFORMED, width=2.5),
                                 name="target triangle", hoverinfo="skip"))
        fig.add_trace(plotting.points_trace(D3, theme.TRANSFORMED, "target", labels=["1′", "2′", "3′"], size=11,
                                            showlegend=False))
        if ok:
            A3 = tf.fit_affine(S3, D3)
            fig.add_trace(plotting.grid_trace(None, 3, 1, theme.GRID_ORIGINAL))
            fig.add_trace(plotting.grid_trace(A3, 3, 1, theme.GRID_TRANSFORMED))
        st.plotly_chart(fig, key="af3_fig")

    r1, r2 = st.columns(2, gap="large")
    with r1:
        st.markdown("**Row 1 only involves a, b, tₓ** (the x′ equations):")
        st.latex(r"\underbrace{" + ui.tex_matrix(M3, 1) + r"}_{M}\begin{bmatrix} a \\ b \\ t_x \end{bmatrix} = "
                 + ui.tex_vec(D3[:, 0], 1))
        st.markdown("**Row 2 only involves c, d, t_y** (the y′ equations) - same M:")
        st.latex(r"M\begin{bmatrix} c \\ d \\ t_y \end{bmatrix} = " + ui.tex_vec(D3[:, 1], 1))
    with r2:
        st.latex(rf"\det M = {detM:.3f} = 2 \times \text{{area}}(\triangle_{{\text{{source}}}})")
        if ok:
            row1 = np.linalg.solve(M3, D3[:, 0])
            row2 = np.linalg.solve(M3, D3[:, 1])
            st.latex(r"\begin{bmatrix} a \\ b \\ t_x \end{bmatrix} = M^{-1}" + ui.tex_vec(D3[:, 0], 1) + " = " +
                     ui.tex_vec(row1, 3) + r",\quad \begin{bmatrix} c \\ d \\ t_y \end{bmatrix} = " +
                     ui.tex_vec(row2, 3))
            ui.show_matrix("A", np.vstack([row1, row2, [0, 0, 1]]), 3)
        else:
            st.error("det M = 0: the three source points are **collinear**. The triangle has no area, M has no "
                     "inverse, and infinitely many affine maps fit - the transformation cannot be determined.",
                     icon=":material/block:")
        st.caption("Try it: drag source point 3 onto the line through points 1 and 2 and watch det M (and the amber "
                   "area) shrink to zero.")

    # the block keeps its size even when det M = 0, so the page does not jump while a point is dragged
    ui.anim_intro("Step by step: what each solved row does",
                  "The amber triangle starts on the source points. Step 2 applies only row 1 (a, b, tₓ); step 3 "
                  "adds row 2 (c, d, t_y). Uses the points you set above.")
    base = plotting.base_figure(3.6, height=540, legend=False)
    base.add_trace(plotting.grid_trace(None, 4, 1, theme.GRID_ORIGINAL))
    tri2 = np.vstack([D3, D3[:1]])
    base.add_trace(go.Scatter(x=tri2[:, 0], y=tri2[:, 1], mode="lines+markers",
                              line=dict(color=theme.TRANSFORMED, width=2.5, dash="dot"),
                              marker=dict(size=10, color=theme.TRANSFORMED), showlegend=False, hoverinfo="skip"))
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        if ok:
            st.plotly_chart(anim.step_player(base, solve_steps(S3, D3), n_sub=16, wrap_at=80), key="af3_steps")
        else:
            src = np.vstack([S3, S3[:1]])
            base.add_trace(go.Scatter(x=src[:, 0], y=src[:, 1], mode="lines+markers",
                                      line=dict(color=theme.HIGHLIGHT, width=2.5),
                                      marker=dict(size=10, color=theme.HIGHLIGHT), showlegend=False, hoverinfo="skip"))
            base.add_annotation(text="<b>det M = 0</b>: the source points are collinear,<br>so there is no unique "
                                     "solution to animate.", x=0.5, y=1.0, xref="paper", yref="paper",
                                xanchor="center", yanchor="bottom", showarrow=False, font=dict(size=13))
            base.update_layout(margin=dict(t=60))
            st.plotly_chart(base, key="af3_steps_none")
    with right:
        ui.callout("intuition",
                   "Because x′ does not depend on c, d, t_y and y′ does not depend on a, b, tₓ, the 6-unknown "
                   "problem splits into two independent 3-unknown problems. In the animation that is visible as "
                   "two separate moves: first purely horizontal, then purely vertical.")


three_points_section()


# ------------------------------------------------------------------ live: shear sweep
@live.section
def shear_sweep_section():
    st.subheader("Live: what a shear keeps and what it breaks", divider="gray")
    ui.anim_intro("Drag the shear k under the picture",
                  "x′ = x + k·y. Segment 2 stays parallel to segment 1 for every k, the midpoint stays the midpoint - "
                  "but the right angle between segments 1 and 3 bends.")
    base = plotting.base_figure(5.0, height=500, legend=False)
    for P in (SEG_A, SEG_B, PERP):
        base.add_trace(plotting.line_trace(P, theme.ORIGINAL, "before", width=1.5, dash="dash", showlegend=False))
    ks = np.round(np.arange(-1.5, 1.5001, 0.05), 2)

    def make(kv):
        Hk = tf.shear(kv)
        trs = [plotting.grid_trace(Hk, 6, 1, theme.GRID_TRANSFORMED)]
        for P, col in [(SEG_A, theme.E1), (SEG_B, theme.E2), (PERP, theme.MAGENTA)]:
            trs.append(plotting.line_trace(tf.apply(Hk, P), col, "seg", width=3.2, showlegend=False))
        trs.append(plotting.points_trace(tf.apply(Hk, [MID]), theme.HIGHLIGHT, "midpoint", size=13, symbol="diamond",
                                         showlegend=False))
        t = invariant_table(Hk)
        cap = (f"k = <b>{kv:+.2f}</b>   ·   angle 1 & 2: <b>{t['after'][0]}</b> (parallel kept)   ·   "
               f"angle 1 & 3: <b>{t['after'][1]}</b> (was 90°)   ·   length ratio 2/1: {t['after'][3]}   ·   area × 1")
        return trs, cap

    st.plotly_chart(anim.scrubber(base, ks, make, active=int(np.argmin(np.abs(ks - 0.6))), label="k", fmt="{:+.2f}",
                                  frame_ms=60), key="af_shear_scrub")


shear_sweep_section()

a, b = st.columns(2, gap="large")
with a:
    ui.preserved_badges("affine")
    st.write("")
    ui.callout("practice",
               "- **Data augmentation**: random shear / aspect-ratio changes (`cv2.warpAffine`).\n"
               "- **Distant planes**: when a planar object is far from the camera, perspective is negligible and a "
               "homography is well approximated by an affine map (the *weak-perspective* camera).\n"
               "- **Affine-covariant regions**: features that adapt an ellipse to the local patch.")
with b:
    ui.callout("pitfall",
               "OpenCV's `warpAffine` takes the **2×3** top part [A | t], while `warpPerspective` takes the full "
               "3×3. And both map **destination pixels back to source** internally (inverse mapping) - see the "
               "*Warping images* lesson.")

ui.whys([
    ("Why do exactly three point pairs determine an affine map?",
     "Six unknowns (a, b, c, d, tₓ, t_y) and each pair gives two equations (one for x′, one for y′). "
     "3 × 2 = 6 - just enough. With two pairs you would have only 4 equations and a whole family of solutions; "
     "with four or more you have more equations than unknowns and use least squares."),
    ("Why must the three source points not be collinear?",
     "det M equals twice the triangle's (signed) area. Collinear points make a flat triangle with zero area, so "
     "det M = 0 and M cannot be inverted. Intuitively, three points on a line only tell you what happens "
     "*along that line* - nothing pins down how the rest of the plane moves sideways."),
    ("Why can the x′ and y′ equations be solved separately?",
     "Look at x′ = a x + b y + tₓ: it contains only a, b, tₓ. And y′ = c x + d y + t_y contains only c, d, t_y. "
     "The two rows of the affine matrix never mix, so we solve two small 3×3 systems with the **same** matrix M."),
    ("Why do parallel lines stay parallel?",
     "A line in direction **v** maps to direction A**v** - the same A everywhere, and t only shifts. Two parallel "
     "lines share direction **v**, so both end up with direction A**v**: still parallel. (In a homography the "
     "division by w′ changes from place to place, which is what breaks this.)"),
    ("Why is the area factor the determinant?",
     "The unit square spanned by e₁ and e₂ becomes the parallelogram spanned by the two columns of A, and the "
     "area of a parallelogram spanned by (a, c) and (b, d) is |ad − bc| = |det A|. Every region is made of tiny "
     "squares, so every area scales the same way."),
    ("Why is the last row (0, 0, 1)?",
     "So that the third coordinate stays 1: w′ = 0·x + 0·y + 1·1 = 1. Nothing needs to be divided, and the map "
     "stays affine. Letting that row change is exactly the step to a projective transformation."),
])

st.divider()
ui.mcq("af1", "Which statement about affine maps is FALSE?",
       ["Parallel lines stay parallel", "The midpoint of a segment maps to the midpoint of its image",
        "A circle always maps to a circle", "Area ratios between regions are preserved"], 2,
       "A circle maps to an ellipse with semi-axes σ₁, σ₂ (the singular values of A). It stays a circle only "
       "when σ₁ = σ₂, i.e. when the map is a similarity.")
ui.mcq("af2", "A = [[2, 1], [0, 3]]. By how much does the area of any region change?",
       ["× 2", "× 3", "× 5", "× 6"], 3,
       "Area scales by |det A| = |2·3 − 1·0| = 6.")

ui.lesson_nav(__file__)
