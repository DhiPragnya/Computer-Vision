import numpy as np
import streamlit as st

import plotly.graph_objects as go

from core import anim, live, plotting, shapes, theme, ui
from core import transforms as tf

ui.lesson_header(
    __file__,
    "Stretch or shrink along the axes: <i>x' = sₓ x</i> and <i>y' = s_y y</i>. Equal factors give "
    "<b>uniform</b> scaling (same shape, new size); different factors give <b>non-uniform</b> scaling; "
    "a negative factor turns the shape into its mirror image.",
)

# ------------------------------------------------------------------ derive it
st.subheader("Derive the matrix - don't memorise it", divider="gray")
d1, d2 = st.columns([1.1, 1], gap="large")
with d1:
    st.markdown("Start from what we **want** each new coordinate to be, and write every term explicitly - "
                "including the ones that are zero:")
    st.latex(r"\begin{aligned} x' &= \color{" + theme.E1 + r"}{s_x}\,x + \color{" + theme.ORIGINAL +
             r"}{0}\,y + \color{" + theme.ORIGINAL + r"}{0}\cdot 1 \\ y' &= \color{" + theme.ORIGINAL +
             r"}{0}\,x + \color{" + theme.E2 + r"}{s_y}\,y + \color{" + theme.ORIGINAL + r"}{0}\cdot 1 \\ "
             r"1 &= 0\,x + 0\,y + 1\cdot 1 \end{aligned}")
    st.markdown("The coefficients, read row by row, **are** the matrix:")
    st.latex(r"\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix} \color{" + theme.E1 + r"}{s_x} & 0 & 0 \\ "
             r"0 & \color{" + theme.E2 + r"}{s_y} & 0 \\ 0 & 0 & 1 \end{bmatrix}\begin{bmatrix} x \\ y \\ 1 \end{bmatrix}")
with d2:
    ui.callout("intuition",
               "This is the recipe for **every** matrix in this topic: write the equations you want, line them up "
               "as 'something × x + something × y + something × 1', and copy the somethings into the rows. "
               "A zero in the matrix simply says *this output does not depend on that input*.")
    ui.callout("note",
               "Row 1 builds x', row 2 builds y', row 3 keeps the homogeneous 1. Column 1 multiplies x, column 2 "
               "multiplies y, column 3 multiplies the 1 - which is why translations live in column 3.",
               title="Rows and columns")

HOUSE = shapes.house(offset=(0.3, 0.3))
EAVE = np.array([2.3, 1.8])


@st.cache_resource(show_spinner=False)
@live.full_quality()          # cached: never store a coarse mid-drag version
def derive_figure(sx=1.6, sy=0.7):
    def state(S):
        trs = plotting.layer_traces(HOUSE, S, grid_extent=5.0, n=2)
        q = tf.apply(S, [EAVE])[0]
        trs.append(go.Scatter(x=[q[0]], y=[q[1]], mode="markers+text", text=[f"({q[0]:.2f}, {q[1]:.2f})"],
                              textposition="top right", textfont=dict(size=13),
                              marker=dict(size=13, color=theme.HIGHLIGHT, line=dict(color="white", width=2)),
                              showlegend=False, hoverinfo="skip"))
        return trs

    H, f = anim.hl, ui.num
    x, y = EAVE
    steps = [
        anim.Step("What should happen to a point?", f"follow the eave  (x, y) = ({f(x)}, {f(y)})",
                  "We want widths multiplied by sₓ = 1.6 and heights by s_y = 0.7. Write it one coordinate at a time.",
                  state(np.eye(3))),
        anim.Step("Equation 1: x′ = sₓ·x + 0·y + 0·1",
                  f"x′ = {H('1.6')}·{f(x)} = <b>{f(1.6 * x)}</b>      (y′ still = y)",
                  "Only x is multiplied, so the house stretches sideways - and moves away from the origin. "
                  "The orange arrow e₁ grows to (sₓ, 0).", state(tf.scaling(sx, 1.0))),
        anim.Step("Equation 2: y′ = 0·x + s_y·y + 0·1",
                  f"y′ = {H('0.7')}·{f(y)} = <b>{f(0.7 * y)}</b>",
                  "Now the heights shrink. The green arrow e₂ becomes (0, s_y). The two axes never mix - "
                  "that is why the off-diagonal entries are 0.", state(tf.scaling(sx, sy))),
        anim.Step("Copy the coefficients into rows",
                  f"S = [{H('sₓ', theme.E1)} 0 0 ; 0 {H('s_y', theme.E2)} 0 ; 0 0 1]  =  [1.6 0 0 ; 0 0.7 0 ; 0 0 1]",
                  "Row 1 = the numbers in front of x, y, 1 in equation 1; row 2 = those of equation 2; row 3 keeps "
                  "the 1. The arrows are the first two COLUMNS.", state(tf.scaling(sx, sy))),
    ]
    base = plotting.base_figure(5.0, height=560, legend=False)
    for tr in plotting.static_layers(HOUSE, grid_extent=5.0):
        base.add_trace(tr)
    return anim.step_player(base, steps, n_sub=14, wrap_at=78)


ui.anim_intro("Step by step: build the scaling matrix from its two equations")
d3, d4 = st.columns([1.45, 1], gap="large")
with d3:
    st.plotly_chart(derive_figure(), key="sc_derive")
with d4:
    ui.callout("try", "Press **2** and **3** in turn and watch the arrows: the orange one only changes in step 2 "
                      "(it is the x-column), the green one only in step 3 (the y-column). Drag the slider under the "
                      "picture back and forth to compare.")

# ------------------------------------------------------------------ interactive
def angle_at(P, i):
    u, v = P[i - 1] - P[i], P[(i + 1) % len(P)] - P[i]
    return np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))


@live.section
def play_section():
    st.subheader("Play with it", divider="gray")
    c_ctrl, c_fig, c_txt = st.columns([0.8, 1.6, 1.05], gap="large")
    with c_ctrl:
        uniform = st.toggle("Uniform (sₓ = s_y)", value=False, key="sc_uni")
        sx = live.slider("sₓ", -2.5, 2.5, 1.6, 0.05, key="sc_sx")
        sy = sx if uniform else live.slider("s_y", -2.5, 2.5, 0.7, 0.05, key="sc_sy")
        about = st.segmented_control("Scale about", ["origin", "house centre"], default="origin", key="sc_about",
                                     required=True)
    centre = np.array([1.0 + 0.3, 1.3 + 0.3])

    def H_at(t):
        S = tf.scaling(1 + t * (sx - 1), 1 + t * (sy - 1))
        if about == "house centre":
            S = tf.translation(*centre) @ S @ tf.translation(*-centre)
        return S

    H = H_at(1.0)
    extra = [plotting.points_trace([centre], theme.HIGHLIGHT, "centre (fixed)", size=12, symbol="x")] \
        if about == "house centre" else [plotting.points_trace([[0, 0]], theme.HIGHLIGHT, "origin (fixed)", size=11)]
    with c_fig:
        fig = plotting.transform_figure(HOUSE, H_at, extent=5.0, height=560, extra_static=extra)
        st.plotly_chart(fig, key="sc_fig")
    with c_txt:
        ui.show_matrix("S", H, 2, ui.column_highlight())
        detA = sx * sy
        kind = "uniform" if abs(sx - sy) < 1e-9 else "non-uniform"
        st.markdown(f"**{kind.capitalize()} scaling** · class: **{tf.classify(H)['name']}**")
        m1, m2 = st.columns(2)
        m1.metric("area ×", f"{abs(detA):.2f}", border=True, help="|sₓ · s_y|")
        m2.metric("sₓ / s_y", f"{sx / sy:.2f}" if abs(sy) > 1e-9 else "∞", border=True, help="change of aspect ratio")
        if abs(detA) < 1e-9:
            st.error("A zero factor squashes the plane onto a line: not invertible.", icon=":material/block:")
        elif sx < 0 and sy < 0:
            st.info("Both factors negative: that is a **rotation by 180°**, not a mirror (det > 0). "
                    "Look - the chimney is still on the right of the door.", icon=":material/rotate_right:")
        elif detA < 0:
            st.warning(f"One negative factor: a **mirror image** across the {'y' if sx < 0 else 'x'}-axis "
                       "(det < 0, the chimney swaps sides).", icon=":material/flip:")
        if about == "origin" and abs(sx - 1) + abs(sy - 1) > 1e-6:
            st.caption("Notice the house also **moves**: every point's distance from the origin is scaled, "
                       "not just the house's size. Switch to 'house centre' to scale in place.")

    # ---------------------------------------------------------------- measure
    st.subheader("Measure it", divider="gray")
    pts = np.array([[0, 0], [2, 0], [2, 1.5], [1.0, 2.5], [0, 1.5]]) + 0.3
    q = tf.apply(H, pts)
    w0, h0 = np.ptp(pts[:, 0]), np.ptp(pts[:, 1])
    w1, h1 = np.ptp(q[:, 0]), np.ptp(q[:, 1])
    a, b, c, d = st.columns(4)
    a.metric("Width", f"{w0:.2f} → {w1:.2f}", delta=f"× {w1 / w0:.2f} = |sₓ|", delta_color="off", border=True)
    b.metric("Height", f"{h0:.2f} → {h1:.2f}", delta=f"× {h1 / h0:.2f} = |s_y|", delta_color="off", border=True)
    roof = f"{angle_at(pts, 3):.0f}° → {angle_at(q, 3):.0f}°" if abs(detA) > 1e-9 else f"{angle_at(pts, 3):.0f}° → -"
    c.metric("Roof angle", roof, border=True, help="Unchanged only for uniform scaling.")
    d.metric("Shape kept?", "yes" if abs(abs(sx) - abs(sy)) < 1e-9 and abs(detA) > 1e-9 else "no", border=True,
             help="Only uniform scaling keeps the shape")

    # ---------------------------------------------------------------- live: sweep sₓ through zero
    ui.anim_intro("Live: sweep sₓ from +2.5 through 0 to −2.5",
                  f"Drag the slider under the picture (s_y stays at your value {ui.num(sy)}). Watch the house "
                  "get thinner, collapse onto a line at sₓ = 0 and come back as its mirror image.")
    base = plotting.base_figure(5.0, height=520, legend=False)
    for tr in plotting.static_layers(HOUSE, grid_extent=5.0):
        base.add_trace(tr)
    sweep = np.round(np.arange(2.5, -2.5001, -0.1), 2)

    def make(v):
        S = tf.scaling(v, sy)
        det = v * sy
        if abs(det) < 1e-9:
            verdict = "<span style='color:#e34948'><b>det = 0: squashed onto a line - cannot be undone</b></span>"
        elif det < 0:
            verdict = "det < 0 → <b>mirror image</b> (the chimney is on the other side)"
        else:
            verdict = "det > 0 → orientation kept" + (" (two negatives = a 180° turn)" if v < 0 else "")
        cap = f"sₓ = <b>{v:+.1f}</b>, s_y = {sy:+.2f}   →   area × |sₓ·s_y| = {abs(det):.2f}   ·   {verdict}"
        return plotting.layer_traces(HOUSE, S, grid_extent=5.0, n=2), cap

    st.plotly_chart(anim.scrubber(base, sweep, make, active=int(np.argmin(np.abs(sweep - 1.6))), label="sₓ",
                                  fmt="{:+.1f}", frame_ms=70), key="sc_sweep")


play_section()

e1, e2 = st.columns([1.1, 1], gap="large")
with e1:
    ui.callout("practice",
               "- **Resizing** an image (`cv2.resize`) is scaling - usually about the top-left corner.\n"
               "- **Image pyramids** for multi-scale detection repeatedly scale by ½.\n"
               "- In camera calibration the focal lengths **fₓ, f_y** turn metres on the sensor into pixels: "
               "a non-uniform scaling inside the camera matrix K.\n"
               "- Flipping an image left↔right for data augmentation is sₓ = −1 (about the centre).")
with e2:
    ui.callout("takeaway",
               "Scaling is a diagonal matrix. Uniform scaling (sₓ = s_y) keeps angles and shape and belongs to the "
               "**similarity** family; non-uniform scaling changes angles and belongs to the **affine** family. "
               "Areas always scale by |sₓ s_y|.")

ui.whys([
    ("Why does a negative scale make a mirror image?",
     "With sₓ = −1, every point (x, y) goes to (−x, y): left and right are swapped, like looking in a mirror "
     "standing on the y-axis. Walking around the outline, the corners now come in the **opposite order** "
     "(clockwise ↔ counter-clockwise) - that is what det < 0 means. No sliding or turning in the plane can do "
     "that; you would have to lift the shape out and flip it over."),
    ("If one negative factor mirrors, why don't two negatives mirror twice?",
     "They do - and two mirrors cancel. sₓ = s_y = −1 sends (x, y) to (−x, −y), which is exactly a rotation by "
     "180°: cos 180° = −1, sin 180° = 0, so R(180°) = diag(−1, −1). Try it above: det = (−1)(−1) = +1."),
    ("Why does the house move when I only asked it to grow?",
     "Because the matrix scales **coordinates**, i.e. distances measured from the origin. A corner at x = 2 goes "
     "to x = 4 - it moves away from the origin. Only the origin itself stays put. To grow 'in place' about a "
     "centre c, move c to the origin, scale, move back: T(c)·S·T(−c)."),
    ("Why are the off-diagonal entries zero?",
     "The off-diagonal entry in row 1 multiplies y. A non-zero value would make x' depend on y - that is a "
     "**shear**, not a scaling. Scaling treats the two axes completely independently, so the matrix is diagonal."),
    ("Why does non-uniform scaling change angles?",
     "A 45° diagonal (1, 1) becomes (sₓ, s_y). If sₓ ≠ s_y its angle is no longer 45° - one direction got "
     "stretched more than the other. Only directions along the axes keep their angle. With sₓ = s_y every "
     "direction is stretched equally, so all angles survive."),
    ("Why is the inverse so easy?",
     "Each axis is scaled independently, so you undo each one independently: S(sₓ, s_y)⁻¹ = S(1/sₓ, 1/s_y). "
     "This also shows why a zero factor is forbidden - you cannot divide by zero, and indeed the squashed "
     "information is gone."),
])

st.divider()
ui.mcq("sc1", "What does S = diag(−1, 1, 1) do?",
       ["Mirror across the x-axis", "Mirror across the y-axis", "Rotate by 180°", "Nothing"], 1,
       "x' = −x, y' = y: left and right swap, so the mirror line is the y-axis.")
ui.mcq("sc2", "A 4×2 rectangle is scaled by sₓ = 0.5, s_y = 3. What is the new area?",
       ["8", "12", "24", "4"], 1, "Area 8 × |0.5 · 3| = 8 × 1.5 = 12 (it becomes 2 × 6).")

ui.lesson_nav(__file__)
