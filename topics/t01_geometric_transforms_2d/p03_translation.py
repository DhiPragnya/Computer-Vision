import numpy as np
import streamlit as st

import plotly.graph_objects as go

from core import anim, live, plotting, shapes, theme, ui
from core import transforms as tf

ui.lesson_header(
    __file__,
    "The simplest move: slide everything by the same vector <b>t</b>. Nothing turns, nothing stretches. "
    "It is the bottom rung of the ladder - 2 degrees of freedom - and the one that needed homogeneous "
    "coordinates to become a matrix.",
)

HOUSE = shapes.house(offset=(-1.0, -1.3))
CORNERS = np.array([[0, 0], [2, 0], [2, 1.5], [1, 2.5], [0, 1.5]]) + np.array([-1.0, -1.3])


def product_steps(tx, ty):
    """x′ = T x, one row of the matrix at a time, for the eave corner - then for every point."""
    v = CORNERS[2]
    vx, vxy = v + [tx, 0], v + [tx, ty]

    def state(house_H, pt, xa, ya, all_s):
        trs = plotting.layer_traces(HOUSE, house_H, grid=False, basis=False)
        trs.append(plotting.arrows_trace([v], [xa], theme.E1, width=3, size=13))
        trs.append(plotting.arrows_trace([vx], [ya], theme.E2, width=3, size=13))
        trs.append(plotting.arrows_trace(CORNERS, CORNERS + all_s * np.array([tx, ty]), theme.HIGHLIGHT, width=1.8))
        trs.append(go.Scatter(x=[pt[0]], y=[pt[1]], mode="markers", marker=dict(size=14, color=theme.HIGHLIGHT,
                              line=dict(color="white", width=2)), showlegend=False, hoverinfo="skip"))
        return trs

    f, H = ui.num, anim.hl
    I = np.eye(3)
    return [
        anim.Step("Pick one point", f"p = (x, y, 1) = ({f(v[0])}, {f(v[1])}, 1)",
                  "We follow the eave of the house through the product T·p, one row at a time.",
                  state(I, v, v, vx, 0.0)),
        anim.Step("Row 1 gives x′", f"x′ = 1·x + 0·y + {H('tₓ·1')} = {f(v[0])} + {H(f(tx))} = <b>{f(vx[0])}</b>",
                  "The identity part copies x; the last column ADDS tₓ (because the third coordinate is 1). "
                  "The point slides horizontally.",
                  state(I, vx, vx, vx, 0.0)),
        anim.Step("Row 2 gives y′", f"y′ = 0·x + 1·y + {H('t_y·1')} = {f(v[1])} + {H(f(ty))} = <b>{f(vxy[1])}</b>",
                  "Same idea one row down: y is copied and t_y is added. The point slides vertically.",
                  state(I, vxy, vx, vxy, 0.0)),
        anim.Step("Row 3 keeps w = 1", "w′ = 0·x + 0·y + 1·1 = " + H("1") + "   →  nothing to divide",
                  "The last row (0, 0, 1) only re-creates the 1, so (x′, y′) can be read off directly.",
                  state(I, vxy, vx, vxy, 0.0)),
        anim.Step("Every point gets the same t", f"T·p = p + t   for EVERY p,  t = ({f(tx)}, {f(ty)})",
                  "The amber arrows are all identical: same length, same direction. That is what makes it a "
                  "translation.",
                  state(tf.translation(tx, ty), vxy, vx, vxy, 1.0)),
    ]


@live.section
def translation_section():
    c_ctrl, c_fig, c_math = st.columns([0.8, 1.6, 1.05], gap="large")
    with c_ctrl:
        st.markdown("**Controls**")
        tx = live.slider("tₓ  (shift right)", -3.0, 3.0, 1.5, 0.1, key="tr_tx")
        ty = live.slider("t_y  (shift up)", -3.0, 3.0, 0.8, 0.1, key="tr_ty")
        show_vec = st.toggle("Show displacement of every corner", value=True, key="tr_vec")
        show_grid = st.toggle("Show the moving grid", value=True, key="tr_grid")
        ui.callout("try", "Drag tₓ and t_y and watch the house and the matrix follow. Set both to 0: the "
                          "matrix becomes the identity. **▶ Play** glides from the identity to your translation.")

    T = tf.translation(tx, ty)

    def displacement(Ht, t):
        if not show_vec:
            return []
        moved = tf.apply(Ht, CORNERS)
        return [plotting.arrow_trace(a, b, theme.HIGHLIGHT, name="displacement t (same for every point)",
                                     width=2, showlegend=(i == 0), size=10) for i, (a, b) in enumerate(zip(CORNERS, moved))]

    with c_fig:
        fig = plotting.transform_figure(HOUSE, lambda t: tf.translation(t * tx, t * ty), extent=4.5,
                                        grid=show_grid, height=540, extra_dynamic=displacement)
        st.plotly_chart(fig, key="tr_fig")

    with c_math:
        st.markdown("**The matrix (live)**")
        ui.show_matrix("T", T, highlight={(0, 2): theme.HIGHLIGHT, (1, 2): theme.HIGHLIGHT})
        v = CORNERS[2]
        st.markdown(f"**Follow one corner** - the eave at ({ui.num(v[0])}, {ui.num(v[1])}):")
        st.latex(rf"{ui.tex_matrix(T)} {ui.tex_vec([v[0], v[1], 1])} = "
                 rf"{ui.tex_vec([v[0] + tx, v[1] + ty, 1])}")
        st.caption("The last column does all the work; the identity block in the top-left leaves "
                   "directions untouched.")
        st.metric("Degrees of freedom", 2, help="tₓ and t_y", border=True)

    ui.anim_intro("Step by step: the matrix-vector product T·p, row by row",
                  "Uses your tₓ and t_y. ▶ Play runs all steps; the numbered buttons replay one step; drag the "
                  "slider under the picture to scrub between steps.")
    base = plotting.base_figure(4.5, height=560, legend=False)
    for tr in plotting.static_layers(HOUSE, grid_extent=5.0, basis=False):
        base.add_trace(tr)
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        st.plotly_chart(anim.step_player(base, product_steps(tx, ty), n_sub=12, wrap_at=72),
                        key="tr_steps")
    with right:
        st.latex(r"\begin{bmatrix}x'\\y'\\1\end{bmatrix} = "
                 r"\begin{bmatrix}1&0&\color{#eda100}{t_x}\\0&1&\color{#eda100}{t_y}\\0&0&1\end{bmatrix}"
                 r"\begin{bmatrix}x\\y\\1\end{bmatrix} = \begin{bmatrix}x+t_x\\y+t_y\\1\end{bmatrix}")
        ui.callout("intuition",
                   "Read a matrix-vector product **one row at a time**: each row builds one coordinate of the "
                   "answer. Row 1 → x′ (orange arrow), row 2 → y′ (green arrow), row 3 → the 1 that keeps "
                   "the point a point.")


translation_section()

st.subheader("What changes and what stays", divider="gray")
a, b = st.columns([1.1, 1], gap="large")
with a:
    ui.preserved_badges("translation")
    st.write("")
    ui.callout("intuition",
               "Every arrow in the figure is identical: same length, same direction. That is the definition of "
               "a translation - it is the **only** transform in the hierarchy that keeps **orientation** "
               "(a line pointing north-east still points north-east).")
with b:
    with st.container(border=True):
        st.markdown(":material/function: **Algebra you get for free**")
        st.latex(r"T(\mathbf{a})\,T(\mathbf{b}) = T(\mathbf{a}+\mathbf{b}) = T(\mathbf{b})\,T(\mathbf{a})")
        st.latex(r"T(\mathbf{t})^{-1} = T(-\mathbf{t})")
        st.caption("Two shifts add up, their order does not matter (they commute), and you undo a shift by "
                   "shifting back. Translations form a group isomorphic to (ℝ², +).")

c1, c2 = st.columns(2, gap="large")
with c1:
    ui.callout("practice",
               "- **Template matching / cross-correlation** searches over all translations of a patch.\n"
               "- **Video stabilisation** first estimates the frame-to-frame shift.\n"
               "- **Optical flow** describes motion *locally* as a tiny translation per pixel.\n"
               "- **Cropping** an image is a translation of the coordinate system.")
with c2:
    ui.callout("pitfall",
               "In **image** coordinates the y-axis points **down** (row index grows downwards). "
               "A positive t_y moves a picture *down* the screen, not up. Always check which convention "
               "a library uses before you trust a sign.")

ui.whys([
    ('Why does translation live in the third column?',
     'The third column multiplies the homogeneous 1. So it adds a constant - tₓ to x′ and t_y to y′ - no matter where the point is. A constant added to every point is exactly a translation.'),
    ("Why don't the orange and green arrows change during a translation?",
     'Those arrows are the first two columns of the matrix, and for a translation they are still (1, 0) and (0, 1). A translation changes no direction and no length - only position.'),
    ('Why is the last row (0, 0, 1)?',
     'It produces the new third coordinate: 0·x + 0·y + 1·1 = 1. Keeping w = 1 means no division is needed and the result is directly the new (x′, y′).'),
    ("Why can't a 2×2 matrix translate?",
     'Any matrix times the zero vector gives zero, so a 2×2 matrix always leaves the origin where it is. A translation must move the origin - impossible without the extra coordinate.'),
])

st.divider()
ui.mcq("tr1", "A translation maps (1, 2) to (4, 0). Where does it map (−2, 5)?",
       ["(1, 3)", "(−5, 7)", "(2, 10)", "(1, 7)"], 0,
       "The displacement is t = (4, 0) − (1, 2) = (3, −2), and it is the same for every point: "
       "(−2, 5) + (3, −2) = (1, 3).")

ui.lesson_nav(__file__)
