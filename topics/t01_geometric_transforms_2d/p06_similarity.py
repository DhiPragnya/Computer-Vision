import numpy as np
import streamlit as st

from core import anim, live, plotting, shapes, theme, ui
from core import transforms as tf

ui.lesson_header(
    __file__,
    "Add one more knob - a uniform scale <b>s</b> - and you get a similarity: the same shape, just bigger or "
    "smaller, turned and moved. It is what happens when an object comes closer to a camera that looks straight "
    "at it. 4 degrees of freedom.",
)

HOUSE = shapes.house(offset=(0.2, 0.2))
CENTRE = np.array([1.2, 1.5])


def sim_H(p: dict, about_centroid: bool) -> np.ndarray:
    M = tf.rotation(p["th"]) @ tf.scaling(p["s"])
    if about_centroid:
        M = tf.translation(*CENTRE) @ M @ tf.translation(*-CENTRE)
    return tf.translation(p["tx"], p["ty"]) @ M


def similarity_steps(s, th, tx, ty, about_centroid):
    def build(p):
        return plotting.layer_traces(HOUSE, sim_H(p, about_centroid), grid_extent=5.0, n=2)

    P = [dict(s=1.0, th=0.0, tx=0.0, ty=0.0), dict(s=s, th=0.0, tx=0.0, ty=0.0),
         dict(s=s, th=th, tx=0.0, ty=0.0), dict(s=s, th=th, tx=tx, ty=ty)]
    a_, b_ = s * np.cos(np.deg2rad(th)), s * np.sin(np.deg2rad(th))
    f, H = ui.num, anim.hl
    where = "about the house centre (amber ×)" if about_centroid else "about the origin"
    info = [
        ("Start", "H = I",
         "Grey dashed = before. Orange / green arrows = where e₁ and e₂ land (the matrix columns)."),
        (f"S(s): scale by s = {f(s)}", f"H = {H('S(s)')}   ·   e₁ → ({f(s)}, 0)",
         f"Every length is multiplied by s, {where}. Angles are untouched - the house only grows or shrinks."),
        (f"R(θ): rotate by θ = {f(th)}°", f"H = {H('R(θ)')} S(s)   ·   e₁ → (a, b) = (s cos θ, s sin θ) = ({f(a_)}, {f(b_)})",
         "The orange arrow is now the complex number a + ib = s·e^{iθ}; the green one is it turned 90°: (−b, a)."),
        (f"T(t): slide by ({f(tx)}, {f(ty)})", f"H = {H('T(t)')} R(θ) S(s) = [a −b tₓ ; b a t_y ; 0 0 1]",
         "Translation moves everything equally - it changes no angle, no length, no ratio. 4 numbers: a, b, tₓ, t_y."),
    ]
    return [anim.Step(t_, e_, n_, build(P[i]), between=None if i == 0 else anim.morph(build, P[i - 1], P[i]))
            for i, (t_, e_, n_) in enumerate(info)]


def corner_angle(P, i):
    u, v = P[i - 1] - P[i], P[(i + 1) % len(P)] - P[i]
    return np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))


def area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * abs(x @ np.roll(y, -1) - y @ np.roll(x, -1))


def lens(P):
    return np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)


@live.section
def similarity_section():
    c_ctrl, c_fig, c_txt = st.columns([0.8, 1.6, 1.05], gap="large")
    with c_ctrl:
        s = live.slider("scale s", 0.2, 2.5, 1.6, 0.05, key="sm_s")
        th = live.slider("rotation θ (degrees)", -180, 180, 30, 1, key="sm_th", format="%d°")
        tx = live.slider("tₓ", -3.0, 3.0, -0.5, 0.1, key="sm_tx")
        ty = live.slider("t_y", -3.0, 3.0, -0.4, 0.1, key="sm_ty")
        about_centroid = st.toggle("Scale & rotate about the house's centre", value=False, key="sm_centroid",
                                   help="Compare with scaling about the origin: the house also *moves* then.")

    def H_at(t):
        return sim_H(dict(s=1 + t * (s - 1), th=t * th, tx=t * tx, ty=t * ty), about_centroid)

    H = H_at(1.0)
    extra = [plotting.points_trace([CENTRE], theme.HIGHLIGHT, "centre", size=12, symbol="x")] if about_centroid else []
    with c_fig:
        fig = plotting.transform_figure(HOUSE, H_at, extent=5.0, height=560, extra_static=extra)
        st.plotly_chart(fig, key="sm_fig")

    with c_txt:
        a_, b_ = s * np.cos(np.deg2rad(th)), s * np.sin(np.deg2rad(th))
        st.markdown("**Two ways to write the same matrix**")
        st.latex(r"H = \begin{bmatrix} sR & \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix} = "
                 r"\begin{bmatrix} \color{" + theme.E1 + r"}{a} & \color{" + theme.E2 + r"}{-b} & t_x \\ \color{" +
                 theme.E1 + r"}{b} & \color{" + theme.E2 + r"}{a} & t_y \\ 0 & 0 & 1 \end{bmatrix}")
        st.latex(rf"a = s\cos\theta = {ui.num(a_, 3)}, \quad b = s\sin\theta = {ui.num(b_, 3)}")
        ui.show_matrix("H", H, 3, ui.column_highlight())
        st.caption("The (a, b) form is **linear** in the four unknowns a, b, tₓ, t_y. That is why fitting a "
                   "similarity to point matches is a tiny linear least-squares problem - two point pairs are enough.")

    ui.anim_intro("Step by step: build H = T(t) · R(θ) · S(s) one factor at a time",
                  "Uses your s, θ, tₓ, t_y. Read the product right to left: scale first, then rotate, then slide.")
    base = plotting.base_figure(5.0, height=560, legend=False)
    for tr in plotting.static_layers(HOUSE, grid_extent=5.0):
        base.add_trace(tr)
    if about_centroid:
        base.add_trace(plotting.points_trace([CENTRE], theme.HIGHLIGHT, "centre", size=12, symbol="x",
                                             showlegend=False))
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        st.plotly_chart(anim.step_player(base, similarity_steps(s, th, tx, ty, about_centroid), n_sub=14,
                                         wrap_at=80), key="sm_steps")
    with right:
        ui.callout("try", "Scrub between steps 2 and 3: the orange arrow keeps its length s and only turns - "
                          "its tip draws an arc of radius s. That arrow's tip is (a, b), the first column.")

    st.subheader("Measure it: what survives a similarity?", divider="gray")
    pts = np.array([[0, 0], [2, 0], [2, 1.5], [1.0, 2.5], [0, 1.5]]) + np.array([0.2, 0.2])
    q = tf.apply(H, pts)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Roof-peak angle (before → after)", f"{corner_angle(pts, 3):.1f}° → {corner_angle(q, 3):.1f}°",
              border=True, help="Angles are preserved.")
    m2.metric("Bottom / wall length ratio", f"{lens(pts)[0] / lens(pts)[1]:.3f} → {lens(q)[0] / lens(q)[1]:.3f}",
              border=True, help="Ratios of lengths are preserved, so the shape is the same.")
    m3.metric("Bottom edge length", f"{lens(pts)[0]:.2f} → {lens(q)[0]:.2f}",
              delta=f"× {lens(q)[0] / lens(pts)[0]:.2f}", delta_color="off", border=True, help="Lengths scale by s.")
    m4.metric("Area", f"{area(pts):.2f} → {area(q):.2f}", delta=f"× {area(q) / area(pts):.2f}  (= s²)",
              delta_color="off", border=True, help="Areas scale by s².")


similarity_section()

a, b = st.columns([1.1, 1], gap="large")
with a:
    ui.preserved_badges("similarity")
    st.write("")
    ui.callout("intuition",
               "A similarity is a **photocopier with a zoom button**: the copy may be larger, rotated and placed "
               "elsewhere, but every angle and every *ratio* of lengths is exactly as before. Two triangles "
               "related by a similarity are what school geometry calls *similar triangles*.")
with b:
    ui.callout("note",
               "Treat a point as a complex number z = x + iy. A similarity is simply "
               "$z' = (a + ib)\\,z + (t_x + i t_y)$. Multiplying by $a + ib = s\\,e^{i\\theta}$ scales by "
               "$|a+ib| = s$ and rotates by $\\arg = \\theta$. The orange arrow in the figure *is* that complex "
               "number: it is where 1 (= e₁) lands.", title="A neat shortcut: complex numbers")

c1, c2 = st.columns(2, gap="large")
with c1:
    ui.callout("practice",
               "- **Face alignment**: map the two eye centres to fixed positions → a similarity.\n"
               "- **Scale-invariant features** (e.g. SIFT keypoints) each carry a position, scale and "
               "orientation - exactly a similarity frame.\n"
               "- A camera looking **straight at** a plane while moving closer / turning about its axis.")
with c2:
    ui.callout("pitfall",
               "Scaling is always **about the origin**. Turn the 'about the house's centre' toggle off and set s = 2: "
               "the house doubles in size *and* runs away from the origin. To zoom in place, conjugate: "
               "T(c) · S(s) · T(−c).")

ui.whys([
    ('Why does a similarity have 4 degrees of freedom?',
     'One for the scale s, one for the angle θ and two for the shift (tₓ, t_y). Equivalently the four numbers a = s cos θ, b = s sin θ, tₓ, t_y in the matrix [[a, −b, tₓ], [b, a, t_y]].'),
    ('Why do areas scale by s² and not by s?',
     'Area is length × length. Both directions are stretched by s, so a 1 × 1 square becomes s × s. The determinant agrees: det [[a, −b], [b, a]] = a² + b² = s².'),
    ('Why is uniform scaling a similarity but non-uniform scaling is not?',
     'A similarity must keep every angle. Equal stretch in all directions keeps angles; unequal stretch tilts diagonals (see the *Scaling* lesson), so non-uniform scaling is only affine.'),
    ('Why can the similarity matrix be written with just a and b?',
     'Because sR = s[[cos θ, −sin θ], [sin θ, cos θ]] always has the pattern [[a, −b], [b, a]]. Two numbers carry both the size (s = √(a² + b²)) and the angle (θ = atan2(b, a)) - and the equations stay linear in a and b.'),
])

st.divider()
ui.mcq("sm1", "A similarity maps a 3 cm segment to a 6 cm segment. What happens to a 10 cm² region?",
       ["Stays 10 cm²", "Becomes 20 cm²", "Becomes 40 cm²", "Cannot be known without θ"], 2,
       "s = 6 / 3 = 2 and areas scale by s² = 4, so 10 cm² → 40 cm². Rotation and translation never change area.")
ui.mcq("sm2", "Minimum number of point correspondences needed to determine a similarity?",
       ["1", "2", "3", "4"], 1,
       "4 unknowns (a, b, tₓ, t_y) and every point pair gives 2 equations → 2 pairs.")

ui.lesson_nav(__file__)
