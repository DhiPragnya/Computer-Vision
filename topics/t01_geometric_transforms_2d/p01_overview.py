import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core import anim, images, live, plotting, shapes, theme, ui
from core import transforms as tf

ui.lesson_header(
    __file__,
    "A geometric transformation changes <b>where</b> things are, not <b>what colour</b> they are: "
    "every point <i>x</i> is moved to <i>x' = f(x)</i>. Five families of such maps - each one a special "
    "case of the next - explain almost everything a camera can do to a flat scene.",
)

# ---------------------------------------------------------------- geometric vs photometric
st.subheader("Geometric vs photometric", divider="gray")
img = images.poster(260)
H = images.about_center(tf.similarity(0.85, 25), img.shape)
geo = images.cv_warp(img, H)
photo = np.clip(img.astype(float) * 0.55 + 60, 0, 255).astype(np.uint8)
c1, c2, c3, c4 = st.columns([1, 1, 1, 1.25], gap="medium")
c1.image(ui.datauri(img), caption="Original image  I(x)")
c2.image(ui.datauri(geo), caption="Geometric: pixels MOVE  I(f⁻¹(x))")
c3.image(ui.datauri(photo), caption="Photometric: pixels CHANGE value  g(I(x))")
with c4:
    ui.callout("intuition",
               "A geometric transform is like printing the image on a rubber sheet and then sliding, "
               "turning, stretching or tilting the sheet. The ink (colours) is untouched - only its "
               "position changes. This whole topic is about the *sheet*.")

# ---------------------------------------------------------------- the family
@live.section
def family_section():
    st.subheader("The family of 2D transformations", divider="gray")
    st.markdown("Drag the slider to blend from **do nothing** (0) to a typical member of each family (1). "
                "Watch which features of the house survive in each panel.")
    t = live.slider("Amount of transformation", 0.0, 1.0, 1.0, 0.05, key="ov_t")

    panels = [
        ("Translation", tf.translation(1.0 * t, 0.7 * t)),
        ("Euclidean (rigid)", tf.euclidean(40 * t, 0.9 * t, 0.2 * t)),
        ("Similarity", tf.similarity(1 - 0.38 * t, 40 * t, 0.9 * t, 0.2 * t)),
        ("Affine", tf.affine(1 - 0.05 * t, 0.55 * t, 0.12 * t, 1 - 0.3 * t, 0.4 * t, 0.2 * t)),
        ("Projective", np.array([[1, 0.15 * t, 0.3 * t], [0.05 * t, 1, 0.1 * t], [0.28 * t, 0.2 * t, 1.0]])),
    ]
    house = shapes.house(0.9, offset=(-0.9, -1.15))
    fig = plotting.small_multiples([(n, house, M) for n, M in panels], extent=2.8, height=300)
    st.plotly_chart(fig, key="ov_panels")
    cols = st.columns(5)
    for col, c in zip(cols, tf.CLASSES):
        with col:
            st.metric("Degrees of freedom", c["dof"], border=True)
            st.caption(f"Signature property kept: **{c['preserves']}**")


family_section()

with st.expander("The matrix of each family", icon=":material/function:", expanded=False):
    mcols = st.columns(5)
    for col, c in zip(mcols, tf.CLASSES):
        col.markdown(f"**{c['name']}**")
        col.latex(c["form"])
    st.caption("I = identity, R = rotation (RᵀR = I, det R = 1), s = scale, A = any invertible 2×2, "
               "t = translation vector. All live in 3×3 homogeneous form - next lesson explains why.")


# ---------------------------------------------------------------- climbing the ladder (step animation)
def ladder_H(p: dict) -> np.ndarray:
    """[[A, t], [v, 1]] with A = s · R(θ) · [[1, k], [0, a]]."""
    th = np.deg2rad(p["th"])
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    A = p["s"] * R @ np.array([[1.0, p["k"]], [0.0, p["a"]]])
    return np.array([[A[0, 0], A[0, 1], p["tx"]], [A[1, 0], A[1, 1], p["ty"]], [p["v1"], p["v2"], 1.0]])


@st.cache_resource(show_spinner=False)
@live.full_quality()          # cached: never store a coarse mid-drag version
def ladder_figure():
    house = shapes.house(0.9, offset=(-0.9, -1.15))
    P0 = dict(tx=0.0, ty=0.0, th=0.0, s=1.0, k=0.0, a=1.0, v1=0.0, v2=0.0)
    P1 = dict(P0, tx=0.9, ty=0.5)
    P2 = dict(P1, th=35.0)
    P3 = dict(P2, s=0.72)
    P4 = dict(P3, k=0.55, a=0.75)
    P5 = dict(P4, v1=0.17, v2=0.1)
    params = [P0, P1, P2, P3, P4, P5]

    def build(p):
        return plotting.layer_traces(house, ladder_H(p), grid_extent=3.0, n=2)

    H, h = anim.hl, anim.faint
    info = [
        ("Do nothing - the identity", f"[1 0 0 ; 0 1 0 ; 0 0 1]  ·  {h('0 numbers to choose')}",
         "Grey dashed = before, blue = after. Orange / green arrows = where (1, 0) and (0, 1) land."),
        ("Translation", f"[1 0 {H('tₓ')} ; 0 1 {H('t_y')} ; 0 0 1]  ·  <b>2 DOF</b>",
         "Only the POSITION changes. Lengths, angles, areas, orientation - everything else survives."),
        ("+ rotation  →  Euclidean (rigid)", f"[{H('cos θ  −sin θ')} tₓ ; {H('sin θ  cos θ')} t_y ; 0 0 1]  ·  <b>3 DOF</b>",
         "Directions change; lengths and angles are still exact. A real object moving on a table does this."),
        ("+ uniform scale  →  Similarity", f"[{H('s')}·cos θ  −{H('s')}·sin θ  tₓ ; {H('s')}·sin θ  {H('s')}·cos θ  t_y ; 0 0 1]  ·  <b>4 DOF</b>",
         "Sizes change, but angles and length RATIOS survive: the shape is the same, only zoomed."),
        ("+ shear & aspect  →  Affine", f"[{H('a  b')} tₓ ; {H('c  d')} t_y ; 0 0 1]  -  any 2×2 block  ·  <b>6 DOF</b>",
         "Angles and length ratios are lost - but parallel lines STAY parallel (look at the grid)."),
        ("+ perspective row  →  Projective", f"[a b tₓ ; c d t_y ; {H('h₃₁  h₃₂')} 1]  ·  <b>8 DOF</b>",
         "Parallel grid lines now converge. Only straight lines (and the cross-ratio) survive."),
    ]
    steps = []
    for i, (title, eq, note) in enumerate(info):
        steps.append(anim.Step(title, eq, note, build(params[i]),
                               between=None if i == 0 else anim.morph(build, params[i - 1], params[i])))
    base = plotting.base_figure(3.0, height=640, legend=False)
    for tr in plotting.static_layers(house, grid_extent=3.0):
        base.add_trace(tr)
    return anim.step_player(base, steps, n_sub=12, wrap_at=80)


st.subheader("Climbing the ladder, one freedom at a time", divider="gray")
ui.anim_intro("Step by step: each family = the previous one + one more kind of move",
              "Every step adds new entries to the matrix (highlighted) and the house morphs accordingly. "
              "▶ Play runs the whole ladder; the numbered buttons replay one step; drag the slider to go back "
              "and forth and compare what is kept and what is lost.")
left, right = st.columns([1.6, 1], gap="large")
with left:
    st.plotly_chart(ladder_figure(), key="ov_ladder")
with right:
    ui.callout("intuition",
               "Keep your eye on three things while you scrub: the **grid squares** (angles, areas), "
               "the **grid lines** (parallel or not) and the **arrows** (the matrix columns). Each step "
               "unlocks exactly one new kind of change - and it is always visible in one of them.")
    ui.callout("note",
               "DOF = degrees of freedom = how many numbers you may choose. 2 → 3 → 4 → 6 → 8: every rung "
               "adds parameters, and each family contains all the previous ones (set the new parameter to "
               "its neutral value and you are back one rung).", title="Counting freedom")

# ---------------------------------------------------------------- nested sets
st.subheader("Each family contains the previous one", divider="gray")
left, right = st.columns([1.35, 1], gap="large")
with left:
    fig = go.Figure()
    adds = ["+ 2 perspective terms → lines may converge",
            "+ 2 shear / aspect terms → angles may change",
            "+ 1 scale → sizes may change",
            "+ 1 rotation → directions may change",
            "2 translation terms"]
    ring = list(reversed(tf.CLASSES))
    for i, c in enumerate(ring):
        x0, x1, y0, y1 = i * 0.8, 10 - i * 0.8, i * 0.22, 6.6 - i * 0.98
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                      line=dict(color=theme.TRANSFORMED, width=1.5),
                      fillcolor=theme.rgba(theme.TRANSFORMED, 0.05 + 0.06 * i), layer="below")
        fig.add_annotation(x=x0 + 0.15, y=y1 - 0.12, xanchor="left", yanchor="top", showarrow=False,
                           text=f"<b>{c['name']}</b>  ·  {c['dof']} DOF", font=dict(size=13))
        fig.add_annotation(x=x0 + 0.15, y=y1 - 0.48, xanchor="left", yanchor="top", showarrow=False,
                           text=adds[i], font=dict(size=11.5), opacity=0.75)
    fig.update_xaxes(visible=False, range=[-0.1, 10.1])
    fig.update_yaxes(visible=False, range=[-0.1, 6.7])
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
    st.plotly_chart(fig, key="ov_nested", config=dict(staticPlot=True))
with right:
    ui.callout("intuition",
               "Think of a ladder of freedom. Every step outward lets the matrix do **one more kind of "
               "move** (more degrees of freedom), and in exchange **one more geometric property is "
               "lost**. A translation keeps everything except position; a homography keeps only "
               "straight lines (and the cross-ratio).")
    ui.callout("takeaway",
               "To recognise a transform from a picture, ask in order: *Are lines still straight?* "
               "*Still parallel?* *Are angles kept?* *Are lengths kept?* *Did directions change?* "
               "The first 'no' tells you the family.")

# ---------------------------------------------------------------- invariants
@live.section
def invariants_section():
    st.subheader("What each family preserves", divider="gray")
    pick = st.segmented_control("Highlight a family", [c["key"] for c in tf.CLASSES],
                                format_func=lambda k: tf.CLASS_BY_KEY[k]["name"].split(" (")[0],
                                default="affine", key="ov_pick")
    ui.invariants_table(pick)
    st.caption("Read a column top-to-bottom: everything below the first ✓ is also preserved. "
               "Read a row left-to-right: once a property is lost, more general families never get it back.")


invariants_section()

# ---------------------------------------------------------------- where in CV
st.subheader("Where you meet these in computer vision", divider="gray")
a, b, c = st.columns(3)
with a:
    ui.callout("practice",
               "- **Translation** - video stabilisation, template matching, small optical-flow motions.\n"
               "- **Euclidean (SE(2))** - pose of a part lying on a conveyor or table; a mobile robot's "
               "planar pose (x, y, θ).", title="Shift & turn")
with b:
    ui.callout("practice",
               "- **Similarity** - aligning faces by their eyes, objects at different distances, image pyramids.\n"
               "- **Affine** - data augmentation, small image patches seen at an angle, "
               "3-point registration.", title="Zoom & stretch")
with c:
    ui.callout("practice",
               "- **Projective (homography)** - panoramas, document scanning, bird's-eye view of a road or "
               "factory floor, AR overlays on planar markers, camera-to-table calibration for a robot arm.",
               title="Perspective")

ui.whys([
    ('Where do the numbers inside a transformation matrix come from?',
     "From the equations you want. Write the new coordinates as 'something·x + something·y + something·1' and copy the somethings into the rows - the *Scaling* lesson does this step by step. Rather than memorising matrices, derive them; then you can build any transform you need."),
    ('Why use matrices at all instead of just writing the formulas?',
     'Because then *everything* is one operation. Chaining moves is multiplying matrices, undoing a move is inverting a matrix, and a single piece of code (or GPU kernel) can apply any of them to millions of pixels.'),
    ('Are these five the only transformations that exist?',
     'No. They are the ones that are *linear in homogeneous coordinates* - a single 3×3 matrix. Lens distortion, thin-plate splines or optical flow fields bend straight lines and need more general (non-linear) warps. The five here are the backbone that those methods build on.'),
    ('Why does a camera give a homography only for flat scenes?',
     'Points at different depths shift by different amounts when the camera moves (parallax). One 3×3 matrix can only describe a scene with no depth variation - a plane - or a camera that rotates without moving, where depth does not matter.'),
])

ui.lesson_nav(__file__)
