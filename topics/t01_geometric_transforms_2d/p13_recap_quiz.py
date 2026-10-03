import numpy as np
import streamlit as st

from core import live, plotting, shapes, ui
from core import transforms as tf

ui.lesson_header(
    __file__,
    "Everything in this topic on one page - then test yourself. The drill generates a fresh transform every time, "
    "so you can practise recognising the five families until it is automatic.",
)

# ------------------------------------------------------------------ cheat sheet
st.subheader("Cheat sheet", divider="gray")
st.markdown(
    "| Family | Matrix (2×3 or 3×3) | DOF | Min. matches | Signature property kept | OpenCV |\n"
    "|---|---|---|---|---|---|\n"
    "| Translation | $[\\,I \\mid \\mathbf{t}\\,]$ | 2 | 1 | orientation | `np.float32([[1,0,tx],[0,1,ty]])` |\n"
    "| Scaling (axis-aligned) | $\\mathrm{diag}(s_x, s_y, 1)$ | 2 | 1 | axis directions | `cv2.resize` |\n"
    "| Euclidean (rigid) | $[\\,R \\mid \\mathbf{t}\\,]$ | 3 | 2 | lengths | `cv2.getRotationMatrix2D(c, θ, 1)` |\n"
    "| Similarity | $[\\,sR \\mid \\mathbf{t}\\,]$ | 4 | 2 | angles | `cv2.estimateAffinePartial2D` |\n"
    "| Affine | $[\\,A \\mid \\mathbf{t}\\,]$ | 6 | 3 | parallelism | `cv2.getAffineTransform`, `cv2.warpAffine` |\n"
    "| Projective | $\\tilde H$ (3×3, up to scale) | 8 | 4 | straight lines | `cv2.findHomography`, `cv2.warpPerspective` |"
)
st.write("")
ui.invariants_table(None)

st.subheader("Five sentences to remember", divider="gray")
st.markdown(
    "1. **Homogeneous coordinates** (x, y, 1) make every 2D transform - even translation and perspective - a single "
    "3×3 matrix; vectors are equal up to scale, so divide by w at the end.\n"
    "2. **The columns of a matrix are where the basis vectors go** - read any matrix by drawing e₁ and e₂.\n"
    "3. The families are **nested**: each adds degrees of freedom and loses one invariant "
    "(orientation → lengths → angles → parallelism → only straight lines).\n"
    "4. **Composition is multiplication, read right to left**; order matters, and (AB)⁻¹ = B⁻¹A⁻¹.\n"
    "5. Warp images with **inverse mapping + interpolation**, and estimate transforms from matches with "
    "**least squares** - or **RANSAC** when some matches are wrong."
)

st.subheader("The lecture-notes summary", divider="gray")
st.markdown(
    "- Homogeneous coordinates let 2D transformations be written as matrix multiplication.\n"
    "- Translation adds tₓ, t_y through the **last column**; scaling follows directly from x′ = sₓx, y′ = s_y y; "
    "the rotation matrix follows from the **angle-addition identities**.\n"
    "- An affine transformation has **6** DOF; each correspondence gives **2** constraints → **3 non-collinear** "
    "points determine it (det M ≠ 0).\n"
    "- A homography has **8** DOF (9 entries, defined up to scale) → **4** correspondences; build **A h = 0** and "
    "solve with the **SVD** (the DLT), then normalise h₃₃ = 1.\n"
    "- Affine ⊂ projective: an affine map is a homography with h₃₁ = h₃₂ = 0."
)

# ------------------------------------------------------------------ name that transform
st.subheader("Drill: name that transform", divider="gray")
ss = st.session_state
ss.setdefault("dr_round", 0)
ss.setdefault("dr_score", 0)
ss.setdefault("dr_total", 0)
ss.setdefault("dr_answered", False)


def make_question(seed: int):
    rng = np.random.default_rng(1000 + seed)
    kind = tf.CLASS_ORDER[rng.integers(0, 5)]
    sgn = lambda: rng.choice([-1, 1])
    t = rng.uniform(0.6, 1.4, 2) * np.array([sgn(), sgn()])
    th = sgn() * rng.uniform(25, 70)
    if kind == "translation":
        H = tf.translation(*t)
    elif kind == "euclidean":
        H = tf.euclidean(th, *t * 0.5)
    elif kind == "similarity":
        s = rng.choice([rng.uniform(0.55, 0.7), rng.uniform(1.35, 1.6)])
        H = tf.similarity(s, th, *t * 0.4)
    elif kind == "affine":
        H = tf.translation(*t * 0.3) @ tf.rotation(th * 0.3) @ tf.shear(sgn() * rng.uniform(0.45, 0.8)) \
            @ tf.scaling(rng.uniform(0.8, 1.3), rng.uniform(0.6, 1.0))
    else:
        H = tf.affine(1, 0, 0, 1, *t * 0.2)
        H[2, :2] = rng.uniform(0.14, 0.24, 2) * np.array([sgn(), sgn()])
    return kind, H


def next_round():
    ss.dr_round += 1
    ss.dr_answered = False
    ss.pop("dr_choice", None)


@live.section
def drill_section():
    kind, Hq = make_question(ss.dr_round)
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        fig = plotting.transform_figure(shapes.house(0.9, offset=(-0.9, -1.1)), Hq, extent=3.5, height=460,
                                        basis=False, grid_step=0.5, grid_extent=2.5, n_frames=24)
        st.plotly_chart(fig, key=f"dr_fig_{ss.dr_round}")
    with right:
        st.markdown(f"**Round {ss.dr_round + 1}** · score **{ss.dr_score} / {ss.dr_total}**")
        st.caption("Look at the blue grid and house. Straight? Parallel? Right angles? Same size? Same direction?")
        choice = st.radio("Which family is the most specific one that fits?", tf.CLASS_ORDER, index=None,
                          format_func=lambda k: tf.CLASS_BY_KEY[k]["name"], key="dr_choice",
                          disabled=ss.dr_answered)
        if choice is not None and not ss.dr_answered:
            ss.dr_answered = True
            ss.dr_total += 1
            ss.dr_last_ok = choice == kind
            if ss.dr_last_ok:
                ss.dr_score += 1
            live.rerun()
        if ss.dr_answered:
            reasons = {
                "translation": "every grid line keeps its direction - only the position changed.",
                "euclidean": "the grid turned but the squares kept their size: lengths are preserved.",
                "similarity": "the squares are still squares (angles kept) but their size changed.",
                "affine": "grid lines are still parallel, but the squares became parallelograms (angles changed).",
                "projective": "grid lines stay straight but converge - parallelism is lost.",
            }
            if ss.get("dr_last_ok"):
                st.success(f"**Correct - {tf.CLASS_BY_KEY[kind]['name']}.** The giveaway: {reasons[kind]}",
                           icon=":material/check_circle:")
            else:
                st.error(f"**It was {tf.CLASS_BY_KEY[kind]['name']}.** The giveaway: {reasons[kind]}",
                         icon=":material/cancel:")
            ui.show_matrix("H", tf.normalize_h(Hq), 2)
        st.button("Next transform", on_click=next_round, icon=":material/skip_next:", type="primary")


drill_section()

# ------------------------------------------------------------------ concept quiz
st.subheader("Concept check", divider="gray")
qs = [
    ("q1", "A camera photographs a chessboard lying on a table from an oblique angle. Which model relates the "
           "board's own coordinates to the image?",
     ["Similarity", "Affine", "Homography", "Euclidean"], 2,
     "Any image of a plane is a homography of that plane (pinhole camera, no lens distortion)."),
    ("q2", "Which property is preserved by an affine map but NOT by a homography?",
     ["Collinearity", "Parallelism", "Cross-ratio", "Incidence of points and lines"], 1,
     "Affine maps send points at infinity to points at infinity, so parallel lines stay parallel; homographies "
     "can move them to finite vanishing points."),
    ("q3", "The 2×2 block of a transform is [[0, −2], [2, 0]]. What does it do?",
     ["Shear by 2", "Rotate 90° and scale by 2", "Mirror and scale by 2", "Scale x by 0 and y by 2"], 1,
     "Columns (0, 2) and (−2, 0): e₁ goes to 2·(0, 1), e₂ to 2·(−1, 0) - a 90° rotation with scale 2 "
     "(a = 0, b = 2 in the similarity form)."),
    ("q4", "You compute x̃' = H x̃ = (6, 3, 3). Where is the point in the image?",
     ["(6, 3)", "(2, 1)", "(18, 9)", "(3, 0)"], 1, "Divide by w = 3: (6/3, 3/3) = (2, 1)."),
    ("q5", "Why is inverse mapping preferred for warping images?",
     ["It is the only one that works for rotations", "Every output pixel gets exactly one value - no holes",
      "It does not need interpolation", "It is invertible"], 1,
     "Pulling a value for each destination pixel guarantees full coverage; interpolation handles the "
     "fractional source positions."),
    ("q6", "RANSAC with 50 % outliers: which model needs the most iterations for the same confidence?",
     ["Translation", "Similarity", "Affine", "Homography"], 3,
     "N grows with the minimal sample size s: (1 − e)^s = 0.5⁴ for a homography vs 0.5² for a similarity."),
]
@live.section
def concept_section():
    score = 0
    answered = 0
    cols = st.columns(2, gap="medium")
    for i, (key, q, opts, ans, expl) in enumerate(qs):
        with cols[i % 2]:
            r = ui.mcq(f"final_{key}", q, opts, ans, expl, scored=True)
            if r is not None:
                answered += 1
                score += int(r)
    if answered:
        st.progress(score / len(qs), text=f"Concept check: {score} / {len(qs)} correct ({answered} answered)")
    if answered == len(qs) and score == len(qs):
        st.success("Full marks - you are ready for the next topic.", icon=":material/military_tech:")


concept_section()

ui.lesson_nav(__file__)
