import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core import live, theme, ui
from core import contour_ui as cui
from core import contours as K

ui.lesson_header(
    __file__,
    "The whole topic on one page, then practise: a <b>polygon-budget challenge</b> (tune ε to hit a vertex count), a "
    "<b>contour-or-Hough?</b> drill, and a concept check.",
)

# ------------------------------------------------------------------ progression + equations
st.subheader("The progression", divider="gray")
with st.container(horizontal=True, gap="small"):
    for i, s_ in enumerate(["Pixels", "Edges", "Contours", "Geometric representation"]):
        st.badge(s_, color="blue")
        if i < 3:
            st.markdown("→")
with st.container(horizontal=True, gap="small"):
    for i, s_ in enumerate(["Edge points", "Parameter space", "Voting", "Geometric model"]):
        st.badge(s_, color="orange")
        if i < 3:
            st.markdown("→")

st.subheader("Key equations", divider="gray")
e1, e2 = st.columns(2, gap="medium")
with e1:
    with st.container(border=True):
        st.markdown("**Contour, connectivity, boundary**")
        st.latex(r"C = \{(x_0, y_0), \dots, (x_{N-1}, y_{N-1})\},\qquad \partial\Omega = \{p \in \Omega : N(p) \cap "
                 r"\Omega^c \neq \emptyset\}")
    with st.container(border=True):
        st.markdown("**Area and perimeter**")
        st.latex(r"A = \tfrac12\Big|\sum_i (x_i y_{i+1} - x_{i+1} y_i)\Big|,\qquad P = \sum_i \lVert p_{i+1} - p_i\rVert,"
                 r"\qquad C = \frac{4\pi A}{P^2}")
    with st.container(border=True):
        st.markdown("**Polygonal approximation (RDP)**")
        st.latex(r"d = \frac{|Ax + By + C|}{\sqrt{A^2 + B^2}},\qquad \text{keep } p_k \text{ if } d_{\max} > \epsilon")
with e2:
    with st.container(border=True):
        st.markdown("**Hough lines**")
        st.latex(r"x\cos\theta + y\sin\theta = \rho,\qquad A(\rho, \theta),\qquad (\rho^*, \theta^*) = "
                 r"\arg\max A(\rho, \theta)")
    with st.container(border=True):
        st.markdown("**Hough circles**")
        st.latex(r"(x - a)^2 + (y - b)^2 = r^2,\qquad A(a, b, r),\qquad (a, b) = (x \pm r\cos\theta,\; y \pm r\sin\theta)")
    with st.container(border=True):
        st.markdown("**General principle**")
        st.latex(r"\boldsymbol\theta^* = \arg\max_{\boldsymbol\theta} A(\boldsymbol\theta)\qquad \text{(voting in "
                 r"parameter space)}")

st.markdown("| | Contours | RDP polygon | Hough |\n|---|---|---|---|\n| Input | binary regions | a contour | edge points |\n"
            "| Output | ordered boundary points | M ≪ N vertices | model parameters |\n"
            "| Needs connectivity | yes | yes (a contour) | no |\n| Shape | any | any | parametric (line, circle, …) |\n"
            "| Key parameter | threshold / connectivity | ε | bin sizes, peak threshold |\n"
            "| Weakness | gaps split objects | greedy, start point on closed curves | dimension explosion |")

# ------------------------------------------------------------------ challenge: polygon budget
st.subheader("Challenge: hit the vertex budget", divider="gray")
ss = st.session_state
ss.setdefault("rq_round", 0)
SHAPES = ["Star", "Blob", "Noisy rectangle", "Hexagon", "Arrow"]


@st.cache_resource(show_spinner=False, max_entries=16)
def challenge(r: int):
    rng = np.random.default_rng(300 + r)
    kind = SHAPES[r % len(SHAPES)]
    img = np.zeros((200, 200), np.uint8)
    if kind == "Star":
        cv2.fillPoly(img, [K._star(100, 104, 85, 38, n=int(rng.integers(5, 8)))], 1)
    elif kind == "Blob":
        a = np.linspace(0, 2 * np.pi, 400, endpoint=False)
        rr = 65 * (1 + 0.25 * np.sin(3 * a + rng.uniform(0, 6)) + 0.1 * np.sin(5 * a))
        cv2.fillPoly(img, [np.int32(np.c_[100 + rr * np.cos(a), 100 + rr * np.sin(a)])], 1)
    elif kind == "Noisy rectangle":
        f = np.zeros((200, 200))
        cv2.rectangle(f, (35, 55), (165, 145), 1.0, -1)
        img = ((cv2.GaussianBlur(f, (0, 0), 1.2) + rng.normal(0, 0.12, f.shape)) > 0.5).astype(np.uint8)
    elif kind == "Hexagon":
        a = np.deg2rad(np.arange(6) * 60 + rng.uniform(0, 60))
        cv2.fillPoly(img, [np.int32(np.c_[100 + 80 * np.cos(a), 100 + 80 * np.sin(a)])], 1)
    else:
        cv2.fillPoly(img, [np.int32([[20, 80], [120, 80], [120, 40], [185, 100], [120, 160], [120, 120], [20, 120]])], 1)
    cs, _ = K.find_contours(img > 0, cv2.RETR_EXTERNAL)
    C = max(cs, key=len)
    L = K.perimeter(C)
    counts = sorted({len(K.rdp_closed(C, f * L)) for f in np.geomspace(0.003, 0.2, 60)})
    reachable = [c for c in counts if 3 <= c <= 30]
    target = int(rng.choice(reachable)) if reachable else counts[0]
    return kind, C, L, target


@live.section
def challenge_section():
    kind, C, L, target = challenge(ss.rq_round)
    c_ctrl, c_fig, c_txt = st.columns([0.85, 1.4, 1.2], gap="large")
    with c_ctrl:
        st.markdown(f"**Round {ss.rq_round + 1}** · {kind}")
        st.markdown(f"Find an ε that gives **exactly {target} vertices**, with the smallest error you can.")
        frac = live.slider("ε / perimeter", 0.002, 0.2, 0.05, 0.001, key=f"rq_eps{ss.rq_round}", format="%.3f")
        if st.button("New shape", icon=":material/refresh:", key="rq_new"):
            ss.rq_round += 1
            live.rerun()
    idx = K.rdp_closed(C, frac * L)
    err = K.max_deviation(C, idx, closed=True)
    Q = np.vstack([C[idx], C[idx][:1]])
    loop = np.vstack([C, C[:1]])
    with c_fig:
        fig = go.Figure(data=[cui.scatter(loop[:, 0], loop[:, 1], theme.ORIGINAL, mode="lines", width=1.5),
                              cui.scatter(Q[:, 0], Q[:, 1], theme.TRANSFORMED, mode="lines+markers", size=9, width=3)
                              .update(fill="toself", fillcolor=theme.rgba(theme.TRANSFORMED, 0.1))],
                        layout=dict(height=360, margin=dict(l=8, r=8, t=10, b=8), showlegend=False,
                                    xaxis=dict(range=[0, 200], visible=False, constrain="domain"),
                                    yaxis=dict(range=[200, 0], visible=False, scaleanchor="x", constrain="domain")))
        st.plotly_chart(fig, key="rq_fig")
    with c_txt:
        a, b = st.columns(2)
        a.metric("your vertices", len(idx), f"{len(idx) - target:+d} vs target", delta_color="off", border=True)
        b.metric("largest error", f"{err:.1f} px", border=True)
        if len(idx) == target:          # always one status line (same height): no layout jump while dragging
            st.success(f"**Hit!** {target} vertices with ε = {frac * L:.1f} px. Can you find the plateau's left end "
                       "(smallest ε, smallest error)?", icon=":material/emoji_events:")
        else:
            st.info(f"{'Too many' if len(idx) > target else 'Too few'} vertices: "
                    f"{'raise' if len(idx) > target else 'lower'} ε.", icon=":material/tune:")
        st.caption("Reminder: larger ε → fewer vertices, larger error. Where M stays constant over a range of ε, the "
                   "shape really is that polygon.")


challenge_section()

# ------------------------------------------------------------------ drill: contour or Hough?
st.subheader("Drill: contours or Hough?", divider="gray")
ss.setdefault("sd_round", 0)
ss.setdefault("sd_score", 0)
ss.setdefault("sd_total", 0)
ss.setdefault("sd_done", False)
TOOLS = ["Contours (+ RDP / measurements)", "Hough lines", "Hough circles"]
CASES = [
    ("Count and measure irregular cells on a clean, high-contrast microscope image.", 0,
     "Closed, well-separated regions of arbitrary shape: threshold, trace contours, measure area and circularity."),
    ("Find the lane markings on a road; they are dashed and partly covered by cars.", 1,
     "A known model (lines) with gaps and occlusion: exactly what voting is good at."),
    ("Locate coins on a cluttered table where coins overlap each other.", 2,
     "Circles with known radius range; overlapping coins break contours, but each visible arc still votes for its centre."),
    ("Decide whether each shape in a children's puzzle is a triangle, square or pentagon.", 0,
     "Contour + RDP vertex count (+ circularity) classifies polygons of any size and rotation."),
    ("Find the straight edges of a building in a photo full of trees and texture.", 1,
     "Texture gives clutter; long straight edges still collect many votes at one (ρ, θ)."),
    ("Find the pupil (a dark disc) in an eye image, partly hidden by the eyelid.", 2,
     "A partial circle with a strong radial gradient: gradient-based circle Hough."),
    ("Extract the outline of a leaf to compare its shape with a database.", 0,
     "Arbitrary shape, no formula: the contour (and its measurements) IS the description."),
    ("Detect the horizon line in a sea photograph with waves and spray.", 1,
     "One dominant straight line, broken and noisy: the strongest Hough peak."),
    ("Measure the holes drilled in a metal plate (seen from above).", 2,
     "Circles of known size; Hough is robust to burrs and reflections. (Contours + circularity also work on clean "
     "images - both answers are defensible, but Hough is the safer choice.)"),
]


def next_case():
    ss.sd_round += 1
    ss.sd_done = False
    ss.pop("sd_choice", None)


@live.section
def drill_section():
    txt, ans, why = CASES[(ss.sd_round * 4) % len(CASES)]
    a, b = st.columns([1.3, 1], gap="large")
    with a:
        with st.container(border=True):
            st.markdown(f"**Case {ss.sd_round + 1}** · score **{ss.sd_score} / {ss.sd_total}**")
            st.markdown(f":material/photo_camera: {txt}")
    with b:
        ch = st.radio("Best tool?", TOOLS, index=None, key="sd_choice", disabled=ss.sd_done)
        if ch is not None and not ss.sd_done:
            ss.sd_done = True
            ss.sd_total += 1
            ss.sd_ok = TOOLS.index(ch) == ans
            ss.sd_score += int(ss.sd_ok)
            live.rerun()
        if ss.sd_done:
            (st.success if ss.get("sd_ok") else st.error)(
                f"**{'Correct' if ss.get('sd_ok') else 'Better: ' + TOOLS[ans]}.** {why}",
                icon=":material/check_circle:" if ss.get("sd_ok") else ":material/cancel:")
        st.button("Next case", on_click=next_case, icon=":material/skip_next:", type="primary", key="sd_next")


drill_section()

# ------------------------------------------------------------------ concept check
st.subheader("Concept check", divider="gray")
qs = [
    ("rq1", "Two object pixels touch only at a corner. With 8-connectivity they are …",
     ["two objects", "one object", "background", "a hole"], 1, "Diagonal neighbours are in N₈."),
    ("rq2", "Why must contour tracing not stop the first time it returns to the start pixel?",
     ["It would be too slow", "The start pixel can be visited twice (thin parts), so part of the contour would be "
      "missed", "OpenCV requires two loops", "To compute the area"], 1,
     "Stop only when the first move repeats (Jacob's criterion)."),
    ("rq3", "The shoelace sum of a contour is negative. This means …",
     ["the area is negative", "the contour runs clockwise (in y-up axes)", "the contour is open", "there is a hole"], 1,
     "The sign gives the orientation; the area is the absolute value."),
    ("rq4", "Increasing ε in RDP …",
     ["keeps more vertices", "keeps fewer vertices and allows a larger error", "has no effect on closed contours",
      "makes the polygon smoother but with more points"], 1, "ε is the allowed deviation: larger ε, coarser polygon."),
    ("rq5", "Why does the Hough transform use x cos θ + y sin θ = ρ instead of y = m x + c?",
     ["It is faster to compute", "Both parameters are bounded and vertical lines are included",
      "It needs fewer points", "It avoids noise"], 1, "m → ∞ for vertical lines; θ ∈ [0°, 180°), |ρ| ≤ diagonal."),
    ("rq6", "Using the gradient direction in circle Hough reduces each point's votes (per radius) from a circle to …",
     ["one or two points", "a line", "a disc", "nothing - it only speeds up Canny"], 0,
     "The centre lies along the gradient at distance r: x ± r cos θ, y ± r sin θ."),
]


@live.section
def concept_section():
    score = answered = 0
    cols = st.columns(2, gap="medium")
    for i, (k, q, opts, ans, ex) in enumerate(qs):
        with cols[i % 2]:
            r = ui.mcq(k, q, opts, ans, ex, scored=True)
            if r is not None:
                answered += 1
                score += int(r)
    if answered:
        st.progress(score / len(qs), text=f"Concept check: {score} / {len(qs)} correct ({answered} answered)")
    if answered == len(qs) and score == len(qs):
        st.success("Full marks - contours, polygons and Hough mastered.", icon=":material/military_tech:")


concept_section()

ui.lesson_nav(__file__)
