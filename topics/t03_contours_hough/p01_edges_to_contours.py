import cv2
import numpy as np
import streamlit as st

from core import anim, live, theme, ui
from core import contour_ui as cui
from core import contours as K
from core import edges as E

ui.lesson_header(
    __file__,
    "An edge detector gives you a <b>set</b> of pixels. To measure a shape - its area, its perimeter, its corners - "
    "you need those pixels as an <b>ordered list</b> that walks around the boundary: a <b>contour</b>. On this page "
    "you will see the jump from a pile of pixels to an ordered chain, and the two roads this topic takes from there.",
)

# ------------------------------------------------------------------ the two roads
st.subheader("Where this topic goes", divider="gray")
c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown("**Road 1 · follow the boundary** (lessons 1-6)")
    with st.container(horizontal=True, gap="small"):
        for i, s_ in enumerate(["Image", "Edges", "Contours", "Geometric representation"]):
            st.badge(s_, color="blue")
            if i < 3:
                st.markdown("→")
    st.caption("Contour detection, then polygonal approximation and measurements. Works for ANY shape.")
with c2:
    st.markdown("**Road 2 · vote for a model** (lessons 7-10)")
    with st.container(horizontal=True, gap="small"):
        for i, s_ in enumerate(["Image", "Edges", "Hough voting", "Geometric primitive"]):
            st.badge(s_, color="orange")
            if i < 3:
                st.markdown("→")
    st.caption("The Hough transform. Only for shapes with a formula (lines, circles), but robust to gaps.")


@st.cache_resource(show_spinner=False, max_entries=8)
def canny_of(name: str) -> np.ndarray:
    return E.canny(K.SCENES[name](), 1.4, 0.1, 0.25)["edges"]


@live.section
def explore_section():
    st.subheader("A set of pixels vs. an ordered contour", divider="gray")
    c_ctrl, c_main = st.columns([0.8, 3], gap="large")
    with c_ctrl:
        B, img, name, t = cui.pick_scene("ec", t_default=0.45)
        st.caption("Thresholding gives the binary image B(x, y): 1 = object, 0 = background. Contours are traced "
                   "around the objects of B.")
    Ed = canny_of(name)
    cs, _ = K.find_contours(B, cv2.RETR_LIST, min_points=8)
    big = sorted(cs, key=len, reverse=True)
    with c_main:
        a, b, c = st.columns(3, gap="small")
        a.image(ui.datauri(K.binary_rgb(B)), caption=f"binary image B: {int(B.sum()):,} object pixels", width="stretch")
        b.image(ui.datauri(E.edges_rgb(Ed)), caption=f"Canny edge map E: {int(Ed.sum()):,} edge pixels - but which "
                "belong together?", width="stretch")
        vis = K.draw_contours(K.gray_rgb(img, 0.55), big, K.PALETTE, scale=2, thickness=2)
        c.image(ui.datauri(vis), caption=f"{len(big)} contours, one colour each: every pixel now knows its "
                "neighbours along the boundary", width="stretch")
        m1, m2, m3 = st.columns(3)
        m1.metric("contours", len(big), border=True)
        m2.metric("points in the longest", len(big[0]) if big else 0, border=True)
        m3.metric("points in all contours", sum(len(q) for q in big), border=True)
        first = big[0][:6].astype(int).tolist() if big else []
        st.code("C = [" + ", ".join(f"({x}, {y})" for x, y in first) + (", ...]" if big else "]") +
                "      # the longest contour, (x, y) in order", language="python")


explore_section()

# ------------------------------------------------------------------ animation
GRID = K.parse_grid([
    "............",
    "...######...",
    "..########..",
    "..#########.",
    "..#########.",
    "...#######..",
    "....####....",
    "............",
])


@st.cache_resource(show_spinner=False)
@live.full_quality()          # cached: never store a coarse mid-drag version
def ordering_figure():
    tr = K.moore_trace(GRID)
    pts = np.array(tr["points"], float)               # (row, col), in walking order
    n = len(pts)
    rng = np.random.default_rng(7)
    shuffled = rng.permutation(n)
    blank = [""] * n
    set_labels = [str(int(v)) for v in shuffled]

    def state(labels, path_f, start_size, close_f, color=theme.HIGHLIGHT):
        closing = np.array([pts[-1], pts[-1] + close_f * (pts[0] - pts[-1])])
        return [cui.markers(pts, color, size=21, text=labels),
                cui.path(cui.grow(pts, path_f), theme.TRANSFORMED, width=4),
                cui.path(closing, theme.TRANSFORMED, width=4, dash="dot"),
                cui.markers(pts[:1], theme.E1, size=start_size, symbol="circle-open")]

    def walk(f):
        k = int(np.floor(f * (n - 1)))
        return [str(i) if i <= k else "" for i in range(n)]

    steps = [
        anim.Step("An edge map is a SET of boundary pixels", "E(x, y) ∈ {0, 1}",
                  "We know WHICH pixels are on the boundary - but nothing says which one comes after which.",
                  state(blank, 0, 0, 0)),
        anim.Step("A set has no order", "{p₀, p₁, …} - any numbering is as good as any other",
                  "Numbered at random: following 0 → 1 → 2 would jump all over the shape. No perimeter, no area yet.",
                  state(set_labels, 0, 0, 0)),
        anim.Step("Pick a start pixel p₀", f"p₀ = ({int(pts[0][1])}, {int(pts[0][0])})  (the first one in raster order)",
                  "Top row first, then left to right: the first object pixel we meet is surely on the boundary.",
                  state(walk(0), 0, 34, 0)),
        anim.Step("Walk to a neighbouring boundary pixel, again and again",
                  f"p₀ → p₁ → p₂ → … ({anim.hl('consecutive points are neighbours')})",
                  "Each step goes to one of the 8 neighbours. The numbers now tell you where you are along the boundary.",
                  state(walk(0.5), 0.5, 34, 0), between=lambda t: state(walk(0.5 * t), 0.5 * t, 34, 0)),
        anim.Step("… all the way round", f"C = {{p₀, p₁, …, p_N−1}},  N = {n}",
                  "The same pixels as in step 1 - but now as an ordered list: a contour.",
                  state(walk(1), 1, 34, 0), between=lambda t: state(walk(0.5 + 0.5 * t), 0.5 + 0.5 * t, 34, 0)),
        anim.Step("Close the loop", f"{anim.hl('p_N = p₀')}: the last point connects back to the first",
                  "A closed contour is a closed chain of connected boundary pixels: now we can add up lengths and areas.",
                  state(walk(1), 1, 34, 1)),
    ]
    base = cui.grid_figure(GRID, height=430)
    return anim.step_player(base, steps, n_sub=12, hold=16, wrap_at=95)


st.subheader("Step by step: from a pile of pixels to an ordered contour", divider="gray")
f1, f2 = st.columns([1.5, 1], gap="large")
with f1:
    ui.anim_intro("Step by step: ordering the boundary pixels")
    st.plotly_chart(ordering_figure(), key="ec_order")
with f2:
    st.markdown("**The notation of the notes**")
    st.latex(r"E(x, y) \in \{0, 1\}, \qquad E(x, y) = 1 \;\Rightarrow\; \text{edge pixel}")
    st.latex(r"C = \{p_0, p_1, \dots, p_{N-1}\}, \qquad p_i = (x_i, y_i)")
    st.latex(r"p_0 \to p_1 \to p_2 \to \cdots \to p_{N-1} \;(\to p_0)")
    ui.callout("intuition", "An edge map is like a bag of beads. A contour is the same beads **on a string**: the "
                            "string tells you which bead comes next, so you can walk around the shape.")
    st.markdown("| | Edge map | Contour |\n|---|---|---|\n| What it is | a set of pixels | an ordered list |\n"
                "| Which pixels belong together? | unknown | one list per boundary |\n"
                "| Perimeter, area | ✗ | ✓ (lesson 4) |\n| Corners, polygon | ✗ | ✓ (lesson 5) |\n"
                "| Holes vs. objects | ✗ | ✓ (lesson 3) |")

ui.callout("pitfall", "Image coordinates: **x = column, y = row, and y points DOWN**. A contour point (x, y) is "
                      "pixel `B[y, x]` in NumPy. Mixing up (row, col) and (x, y) is the most common contour bug.")
ui.callout("takeaway", "Edges say *where* intensity changes. Contours say *which* of those pixels form one "
                       "boundary, and in *what order* - and the order is what makes geometry possible.")

ui.whys([
    ("Why isn't an edge map enough?",
     "Because it is only a set: 5,000 white pixels with no grouping. To ask 'how big is the coin?' you must first "
     "know which pixels belong to the coin's boundary and how they connect. A contour answers both - one list per "
     "boundary, in walking order."),
    ("Why does the order matter?",
     "Every measurement in this topic uses neighbours along the boundary: the perimeter adds up ‖p_{i+1} − p_i‖, the "
     "shoelace area uses x_i y_{i+1} − x_{i+1} y_i, RDP needs the first and last point of a stretch. Shuffle the "
     "points and all of these give nonsense."),
    ("Why threshold first instead of using Canny edges?",
     "Contour tracing needs **closed regions**. Thresholding gives filled objects whose boundaries are closed by "
     "construction. A Canny edge is a 1-pixel-thin *curve* with gaps; OpenCV treats it as a very thin object and "
     "walks around it - down one side and back up the other - so you get roughly twice as many points as edge pixels, "
     "and an open edge never encloses an area. Try both in lesson 3."),
    ("Is a contour the same as an edge?",
     "No. An edge is a *local* property (big intensity change here). A contour is a *global* object (this whole closed "
     "chain is one boundary). Edges are the raw material; contours are the organised result."),
    ("Why does y point down?",
     "Images are stored row by row from the top. Row 0 is the top line, so the row index - our y - grows downwards. "
     "OpenCV, NumPy and this course all use that convention for images."),
    ("What if an object touches the image border?",
     "Pixels outside the image count as background, so the contour simply runs along the border. Measurements of such "
     "an object are measurements of the *visible part* only - a common source of wrong areas."),
])

st.divider()
ui.mcq("ec1", "What turns a set of boundary pixels into a contour?",
       ["Sorting them by x", "Ordering them so that consecutive points are neighbours along the boundary",
        "Thresholding them", "Smoothing them"], 1,
       "A contour is an ordered sequence p₀ → p₁ → … in which consecutive points are neighbouring boundary pixels.")
ui.mcq("ec2", "A contour point is (x, y) = (40, 12). Which NumPy element is it?",
       ["B[40, 12]", "B[12, 40]", "B[−12, 40]", "B[40, −12]"], 1,
       "x is the column and y is the row: B[row, col] = B[y, x] = B[12, 40].")

ui.lesson_nav(__file__)
