import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import anim, live, theme, ui
from core import contour_ui as cui
from core import contours as K

ui.lesson_header(
    __file__,
    "Finding the boundary pixels is not enough - we must <b>walk</b> along them in order. Watch Moore-neighbour "
    "tracing turn around each pixel until it finds the next one, see why it must not stop the first time it is back "
    "at the start, and how OpenCV tells <b>holes</b> from <b>objects</b> and builds a contour hierarchy.",
)

TINY = K.parse_grid([
    "...........",
    "....#####..",
    "...#.......",
    "..###......",
    "..####.....",
    "..###......",
    "...........",
])
TR = K.moore_trace(TINY)
PTS = np.array(TR["points"], float)
LOG = TR["log"]


def _check_state(move, k_shown, p_at=None, b_at=None):
    """Markers for one Moore move: p (orange square), b (red ×) and the first k_shown checked neighbours."""
    p = move["p"] if p_at is None else p_at
    b = move["back"] if b_at is None else b_at
    checks = move["checks"] + [move["p"]] * (8 - len(move["checks"]))
    found_i = len(move["checks"]) - 1
    cols = [theme.E2 if i == found_i else theme.OUTLIER for i in range(8)]
    sizes = np.array([18.0 if i < k_shown and i < len(move["checks"]) else 0.0 for i in range(8)])
    labels = [str(i + 1) if i < k_shown and i < len(move["checks"]) else "" for i in range(8)]
    return [cui.markers(checks, cols, size=sizes, symbol="circle", text=labels, textcolor="white"),
            cui.markers([b], theme.OUTLIER, size=20, symbol="x"),
            cui.markers([p], theme.E1, size=30, symbol="square-open")]


@st.cache_resource(show_spinner=False)
@live.full_quality()
def one_step_figure():
    j = max(range(1, len(LOG)), key=lambda i: len(LOG[i]["checks"]))     # a move with many checks
    m = LOG[j]
    done = PTS[:j + 1]
    ctx = cui.path(np.vstack([done, np.full((len(PTS) - len(done), 2), np.nan)]), theme.TRANSFORMED, width=3)
    names = ["up", "up-right", "right", "down-right", "down", "down-left", "left", "up-left"]
    d0 = K.N8.index((m["back"][0] - m["p"][0], m["back"][1] - m["p"][1]))
    steps = [anim.Step("We stand on p, and we know one background neighbour b",
                       f"p = ({m['p'][1]}, {m['p'][0]}),  b = ({m['back'][1]}, {m['back'][0]}) - the pixel we came from",
                       "b is background for sure: it was checked (and rejected) in the previous move.",
                       [ctx] + _check_state(m, 0))]
    for k in range(1, len(m["checks"]) + 1):
        q = m["checks"][k - 1]
        fg = k == len(m["checks"])
        dname = names[(d0 + k) % 8]
        steps.append(anim.Step(f"Check neighbour {k}, turning clockwise from b",
                               f"{dname}: ({q[1]}, {q[0]}) is " + (anim.hl("FOREGROUND", theme.E2) if fg else
                                                                   anim.hl("background", theme.OUTLIER)),
                               "Found the next contour pixel - stop turning." if fg else
                               "Background - keep turning clockwise.", [ctx] + _check_state(m, k)))
    prev = m["checks"][-2] if len(m["checks"]) > 1 else m["back"]
    steps.append(anim.Step("Move: the found pixel becomes p, the last background check becomes b",
                           f"p ← ({m['found'][1]}, {m['found'][0]}),   b ← ({prev[1]}, {prev[0]})",
                           "Then repeat. Because b is always outside and we turn clockwise from it, we hug the object.",
                           [ctx] + _check_state(m, len(m["checks"]), p_at=m["found"], b_at=prev)))
    return anim.step_player(cui.grid_figure(TINY, height=420), steps, n_sub=10, hold=14, wrap_at=58)


@st.cache_resource(show_spinner=False)
@live.full_quality()
def full_trace_figure():
    n = len(LOG)
    s = tuple(PTS[0].astype(int))
    first = LOG[0]["found"]
    codes = K.chain_code(TR["points"])

    def make_state(i):
        i = int(i)
        if i < n:
            m = LOG[i]
            trs = _check_state(m, 8)
        else:
            m = LOG[-1]
            trs = _check_state(dict(p=s, back=(s[0], s[1] - 1), checks=[first], found=first), 1)
        upto = PTS[:i + 1] if i < n else np.vstack([PTS, PTS[:1]])
        line = np.vstack([upto, np.full((len(PTS) + 1 - len(upto), 2), np.nan)])
        start = cui.markers([PTS[0]], theme.HIGHLIGHT, size=34, symbol="circle-open")
        chain = "".join(K.FREEMAN_ARROWS[c] for c in codes[:i])
        if i < n:
            at_s = tuple(m["p"]) == s and i > 0
            cap = (f"move {i + 1} of {n} · chain so far: {chain or '-'}" +
                   ("   ·   <b>back at p₀ - but the next move is NOT the first move: keep going</b>" if at_s else ""))
        else:
            cap = (f"back at p₀ AND the next move would repeat the first one → <b>stop</b> · {len(PTS)} contour points "
                   f"· chain: {chain}")
        return [cui.path(line, theme.TRANSFORMED, width=3)] + trs + [start], cap

    return anim.scrubber(cui.grid_figure(TINY, height=420), range(n + 1), make_state, active=0, label="move",
                         fmt="{:.0f}", frame_ms=180, bounce=False, wrap_at=62)


st.subheader("Moore-neighbour tracing", divider="gray")
c1, c2 = st.columns(2, gap="large")
with c1:
    ui.anim_intro("Step by step: one move of the tracer")
    st.plotly_chart(one_step_figure(), key="ct_one")
with c2:
    ui.anim_intro("Sweep: the whole contour, move by move",
                  "▶ Play or drag. Numbers = the order in which the 8 neighbours are checked; green = the one taken.")
    st.plotly_chart(full_trace_figure(), key="ct_full")

naive = K.moore_trace(TINY, naive=True)
cv_cs, _ = K.find_contours(TINY, cv2.RETR_EXTERNAL)
m1, m2, m3 = st.columns(3)
m1.metric("stop at the FIRST return to p₀", f"{len(naive['points'])} points", border=True,
          help="Misses everything after the first visit back to the start")
m2.metric("stop when the first MOVE repeats", f"{len(PTS)} points", border=True)
m3.metric("cv2.findContours (same object)", f"{len(cv_cs[0])} points", border=True)
st.caption("The start pixel sits where two parts of the object meet, so the tracer passes through it twice. Stopping "
           "the first time loses the whole lower part. The safe rule (Jacob's stopping criterion): stop only when "
           "you are back at p₀ **and** about to repeat the first move. Our tracer and OpenCV visit the same pixels.")

with st.expander("The algorithm in code", icon=":material/code:"):
    st.code("""s = first object pixel in raster order;  b = its left neighbour (background)
p, contour = s, [s]
while True:
    # turn clockwise around p, starting just after b
    for q in clockwise_neighbours(p, starting_after=b):
        if B[q] == 1:
            break                 # q = next contour pixel
        b_last = q                # remember the last background pixel we checked
    if p == s and q == first_move and len(contour) > 1:
        break                     # back at the start, about to repeat the first move
    contour.append(q)
    p, b = q, b_last""", language="python")

# ------------------------------------------------------------------ holes and hierarchy
DEPTH_COL = [K.BLUE, K.ORANGE, K.GREEN, K.MAGENTA, K.AMBER, K.RED]
DEPTH_HEX = [theme.TRANSFORMED, theme.E1, theme.E2, theme.MAGENTA, theme.HIGHLIGHT, theme.OUTLIER]


def tree_layout(hier: np.ndarray):
    """x, y positions for the contour tree: depth goes down, siblings side by side (leaves spaced evenly)."""
    n = len(hier)
    kids = {i: [] for i in range(-1, n)}
    for i in range(n):
        kids[hier[i][3]].append(i)
    pos, x = {}, [0]

    def place(i, d):
        if not kids[i]:
            pos[i] = (x[0], -d)
            x[0] += 1
            return pos[i][0]
        xs = [place(c, d + 1) for c in kids[i]]
        pos[i] = (float(np.mean(xs)), -d)
        return pos[i][0]

    for r in kids[-1]:
        place(r, 0)
    return pos


@st.cache_resource(show_spinner=False)
@live.full_quality()
def tree_build_figure():
    B = K.binarize(K.ring_scene(), 0.5)
    cs, hier = K.find_contours(B, cv2.RETR_TREE)
    dep = K.depths(hier)
    pos = tree_layout(hier)
    order = sorted(range(len(cs)), key=lambda i: (dep[i], pos[i][0]))
    n = len(cs)
    xy = np.array([pos[i] for i in range(n)], float)
    par = [hier[i][3] for i in range(n)]
    kind = ["object" if d % 2 == 0 else "hole" for d in dep]

    def state(shown):
        sizes = np.array([30.0 if i in shown else 0.0 for i in range(n)])
        ex, ey = [], []
        for i in range(n):
            p = par[i]
            a = xy[p] if p >= 0 else xy[i]
            b = xy[i] if (i in shown and p >= 0) else a
            ex += [a[0], b[0], np.nan]
            ey += [a[1], b[1], np.nan]
        text = [f"C{i}" if i in shown else "" for i in range(n)]
        return [cui.scatter(ex, ey, theme.ORIGINAL, mode="lines", width=2),
                dict(type="scatter", x=xy[:, 0], y=xy[:, 1], mode="markers+text", text=text,
                     textfont=dict(size=11, color="white"),
                     marker=dict(size=sizes, color=[DEPTH_HEX[d % 6] for d in dep], line=dict(color="white", width=1)),
                     hoverinfo="skip", showlegend=False)]

    steps = [anim.Step("Start with no contours", "hierarchy[i] = [next, previous, first_child, parent]",
                       "OpenCV returns one row per contour; parent = −1 means 'not inside anything'.", state(set()))]
    shown = set()
    for d in range(max(dep) + 1):
        new = [i for i in order if dep[i] == d]
        shown |= set(new)
        what = {0: "outer boundaries of top-level objects", 1: "holes inside those objects",
                2: "objects sitting inside holes", 3: "holes of those objects"}.get(d, "deeper levels")
        steps.append(anim.Step(f"Depth {d}: {what}",
                               "  ".join(f"C{i}: parent = {par[i]}" for i in new),
                               ("Even depth = an object boundary, odd depth = a hole. " if d else
                                "Contours not enclosed by anything. ") + f"{len(new)} contour(s) at this level.",
                               state(set(shown))))
    base = go.Figure(layout=dict(height=470, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                                 xaxis=dict(visible=False, range=[xy[:, 0].min() - 0.8, xy[:, 0].max() + 0.8]),
                                 yaxis=dict(visible=False, range=[xy[:, 1].min() - 0.6, 0.6])))
    vis = K.draw_contours(K.gray_rgb(K.ring_scene(), 0.5), cs, [DEPTH_COL[d % 6] for d in dep], scale=2, thickness=3)
    for i, c in enumerate(cs):
        x, y = c[np.argmin(c[:, 1] * 1000 + c[:, 0])]
        cv2.putText(vis, f"C{i}", (int(x * 2) + 4, int(y * 2) + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    DEPTH_COL[dep[i] % 6], 2, cv2.LINE_AA)
    return anim.step_player(base, steps, n_sub=10, hold=16, wrap_at=62), vis, kind


st.subheader("Outer contours, holes and the contour hierarchy", divider="gray")
st.latex(r"\text{Outer contour} \to \text{Inner contour}, \qquad C_0 \to \{C_1, C_2, \dots, C_k\}")
fig_tree, vis_tree, _ = tree_build_figure()
t1, t2 = st.columns([1, 1.2], gap="large")
with t1:
    st.image(ui.datauri(vis_tree), caption="cv2.findContours(…, RETR_TREE): blue = depth 0, orange = 1, green = 2, "
             "magenta = 3", width="stretch")
with t2:
    ui.anim_intro("Step by step: building the hierarchy tree")
    st.plotly_chart(fig_tree, key="ct_tree")


@live.section
def opencv_section():
    st.markdown("**Try OpenCV's retrieval modes on any image:**")
    c_ctrl, c_img, c_tab = st.columns([0.8, 1.4, 1.3], gap="large")
    with c_ctrl:
        B, img, name, t = cui.pick_scene("ct", default="Nested rings (synthetic)", t_default=0.5)
        mode_name = st.selectbox("Retrieval mode", list(K.MODES), index=3, key="ct_mode")
        approx_name = st.radio("Points kept", list(K.APPROX), index=0, key="ct_approx")
    cs, hier = K.find_contours(B, K.MODES[mode_name], K.APPROX[approx_name], min_points=1)
    keep = [i for i, c in enumerate(cs) if len(c) >= 3 or K.area(c) > 0]
    dep = K.depths(hier) if len(hier) else []
    with c_img:
        vis = K.draw_contours(K.gray_rgb(img, 0.5), cs, [DEPTH_COL[d % 6] for d in dep] or None, scale=2, thickness=2,
                              dots=K.APPROX[approx_name] == cv2.CHAIN_APPROX_SIMPLE)
        st.image(ui.datauri(vis), caption="contour colour = depth (blue 0, orange 1, green 2, magenta 3)",
                 width="stretch")
        n_none = sum(len(c) for c in K.find_contours(B, K.MODES[mode_name], cv2.CHAIN_APPROX_NONE)[0])
        n_simple = sum(len(c) for c in K.find_contours(B, K.MODES[mode_name], cv2.CHAIN_APPROX_SIMPLE)[0])
        st.caption(f"Measured: CHAIN_APPROX_NONE stores **{n_none:,}** points, SIMPLE only **{n_simple:,}** "
                   f"({100 * n_simple / max(n_none, 1):.0f} %) - the same boundaries, without the points in the middle "
                   "of straight horizontal, vertical or diagonal runs.")
    with c_tab:
        rows = []
        for i, c in enumerate(cs[:200]):
            sa = K.signed_area(c)
            rows.append(dict(contour=f"C{i}", points=len(c), parent=int(hier[i][3]), child=int(hier[i][2]),
                             depth=dep[i], kind=("hole" if dep[i] % 2 else "object") if mode_name.startswith(("CCOMP", "TREE"))
                             else "-", area=round(abs(sa), 1),
                             turns="↺ on screen" if sa < 0 else ("↻ on screen" if sa > 0 else "-")))
        df = pd.DataFrame(rows, columns=["contour", "points", "parent", "child", "depth", "kind", "area", "turns"])
        st.dataframe(df, hide_index=True, height=300, width="stretch")
        st.caption(f"{len(cs)} contours ({len(keep)} enclose an area). parent/child = −1: none. *turns*: the sign of "
                   "the shoelace area in image coordinates (lesson 4) - OpenCV walks around objects one way and around "
                   "holes the other way.")


opencv_section()
st.markdown("| Mode | What you get | Use it when |\n|---|---|---|\n"
            "| EXTERNAL | only the outermost boundaries | counting / measuring separate objects |\n"
            "| LIST | every contour, no parents | you just need all boundaries |\n"
            "| CCOMP | 2 levels: objects and their holes | area of objects *minus* their holes |\n"
            "| TREE | the full nesting | objects inside holes inside objects (documents, targets, nested parts) |")

# ------------------------------------------------------------------ chain code
st.subheader("Storing a contour compactly: Freeman chain code", divider="gray")
codes = K.chain_code(TR["points"])
cc1, cc2 = st.columns([1, 1.6], gap="large")
with cc1:
    Z = np.array([[3, 2, 1], [4, -1, 0], [5, 6, 7]], float)
    txt = [[f"{int(v)} {K.FREEMAN_ARROWS[int(v)]}" if v >= 0 else "p" for v in r] for r in Z]
    fig = go.Figure(go.Heatmap(z=np.where(Z < 0, 1, 0.4), text=txt, texttemplate="%{text}",
                               textfont=dict(size=17, color="#1a1a19"), colorscale=[[0, theme.rgba(theme.E2, 0.55)],
                                                                                   [1, theme.HIGHLIGHT]],
                               zmin=0, zmax=1, showscale=False, xgap=4, ygap=4, hoverinfo="skip"),
                    layout=dict(height=240, margin=dict(l=30, r=30, t=10, b=0), xaxis=dict(visible=False),
                                yaxis=dict(visible=False, autorange="reversed", scaleanchor="x")))
    st.plotly_chart(fig, key="ct_freeman")
with cc2:
    st.markdown("Instead of storing every (x, y), store the start point and **one direction number (0-7) per move**. "
                "For the little shape above:")
    st.code(f"start = ({int(PTS[0][1])}, {int(PTS[0][0])})\ncode  = {''.join(map(str, codes))}\n"
            f"        {''.join(K.FREEMAN_ARROWS[c] for c in codes)}", language="text")
    n = len(codes)
    st.markdown(f"{n} moves × **3 bits** = {3 * n} bits, against {n} points × 2 coordinates × 32 bits = {64 * n} bits "
                f"→ **{64 * n / (3 * n + 64):.0f}× smaller** (counting 64 bits for the start point). Chain codes also make "
                "some measurements easy: the number of odd codes is the number of diagonal steps (length √2).")

ui.callout("pitfall", "cv2.findContours treats every **non-zero** pixel as object and the area outside the image as "
                      "background. Dark objects on a bright page give ONE big contour around the whole image (try "
                      "*invert* above). Make the objects white on black first.")
ui.callout("takeaway", "Tracing = turn clockwise around the current pixel from a known background pixel; the first "
                       "object pixel is the next contour point. Stop only when the first move repeats. Holes are "
                       "traced too, and the hierarchy records who is inside whom.")

ui.whys([
    ("Why is stopping at the first pixel not enough?",
     "The start pixel can be a 'junction' that the contour passes through more than once (a one-pixel-thin neck or "
     "spur). The metrics above show it: stopping at the first return gives a contour of only part of the object. "
     "Requiring the first *move* to repeat guarantees the whole loop is done."),
    ("How does OpenCV know a hole from an object?",
     "OpenCV uses Suzuki & Abe's border-following algorithm (1985). It scans the image row by row; a 0→1 step "
     "starts an *outer* border, a 1→0 step (with object to the left) starts a *hole* border. While following each "
     "border it records which border it was found inside - that gives the parent, and the hierarchy."),
    ("Why does OpenCV want white objects on a black background?",
     "Because 'object' means non-zero and the outside of the image counts as zero. If the background is white, the "
     "background itself is the biggest 'object' and the real objects become its holes."),
    ("Which way does the contour run?",
     "Measured on the table above: in image coordinates (y down) OpenCV's object contours have a negative shoelace "
     "area - they run anti-clockwise on screen - and holes run the other way. Our Moore tracer turns clockwise, so it "
     "walks object boundaries clockwise on screen. Either is fine as long as you know which."),
    ("Is the contour on the pixels or between them?",
     "On the pixels: OpenCV returns the centres of the object's boundary pixels. Walking along the cracks *between* "
     "pixels is another option (crack following); its length is longer (lesson 4 measures both)."),
    ("Why does one object sometimes give several contours?",
     "Holes give their own contours, and with 8-connectivity two blobs touching at one corner are ONE object. Noise "
     "specks give tiny contours: filter them by area or by the number of points."),
])

st.divider()
ui.mcq("ct1", "In Moore tracing, where does the clockwise search around p start?",
       ["At the pixel above p", "Just after the background pixel b we came from", "At the start pixel p₀",
        "At a random neighbour"], 1,
       "Starting from the known background neighbour and turning clockwise makes the tracer hug the object.")
ui.mcq("ct2", "With RETR_TREE, a contour at depth 1 is …",
       ["an object inside a hole", "a hole in a top-level object", "a top-level object", "noise"], 1,
       "Depth 0 = outermost objects, 1 = their holes, 2 = objects inside those holes, …")

ui.lesson_nav(__file__)
