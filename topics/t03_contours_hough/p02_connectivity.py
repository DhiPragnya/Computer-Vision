import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core import anim, live, theme, ui
from core import contour_ui as cui
from core import contours as K

ui.lesson_header(
    __file__,
    "Before we can follow a boundary we must agree what 'next to' means for pixels. With <b>4-connectivity</b> only "
    "up, down, left and right count; with <b>8-connectivity</b> the diagonals count too. That one choice changes how "
    "many objects you see, which pixels are on the boundary - and whether a ring has an inside.",
)

# ------------------------------------------------------------------ N4 and N8
st.subheader("Two kinds of neighbours", divider="gray")
c1, c2 = st.columns([1.1, 1], gap="large")
with c1:
    st.latex(r"N_4(x, y) = \{(x-1, y),\ (x+1, y),\ (x, y-1),\ (x, y+1)\}")
    st.latex(r"N_8(x, y) = \{(x+i,\ y+j) : i, j \in \{-1, 0, 1\},\ (i, j) \neq (0, 0)\}")
    st.markdown("Two pixels are **connected** if you can walk from one to the other through object pixels, each step "
                "going to a neighbour. *Which* neighbours you allow decides what counts as one object.")
with c2:
    Z = np.array([[0, 1, 0, -1, 1, 1, 1], [1, 2, 1, -1, 1, 2, 1], [0, 1, 0, -1, 1, 1, 1]], float)
    txt = [["", "↑", "", "", "↖", "↑", "↗"], ["←", "p", "→", "", "←", "p", "→"], ["", "↓", "", "", "↙", "↓", "↘"]]
    fig = go.Figure(go.Heatmap(z=Z, text=txt, texttemplate="%{text}", textfont=dict(size=20, color="#1a1a19"),
                               colorscale=[[0, "rgba(0,0,0,0)"], [0.33, cui.GRID_BG], [0.66, theme.rgba(theme.E2, 0.75)],
                                           [1, theme.HIGHLIGHT]], zmin=-1, zmax=2, showscale=False, xgap=4, ygap=4,
                               hoverinfo="skip"),
                    layout=dict(height=230, margin=dict(l=10, r=10, t=30, b=0),
                                xaxis=dict(visible=False), yaxis=dict(visible=False, autorange="reversed",
                                                                      scaleanchor="x"),
                                annotations=[dict(text="N₄: 4 neighbours", x=1, y=-0.75, showarrow=False,
                                                  font=dict(size=13)),
                                             dict(text="N₈: 8 neighbours", x=5, y=-0.75, showarrow=False,
                                                  font=dict(size=13))]))
    st.plotly_chart(fig, key="px_n48")

# ------------------------------------------------------------------ labelling animation
SCENE = K.parse_grid([
    "............",
    ".##.....#...",
    ".##....#.#..",
    "...#..#...#.",
    "....#..#.#..",
    ".....#..#...",
    "..##..#.....",
    "..##........",
])


@st.cache_resource(show_spinner=False)
@live.full_quality()
def flood_figure():
    h, w = SCENE.shape
    off = w + 1                                           # the N8 copy sits to the right of the N4 copy
    lab4, dep4 = K.label_layers(SCENE, 4)
    lab8, dep8 = K.label_layers(SCENE, 8)
    rr, cc = np.nonzero(SCENE)
    col4 = [_hex(K.PALETTE[lab4[r, c] % 8]) for r, c in zip(rr, cc)]
    col8 = [_hex(K.PALETTE[lab8[r, c] % 8]) for r, c in zip(rr, cc)]
    n4, n8 = lab4.max() + 1, lab8.max() + 1
    big = 26

    def state(k):
        s4 = np.where(dep4[rr, cc] <= k, big, 0).astype(float)
        s8 = np.where(dep8[rr, cc] <= k, big, 0).astype(float)
        return [cui.markers(np.c_[rr, cc], col4, size=s4), cui.markers(np.c_[rr, cc + off], col8, size=s8)]

    kmax = int(max(dep4.max(), dep8.max()))
    steps = [anim.Step("Start: every object pixel is still unlabelled",
                       "Scan in raster order; the first unlabelled object pixel starts a new label",
                       "Left: 4-connectivity. Right: the SAME image with 8-connectivity.", state(-1)),
             anim.Step("Seeds: one new label per object found", "step 0",
                       "Each colour is one label. A seed is placed whenever the scan meets a pixel nobody reached yet.",
                       state(0))]
    for k in range(1, kmax + 1):
        steps.append(anim.Step(f"Grow by one ring of neighbours (step {k})",
                               f"label(q) = label(p)  for every q ∈ N(p) ∩ Ω  ({anim.hl('N₄ left, N₈ right')})",
                               "Paint flows only to neighbours. Diagonal steps are allowed on the right only.",
                               state(k)))
    steps.append(anim.Step("Done: count the colours",
                           f"N₄: {anim.hl(f'{n4} objects', theme.E1)}   ·   N₈: {anim.hl(f'{n8} objects', theme.E1)}",
                           "Pixels that touch only at a corner are separate objects under N₄ but one object under N₈.",
                           state(kmax)))
    Z = np.hstack([SCENE, np.full((h, 1), np.nan), SCENE]).astype(float)
    base = go.Figure(data=[cui.grid_heatmap(Z)], layout=cui.grid_layout(Z.shape, height=470))
    return anim.step_player(base, steps, n_sub=8, hold=12, wrap_at=100)


def _hex(rgb):
    return "#%02x%02x%02x" % tuple(rgb)


st.subheader("Same pixels, different objects", divider="gray")
ui.anim_intro("Step by step: labelling connected components (flood fill)",
              "▶ Play: each object is flooded from its first pixel, one ring of neighbours per step. "
              "Left uses N₄, right uses N₈.")
st.plotly_chart(flood_figure(), key="px_flood")
n4, _ = K.label(SCENE, 4)
n8, _ = K.label(SCENE, 8)
st.caption(f"Measured with cv2.connectedComponents on the same grid: **{n4} objects with N₄**, **{n8} with N₈**.")

# ------------------------------------------------------------------ boundary pixels
SHAPE = K.parse_grid([
    "..............",
    ".....####.....",
    "...########...",
    "..##########..",
    "..##########..",
    "..##########..",
    "...########...",
    ".....####.##..",
    "..........##..",
    "..............",
])


@st.cache_resource(show_spinner=False, max_entries=4)
@live.full_quality()
def erosion_figure(conn: int):
    """∂Ω = Ω \\ (Ω ⊖ N): the boundary is what is left after removing the interior."""
    rr, cc = np.nonzero(SHAPE)
    bd = K.boundary(SHAPE, conn)
    interior = SHAPE & ~bd
    big = 22

    def state(show_fg, show_int, show_bd):
        return [cui.markers(np.c_[rr, cc], theme.TRANSFORMED, size=np.where(show_fg, big, 0).astype(float)),
                cui.markers(np.c_[rr, cc], theme.E2, size=np.where(show_int & interior[rr, cc], big, 0).astype(float)),
                cui.markers(np.c_[rr, cc], theme.HIGHLIGHT, size=np.where(show_bd & bd[rr, cc], big, 0).astype(float))]

    N = "N₄" if conn == 4 else "N₈"
    all_ = np.ones(len(rr), bool)
    none = np.zeros(len(rr), bool)
    steps = [
        anim.Step("The object", "Ω = {(x, y) : B(x, y) = 1}", "Blue = every object pixel.", state(all_, none, none)),
        anim.Step(f"Which pixels have ALL their {N} neighbours inside Ω?", f"N(p) ⊆ Ω  ({N})",
                  "Green = interior pixels: no step in any allowed direction leaves the object.",
                  state(all_, all_, none)),
        anim.Step("That set is the erosion of Ω", f"{anim.hl('Ω ⊖ N')} = {{p : N(p) ⊆ Ω}}",
                  "Erosion with the neighbourhood as structuring element (cv2.erode does exactly this).",
                  state(none, all_, none)),
        anim.Step("Remove the interior", f"∂Ω = Ω \\ (Ω ⊖ N)", "What is left are object pixels with at least one "
                  "background neighbour.", state(none, none, all_)),
        anim.Step("… which is the definition in the notes", f"∂Ω = {{p ∈ Ω : {anim.hl('N(p) ∩ Ωᶜ ≠ ∅')}}}",
                  f"Boundary pixels = foreground pixels that touch the background ({int(bd.sum())} pixels with {N}).",
                  state(none, none, all_)),
    ]
    base = cui.grid_figure(SHAPE, height=440)
    return anim.step_player(base, steps, n_sub=10, hold=14, wrap_at=48)


@st.cache_resource(show_spinner=False, max_entries=4)
@live.full_quality()
def sweep_figure(conn: int):
    """The boundary test, pixel by pixel."""
    offs = K.N4 if conn == 4 else K.N8
    rr, cc = np.nonzero(SHAPE)
    bd = K.boundary(SHAPE, conn)
    is_b = bd[rr, cc]
    base = cui.grid_figure(SHAPE, height=440)

    def make_state(i):
        i = int(i)
        r, c = rr[i], cc[i]
        nb = np.array([(r + dr, c + dc) for dr, dc in offs] + [(r, c)] * (8 - len(offs)), float)
        inside = [0 <= a < SHAPE.shape[0] and 0 <= b < SHAPE.shape[1] and SHAPE[int(a), int(b)] for a, b in nb]
        sizes = np.array([16.0] * len(offs) + [0.0] * (8 - len(offs)))
        cols = [theme.E2 if v else theme.OUTLIER for v in inside]
        done = np.where((np.arange(len(rr)) <= i) & is_b, 22.0, 0.0)
        n_out = sum(1 for v in inside[:len(offs)] if not v)
        verdict = (f"<b style='color:{theme.HIGHLIGHT}'>boundary</b>: {n_out} neighbour(s) in the background"
                   if is_b[i] else f"<b style='color:{theme.E2}'>interior</b>: every neighbour is inside Ω")
        cap = f"pixel ({c}, {r}), {i + 1} of {len(rr)}<br>{verdict}"
        return [cui.markers(np.c_[rr, cc], theme.HIGHLIGHT, size=done),
                cui.markers(nb, cols, size=sizes, symbol="circle"),
                cui.markers([(r, c)], theme.E1, size=26, symbol="square-open")], cap

    return anim.scrubber(base, range(len(rr)), make_state, active=0, label="pixel", fmt="{:.0f}", frame_ms=90,
                         bounce=False, wrap_at=48)


@live.section
def boundary_section():
    st.subheader("Boundary pixels: object pixels that touch the background", divider="gray")
    st.latex(r"\Omega = \{(x, y) : B(x, y) = 1\}, \qquad \partial\Omega = \{p \in \Omega : N(p) \cap \Omega^c \neq "
             r"\emptyset\}")
    c_ctrl, c_a, c_b = st.columns([0.7, 1.3, 1.3], gap="large")
    with c_ctrl:
        conn = 8 if st.segmented_control("Neighbourhood N", ["N₄", "N₈"], default="N₄", key="px_bconn") == "N₈" else 4
        b4, b8 = K.boundary(SHAPE, 4).sum(), K.boundary(SHAPE, 8).sum()
        st.metric("boundary pixels with N₄", int(b4), border=True)
        st.metric("boundary pixels with N₈", int(b8), border=True)
        st.caption("Surprise: testing MORE neighbours (N₈) finds MORE boundary pixels - the corner pixels that only "
                   "touch the background diagonally.")
    with c_a:
        ui.anim_intro("Step by step: boundary = object minus interior")
        st.plotly_chart(erosion_figure(conn), key="px_erode")
    with c_b:
        ui.anim_intro("Sweep: the test for every pixel",
                      "▶ Play or drag: the orange square is the pixel under test; green / red dots are its neighbours "
                      "inside / outside the object.")
        st.plotly_chart(sweep_figure(conn), key="px_sweep")

    st.markdown("**The same on a real image** (zoomed in on a diagonal part of the longest boundary):")
    c_ctrl, c1, c2, c3 = st.columns([0.7, 1, 1, 1], gap="medium")
    with c_ctrl:
        B, img, name, t = cui.pick_scene("px", t_default=0.45, invert=False)
    cs, _ = K.find_contours(B, min_points=8)
    if cs:
        top = max(cs, key=len)
        x0, y0 = (int(v) for v in top[np.argmax(top[:, 0] + top[:, 1])])      # a diagonal stretch of the boundary
    else:
        x0, y0 = B.shape[1] // 2, B.shape[0] // 2
    h, w = 36, 48
    y0 = int(np.clip(y0 - h // 2, 0, B.shape[0] - h))
    x0 = int(np.clip(x0 - w // 2, 0, B.shape[1] - w))
    crop = B[y0:y0 + h, x0:x0 + w]
    for col, conn_, txt in [(c1, None, "object pixels Ω"), (c2, 4, "∂Ω with N₄"), (c3, 8, "∂Ω with N₈")]:
        vis = K.binary_rgb(crop, fg=(80, 80, 78))
        if conn_ is None:
            vis = K.binary_rgb(crop)
            cnt = int(B.sum())
        else:
            bd = K.boundary(crop, conn_)
            vis[bd] = K.AMBER
            cnt = int(K.boundary(B, conn_).sum())
        col.image(ui.datauri(K.draw_contours(vis, [], scale=7)), caption=f"{txt}: {cnt:,} pixels in the whole image",
                  width="stretch")


boundary_section()

with st.expander("Why does N₄ give a thin boundary and N₈ a thick one?", icon=":material/psychology_alt:"):
    st.markdown("Testing with **N₄** marks a pixel only if it touches the background through a *side*. The result is "
                "a chain whose pixels connect diagonally - an **8-connected**, 1-pixel-thin boundary. Testing with **N₈** "
                "also marks pixels that touch the background only at a *corner*; those fill in the diagonal steps, so the "
                "boundary becomes **4-connected** and slightly thicker. Rule of thumb: *test with N₄ → trace with N₈* "
                "(and vice versa).")

# ------------------------------------------------------------------ the paradox
RING = K.parse_grid([
    ".........",
    "....#....",
    "...#.#...",
    "..#...#..",
    ".#.....#.",
    "..#...#..",
    "...#.#...",
    "....#....",
    ".........",
])


@live.section
def paradox_section():
    st.subheader("The connectivity paradox: does a ring have an inside?", divider="gray")
    c_ctrl, c_fig, c_txt = st.columns([0.8, 1.3, 1.3], gap="large")
    with c_ctrl:
        fg = 4 if st.segmented_control("object (foreground) uses", ["N₄", "N₈"], default="N₈", key="px_fg") == "N₄" else 8
        bg = 8 if st.segmented_control("background uses", ["N₄", "N₈"], default="N₄", key="px_bg") == "N₈" else 4
    n_fg, lab_fg = K.label(RING, fg)
    n_bg, lab_bg = K.label(~RING, bg)
    closed_curve = n_fg == 1
    separates = n_bg >= 2
    ok = closed_curve == separates
    with c_fig:
        fig = go.Figure(data=[cui.grid_heatmap(np.zeros(RING.shape))], layout=cui.grid_layout(RING.shape, height=360))
        rr, cc = np.nonzero(~RING)
        bg_cols = [theme.rgba(_hex(K.PALETTE[(lab_bg[r, c] + 3) % 8]), 0.55) for r, c in zip(rr, cc)]
        fig.add_trace(cui.markers(np.c_[rr, cc], bg_cols, size=24, line_color="rgba(0,0,0,0)"))
        rr, cc = np.nonzero(RING)
        fg_cols = [_hex(K.PALETTE[(lab_fg[r, c] - 1) % 8]) for r, c in zip(rr, cc)]
        fig.add_trace(cui.markers(np.c_[rr, cc], fg_cols, size=26, line_color="white"))
        st.plotly_chart(fig, key="px_ring")
    with c_txt:
        st.metric("ring (object) pieces", n_fg, border=True)
        st.metric("background regions", n_bg, border=True)
        if ok:
            st.success(f"**Consistent.** {'One closed ring' if closed_curve else 'Broken pieces'} and "
                       f"{'an inside separated from the outside' if separates else 'a single background'}: what you "
                       "see matches what you count.", icon=":material/check_circle:")
        else:
            st.error("**Paradox.** " + ("The ring is one closed curve, yet inside and outside are the SAME region - "
                                        "the background leaks through the diagonal gaps." if closed_curve else
                                        "The ring falls apart into separate pixels, yet it still cuts the background "
                                        "in two - a broken fence that keeps the sheep in."), icon=":material/cancel:")
    st.markdown("Each colour is one connected region. Only the **mixed** choices (N₈ for objects with N₄ for the "
                "background, or the reverse) behave like real curves: a closed curve separates inside from outside "
                "(the digital Jordan curve theorem). That is why tracing and hole-finding (cv2.findContours) use "
                "8-connected objects and 4-connected holes.")


paradox_section()

ui.callout("intuition", "N₄ is a rook on a chess board, N₈ is a king. The king can slip through a diagonal gap; the "
                        "rook cannot. Use the king for the objects and the rook for the background, and every closed "
                        "fence keeps its inside in.")
ui.callout("pitfall", "Counting objects with N₄ on a thin diagonal line gives one object per pixel. If your object count "
                      "is suspiciously high, check the connectivity.")
ui.callout("takeaway", "Connectivity decides what is 'one object' and what is 'the boundary'. Trace object boundaries "
                       "with N₈; pair it with N₄ for the background so that holes stay holes.")

ui.whys([
    ("Why is 8-connectivity preferred for boundary tracing?",
     "A boundary drawn on a pixel grid steps diagonally whenever it is not horizontal or vertical. Under N₄ those "
     "diagonal steps are not connections, so a perfectly fine diagonal boundary falls apart into single pixels. N₈ "
     "keeps it one chain - which is also why Canny's hysteresis uses N₈."),
    ("Why can a diagonal line 'leak'?",
     "Two background pixels that touch only at a corner are neighbours under N₈. A diagonal fence of object pixels "
     "has exactly such corner gaps, so an N₈ background flows straight through it. With an N₄ background it cannot."),
    ("Why not just use N₈ for everything?",
     "Because of the paradox above: with N₈ for both the object AND the background, a closed diagonal ring would not "
     "separate its inside from its outside. One of the two must be N₄."),
    ("Does the image border count as background?",
     "In this course and in OpenCV, yes: pixels outside the image are treated as 0. So an object touching the border "
     "gets boundary pixels along the border."),
    ("Is there a grid without this problem?",
     "Yes - a hexagonal grid. Every hexagon shares a full side with each of its 6 neighbours, so there are no "
     "'corner-only' neighbours and no paradox. Cameras use square pixels, so we live with N₄/N₈."),
    ("Isn't the boundary just the edge map from Canny?",
     "No. ∂Ω is defined on a binary *region*: it is exactly the region's outer layer, always closed, never broken. "
     "Canny works on intensities and can leave gaps or double lines. Contour tracing therefore starts from regions."),
])

st.divider()
ui.mcq("px1", "Two object pixels touch only at a corner. How many objects are they?",
       ["Always one", "One with N₈, two with N₄", "Two with N₈, one with N₄", "Always two"], 1,
       "Diagonal neighbours belong to N₈ but not to N₄.")
ui.mcq("px2", "Which pixels form ∂Ω?",
       ["All pixels of the image whose value changes", "Object pixels with at least one background neighbour",
        "Background pixels next to the object", "Pixels with a large gradient"], 1,
       "∂Ω = {p ∈ Ω : N(p) ∩ Ωᶜ ≠ ∅} - foreground pixels that touch the background.")

ui.lesson_nav(__file__)
