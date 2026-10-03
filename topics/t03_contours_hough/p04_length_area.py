import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import anim, live, plotting, theme, ui
from core import contour_ui as cui
from core import contours as K

ui.lesson_header(
    __file__,
    "Once the boundary is an ordered list, two numbers come almost for free: the <b>perimeter</b> (add up the steps) "
    "and the <b>area</b> (the shoelace formula). You will see <i>why</i> the shoelace formula works - a fan of signed "
    "triangles whose overhangs cancel - and why pixel shapes always look a bit longer than they really are.",
)

POLYS = {
    "Arrow (concave)": [(1, 2.5), (4, 2.5), (4, 1), (7, 3.5), (4, 6), (4, 4.5), (1, 4.5)],
    "House": [(1, 1), (5, 1), (5, 4), (3, 6), (1, 4)],
    "L-shape": [(1, 1), (6, 1), (6, 3), (3, 3), (3, 6), (1, 6)],
    "Star": [(4 + (2.8 if k % 2 == 0 else 1.2) * np.cos(np.pi / 2 + k * np.pi / 5),
              3.6 + (2.8 if k % 2 == 0 else 1.2) * np.sin(np.pi / 2 + k * np.pi / 5)) for k in range(10)],
}
EXT = (-1.5, 8.5, -1.5, 7.5)


def fan_traces(P, O, f=None):
    """One filled triangle O p_i p_{i+1} per edge; blue if its signed area is positive, red if negative.
    f[i] in [0, 1] grows triangle i out of O (for the animation)."""
    P = np.asarray(P, float)
    O = np.asarray(O, float)
    n = len(P)
    out = []
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        s = (a[0] - O[0]) * (b[1] - O[1]) - (b[0] - O[0]) * (a[1] - O[1])
        g = 1.0 if f is None else f[i]
        tri = np.array([O, O + g * (a - O), O + g * (b - O), O])
        col = theme.TRANSFORMED if s >= 0 else theme.OUTLIER
        out.append(cui.scatter(tri[:, 0], tri[:, 1], col, mode="lines", width=1.2, opacity=0.9)
                   .update(fill="toself", fillcolor=theme.rgba(col, 0.22)))
    return out


def poly_traces(P, labels=True):
    P = np.asarray(P, float)
    Q = np.vstack([P, P[:1]])
    out = [cui.scatter(Q[:, 0], Q[:, 1], "#7d7b76", mode="lines", width=2.6)]
    if labels:
        out.append(dict(type="scatter", x=P[:, 0], y=P[:, 1], mode="markers+text",
                        text=[f"p{i}" for i in range(len(P))], textposition="top center",
                        marker=dict(size=8, color="#7d7b76"), textfont=dict(size=12), hoverinfo="skip",
                        showlegend=False))
    return out


@live.section
def explore_section():
    st.subheader("The shoelace formula, live", divider="gray")
    st.latex(r"A = \frac{1}{2}\left|\sum_{i=0}^{N-1} \left(x_i y_{i+1} - x_{i+1} y_i\right)\right|, \qquad "
             r"x_N = x_0,\; y_N = y_0")
    c_ctrl, c_fig, c_tab = st.columns([0.8, 1.5, 1.05], gap="large")
    with c_ctrl:
        name = st.selectbox("Polygon", list(POLYS), key="pa_poly")
        ox = live.slider("reference point O: x", -1.0, 8.0, 0.0, 0.1, key="pa_ox")
        oy = live.slider("reference point O: y", -1.0, 7.0, 0.0, 0.1, key="pa_oy")
        rev = st.toggle("walk the other way round", False, key="pa_rev")
        st.caption("The formula uses the origin as O. Moving O = shifting all coordinates: the triangles change, "
                   "the total does not.")
    P = np.array(POLYS[name], float)
    if rev:
        P = P[::-1]
    O = np.array([ox, oy])
    Q = P - O
    terms = K.shoelace_terms(Q)
    sa = 0.5 * terms.sum()
    with c_fig:
        fig = plotting.base_figure(EXT, height=470, legend=False)
        fig.add_traces(fan_traces(P, O) + poly_traces(P) +
                       [cui.scatter([ox], [oy], theme.HIGHLIGHT, size=13, mode="markers+text").update(
                           text=["O"], textposition="bottom left", textfont=dict(size=14))])
        st.plotly_chart(fig, key="pa_fig")
    with c_tab:
        rows = [dict(i=i, term=f"x{i}·y{(i + 1) % len(P)} − x{(i + 1) % len(P)}·y{i}", value=round(float(v), 2),
                     sign="+" if v >= 0 else "−") for i, v in enumerate(terms)]
        st.dataframe(pd.DataFrame(rows), hide_index=True, height=300, width="stretch")
        st.metric("½ Σ (signed area)", f"{sa:.2f}", border=True)
        st.metric("area A = |½ Σ|", f"{abs(sa):.2f}", border=True)
        st.caption(f"Positive = counter-clockwise (maths axes, y up). Now: **{'counter-clockwise' if sa > 0 else 'clockwise'}**. "
                   "Blue triangles add, red ones subtract.")


explore_section()


# ------------------------------------------------------------------ derivation 1: one term = a cross product
@st.cache_resource(show_spinner=False)
@live.full_quality()
def cross_figure():
    a, b = np.array([4.5, 1.2]), np.array([1.6, 3.4])
    O = np.zeros(2)
    ah = a / np.hypot(*a)
    b_perp = b - (b @ ah) * ah                          # shear b along a until it is perpendicular to a
    det = a[0] * b[1] - b[0] * a[1]

    def par(bb, show=1.0):
        return np.array([O, show * a, show * (a + bb), show * bb, O])

    def state(para, tri_f, tri_color, b_vec=b, a_first=True):
        t = np.array([O, O + tri_f * a, O + tri_f * b_vec, O])
        col = tri_color
        return [cui.scatter(para[:, 0], para[:, 1], theme.ORIGINAL, mode="lines", width=1.5)
                .update(fill="toself", fillcolor=theme.rgba(theme.E2, 0.18)),
                cui.scatter(t[:, 0], t[:, 1], col, mode="lines", width=2)
                .update(fill="toself", fillcolor=theme.rgba(col, 0.3)),
                plotting.arrow_trace(O, a, theme.E1, "a = p_i", showlegend=False),
                plotting.arrow_trace(O, b_vec, theme.E2, "b = p_{i+1}", showlegend=False)]

    zero = np.array([O] * 5)
    steps = [
        anim.Step("Two consecutive contour points, seen from the origin O",
                  "a = p<sub>i</sub> = (x<sub>i</sub>, y<sub>i</sub>),   b = p<sub>i+1</sub> = (x<sub>i+1</sub>, y<sub>i+1</sub>)",
                  "Each term of the shoelace sum only involves two neighbouring points.",
                  state(zero, 0, theme.TRANSFORMED)),
        anim.Step("They span a parallelogram", "corners O, a, a + b, b",
                  "Its area is what the 2×2 determinant measures.", state(par(b), 0, theme.TRANSFORMED)),
        anim.Step("Shear it: slide b along a - the area does not change",
                  f"area = base × height = |a| · h = {anim.hl(f'{abs(det):.2f}')}",
                  "Sliding the top edge parallel to the base keeps base and height, so the area stays the same.",
                  state(par(b_perp), 0, theme.TRANSFORMED, b_vec=b_perp)),
        anim.Step("… and that number is the determinant",
                  f"det [a b] = x<sub>i</sub>y<sub>i+1</sub> − x<sub>i+1</sub>y<sub>i</sub> = {a[0]}·{b[1]} − {b[0]}·{a[1]} = {anim.hl(f'{det:.2f}')}",
                  "The 2D cross product: the signed area of the parallelogram.", state(par(b), 0, theme.TRANSFORMED)),
        anim.Step("The triangle O p<sub>i</sub> p<sub>i+1</sub> is half of it",
                  f"area = {anim.hl('½')}(x<sub>i</sub>y<sub>i+1</sub> − x<sub>i+1</sub>y<sub>i</sub>) = {det / 2:.2f}",
                  "A diagonal cuts every parallelogram into two equal triangles.", state(par(b), 1, theme.TRANSFORMED)),
        anim.Step("The sign remembers the direction",
                  f"swap the order: x<sub>i+1</sub>y<sub>i</sub> − x<sub>i</sub>y<sub>i+1</sub> = {anim.hl(f'{-det:.2f}', theme.OUTLIER)}",
                  "Turning anti-clockwise from a to b gives +, clockwise gives −. That sign is what makes the "
                  "shoelace sum work for ANY polygon.", state(par(b), 1, theme.OUTLIER)),
    ]
    base = plotting.base_figure((-1, 6.5, -0.8, 4.6), height=430, legend=False)
    return anim.step_player(base, steps, n_sub=14, hold=16, wrap_at=58)


# ------------------------------------------------------------------ derivation 2: the fan of triangles
@st.cache_resource(show_spinner=False)
@live.full_quality()
def fan_figure():
    P = np.array(POLYS["Arrow (concave)"], float)
    O = np.array([0.0, 0.0])
    n = len(P)
    terms = K.shoelace_terms(P)
    base = plotting.base_figure(EXT, height=470, legend=False)
    for tr in poly_traces(P):
        base.add_trace(tr)
    base.add_trace(cui.scatter([0], [0], theme.HIGHLIGHT, size=12, mode="markers+text").update(
        text=["O"], textposition="bottom left", textfont=dict(size=14)))
    steps = [anim.Step("A closed contour and the origin O", f"N = {n} points, p_N = p₀",
                       "O is outside the shape on purpose: the trick still works.", fan_traces(P, O, np.zeros(n)))]
    run = 0.0
    for i in range(n):
        run += terms[i]
        f = np.array([1.0 if k <= i else 0.0 for k in range(n)])
        sign = "+" if terms[i] >= 0 else "−"
        steps.append(anim.Step(f"Triangle O p<sub>{i}</sub> p<sub>{(i + 1) % n}</sub>",
                               f"½(x<sub>{i}</sub>y<sub>{(i + 1) % n}</sub> − x<sub>{(i + 1) % n}</sub>y<sub>{i}</sub>) = "
                               f"{anim.hl(f'{sign}{abs(terms[i]) / 2:.2f}', theme.TRANSFORMED if terms[i] >= 0 else theme.OUTLIER)}"
                               f"     running sum = {run / 2:.2f}",
                               ("Counter-clockwise seen from O: it ADDS area." if terms[i] >= 0 else
                                "Clockwise seen from O: it SUBTRACTS - this triangle covers ground outside the shape."),
                               fan_traces(P, O, f)))
    steps.append(anim.Step("Everything outside cancels", f"A = ½ |Σ| = {anim.hl(f'{abs(terms.sum()) / 2:.2f}')}",
                           "Every point outside the polygon is covered by as many red as blue triangles; every point "
                           "inside by one more blue. What is left is exactly the area.", fan_traces(P, O)))
    return anim.step_player(base, steps, n_sub=12, hold=14, wrap_at=58)


st.subheader("Step by step: why the shoelace formula works", divider="gray")
d1, d2 = st.columns(2, gap="large")
with d1:
    ui.anim_intro("Step 1: one term = a signed triangle")
    st.plotly_chart(cross_figure(), key="pa_cross")
with d2:
    ui.anim_intro("Step 2: add up the triangles of all edges")
    st.plotly_chart(fan_figure(), key="pa_fan")

# ------------------------------------------------------------------ perimeter on pixels
st.subheader("Contour length - and the staircase problem", divider="gray")
st.latex(r"\lVert p_{i+1} - p_i \rVert = \sqrt{(x_{i+1}-x_i)^2 + (y_{i+1}-y_i)^2}, \qquad "
         r"P = \sum_{i=0}^{N-2}\lVert p_{i+1}-p_i\rVert + \lVert p_0 - p_{N-1}\rVert")


@st.cache_resource(show_spinner=False)
def ratio_curve():
    rs = np.arange(2, 81)
    rows = [K.staircase(K.disk(r)) for r in rs]
    return rs, np.array([s["crack"] for s in rows]) / (2 * np.pi * rs), np.array([s["chain8"] for s in rows]) / (2 * np.pi * rs)


@st.cache_resource(show_spinner=False)
@live.full_quality()
def perimeter_sweep():
    D = K.disk(5)
    s = K.staircase(D)
    c = s["contour"]
    rc = np.c_[c[:, 1], c[:, 0]]
    loop = np.vstack([rc, rc[:1]])
    seg = np.hypot(*np.diff(loop, axis=0).T)

    def make_state(i):
        i = int(i)
        L = seg[:i].sum()
        upto = np.vstack([loop[:i + 1], np.full((len(loop) - i - 1, 2), np.nan)])
        n_d = int((np.abs(seg[:i] - np.sqrt(2)) < 1e-6).sum())
        cap = (f"after {i} of {len(seg)} steps:  P = {i - n_d} × 1 + {n_d} × √2 = <b>{L:.2f}</b>"
               + (f"   ·   true circle 2πr = {2 * np.pi * 5:.2f}" if i == len(seg) else ""))
        return [cui.path(upto, theme.TRANSFORMED, width=4), cui.markers(loop[i:i + 1], theme.E1, size=16,
                                                                         symbol="circle")], cap

    return anim.scrubber(cui.grid_figure(D, height=420), range(len(seg) + 1), make_state, active=0, label="step",
                         fmt="{:.0f}", frame_ms=110, bounce=False)


def crack_xy(B):
    """The pixel SIDES between object and background, as one NaN-separated polyline (x = col, y = row)."""
    xs, ys = [], []
    P = np.pad(B, 1)
    for r, c in zip(*np.nonzero(B)):
        R, C = r + 1, c + 1
        for dr, dc, (x0, y0, x1, y1) in [(-1, 0, (-.5, -.5, .5, -.5)), (1, 0, (-.5, .5, .5, .5)),
                                         (0, -1, (-.5, -.5, -.5, .5)), (0, 1, (.5, -.5, .5, .5))]:
            if not P[R + dr, C + dc]:
                xs += [c + x0, c + x1, np.nan]
                ys += [r + y0, r + y1, np.nan]
    return xs, ys


@live.section
def staircase_section():
    c_ctrl, c_fig, c_plot = st.columns([0.8, 1.3, 1.3], gap="large")
    with c_ctrl:
        r = live.slider("disc radius r (pixels)", 2, 15, 6, 1, key="pa_r")
        D = K.disk(r)
        s = K.staircase(D)
        true = 2 * np.pi * r
        st.metric("true circumference 2πr", f"{true:.1f}", border=True)
        st.metric("8-connected chain (cv2.arcLength)", f"{s['chain8']:.1f}", f"{100 * (s['chain8'] / true - 1):+.1f} %",
                  delta_color="off", border=True)
        st.metric("crack length (pixel sides)", f"{s['crack']}", f"{100 * (s['crack'] / true - 1):+.1f} %",
                  delta_color="off", border=True)
    with c_fig:
        c = s["contour"]
        loop = np.vstack([c, c[:1]])
        xs, ys = crack_xy(D)
        fig = cui.grid_figure(D, height=430, legend=True)
        fig.add_traces([cui.scatter(xs, ys, theme.E1, mode="lines", width=2.5, name="crack (pixel sides)",
                                    showlegend=True),
                        cui.scatter(loop[:, 0], loop[:, 1], theme.TRANSFORMED, mode="lines+markers", size=5, width=2.5,
                                    name="8-connected chain (pixel centres)", showlegend=True)])
        rr = np.linspace(0, 2 * np.pi, 120)
        n0 = (D.shape[0] - 1) / 2
        fig.add_trace(cui.scatter(n0 + r * np.cos(rr), n0 + r * np.sin(rr), theme.HIGHLIGHT, mode="lines", width=2,
                                  dash="dot", name="true circle", showlegend=True))
        st.plotly_chart(fig, key="pa_disc")
    with c_plot:
        rs, crack, chain = ratio_curve()
        f2 = go.Figure(data=[cui.scatter(rs, crack, theme.E1, mode="lines", width=2.5, name="crack / 2πr",
                                         showlegend=True, hover=True),
                             cui.scatter(rs, chain, theme.TRANSFORMED, mode="lines", width=2.5, name="chain / 2πr",
                                         showlegend=True, hover=True),
                             cui.scatter([r, r], [crack[r - 2], chain[r - 2]], theme.HIGHLIGHT, size=11,
                                         name="your r")],
                       layout=dict(height=430, margin=dict(l=8, r=8, t=30, b=8), hovermode="x unified",
                                   legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom"),
                                   xaxis=dict(title="radius r (pixels)", gridcolor=theme.GRID),
                                   yaxis=dict(title="measured length / true 2πr", gridcolor=theme.GRID,
                                              range=[0.95, 1.55])))
        f2.add_hline(y=4 / np.pi, line=dict(color=theme.E1, dash="dot", width=1), annotation_text="4/π ≈ 1.27",
                     annotation_position="top right")
        f2.add_hline(y=1.0, line=dict(color=theme.ORIGINAL, dash="dash", width=1))
        st.plotly_chart(f2, key="pa_ratio")
    st.caption("Measured on digital discs up to r = 80: the crack length heads for **4/π ≈ 1.27 × 2πr** (a staircase "
               "is as long as the bounding box it climbs, whatever the step size) and the 8-connected chain stays about "
               "**5 % too long**. Making the pixels smaller does NOT fix this.")


st.markdown("A pixel boundary is a staircase. How long is it? Drag the radius:")
staircase_section()

p1, p2 = st.columns([1.2, 1], gap="large")
with p1:
    ui.anim_intro("Sweep: adding up the perimeter step by step",
                  "Every move is 1 (horizontal / vertical) or √2 (diagonal).")
    st.plotly_chart(perimeter_sweep(), key="pa_sweep")
with p2:
    st.markdown("**Area on pixels: three answers, one theorem**")
    rows = []
    for r in (3, 6, 10, 25):
        s = K.staircase(K.disk(r))
        Bn = len(s["contour"])
        rows.append(dict(r=r, **{"pixels": s["pixels"], "shoelace (cv2.contourArea)": s["poly_area"],
                                 "pixels − B/2 − 1": s["pixels"] - Bn / 2 - 1, "πr²": round(np.pi * r * r, 1)}))
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown("The shoelace area runs through the pixel **centres**, so it misses half of every boundary pixel. "
                "**Pick's theorem** for polygons with corners on the pixel grid says exactly how much:")
    st.latex(r"A_{\text{shoelace}} = I + \frac{B}{2} - 1 = \text{pixels} - \frac{B}{2} - 1")
    st.caption("I = pixels strictly inside, B = boundary pixels on the contour. Measured above: the two columns agree "
               "exactly.")

ui.callout("intuition", "Shoelace = stand at O and sweep a torch along the boundary. Every bit of floor the beam crosses "
                        "while turning one way is added, the other way subtracted. Floor outside the shape is crossed "
                        "both ways and cancels; floor inside is crossed once more one way. What's left is the area.")
ui.callout("pitfall", "cv2.contourArea and the number of object pixels are NOT the same thing - they differ by about "
                      "half the perimeter (Pick's theorem). For small objects that is a big difference.")
ui.callout("takeaway", "Perimeter = sum of the step lengths. Area = ½ |Σ (x_i y_{i+1} − x_{i+1} y_i)|, a sum of "
                       "signed triangles. Both need the ORDERED contour.")

ui.whys([
    ("Why the absolute value?",
     "The signed sum is positive if the contour runs counter-clockwise and negative if it runs clockwise (try the "
     "toggle). Area should not depend on the walking direction, so we take |·|. Keep the sign when you need the "
     "direction - e.g. to tell holes from objects."),
    ("Why can a term be negative?",
     "Each term is twice the signed area of the triangle O p_i p_{i+1}. If, seen from O, the edge runs clockwise, "
     "the term is negative. Those triangles cover ground outside the shape and are cancelled by positive ones."),
    ("Does the starting point or the position of O matter?",
     "No. The sum is cyclic, so any p_i can be p₀. Moving O shifts every coordinate; the extra terms telescope away "
     "(drag O above: the triangles change, the total does not)."),
    ("Why doesn't a pixel circle's perimeter approach 2πr?",
     "Because a staircase never becomes a straight line, however small the steps. Measured above: walking along the "
     "pixel sides gives about 1.27 × 2πr (4/π), walking through pixel centres with 1 and √2 steps about 1.05 × 2πr. "
     "Better estimators fit a smooth curve first or use corrected step weights."),
    ("In image coordinates (y down) is the sign flipped?",
     "Yes. Mirroring y swaps clockwise and counter-clockwise. The same formula on OpenCV's (x, y) points gives a "
     "negative value for object contours and a positive one for holes (lesson 3's table). cv2.contourArea(c, "
     "oriented=True) returns exactly this signed value."),
    ("Does the shoelace formula work for concave shapes and holes?",
     "For any simple polygon (no self-crossings), concave or not - the cancelling triangles take care of it. "
     "For a region with holes, subtract the holes' areas (or add their signed areas, which have the opposite sign)."),
])

st.divider()
ui.mcq("pa1", "The contour (0,0) → (4,0) → (4,3) → (0,3) has shoelace sum Σ = …",
       ["12", "24", "−24", "7"], 1,
       "Terms: 0·0−4·0 = 0, 4·3−4·0 = 12, 4·3−0·3 = 12, 0·0−0·3 = 0 → Σ = 24, area = ½·24 = 12 (counter-clockwise).")
ui.mcq("pa2", "Walking along the pixel sides of a large digital circle, the perimeter comes out about …",
       ["exactly 2πr", "1.05 × 2πr", "1.27 × 2πr", "0.5 × 2πr"], 2,
       "A staircase is as long as the bounding box it climbs: 8r = (4/π)·2πr ≈ 1.27 × 2πr.")

ui.lesson_nav(__file__)
