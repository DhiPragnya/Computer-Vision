import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import anim, live, plotting, theme, ui
from core import contour_ui as cui
from core import contours as K

ui.lesson_header(
    __file__,
    "ε decides how simple the polygon gets: small ε keeps the detail, large ε keeps the gist. Once the shape is "
    "a clean contour, a handful of numbers - area, perimeter, bounding box, centroid, convexity, <b>circularity</b> - "
    "tell a triangle from a coin from a bar. Here you derive circularity and build a tiny shape classifier.",
)


# ------------------------------------------------------------------ objects
def _blob():
    a = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    r = 62 * (1 + 0.22 * np.sin(3 * a) + 0.08 * np.sin(7 * a + 1))
    return np.int32(np.c_[100 + r * np.cos(a), 100 + r * np.sin(a)])


@st.cache_resource(show_spinner=False, max_entries=8)
def obj_contour(name: str) -> np.ndarray:
    img = np.zeros((200, 200))
    if name.startswith("Rectangle"):
        cv2.rectangle(img, (40, 60), (159, 139), 1.0, -1)
        img = cv2.GaussianBlur(img, (0, 0), 1.2) + np.random.default_rng(3).normal(0, 0.1, img.shape)
    elif name == "Disc":
        cv2.circle(img, (100, 100), 70, 1.0, -1)
    elif name == "Star":
        cv2.fillPoly(img, [K._star(100, 104, 82, 36)], 1.0)
    else:
        cv2.fillPoly(img, [_blob()], 1.0)
    B = img > 0.5
    cs, _ = K.find_contours(B, cv2.RETR_EXTERNAL)
    return max(cs, key=len)


OBJECTS = ["Rectangle (≈ 400 contour pixels)", "Disc", "Star", "Blob"]


@st.cache_resource(show_spinner=False, max_entries=8)
@live.full_quality()
def eps_figure(name: str):
    C = obj_contour(name)
    L = K.perimeter(C)
    fracs = np.round(np.geomspace(0.002, 0.15, 36), 4)
    base = go.Figure(layout=dict(height=460, margin=dict(l=8, r=8, t=10, b=8), showlegend=False,
                                 xaxis=dict(range=[0, 200], visible=False, constrain="domain"),
                                 yaxis=dict(range=[200, 0], visible=False, scaleanchor="x", constrain="domain")))
    loop = np.vstack([C, C[:1]])
    base.add_trace(cui.scatter(loop[:, 0], loop[:, 1], theme.ORIGINAL, mode="lines+markers", size=3, width=1))

    def make_state(f):
        idx = K.rdp_closed(C, f * L)
        Q = C[idx]
        Qc = np.vstack([Q, Q[:1]])
        err = K.max_deviation(C, idx, closed=True)
        cap = (f"ε = {f:.3f} × perimeter = <b>{f * L:.1f} px</b>   ·   {len(C)} points → "
               f"<b style='color:{theme.TRANSFORMED}'>{len(idx)} vertices</b>   ·   largest error {err:.1f} px")
        return [cui.scatter(Qc[:, 0], Qc[:, 1], theme.TRANSFORMED, mode="lines+markers", size=9, width=3)
                .update(fill="toself", fillcolor=theme.rgba(theme.TRANSFORMED, 0.1))], cap

    fig = anim.scrubber(base, list(fracs), make_state, active=int(np.argmin(np.abs(fracs - 0.02))), label="ε / P",
                        fmt="{:.3f}", frame_ms=120, wrap_at=52)
    counts = [len(K.rdp_closed(C, f * L)) for f in fracs]
    errs = [K.max_deviation(C, K.rdp_closed(C, f * L), closed=True) for f in fracs]
    return fig, fracs, counts, errs, len(C)


st.subheader("The ε trade-off", divider="gray")
st.latex(r"\epsilon \downarrow \;\Rightarrow\; \text{more vertices, higher fidelity} \qquad\qquad "
         r"\epsilon \uparrow \;\Rightarrow\; \text{fewer vertices, more simplification}")
e_ctrl, e_fig, e_plot = st.columns([0.7, 1.5, 1.2], gap="large")
with e_ctrl:
    obj = st.selectbox("Object", OBJECTS, key="ms_obj")
    st.caption("ε is set as a fraction of the perimeter P, so the same setting works for big and small objects.")
fig_eps, fracs, counts, errs, n_pts = eps_figure(obj)
with e_fig:
    ui.anim_intro("Sweep: drag ε", "▶ Play sweeps ε from tiny to large and back; or drag the slider under the figure.")
    st.plotly_chart(fig_eps, key="ms_eps")
with e_plot:
    f2 = go.Figure(data=[cui.scatter(fracs, counts, theme.TRANSFORMED, mode="lines+markers", size=6, width=2.5,
                                     name="vertices M", showlegend=True, hover=True),
                         cui.scatter(fracs, errs, theme.OUTLIER, mode="lines", width=2, dash="dot",
                                     name="largest error (px)", showlegend=True, hover=True).update(yaxis="y2")],
                   layout=dict(height=420, margin=dict(l=8, r=8, t=30, b=8), hovermode="x unified",
                               legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0),
                               xaxis=dict(type="log", title="ε / perimeter", gridcolor=theme.GRID),
                               yaxis=dict(type="log", title="vertices M", gridcolor=theme.GRID),
                               yaxis2=dict(overlaying="y", side="right", title="largest error (px)", showgrid=False)))
    st.plotly_chart(f2, key="ms_curve")
    if obj.startswith("Rectangle"):
        st.markdown(f"**The notes' example:** {n_pts} contour pixels → **4 vertices** over a wide range of ε - the "
                    "plateau in the blue curve. A plateau means the shape *is* that polygon.")
    else:
        st.markdown("Look for **plateaus** in the blue curve: a range of ε where M does not change means the shape "
                    "really is that polygon. A disc has no plateau - it is no polygon at all.")


# ------------------------------------------------------------------ circularity derivation
def _resample(poly, n=240):
    """Points along a closed polygon at equal arc-length steps, starting at its right-most point."""
    poly = np.asarray(poly, float)
    poly = np.roll(poly, -int(np.argmax(poly[:, 0] - 1e-6 * poly[:, 1])), axis=0)
    loop = np.vstack([poly, poly[:1]])
    seg = np.hypot(*np.diff(loop, axis=0).T)
    s = np.r_[0, np.cumsum(seg)]
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.c_[np.interp(t, s, loop[:, 0]), np.interp(t, s, loop[:, 1])]


@st.cache_resource(show_spinner=False)
@live.full_quality()
def circularity_figure():
    r = 2.0
    P = 2 * np.pi * r
    a = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    circle = np.c_[r * np.cos(a), r * np.sin(a)]

    def rect(w, h):
        return _resample(np.array([[w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2], [-w / 2, -h / 2]]))

    s = P / 4
    a2 = P / 6
    a6 = P / 14
    shapes_ = [("circle", circle), ("square", rect(s, s)), ("rectangle 2 : 1", rect(2 * a2, a2)),
               ("bar 6 : 1", rect(6 * a6, a6))]

    def tr(Q, col):
        Qc = np.vstack([Q, Q[:1]])
        return [cui.scatter(Qc[:, 0], Qc[:, 1], col, mode="lines", width=3).update(
            fill="toself", fillcolor=theme.rgba(col, 0.18))]

    vals = {nm: (K.area(Q), K.perimeter(Q)) for nm, Q in shapes_}
    circ = {nm: 4 * np.pi * A / L ** 2 for nm, (A, L) in vals.items()}
    Ac, Lc = vals["circle"]
    c_circ, c_sq, c_r2, c_bar = (circ[k] for k in ("circle", "square", "rectangle 2 : 1", "bar 6 : 1"))
    steps = [
        anim.Step("A circle of radius r", f"A = πr² = {Ac:.2f},   P = 2πr = {Lc:.2f}",
                  "Area grows with r², perimeter with r: to compare shapes of any size we need a ratio without units.",
                  tr(circle, theme.TRANSFORMED)),
        anim.Step("Square the perimeter", f"P² = 4π²r²   and   4πA = 4π · πr² = {anim.hl('4π²r²')}",
                  "Both are 'length²', so their ratio has no units: it does not change when you scale the shape.",
                  tr(circle, theme.TRANSFORMED)),
        anim.Step("Circularity of the circle", f"C = 4πA / P² = {anim.hl(f'{c_circ:.3f}')}",
                  "The 4π is there to make the circle score exactly 1.", tr(circle, theme.TRANSFORMED)),
        anim.Step("Same perimeter, bent into a square", f"A = (P/4)² = P²/16   ⇒   C = π/4 = "
                  f"{anim.hl(f'{c_sq:.3f}')}",
                  "Same length of fence, less area inside: the corners waste it.", tr(shapes_[1][1], theme.E1)),
        anim.Step("A 2 : 1 rectangle", f"C = 8π/36 = {anim.hl(f'{c_r2:.3f}')}",
                  "The longer and thinner, the less area per unit of perimeter.", tr(shapes_[2][1], theme.E1)),
        anim.Step("A 6 : 1 bar", f"C = 24π/196 = {anim.hl(f'{c_bar:.3f}', theme.OUTLIER)}",
                  "Elongated shapes score low.", tr(shapes_[3][1], theme.OUTLIER)),
        anim.Step("The isoperimetric inequality", f"{anim.hl('4πA ≤ P²')}   ⇒   0 < C ≤ 1, and C = 1 only for a circle",
                  "Of all closed curves with the same length, the circle encloses the most area (known since the "
                  "ancient Greeks, proved in the 19th century).", tr(circle, theme.TRANSFORMED)),
    ]
    base = plotting.base_figure((-4.6, 4.6, -3.0, 3.0), height=420, legend=False)
    return anim.step_player(base, steps, n_sub=14, hold=18, wrap_at=62)


st.subheader("Step by step: where circularity 4πA / P² comes from", divider="gray")
c1, c2 = st.columns([1.2, 1], gap="large")
with c1:
    ui.anim_intro("Step by step: same perimeter, different shapes")
    st.plotly_chart(circularity_figure(), key="ms_circ")
with c2:
    st.latex(r"C = \frac{4\pi A}{P^2}, \qquad C = 1 \text{ (circle)}, \qquad C < 1 \text{ (anything else)}")
    rows = []
    for r in (3, 5, 10, 20, 40, 80):
        s = K.staircase(K.disk(r))
        rows.append(dict(r=r, circularity=round(4 * np.pi * s["poly_area"] / s["chain8"] ** 2, 3)))
    st.markdown("**But on pixels** (digital discs, contour through pixel centres, measured):")
    st.dataframe(pd.DataFrame(rows).set_index("r").T, width="stretch")
    st.caption("A pixel circle never reaches 1: its perimeter is ≈ 5 % too long (the staircase of lesson 4) and C uses "
               "P², so C ≈ 1/1.05² ≈ 0.9 for large discs and much less for tiny ones. Use thresholds like C > 0.8 "
               "for 'round', not C ≈ 1.")


# ------------------------------------------------------------------ measurement dashboard
LABEL_COL = {"triangle": K.ORANGE, "square": K.GREEN, "rectangle": K.GREEN, "bar": K.MAGENTA, "circle": K.BLUE,
             "pentagon": K.AMBER, "concave (star-like)": K.RED, "ellipse / blob": (0, 160, 190)}


@live.section
def dashboard_section():
    st.subheader("Contour-based shape measurements", divider="gray")
    c_ctrl, c_img, c_info = st.columns([0.8, 1.5, 1.1], gap="large")
    with c_ctrl:
        B, img, name, t = cui.pick_scene("ms", t_default=0.45)
        frac = live.slider("ε / perimeter (for the vertex count)", 0.005, 0.08, 0.02, 0.005, key="ms_frac",
                           format="%.3f")
    cs, _ = K.find_contours(B, cv2.RETR_EXTERNAL, min_points=12)
    cs = [c for c in cs if K.area(c) >= 40]
    cs = sorted(cs, key=lambda c: (c[:, 1].min() // 60, c[:, 0].min()))
    ms = [K.measure(c, frac) for c in cs]
    labels = [K.classify(m) for m in ms]
    with c_ctrl:
        # always the same widget (disabled when there is nothing to choose), so the page height never jumps
        sel = live.slider("highlight object #", 0, max(len(cs) - 1, 1), 0, 1, key="ms_sel", disabled=len(cs) < 2)
    sel = max(0, min(sel, len(cs) - 1))
    scale = 2
    vis = K.gray_rgb(img, 0.45)
    vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    for i, (c, m, lab) in enumerate(zip(cs, ms, labels)):
        col = LABEL_COL.get(lab, K.GREY)
        poly = np.int32((c[m["approx"]] + 0.5) * scale).reshape(-1, 1, 2)
        cv2.polylines(vis, [poly], True, col, 3 if i == sel else 2, cv2.LINE_AA)
        x, y, w, h = m["bbox"]
        cv2.putText(vis, f"{i}: {lab}", (int(x * scale), max(14, int(y * scale) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    col, 1, cv2.LINE_AA)
    if cs:
        c, m = cs[sel], ms[sel]
        x, y, w, h = m["bbox"]
        cv2.rectangle(vis, (int(x * scale), int(y * scale)), (int((x + w + 1) * scale), int((y + h + 1) * scale)),
                      (255, 255, 255), 1, cv2.LINE_AA)
        box = cv2.boxPoints(cv2.minAreaRect(c.astype(np.float32)))
        cv2.polylines(vis, [np.int32((box + 0.5) * scale).reshape(-1, 1, 2)], True, K.MAGENTA, 1, cv2.LINE_AA)
        hull = np.int32((m["hull"] + 0.5) * scale).reshape(-1, 1, 2)
        cv2.polylines(vis, [hull], True, K.AMBER, 1, cv2.LINE_AA)
        cx, cy = m["centroid"]
        cv2.drawMarker(vis, (int((cx + 0.5) * scale), int((cy + 0.5) * scale)), K.AMBER, cv2.MARKER_CROSS, 16, 2)
    with c_img:
        st.image(ui.datauri(vis), caption="coloured polygon = RDP approximation, label = rule-based class. Highlighted "
                 "object: white = bounding box, magenta = rotated (minimum-area) box, amber = convex hull, + = centroid",
                 width="stretch")
    with c_info:
        if cs:
            m = ms[sel]
            a1, a2 = st.columns(2)
            a1.metric("area A", f"{m['area']:.0f}", border=True)
            a2.metric("perimeter P", f"{m['perimeter']:.0f}", border=True)
            a1.metric("circularity 4πA/P²", f"{m['circularity']:.2f}", border=True)
            a2.metric("solidity A/A_hull", f"{m['solidity']:.2f}", border=True)
            a1.metric("aspect w/h (box)", f"{m['aspect']:.2f}", border=True)
            a2.metric("vertices (RDP)", f"{m['vertices']}", border=True)
            st.markdown(f"**#{sel} → {labels[sel]}** · centroid ({m['centroid'][0]:.1f}, {m['centroid'][1]:.1f}) · "
                        f"rotated aspect {m['rot_aspect']:.2f} · extent {m['extent']:.2f}")
        else:
            st.info("No objects at this threshold - move t.", icon=":material/info:")
    df = pd.DataFrame([dict(object=i, label=lab, area=round(m["area"]), perimeter=round(m["perimeter"], 1),
                            circularity=round(m["circularity"], 3), solidity=round(m["solidity"], 3),
                            aspect=round(m["aspect"], 2), rotated_aspect=round(m["rot_aspect"], 2),
                            extent=round(m["extent"], 2), vertices=m["vertices"])
                       for i, (m, lab) in enumerate(zip(ms, labels))],
                      columns=["object", "label", "area", "perimeter", "circularity", "solidity", "aspect",
                               "rotated_aspect", "extent", "vertices"])
    st.dataframe(df, hide_index=True, height=250, width="stretch")


dashboard_section()

with st.expander("The classifier used above (core/contours.py · classify)", icon=":material/code:"):
    st.code("""if solidity < 0.8:            label = "concave (star-like)"   # much smaller than its convex hull
elif vertices == 3:            label = "triangle"
elif vertices == 4:            label = "square" if rotated_aspect < 1.15 else ("bar" if rotated_aspect > 3 else "rectangle")
elif circularity > 0.85:       label = "circle"
elif vertices == 5:            label = "pentagon"
else:                          label = "ellipse / blob" if circularity > 0.6 else f"{vertices}-gon\"""", language="python")

m1, m2 = st.columns(2, gap="large")
with m1:
    st.markdown("**Centroid from moments**")
    st.latex(r"m_{pq} = \iint_{\text{shape}} x^p y^q \, dx\, dy, \qquad \bar x = \frac{m_{10}}{m_{00}},\quad "
             r"\bar y = \frac{m_{01}}{m_{00}}")
    st.caption("m₀₀ is the area. cv2.moments computes these integrals straight from the contour polygon (with Green's "
               "theorem - the same trick as the shoelace formula).")
with m2:
    st.markdown("**Convexity**")
    st.latex(r"\text{solidity} = \frac{A}{A_{\text{hull}}} \le 1, \qquad \text{extent} = \frac{A}{w\,h}")
    st.caption("The convex hull is the rubber band around the shape. Solidity 1 = convex; a star scores much less. "
               "Extent compares with the bounding box instead.")


# ------------------------------------------------------------------ invariance (measured)
@st.cache_resource(show_spinner=False)
def invariance_table():
    def star_img(angle, scale):
        img = np.zeros((420, 420), np.uint8)
        pts = K._star(0, 0, 70, 30).astype(float)
        a = np.deg2rad(angle)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        q = (pts @ R.T) * scale + 210
        cv2.fillPoly(img, [np.int32(np.round(q))], 1)
        cs, _ = K.find_contours(img > 0, cv2.RETR_EXTERNAL)
        return K.measure(max(cs, key=len))

    a, b, c = star_img(0, 1), star_img(30, 1), star_img(30, 2)
    keys = [("area", "area"), ("perimeter", "perimeter"), ("circularity", "circularity"), ("solidity", "solidity"),
            ("aspect", "aspect w/h (axis-aligned box)"), ("rot_aspect", "rotated aspect"), ("extent", "extent"),
            ("vertices", "vertices (RDP)")]
    rows = []
    for k, nm in keys:
        rows.append({"measure": nm, "original": round(float(a[k]), 3), "rotated 30°": round(float(b[k]), 3),
                     "rotated 30° and scaled ×2": round(float(c[k]), 3)})
    return pd.DataFrame(rows)


st.subheader("Which measurements survive rotation and scaling?", divider="gray")
st.dataframe(invariance_table(), hide_index=True, width="stretch")
st.caption("Measured on a star drawn at three poses. Area ×4 and perimeter ×2 under scaling (as expected). Circularity, "
           "solidity, rotated aspect and the vertex count stay (almost) the same - they describe *shape*, not size or "
           "pose. The axis-aligned aspect ratio and the extent change with rotation: they depend on the image axes. "
           "This is topic 1's idea of invariants, applied to measurements.")

ui.callout("pitfall", "A fixed ε in pixels makes small objects look like triangles and big ones like 50-gons. Set "
                      "ε relative to the perimeter (OpenCV examples use 0.01-0.05 × arcLength).")
ui.callout("takeaway", "Pick measurements that are invariant to what should NOT matter (position, rotation, size) and "
                       "sensitive to what should (shape): circularity, solidity, rotated aspect ratio and the RDP "
                       "vertex count.")

ui.whys([
    ("Why the 4π in circularity?",
     "The ratio A / P² has the same value for every circle: πr² / (2πr)² = 1/(4π). Multiplying by 4π just "
     "rescales it so that a perfect circle scores 1 - easier to read than 0.0796."),
    ("Why does a pixel circle not score exactly 1?",
     "Its perimeter is measured along a staircase, which is about 5 % too long (lesson 4), and circularity uses P². "
     "Measured in the table above: about 0.89 for large discs and much lower for tiny ones."),
    ("Why set ε relative to the perimeter?",
     "Because ε is a distance in pixels. The same object photographed twice as large has twice the wiggles in pixels. "
     "ε = 0.02 × P scales with the object, so the vertex count describes the shape, not the zoom level."),
    ("Bounding box or rotated box?",
     "cv2.boundingRect is aligned with the image axes: rotate a bar by 45° and its box becomes a square. "
     "cv2.minAreaRect turns with the object, so its aspect ratio is rotation-invariant (measured above)."),
    ("Circularity vs. solidity - what's the difference?",
     "Circularity punishes any deviation from a circle, including corners and elongation (a square: 0.785). Solidity "
     "punishes only *dents* (a square: 1.0, a star: ≈ 0.6). Use both to separate 'round', 'polygonal' and 'concave'."),
    ("How do I handle holes when measuring area?",
     "Retrieve with RETR_CCOMP or RETR_TREE, then subtract the areas of each object's holes (its children). With "
     "RETR_EXTERNAL - as in the dashboard - a ring is measured as a full disc."),
])

st.divider()
ui.mcq("ms1", "Which shape has the highest circularity 4πA / P²?",
       ["A square", "A 2 : 1 rectangle", "A circle", "A star"], 2,
       "The isoperimetric inequality 4πA ≤ P² is an equality only for the circle.")
ui.mcq("ms2", "Which measurement changes when you rotate an object?",
       ["Circularity", "Solidity", "Aspect ratio of the axis-aligned bounding box", "Area"], 2,
       "The axis-aligned box follows the image axes, not the object; use the minimum-area rectangle instead.")

ui.lesson_nav(__file__)
