import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import anim, live, plotting, theme, ui
from core import contour_ui as cui
from core import contours as K
from core import edges as E
from core import hough as Hh

ui.lesson_header(
    __file__,
    "A circle needs three numbers - centre (a, b) and radius r - so the accumulator becomes 3D. For a known radius, "
    "every edge point votes for a <b>circle of possible centres</b>; the true centre is where those circles meet. "
    "The gradient direction then cuts the votes from a whole circle down to <b>one or two points</b>.",
)
st.latex(r"(x - a)^2 + (y - b)^2 = r^2 \qquad \Rightarrow \qquad (a, b, r), \qquad A(a, b, r)")


def circle_xy(c, r, n=61):
    t = np.linspace(0, 2 * np.pi, n)
    return c[0] + r * np.cos(t), c[1] + r * np.sin(t)


# ------------------------------------------------------------------ fixed radius: point -> circle of centres
@st.cache_resource(show_spinner=False)
@live.full_quality()
def fixed_r_figure():
    w, h = 160, 120
    C, R = np.array([80.0, 60.0]), 35.0
    ang = np.deg2rad([200, 245, 290, 335, 20, 70])
    P = np.c_[C[0] + R * np.cos(ang), C[1] + R * np.sin(ang)]
    n = len(P)
    A_full = Hh.accumulator_circle(P, R, w, h, 180)

    def state(k, grow=1.0, heat=False, star=0.0):
        xs, ys = [], []
        for i in range(n):
            rr = R if i < k - 1 else (R * grow if i == k - 1 else 0.0)
            x, y = circle_xy(P[i], rr)
            xs += list(x) + [np.nan]
            ys += list(y) + [np.nan]
        z = A_full.astype(np.uint8) if heat else np.zeros_like(A_full, dtype=np.uint8)
        sizes = np.array([12.0 if i < k else 0.0 for i in range(n)])
        return [cui.heat_trace(z, np.arange(w), np.arange(h), zmax=n, hover="a=%{x}, b=%{y}: %{z} votes"),
                cui.scatter(xs, ys, theme.rgba("#6fa8ee", 0.95), mode="lines", width=1.8),
                cui.scatter(P[:, 0], P[:, 1], theme.E1, mode="markers").update(
                    marker=dict(size=sizes, color=theme.E1, line=dict(color="white", width=1))),
                cui.scatter([C[0]], [C[1]], theme.HIGHLIGHT, size=20 * star, symbol="star")]

    steps = [anim.Step("Edge points of a circle of known radius r", f"r = {R:g}; centre (a, b) unknown",
                       "Only part of the circle is visible - Hough does not mind.", state(0))]
    for k in range(1, n + 1):
        steps.append(anim.Step(f"Point {k}: which centres are possible?",
                               f"(x<sub>{k}</sub> − a)² + (y<sub>{k}</sub> − b)² = r²   ⇒   "
                               f"{anim.hl('a circle of radius r around the point')}",
                               "Any centre at distance r from this edge point could have produced it.",
                               state(k), between=lambda t, k=k: state(k, t)))
    steps.append(anim.Step("The vote circles meet at the true centre", "image point → circle in (a, b) space",
                           "Every vote circle passes through the real centre ★.", state(n, 1.0, False, 1.0)))
    steps.append(anim.Step("Voting: the accumulator A(a, b)", f"(a*, b*) = arg max A = ({C[0]:g}, {C[1]:g}) with "
                           f"{anim.hl(f'{int(A_full.max())} votes')}", "Elsewhere the circles overlap at most twice.",
                           state(n, 1.0, True, 1.0)))
    base = cui.image_axes(w, h, height=460, title_x="a  (and x)", title_y="b  (and y, down)")
    x, y = circle_xy(C, R, 120)
    base.add_trace(cui.scatter(x, y, theme.ORIGINAL, mode="lines", dash="dash", width=1.5))
    return anim.step_player(base, steps, n_sub=12, hold=14, wrap_at=62)


@st.cache_resource(show_spinner=False)
@live.full_quality()
def cone_figure():
    w, h = 120, 100
    C, R = np.array([60.0, 50.0]), 30.0
    P, _ = Hh.circle_points(C[0], C[1], R, n=40, arc=(0, 300), noise=0.4, seed=3)
    P = np.vstack([P, Hh.clutter(10, w, h, seed=4)])
    radii = list(range(10, 52, 2))
    base = cui.image_axes(w, h, height=440, title_x="a", title_y="b (down)")
    vmax = len(P) * 0.45

    def make_state(r):
        A = Hh.accumulator_circle(P, r, w, h, 120)
        i, j = np.unravel_index(int(np.argmax(A)), A.shape)
        cap = (f"slice r = {r}:  max votes {int(A.max())} at (a, b) = ({j}, {i})" +
               (f"   ·   <b style='color:{theme.HIGHLIGHT}'>the right radius: all votes meet</b>" if abs(r - R) < 1.5
                else ("   ·   too small: the vote circles do not reach the centre" if r < R else
                      "   ·   too large: the vote circles overshoot - a ring, no peak")))
        return [cui.heat_trace(np.minimum(A, 255).astype(np.uint8), np.arange(w), np.arange(h), zmax=vmax,
                               hover="a=%{x}, b=%{y}: %{z} votes"),
                cui.scatter(P[:, 0], P[:, 1], theme.E1, size=5)], cap

    return anim.scrubber(base, radii, make_state, active=radii.index(20), label="r", fmt="{:.0f}", frame_ms=130,
                         wrap_at=62)


st.subheader("Step by step: a point votes for a circle of centres", divider="gray")
c1, c2 = st.columns(2, gap="large")
with c1:
    ui.anim_intro("Step by step: fixed radius r")
    st.plotly_chart(fixed_r_figure(), key="hc_fixed")
with c2:
    ui.anim_intro("Sweep: unknown radius - slice the 3D accumulator",
                  "Each r is one slice A(a, b, r). Only the right r focuses the votes into a sharp peak.")
    st.plotly_chart(cone_figure(), key="hc_cone")
st.markdown("With r unknown, each edge point votes for **all** (a, b, r) with (x − a)² + (y − b)² = r²: a **cone** in "
            "(a, b, r) space (a circle of centres for every radius). The cones of the circle's points all pass through "
            "(a*, b*, r*).")
st.latex(r"(a^*, b^*, r^*) = \arg\max_{a, b, r} A(a, b, r)")


# ------------------------------------------------------------------ live: fixed-r explorer + radius profile
@st.cache_resource(show_spinner=False, max_entries=32)
def profile(r_true, n, arc, noise):
    P, _ = Hh.circle_points(80, 60, r_true, n=n, arc=(0, arc), noise=noise, seed=2)
    radii = np.arange(10, 61)
    A3 = Hh.accumulator_circle_3d(P, radii, 160, 120, 90)
    return P, radii, A3.reshape(len(radii), -1).max(1)


@live.section
def explore_section():
    st.subheader("Fixed-radius voting, live", divider="gray")
    c_ctrl, c_fig = st.columns([0.8, 3], gap="large")
    with c_ctrl:
        r_guess = live.slider("radius r used for voting", 10, 60, 25, 1, key="hc_rg")
        r_true = live.slider("true radius", 15, 55, 35, 1, key="hc_rt")
        n = live.slider("edge points", 6, 90, 40, 2, key="hc_n")
        arc = live.slider("visible arc (degrees)", 40, 360, 270, 10, key="hc_arc")
        noise = live.slider("noise σ (px)", 0.0, 3.0, 0.5, 0.1, key="hc_noise")
    P, radii, prof = profile(r_true, n, arc, round(noise, 1))
    A = Hh.accumulator_circle(P, r_guess, 160, 120, 120)
    i, j = np.unravel_index(int(np.argmax(A)), A.shape)
    fig = cui.two_panel("image: edge points", f"accumulator A(a, b) for r = {r_guess}", height=420,
                        widths=(0.5, 0.5), left_range=(0, 160, 0, 120), left_ydown=True, right_range=(0, 160, 120, 0),
                        left_title_x="x", left_title_y="y", right_title_x="a", right_title_y="b")
    fig.update_layout(yaxis2=dict(scaleanchor="x2", constrain="domain"), xaxis2=dict(constrain="domain"))
    xg, yg = circle_xy((j, i), r_guess, 90)
    xt, yt = circle_xy((80, 60), r_true, 90)
    fig.add_traces([
        cui.scatter(xt, yt, theme.ORIGINAL, mode="lines", dash="dash", width=1.5, xaxis="x", yaxis="y"),
        cui.scatter(xg, yg, theme.HIGHLIGHT, mode="lines", width=2.5, xaxis="x", yaxis="y"),
        cui.scatter(P[:, 0], P[:, 1], theme.E1, size=6, xaxis="x", yaxis="y"),
        cui.heat_trace(A, np.arange(160), np.arange(120), xaxis="x2", yaxis="y2", hover="a=%{x}, b=%{y}: %{z} votes"),
        cui.scatter([j], [i], theme.HIGHLIGHT, size=14, symbol="circle-open", xaxis="x2", yaxis="y2").update(
            marker=dict(line=dict(width=2, color=theme.HIGHLIGHT))),
    ])
    pf = go.Figure(data=[cui.scatter(radii, prof, theme.TRANSFORMED, mode="lines+markers", size=4, width=2.5,
                                     name="best votes at radius r", hover=True),
                         cui.scatter([r_guess], [prof[r_guess - 10]], theme.HIGHLIGHT, size=13, name="your r")],
                   layout=dict(height=220, margin=dict(l=8, r=8, t=10, b=8), showlegend=False,
                               xaxis=dict(title="radius r", gridcolor=theme.GRID),
                               yaxis=dict(title="max over (a, b)", gridcolor=theme.GRID)))
    pf.add_vline(x=r_true, line=dict(color=theme.ORIGINAL, dash="dash", width=1))
    with c_fig:
        st.plotly_chart(fig, key="hc_fig")
        st.plotly_chart(pf, key="hc_prof")
    st.caption(f"Peak {int(A.max())} votes at (a, b) = ({j}, {i}) for r = {r_guess}. The lower plot is the 3D "
               "accumulator seen from the side: the best number of votes in each r-slice. It peaks at the true radius "
               "(dashed) - that is how the radius is found.")


explore_section()


# ------------------------------------------------------------------ gradient voting
@st.cache_resource(show_spinner=False)
@live.full_quality()
def gradient_figure():
    w, h = 160, 120
    C, R = np.array([80.0, 60.0]), 32.0
    P, th = Hh.circle_points(C[0], C[1], R, n=10, arc=(0, 360), outward=True)
    n = len(P)
    g = np.c_[np.cos(th), np.sin(th)]
    minus = P - R * g
    plus = P + R * g
    A = Hh.accumulator_circle_gradient(P, th, R, w, h, "both")

    def state(arrows=0.0, rays=0.0, dots=0, heat=False):
        rx, ry = [], []
        for p, d in zip(P, g):
            a, b = p - rays * 1.25 * R * d, p + rays * 1.25 * R * d
            rx += [a[0], b[0], np.nan]
            ry += [a[1], b[1], np.nan]
        sm = np.array([11.0 if i < dots else 0.0 for i in range(n)])
        z = A.astype(np.uint8) if heat else np.zeros_like(A, dtype=np.uint8)
        return [cui.heat_trace(z, np.arange(w), np.arange(h), zmax=n, hover="a=%{x}, b=%{y}: %{z} votes"),
                cui.scatter(rx, ry, theme.ORIGINAL, mode="lines", width=1, dash="dot"),
                plotting.arrows_trace(P, P + arrows * 16 * g, theme.E1, size=9 if arrows else 0),
                cui.scatter(minus[:, 0], minus[:, 1], theme.E2, mode="markers").update(
                    marker=dict(size=sm, color=theme.E2, line=dict(color="white", width=1))),
                cui.scatter(plus[:, 0], plus[:, 1], theme.ORIGINAL, mode="markers").update(
                    marker=dict(size=sm, color=theme.ORIGINAL, symbol="x"))]

    steps = [
        anim.Step("Edge points and their gradients", "∇I points from dark to bright, ACROSS the edge",
                  "A dark disc on a bright background: every gradient points away from the centre.", state(1.0)),
        anim.Step("The centre must lie on the gradient line", "for a circle the gradient is RADIAL",
                  "The normal of a circle always passes through its centre.", state(1.0, 1.0)),
        anim.Step("… at distance r, one way or the other", f"(a, b) = (x {anim.hl('∓')} r cos θ, y ∓ r sin θ)",
                  "Two candidates per point: against the gradient (green) or along it (grey ×).", state(1.0, 1.0, 1)),
        anim.Step("All points vote twice instead of along a whole circle", "2 votes per point instead of ~2πr",
                  "The green votes pile up on the centre; the grey ones scatter on a circle of radius 2r.",
                  state(1.0, 1.0, n)),
        anim.Step("The accumulator: one sharp peak", f"A({C[0]:g}, {C[1]:g}) = {anim.hl(f'{int(A.max())} votes')}",
                  "Far fewer votes, far less clutter. If you know the disc is dark (or bright), keep only one sign.",
                  state(1.0, 0.0, n, True)),
    ]
    base = cui.image_axes(w, h, height=450, title_x="x  /  a", title_y="y  /  b (down)")
    x, y = circle_xy(C, R, 120)
    base.add_trace(cui.scatter(x, y, theme.ORIGINAL, mode="lines", dash="dash", width=1.2))
    base.add_trace(cui.scatter(P[:, 0], P[:, 1], theme.TRANSFORMED, size=9))
    return anim.step_player(base, steps, n_sub=12, hold=16, wrap_at=62)


st.subheader("Using the gradient direction", divider="gray")
g1, g2 = st.columns([1.15, 1], gap="large")
with g1:
    ui.anim_intro("Step by step: gradient-guided voting")
    st.plotly_chart(gradient_figure(), key="hc_grad")
with g2:
    st.latex(r"a = x - r\cos\theta,\quad b = y - r\sin\theta \qquad\text{(gradient points outward)}")
    st.latex(r"a = x + r\cos\theta,\quad b = y + r\sin\theta \qquad\text{(gradient points inward)}")
    st.latex(r"(a, b) = (x \pm r\cos\theta,\; y \pm r\sin\theta),\qquad A(a, b) \leftarrow A(a, b) + 1")
    st.markdown("θ is the gradient direction θ = atan2(I_y, I_x) from Sobel (topic 2). Which sign is right depends on "
                "the contrast: a **dark** disc on a bright background has gradients pointing **outward** (use −), a "
                "**bright** disc has them pointing **inward** (use +). Not sure? Vote both ways.")


# ------------------------------------------------------------------ real images
SIGN = {"both (unknown contrast)": "both", "against ∇I (dark discs)": "minus", "along ∇I (bright discs)": "plus"}


@st.cache_resource(show_spinner=False, max_entries=12)
def edge_data(name: str, sigma: float):
    img = Hh.circles_image() if name.startswith("Discs") else E.photo("coins.png")
    R = E.canny(img, sigma, 0.1, 0.25)
    ys, xs = np.nonzero(R["edges"])
    return img, R["edges"], np.c_[xs, ys].astype(float), R["theta"][ys, xs]


@st.cache_resource(show_spinner=False, max_entries=24)
def acc3d(name, sigma, r0, r1, use_grad, sign):
    img, Ed, P, th = edge_data(name, sigma)
    h, w = img.shape
    radii = np.arange(r0, r1 + 1)
    if use_grad:
        A3 = Hh.accumulator_circle_gradient_3d(P, th, radii, w, h, sign)
        votes = len(P) * len(radii) * (2 if sign == "both" else 1)
    else:
        A3 = Hh.accumulator_circle_3d(P, radii, w, h, 60)
        votes = int(sum(len(P) * max(16, int(60 * r / max(radii))) for r in radii))
    return A3, radii, votes


@live.section
def real_section():
    st.subheader("On real images - and OpenCV's HoughCircles", divider="gray")
    c_ctrl, c1, c2 = st.columns([0.85, 1.5, 1.2], gap="large")
    with c_ctrl:
        name = st.selectbox("Image", ["Discs (synthetic, one occluded)", "Coins (photo)"], key="hc_img")
        sigma = live.slider("Canny σ", 1.0, 4.0, 2.0, 0.1, key="hc_sigma")
        r0, r1 = live.slider("radius range r", 8, 60, (15, 45) if name.startswith("Discs") else (15, 32), 1,
                             key=f"hc_r_{name[:5]}")
        use_grad = st.toggle("use the gradient direction", True, key="hc_usegrad")
        sign_name = st.radio("centre lies …", list(SIGN), key="hc_sign", disabled=not use_grad)
        k = live.slider("circles to report", 1, 25, 2 if name.startswith("Discs") else 20, 1, key=f"hc_k_{name[:5]}")
    img, Ed, P, th = edge_data(name, round(sigma, 1))
    A3, radii, votes = acc3d(name, round(sigma, 1), r0, r1, use_grad, SIGN[sign_name])
    best = A3.max(0)
    best_r = radii[A3.argmax(0)]
    pk = Hh.find_peaks(best, k, threshold=4, nms=(2 * r0 - 1, 2 * r0 - 1))
    found = [(float(c), float(r), float(best_r[r, c])) for r, c, v in pk]
    cv = Hh.cv_hough_circles(img, r0, r1, min_dist=max(10, r0 * 1.5), acc_thresh=20 if name.startswith("Coins") else 25)
    vis = E.overlay(img, Ed, color=(120, 120, 120), dim=0.7)
    vis = Hh.draw_circles_rgb(vis, cv, color=K.BLUE, thickness=1)
    vis = Hh.draw_circles_rgb(vis, found, color=K.AMBER, thickness=2)
    with c1:
        st.image(ui.datauri(vis), caption=f"amber = our Hough peaks ({len(found)}) · blue = cv2.HoughCircles ({len(cv)})",
                 width="stretch")
    with c2:
        fig = go.Figure(data=[cui.heat_trace(np.sqrt(best), np.arange(best.shape[1]), np.arange(best.shape[0]),
                                             hover="a=%{x}, b=%{y}: √votes %{z:.1f}")],
                        layout=dict(height=300, margin=dict(l=8, r=8, t=10, b=8),
                                    xaxis=dict(title="a", constrain="domain"),
                                    yaxis=dict(title="b", autorange="reversed", scaleanchor="x", constrain="domain")))
        st.plotly_chart(fig, key="hc_best")
        h_, w_ = img.shape
        st.dataframe(pd.DataFrame([{"": "edge points", "value": f"{len(P):,}"},
                                   {"": "votes cast", "value": f"{votes:,}"},
                                   {"": "accumulator cells a × b × r", "value": f"{w_} × {h_} × {len(radii)} = "
                                                                             f"{w_ * h_ * len(radii):,}"}]),
                     hide_index=True, width="stretch")
    st.caption("Heat-map: for each candidate centre, its best vote count over all radii (√ scale). Switch the gradient "
               "off: the number of votes jumps by an order of magnitude and the peaks drown in clutter. OpenCV's "
               "HOUGH_GRADIENT is two-stage: centres first (votes along gradient lines), then the best radius per "
               "centre from a histogram of distances - so it never builds the full 3D array.")


real_section()

ui.callout("intuition", "Each edge point shouts 'the centre is r away from me!'. Without the gradient it can't say in "
                        "which direction, so it shouts in every direction (a circle). With the gradient it points: "
                        "'there - or exactly opposite'.")
ui.callout("pitfall", "A wrong radius range finds nothing - or finds circles of the wrong size whose vote rings happen "
                      "to overlap. Set r_min, r_max from what you know about the objects.")
ui.callout("takeaway", "Circle: (x − a)² + (y − b)² = r², three parameters, a 3D accumulator. A point votes for a "
                       "circle (fixed r) or a cone (unknown r) of centres; with the gradient, for one or two points.")

ui.whys([
    ("Why the ± in the gradient formula?",
     "The gradient is perpendicular to the edge, so it lies along the radius - but it points from dark to bright. "
     "For a dark coin on a bright table it points outward (centre = point − r·direction); for a bright coin on a dark "
     "table it points inward (+). If the contrast is unknown, vote both ways (the wrong one scatters)."),
    ("Why is the circle accumulator 3D?",
     "Because a circle has three degrees of freedom (a, b, r). Each extra parameter multiplies memory and time by the "
     "number of its bins: a 360 × 284 image with 30 radii already has over 3 million cells."),
    ("Why does using the gradient make it faster AND cleaner?",
     "Faster: 2 votes per point and radius instead of ~2πr. Cleaner: fewer wrong votes means less background "
     "clutter in the accumulator, so false peaks (where unrelated vote circles happen to overlap) become rare."),
    ("How does cv2.HoughCircles work?",
     "The 'Hough gradient method': Canny edges, then each edge point votes along its gradient line for centres "
     "(2D accumulator, all radii at once), then for each strong centre a histogram of distances to the edge points "
     "picks the radius. Its parameters: dp (accumulator resolution), minDist (between centres), param1 (Canny high "
     "threshold), param2 (centre votes threshold)."),
    ("What about ellipses?",
     "An ellipse needs 5 parameters (centre, two axes, angle): a 5D accumulator is huge. In practice one fits "
     "ellipses to contours (cv2.fitEllipse) or uses clever reduced Hough variants."),
    ("What if only part of the circle is visible?",
     "Fewer votes, but they still meet at the centre - try a small arc above. Hough finds partly occluded circles "
     "(the synthetic image has one) as long as enough of the boundary survives the threshold."),
])

st.divider()
ui.mcq("hc1", "With a FIXED radius r, an edge point votes for …",
       ["one centre", "a circle of radius r of possible centres", "a line of centres", "all centres"], 1,
       "Every centre at distance r from the point could have produced it.")
ui.mcq("hc2", "Using the gradient direction, how many centre candidates does a point have (per radius)?",
       ["One or two", "About 2πr", "r²", "Infinitely many"], 0,
       "The centre lies on the gradient line at distance r: x ± r cos θ, y ± r sin θ.")

ui.lesson_nav(__file__)
