import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import anim, live, theme, ui
from core import contour_ui as cui
from core import contours as K
from core import edges as E
from core import hough as Hh

ui.lesson_header(
    __file__,
    "Lines and circles are two cases of one idea: <b>every edge point votes for every model that could have produced "
    "it, and models are peaks</b>. This page shows why that survives broken edges, where it breaks down (noise, bins, "
    "dimensions), and when to choose contours, polygons or Hough.",
)

# ------------------------------------------------------------------ the general principle
st.subheader("The general Hough principle", divider="gray")
g1, g2 = st.columns([1, 1.2], gap="large")
with g1:
    st.latex(r"\boldsymbol\theta = (\theta_1, \theta_2, \dots, \theta_d)")
    st.latex(r"\text{for every edge point: } A(\boldsymbol\theta) \leftarrow A(\boldsymbol\theta) + 1 "
             r"\;\;\forall \boldsymbol\theta \text{ that could produce it}")
    st.latex(r"\boldsymbol\theta^* = \arg\max_{\boldsymbol\theta} A(\boldsymbol\theta)")
    st.markdown("**Hough transform = voting in parameter space.**")
with g2:
    st.markdown("| | Image space | Hough space |\n|---|---|---|\n| Representation | pixels (x, y) | shape parameters |\n"
                "| Line | points forming a line | peak (ρ, θ) |\n| Circle | boundary points | peak (a, b, r) |\n"
                "| Noise | scattered points | diffuse votes |\n"
                "| True geometric structure | aligned boundary points | concentrated peak |")
    st.latex(r"\text{Geometric structure in image space} \;\longrightarrow\; \text{concentration of votes in parameter space}")

# ------------------------------------------------------------------ broken edges, step by step
W, H = 160, 110


@st.cache_resource(show_spinner=False)
@live.full_quality()
def fragments_figure():
    th0, rho0 = 65.0, 70.0
    F1 = Hh.line_points(th0, rho0, W, H, n=7, span=(0.04, 0.36))
    F2 = Hh.line_points(th0, rho0, W, H, n=7, span=(0.64, 0.96))
    allp = np.vstack([F1, F2])
    n1, n = len(F1), len(F1) + len(F2)
    nt, rr = 60, 2.0
    full_t = np.arange(0, 180.5, 1.0)
    Afull, ths, rhos = Hh.accumulator_rt(allp, W, H, nt, rr)
    r_, c_ = np.unravel_index(int(np.argmax(Afull)), Afull.shape)
    th_s, rho_s = ths[c_], rhos[r_]
    seg = np.array(Hh.clip_line(th_s, rho_s, W, H))

    def boxes(pts):
        lo, hi = pts.min(0) - 4, pts.max(0) + 4
        return np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]], [lo[0], lo[1]]])

    b1, b2 = boxes(F1), boxes(F2)

    def state(g1, g2, contours=0.0, peak=0.0):
        sx, sy = [], []
        for i in range(n):
            f = g1 if i < n1 else g2
            tt = full_t[full_t <= 180 * f] if f > 0 else full_t[:1]
            tt = np.r_[tt, np.full(len(full_t) - len(tt), tt[-1])]
            sx += list(tt) + [np.nan]
            sy += list(Hh.rho_of(allp[i, 0], allp[i, 1], np.deg2rad(tt))) + [np.nan]
        voters = [allp[i] for i in range(n) if (g1 >= 1 if i < n1 else g2 >= 1)]
        Acc = Hh.accumulator_rt(np.array(voters).reshape(-1, 2), W, H, nt, rr)[0].astype(np.uint8)
        c1, c2 = b1[:1] + contours * (b1 - b1[:1]), b2[:1] + contours * (b2 - b2[:1])
        L = seg[:1] + peak * (seg - seg[:1])
        return [cui.heat_trace(Acc, ths, rhos, zmax=n * 0.8, xaxis="x2", yaxis="y2"),
                cui.scatter(sx[:len(sx) * n1 // n], sy[:len(sy) * n1 // n], "#9ec5f4", mode="lines", width=1.4,
                            xaxis="x2", yaxis="y2"),
                cui.scatter(sx[len(sx) * n1 // n:], sy[len(sy) * n1 // n:], "#7fe0bb", mode="lines", width=1.4,
                            xaxis="x2", yaxis="y2"),
                cui.scatter(c1[:, 0], c1[:, 1], theme.TRANSFORMED, mode="lines", width=2, dash="dot", xaxis="x",
                            yaxis="y"),
                cui.scatter(c2[:, 0], c2[:, 1], theme.E2, mode="lines", width=2, dash="dot", xaxis="x", yaxis="y"),
                cui.scatter(L[:, 0], L[:, 1], theme.HIGHLIGHT, mode="lines", width=3, xaxis="x", yaxis="y"),
                cui.scatter([th_s], [rho_s], theme.HIGHLIGHT, size=18 * peak, symbol="star", xaxis="x2", yaxis="y2")]

    steps = [
        anim.Step("A line, partly hidden behind an obstacle", "two fragments, one gap",
                  "Blue and green points lie on the SAME line; the grey box hides the middle.", state(0, 0)),
        anim.Step("Contour tracing sees two objects", "fragment 1 ≠ fragment 2",
                  "The fragments are not connected, so tracing gives two separate contours (dotted boxes).",
                  state(0, 0, 1.0)),
        anim.Step("Fragment 1 votes", "ρ = x cos θ + y sin θ  for its points",
                  "Its sinusoids cross at the line's (θ, ρ).", state(1, 0, 1.0), between=lambda t: state(t, 0, 1.0)),
        anim.Step("Fragment 2 votes", f"every visible point still satisfies {anim.hl('x cos θ + y sin θ = ρ')}",
                  "Its sinusoids cross at the SAME place: the gap does not matter.", state(1, 1, 1.0),
                  between=lambda t: state(1, t, 1.0)),
        anim.Step("One common peak = one line", f"(θ*, ρ*) = ({th_s:.0f}°, {rho_s:.0f}) with "
                  f"{anim.hl(f'{int(Afull.max())} votes')} from BOTH fragments",
                  "Disconnected edge fragments → common peak in parameter space.", state(1, 1, 1.0, 1.0)),
    ]
    base = cui.two_panel("", "", height=460, widths=(0.42, 0.58), left_range=(-4, W + 4, -4, H + 4), left_ydown=True,
                         right_range=(0, 180, rhos[0], rhos[-1]), left_title_x="x   ·   IMAGE SPACE",
                         left_title_y="y (down)", right_title_x="θ (degrees)   ·   ACCUMULATOR", right_title_y="ρ")
    gap = Hh.line_points(th0, rho0, W, H, n=2, span=(0.4, 0.6))
    mid = gap.mean(0)
    base.add_shape(type="rect", x0=mid[0] - 16, x1=mid[0] + 16, y0=mid[1] - 18, y1=mid[1] + 18, xref="x", yref="y",
                   fillcolor=theme.rgba(theme.ORIGINAL, 0.45), line=dict(width=0))
    base.add_trace(cui.scatter(F1[:, 0], F1[:, 1], theme.TRANSFORMED, size=9))
    base.add_trace(cui.scatter(F2[:, 0], F2[:, 1], theme.E2, size=9))
    return anim.step_player(base, steps, n_sub=10, hold=16, wrap_at=105)


st.subheader("Step by step: why Hough handles broken edges", divider="gray")
ui.anim_intro("Step by step: two fragments, one peak")
st.plotly_chart(fragments_figure(), key="hg_frag")


# ------------------------------------------------------------------ robustness lab
@live.section
def lab_section():
    st.subheader("Robustness lab: gaps, noise, clutter and bin size", divider="gray")
    c_ctrl, c_img, c_acc = st.columns([0.85, 1.4, 1.5], gap="large")
    with c_ctrl:
        gap = live.slider("gap (fraction of the line hidden)", 0.0, 0.8, 0.3, 0.05, key="hg_gap")
        noise = live.slider("position noise σ (px)", 0.0, 4.0, 0.5, 0.1, key="hg_noise")
        n_clut = live.slider("clutter points", 0, 600, 150, 25, key="hg_clutter")
        n_theta = live.slider("θ bins", 18, 360, 180, 6, key="hg_nt")
        rho_res = live.slider("ρ bin size (px)", 0.5, 8.0, 1.0, 0.5, key="hg_rho")
        thr = live.slider("peak threshold (votes)", 5, 120, 40, 5, key="hg_thr")
    w, h, th0, rho0 = 200, 150, 120.0, -30.0
    line_img = np.zeros((h, w), np.uint8)
    seg = Hh.clip_line(th0, rho0, w - 1, h - 1)
    (x0, y0), (x1, y1) = seg
    cv2.line(line_img, (int(x0), int(y0)), (int(x1), int(y1)), 1, 1)
    if gap > 0:
        mid = (np.array([x0, y0]) + np.array([x1, y1])) / 2
        half = gap * np.hypot(x1 - x0, y1 - y0) / 2
        yy, xx = np.mgrid[0:h, 0:w]
        line_img[np.hypot(xx - mid[0], yy - mid[1]) < half] = 0
    ys, xs = np.nonzero(line_img)
    rng = np.random.default_rng(3)
    P = np.c_[xs, ys].astype(float) + rng.normal(0, noise, (len(xs), 2)) * (noise > 0)
    allp = np.vstack([P, Hh.clutter(n_clut, w, h, seed=9)]) if n_clut else P
    n_pieces, _ = K.label(line_img > 0, 8)
    A, ths, rhos = Hh.accumulator_rt(allp, w, h, n_theta, rho_res)
    nms = (max(3, int(round(12 / rho_res)) | 1), max(3, int(round(n_theta / 18)) | 1))
    pk = Hh.find_peaks(A, 12, threshold=thr, nms=nms, wrap_cols=True)
    good = [p for p in pk if Hh.angle_error(ths[p[1]], th0) < 3 and abs(rhos[p[0]] - rho0) < 4 + rho_res]
    spurious = [p for p in pk if p not in good]
    rth, rrho, inl = Hh.ransac_line(allp, 300, 2.0, seed=1)
    with c_img:
        big = cv2.resize(K.binary_rgb(Hh.points_image(allp, w, h) > 0, fg=(200, 200, 196)), (w * 2, h * 2),
                         interpolation=cv2.INTER_NEAREST)
        for peaks, col, thick in ((spurious, K.RED, 1), (good, K.AMBER, 2)):
            for r, c, _ in peaks:
                s_ = Hh.clip_line(ths[c], rhos[r], w - 1, h - 1)
                if s_ is not None:
                    (ax, ay), (bx, by) = s_
                    cv2.line(big, (int(ax * 2 + 1), int(ay * 2 + 1)), (int(bx * 2 + 1), int(by * 2 + 1)), col, thick,
                             cv2.LINE_AA)
        st.image(ui.datauri(big), caption="amber = Hough peak on the true line · red = spurious peaks above the threshold",
                 width="stretch")
    with c_acc:
        fig = go.Figure(data=[cui.heat_trace(np.sqrt(A), ths, rhos, hover="θ=%{x:.0f}°, ρ=%{y:.0f}: √votes %{z:.1f}"),
                              cui.scatter([ths[c] for _, c, _ in good], [rhos[r] for r, _, _ in good], theme.HIGHLIGHT,
                                          size=15, symbol="circle-open").update(
                                  marker=dict(line=dict(width=2.5, color=theme.HIGHLIGHT))),
                              cui.scatter([ths[c] for _, c, _ in spurious], [rhos[r] for r, _, _ in spurious],
                                          theme.OUTLIER, size=13, symbol="x-open").update(
                                  marker=dict(line=dict(width=2, color=theme.OUTLIER)))],
                        layout=dict(height=330, margin=dict(l=8, r=8, t=10, b=8), showlegend=False,
                                    xaxis=dict(title="θ (degrees)"), yaxis=dict(title="ρ")))
        st.plotly_chart(fig, key="hg_acc")
    m = st.columns(5)
    m[0].metric("contour pieces of the line", n_pieces, border=True)
    m[1].metric("votes at the true line", int(max([v for *_, v in good], default=0)), f"of {len(P)} line pixels",
                delta_color="off", border=True)
    m[2].metric("true line found", "yes" if good else "no", border=True)
    m[3].metric("spurious peaks", len(spurious), border=True)
    m[4].metric("RANSAC angle error", f"{Hh.angle_error(rth, th0):.1f}°", f"{int(inl.sum())} inliers",
                delta_color="off", border=True)
    st.caption("Raise the gap: contour tracing breaks the line into pieces, Hough still finds it (with fewer votes). "
               "Raise the clutter and lower the threshold: random alignments of clutter points form spurious peaks. "
               "Shrink the bins with noise on: the true peak splits and can drop below the threshold. RANSAC (topic 1) "
               "is the other classic robust method: random pairs instead of exhaustive voting.")


lab_section()


# ------------------------------------------------------------------ limitations and cost
st.subheader("Limitations - and what they cost", divider="gray")
l1, l2 = st.columns([1, 1.2], gap="large")
with l1:
    st.markdown("1. The parameter space can become **very large**.\n2. Computational cost **grows with the number of "
                "parameters**.\n3. The **resolution** of the accumulator limits the accuracy.\n4. Noise can produce "
                "**spurious peaks**.\n5. Nearby shapes produce **overlapping peaks**.\n6. Choosing the **peak threshold** "
                "matters.")
with l2:
    w_, h_ = 640, 480
    Dg = int(np.ceil(np.hypot(w_, h_)))
    rows = [("line (ρ, θ)", "2D", f"{2 * Dg + 1} × 180", (2 * Dg + 1) * 180, 180),
            ("circle (a, b, r), r = 5…100", "3D", f"{w_} × {h_} × 96", w_ * h_ * 96, 96 * 360),
            ("circle with gradient", "3D", f"{w_} × {h_} × 96", w_ * h_ * 96, 96 * 2),
            ("ellipse (a, b, axes, angle)", "5D", f"{w_} × {h_} × 100 × 100 × 180", w_ * h_ * 100 * 100 * 180,
             100 * 100 * 180)]
    df = pd.DataFrame([{"model": r[0], "dims": r[1], "bins (1-pixel / 1° cells)": r[2],
                        "cells": f"{r[3]:,}", "memory (int32)": f"{r[3] * 4 / 2 ** 20:,.1f} MB" if r[3] * 4 < 2 ** 40
                        else f"{r[3] * 4 / 2 ** 40:,.1f} TB", "votes per edge point": f"{r[4]:,}"} for r in rows])
    st.dataframe(df, hide_index=True, width="stretch")
    st.caption("For a 640 × 480 image. Every extra parameter multiplies the accumulator by its number of bins: lines are "
               "cheap, circles feasible, ellipses impossible by brute force.")
st.latex(r"A(\rho, \theta)\ \text{is 2D}, \qquad A(a, b, r)\ \text{is 3D}, \qquad \text{more complex models} \Rightarrow "
         r"\text{higher-dimensional parameter spaces}")

# ------------------------------------------------------------------ comparisons and pipelines
st.subheader("Contours, polygons or Hough?", divider="gray")
k1, k2 = st.columns(2, gap="large")
with k1:
    st.markdown("**Contour detection vs. Hough transform**")
    st.markdown("| | Contour detection | Hough transform |\n|---|---|---|\n| Main idea | follow connected boundaries | "
                "vote for geometric models |\n| Connectivity | usually important | not required |\n| Output | ordered "
                "contour points | geometric parameters |\n| Good for | arbitrary shapes | parametric shapes |\n"
                "| Handles gaps | less robust | more robust |\n| Representation | boundary-based | model-based |\n"
                "| Examples | objects, polygons | lines, circles, ellipses |")
    st.markdown("Contour detection → *“Where is the boundary?”*  \nHough transform → *“Which geometric model explains "
                "the boundary?”*")
with k2:
    st.markdown("**Polygon approximation vs. Hough transform**")
    st.markdown("Polygonal approximation asks: *can the boundary be represented with fewer vertices?* "
                "Hough asks: *can the boundary be explained by a known parametric model?*")
    st.latex(r"\text{Circle} \to \text{many contour points} \to \text{polygon} \qquad "
             r"(\text{non-parametric simplification})")
    st.latex(r"\text{Circle} \to \text{edge points} \to \text{Hough} \to (a, b, r) \qquad (\text{parametric detection})")


@st.cache_resource(show_spinner=False)
@live.full_quality()
def pipeline_figure():
    img = K.shapes_scene()
    h, w = img.shape
    R = E.canny(img, 1.4, 0.1, 0.25)
    B = K.binarize(img, 0.45)
    cs, _ = K.find_contours(B, cv2.RETR_EXTERNAL, min_points=12)
    contours_vis = K.draw_contours(K.gray_rgb(img, 0.35), cs, K.PALETTE, scale=1, thickness=1)
    ms = [K.measure(c) for c in cs]
    poly = K.gray_rgb(img, 0.35)
    for c, m in zip(cs, ms):
        cv2.polylines(poly, [np.int32(c[m["approx"]]).reshape(-1, 1, 2)], True, K.BLUE, 2, cv2.LINE_AA)
        x, y, *_ = m["bbox"]
        cv2.putText(poly, K.classify(m), (int(x), max(10, int(y) - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, K.AMBER, 1,
                    cv2.LINE_AA)
    ys, xs = np.nonzero(R["edges"])
    P = np.c_[xs, ys].astype(float)
    A, ths, rhos = Hh.accumulator_rt(P, w, h, 180, 1.0)
    acc = (np.sqrt(A / A.max()) * 255).astype(np.uint8)
    acc_rgb = cv2.applyColorMap(cv2.resize(acc, (w, h)), cv2.COLORMAP_INFERNO)[..., ::-1]
    pk = Hh.find_peaks(A, 6, threshold=10, nms=(15, 9), wrap_cols=True)
    acc_pk = acc_rgb.copy()
    for r, c, _ in pk:
        cv2.circle(acc_pk, (int(c * w / A.shape[1]), int(r * h / A.shape[0])), 6, (255, 255, 255), 1, cv2.LINE_AA)
    lines = Hh.draw_lines_rgb(K.gray_rgb(img, 0.35), [(float(rhos[r]), float(ths[c])) for r, c, _ in pk],
                              color=K.AMBER, thickness=1)
    stages = [
        ("Image", "both roads start here", K.gray_rgb(img)),
        ("Edge detection", "Canny (topic 2) → binary edge map", E.edges_rgb(R["edges"])),
        ("Road 1 · contour detection", "here on the thresholded regions: one ordered contour per boundary",
         contours_vis),
        ("Road 1 · polygonal approximation + measurements → classification", "RDP with ε = 0.02 P, then the rules of "
         "lesson 6", poly),
        ("Road 2 · Hough voting", "edge points → accumulator A[ρ, θ] (stretched to the image size)", acc_rgb),
        ("Road 2 · accumulator peaks", "non-maximum suppression keeps the local maxima", acc_pk),
        ("Road 2 · geometric primitives", "the 6 strongest lines - infinite lines, no segments", lines),
    ]
    steps = [anim.Step(t, eq, "", [dict(type="image", source=ui.datauri(im), x0=0, y0=0, dx=1, dy=1,
                                        hoverinfo="skip")]) for t, eq, im in stages]
    base = go.Figure(layout=dict(height=470, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
                                 xaxis=dict(visible=False, range=[-0.5, w - 0.5], constrain="domain"),
                                 yaxis=dict(visible=False, range=[h - 0.5, -0.5], scaleanchor="x", constrain="domain")))
    return anim.step_player(base, steps, n_sub=1, hold=30, wrap_at=100)


st.subheader("Step by step: the two complete pipelines", divider="gray")
st.latex(r"\text{Image} \to \text{Edges} \to \text{Binary edge map} \to \text{Contours} \to \text{Representation} \to "
         r"\{\text{Polygon},\ \text{Measurements},\ \text{Classification}\}")
st.latex(r"\text{Image} \to \text{Edges} \to \text{Edge points} \to \text{Hough voting} \to \text{Accumulator peaks} \to "
         r"\text{Geometric primitives}")
ui.anim_intro("Step by step: both roads on one image")
st.plotly_chart(pipeline_figure(), key="hg_pipe")

ui.callout("intuition", "A contour is a detective following footprints: lose the trail (a gap) and the case splits in "
                        "two. Hough is an election: every footprint votes, and the winner is clear even if some voters "
                        "never showed up.")
ui.callout("pitfall", "Hough only finds shapes you can write as a formula with few parameters, and it reports infinite "
                      "lines / full circles - not where the visible part starts and ends.")
ui.callout("takeaway", "Use contours (+ RDP + measurements) for arbitrary, well-separated shapes; use Hough for "
                       "parametric shapes, especially when edges are broken, occluded or cluttered.")

ui.whys([
    ("When should I use contours, and when Hough?",
     "Contours: the object has a clean, closed boundary and any shape (parts, coins on a table, cells, documents). "
     "Hough: you know the model (lane lines, a clock face, pipes) and the edges are broken or cluttered. Many systems "
     "use both: Hough to find candidates, contours to measure them."),
    ("Why does noise spread the votes?",
     "A point that is slightly off the line draws a sinusoid that misses the exact crossing cell. Many such points "
     "smear the peak over neighbouring cells; clutter points add random votes everywhere, and occasionally a few line "
     "up by chance - a spurious peak. Bigger bins and a sensible threshold help."),
    ("Why don't we Hough-detect arbitrary shapes?",
     "We can - the *generalized Hough transform* (Ballard, 1981) stores the shape as an 'R-table' of offsets from "
     "each boundary direction to a reference point; edge points vote for the reference point. But with rotation and "
     "scale the parameter space grows to 4D, so it is used for fixed templates."),
    ("Hough or RANSAC?",
     "Both are robust to outliers. Hough is exhaustive and finds ALL models at once (several peaks) but needs a "
     "discretised parameter space. RANSAC samples minimal sets (2 points for a line) and needs no accumulator, which "
     "scales to many parameters (homographies in topic 1), but finds one model per run."),
    ("How do I choose the bin size?",
     "Bins about as large as the expected position error: ρ ≈ 1-2 px, θ ≈ 0.5-1°. Too fine splits the peak (noise), "
     "too coarse merges nearby lines and loses precision. The lab above shows both effects."),
    ("Why are nearby lines a problem?",
     "Two almost-parallel lines close together produce two peaks only a few cells apart; their butterflies overlap "
     "and NMS may keep just one. Finer bins or a smaller NMS window separate them - at the risk of noisy peaks."),
])

st.divider()
ui.mcq("hg1", "Why does the Hough transform cope with a line that is broken into fragments?",
       ["It fills the gaps with interpolation", "Every visible point still votes for the same (θ, ρ)",
        "It traces each fragment", "It uses a larger σ"], 1,
       "Disconnected edge fragments → common peak in parameter space.")
ui.mcq("hg2", "Detecting ellipses with a brute-force Hough transform is hard because …",
       ["ellipses have no edges", "the accumulator is 5-dimensional", "the gradient is undefined",
        "ellipses are not closed"], 1, "Centre (2) + two axes + angle = 5 parameters: the number of cells explodes.")

ui.lesson_nav(__file__)
