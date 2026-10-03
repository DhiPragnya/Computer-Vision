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
    "Describe a line by its <b>normal</b>: the angle θ of the perpendicular from the origin and the distance ρ along "
    "it. Both are bounded, vertical lines are no problem, and every edge point now votes along a <b>sinusoid</b>. "
    "This is the Hough transform that OpenCV implements.",
)
st.info("Coordinates on this page are **image coordinates**: origin at the top-left pixel, x to the right, **y down** "
        "- the convention of cv2.HoughLines. θ is therefore measured from the x-axis turning *down* (clockwise on "
        "screen).", icon=":material/info:")

W, H = 200, 150
D = float(np.hypot(W, H))


def img_axes_figure(w=W, h=H, height=430, pad=12):
    return go.Figure(layout=dict(height=height, margin=dict(l=8, r=8, t=10, b=8), showlegend=False,
                                 xaxis=dict(range=[-pad, w + pad], gridcolor=theme.GRID, zeroline=True,
                                            zerolinecolor=theme.AXIS, constrain="domain", title="x"),
                                 yaxis=dict(range=[h + pad, -pad], gridcolor=theme.GRID, zeroline=True,
                                            zerolinecolor=theme.AXIS, scaleanchor="x", constrain="domain",
                                            title="y (down)")))


# ------------------------------------------------------------------ derivation of the normal form
@st.cache_resource(show_spinner=False)
@live.full_quality()
def normal_form_figure():
    th0, rho0 = 35.0, 95.0

    def scene(th, rho, line_on=1.0, of_f=0.0, n_on=0.0, p_s=None, proj_on=0.0):
        t = np.deg2rad(th)
        n = np.array([np.cos(t), np.sin(t)])
        u = np.array([-n[1], n[0]])                       # along the line
        F = rho * n
        L = np.array([F - line_on * 260 * u, F + line_on * 260 * u])
        OF = np.array([[0, 0], of_f * F])
        arr = 45 * n_on * n
        arc_t = np.linspace(0, t * n_on, 30)
        arc = np.c_[28 * np.cos(arc_t), 28 * np.sin(arc_t)]
        if p_s is None:
            p, ps = F, 0
        else:
            p, ps = F + p_s * u, 12
        vp = np.array([[0, 0], p]) if p_s is not None else np.array([[0, 0], [0, 0]])
        pj = np.array([p, p + proj_on * (F - p)])
        return [cui.scatter(L[:, 0], L[:, 1], theme.TRANSFORMED, mode="lines", width=3),
                cui.scatter(OF[:, 0], OF[:, 1], theme.HIGHLIGHT, mode="lines", width=4),
                plotting.arrow_trace([0, 0], arr, theme.E2, "n", showlegend=False, size=12 if n_on else 0),
                cui.scatter(arc[:, 0], arc[:, 1], theme.E2, mode="lines", width=2),
                cui.scatter(vp[:, 0], vp[:, 1], theme.ORIGINAL, mode="lines", width=2, dash="dash"),
                cui.scatter(pj[:, 0], pj[:, 1], theme.HIGHLIGHT, mode="lines", width=2, dash="dot"),
                cui.scatter([p[0]], [p[1]], theme.E1, size=ps),
                cui.scatter([F[0]], [F[1]], theme.HIGHLIGHT, size=10 * of_f)]

    steps = [
        anim.Step("A line in the image; the origin O is the top-left pixel", "how do we describe it with two bounded "
                  "numbers?", "Slope and intercept fail for vertical lines. Use the line's distance and direction "
                  "from O instead.", scene(th0, rho0, of_f=0.0)),
        anim.Step("Drop the perpendicular from O to the line", f"ρ = |OF| = {anim.hl(f'{rho0:g}')} pixels",
                  "F is the closest point of the line to the origin: ρ is the perpendicular distance.",
                  scene(th0, rho0, of_f=1.0)),
        anim.Step("Its direction: the unit normal n", f"n = (cos θ, sin θ),   θ = {anim.hl(f'{th0:g}°')}",
                  "θ = orientation of the normal (from the x-axis, turning towards +y). So F = ρ n.",
                  scene(th0, rho0, of_f=1.0, n_on=1.0)),
        anim.Step("Take ANY point p of the line", "p = (x, y)", "Where p sits on the line does not matter - watch.",
                  scene(th0, rho0, of_f=1.0, n_on=1.0, p_s=-70)),
        anim.Step("Project p onto the normal", f"n<sup>T</sup> p = {anim.hl('ρ')}",
                  "The projection of p onto n lands exactly on F (the dotted path runs along the line, which is ⟂ n).",
                  scene(th0, rho0, of_f=1.0, n_on=1.0, p_s=-70, proj_on=1.0)),
        anim.Step("Slide p along the line: the projection never changes",
                  f"[cos θ  sin θ] [x  y]<sup>T</sup> = ρ   ⇒   {anim.hl('x cos θ + y sin θ = ρ')}",
                  "One equation, valid for every point of the line: the normal form.",
                  scene(th0, rho0, of_f=1.0, n_on=1.0, p_s=60, proj_on=1.0),
                  between=lambda t: scene(th0, rho0, of_f=1.0, n_on=1.0, p_s=-70 + 130 * t, proj_on=1.0)),
        anim.Step("A vertical line is no problem", f"θ = {anim.hl('0°')},  ρ = x:   x cos 0 + y sin 0 = x = ρ",
                  "θ stays in [0°, 180°) and |ρ| ≤ image diagonal: a bounded accumulator holds EVERY line.",
                  scene(0.0, 120.0, of_f=1.0, n_on=1.0, p_s=40, proj_on=1.0),
                  between=lambda t: scene(th0 * (1 - t), rho0 + (120 - rho0) * t, of_f=1.0, n_on=1.0,
                                          p_s=60 - 20 * t, proj_on=1.0)),
    ]
    base = img_axes_figure(height=440)
    base.add_trace(cui.scatter([0], [0], "#7d7b76", size=10, mode="markers+text").update(
        text=["O"], textposition="top left", textfont=dict(size=14)))
    return anim.step_player(base, steps, n_sub=14, hold=16, wrap_at=58)


@st.cache_resource(show_spinner=False)
@live.full_quality()
def sinusoid_figure():
    px, py = 120.0, 60.0
    thetas = np.arange(0, 181, 4.0)
    full = np.arange(0, 180.5, 0.5)
    base = cui.two_panel("", "", height=440, widths=(0.45, 0.55), left_range=(-12, W + 12, -12, H + 12),
                         left_ydown=True, right_range=(0, 180, -D, D), left_title_x="x   ·   IMAGE SPACE",
                         left_title_y="y (down)", right_title_x="θ (degrees)   ·   PARAMETER SPACE",
                         right_title_y="ρ (pixels)")
    base.add_trace(cui.scatter(full, Hh.rho_of(px, py, np.deg2rad(full)), theme.rgba(theme.TRANSFORMED, 0.18),
                               mode="lines", width=6, xaxis="x2", yaxis="y2"))
    base.add_trace(cui.scatter([px], [py], theme.E1, size=13, xaxis="x", yaxis="y"))
    amp, ph = np.hypot(px, py), np.degrees(np.arctan2(py, px))

    def make_state(th):
        t = np.deg2rad(th)
        n = np.array([np.cos(t), np.sin(t)])
        u = np.array([-n[1], n[0]])
        rho = px * n[0] + py * n[1]
        F = rho * n
        L = np.array([F - 400 * u, F + 400 * u])
        upto = full[full <= th]
        cap = (f"θ = {th:.0f}°:  ρ = x cos θ + y sin θ = {px:g}·{np.cos(t):.2f} + {py:g}·{np.sin(t):.2f} = "
               f"<b>{rho:.1f}</b>   ·   a sinusoid of amplitude √(x² + y²) = {amp:.1f}, peaking at θ = atan2(y, x) "
               f"= {ph:.1f}°")
        return [cui.scatter(L[:, 0], L[:, 1], theme.TRANSFORMED, mode="lines", width=2, xaxis="x", yaxis="y"),
                cui.scatter([0, F[0]], [0, F[1]], theme.HIGHLIGHT, mode="lines", width=3, xaxis="x", yaxis="y"),
                cui.scatter(upto, Hh.rho_of(px, py, np.deg2rad(upto)), theme.TRANSFORMED, mode="lines", width=3,
                            xaxis="x2", yaxis="y2"),
                cui.scatter([th], [rho], theme.HIGHLIGHT, size=14, xaxis="x2", yaxis="y2")], cap

    return anim.scrubber(base, list(thetas), make_state, active=9, label="θ", fmt="{:.0f}°", frame_ms=90, wrap_at=110)


st.subheader("Step by step: the normal form x cos θ + y sin θ = ρ", divider="gray")
c1, c2 = st.columns([1, 1.15], gap="large")
with c1:
    ui.anim_intro("Step by step: deriving the normal form")
    st.plotly_chart(normal_form_figure(), key="hn_normal")
with c2:
    st.latex(r"n = \begin{bmatrix}\cos\theta \\ \sin\theta\end{bmatrix},\qquad n^T p = \rho")
    st.latex(r"\begin{bmatrix}\cos\theta & \sin\theta\end{bmatrix}\begin{bmatrix}x\\y\end{bmatrix} = \rho "
             r"\;\;\Longrightarrow\;\; x\cos\theta + y\sin\theta = \rho")
    st.markdown("**A point in normal Hough space.** Fix the edge point (x_i, y_i) and let θ vary:")
    st.latex(r"\rho(\theta) = x_i \cos\theta + y_i \sin\theta \;=\; \sqrt{x_i^2 + y_i^2}\,\cos(\theta - \varphi_i),"
             r"\quad \varphi_i = \operatorname{atan2}(y_i, x_i)")
    st.markdown("A **sinusoid**: one point in the image → one sinusoid in (θ, ρ). Collinear points → sinusoids that "
                "all pass through the same (θ*, ρ*).")

ui.anim_intro("Sweep: one point, every line through it",
              "▶ Play or drag θ: left, the line through the point with normal angle θ; right, its (θ, ρ) - tracing a "
              "sinusoid.")
st.plotly_chart(sinusoid_figure(), key="hn_sin")


# ------------------------------------------------------------------ voting replay
@st.cache_resource(show_spinner=False)
@live.full_quality()
def replay_figure():
    w, h = 100, 80
    P = Hh.line_points(60, 50, w, h, n=7, span=(0.1, 0.9))
    O = np.array([[20.0, 15.0], [80.0, 65.0]])
    allp = np.vstack([P, O])
    n = len(allp)
    nt, rr = 60, 2.0
    A_full, ths, rhos = Hh.accumulator_rt(allp, w, h, nt, rr)
    full_t = np.arange(0, 180.5, 1.0)
    pk = Hh.find_peaks(A_full, 1)[0]
    th_s, rho_s = ths[pk[1]], rhos[pk[0]]

    def state(k, grow=1.0, show_peak=0.0):
        sx, sy = [], []
        for i in range(n):
            f = 1.0 if i < k - 1 else (grow if i == k - 1 else 0.0)
            tt = full_t[full_t <= 180 * f] if f > 0 else full_t[:1]
            tt = np.r_[tt, np.full(len(full_t) - len(tt), tt[-1])]
            sx += list(tt) + [np.nan]
            sy += list(Hh.rho_of(allp[i, 0], allp[i, 1], np.deg2rad(tt))) + [np.nan]
        Acc = Hh.accumulator_rt(allp[:max(k - (1 if grow < 1 else 0), 0)], w, h, nt, rr)[0].astype(np.uint8)
        sizes = np.array([12.0 if i < k else 0.0 for i in range(n)])
        cols = [theme.TRANSFORMED] * len(P) + [theme.OUTLIER] * len(O)
        seg = Hh.clip_line(th_s, rho_s, w, h)
        L = np.array(seg) if seg is not None else np.zeros((2, 2))
        L = L[:1] + show_peak * (L - L[:1])
        return [cui.heat_trace(Acc, ths, rhos, zmax=len(P), xaxis="x2", yaxis="y2"),
                cui.scatter(sx, sy, theme.rgba("#9ec5f4", 0.85), mode="lines", width=1.4, xaxis="x2", yaxis="y2"),
                cui.scatter([th_s], [rho_s], theme.HIGHLIGHT, size=18 * show_peak, symbol="star-open", xaxis="x2",
                            yaxis="y2").update(marker=dict(line=dict(width=2, color=theme.HIGHLIGHT))),
                cui.scatter(L[:, 0], L[:, 1], theme.HIGHLIGHT, mode="lines", width=3, xaxis="x", yaxis="y"),
                cui.scatter(allp[:, 0], allp[:, 1], cols, xaxis="x", yaxis="y").update(
                    marker=dict(size=sizes, color=cols, line=dict(color="white", width=1)))]

    steps = [anim.Step("An empty accumulator A[ρ, θ]", f"{nt} θ-bins of {180 // nt}°  ×  {len(rhos)} ρ-bins of "
                       f"{rr:g} px, all zero", "Each cell is one candidate line.", state(0))]
    for k in range(1, n + 1):
        out = k > len(P)
        steps.append(anim.Step(f"Point {k} votes" + (" (an outlier)" if out else ""),
                               f"for every θ<sub>k</sub>: ρ = x cos θ<sub>k</sub> + y sin θ<sub>k</sub> → "
                               f"A[ρ, θ<sub>k</sub>] += 1", "Its sinusoid adds one vote to every cell it passes."
                               if not out else "Its sinusoid misses the crossing: one vote in many cells, a peak in none.",
                               state(k), between=lambda t, k=k: state(k, t)))
    steps.append(anim.Step("The peak", f"(θ*, ρ*) = arg max A = ({th_s:.0f}°, {rho_s:.0f})  with "
                           f"{anim.hl(f'{int(A_full.max())} votes')}",
                           "Back in the image: the line with the most supporting edge points.",
                           state(n, 1.0, 1.0)))
    base = cui.two_panel("", "", height=460, widths=(0.4, 0.6), left_range=(-5, w + 5, -5, h + 5), left_ydown=True,
                         right_range=(0, 180, rhos[0], rhos[-1]), left_title_x="x   ·   IMAGE SPACE",
                         left_title_y="y (down)", right_title_x="θ (degrees)   ·   ACCUMULATOR A[ρ, θ]",
                         right_title_y="ρ")
    return anim.step_player(base, steps, n_sub=8, hold=14, wrap_at=105)


st.subheader("Step by step: the discrete accumulator", divider="gray")
st.latex(r"\theta \in \{\theta_1, \dots, \theta_K\},\; \rho \in \{\rho_1, \dots, \rho_L\};\qquad "
         r"\rho = x_i\cos\theta_k + y_i\sin\theta_k \;\Rightarrow\; A[k, l] \leftarrow A[k, l] + 1;\qquad "
         r"(\theta^*, \rho^*) = \arg\max A")
ui.anim_intro("Step by step: points vote one after another")
st.plotly_chart(replay_figure(), key="hn_replay")

# ------------------------------------------------------------------ live accumulator
LINES = [(30.0, 90.0), (100.0, 60.0), (160.0, -60.0)]


@live.section
def accumulator_section():
    st.subheader("The accumulator, live - and OpenCV on the same points", divider="gray")
    c_ctrl, c_fig = st.columns([0.8, 3], gap="large")
    with c_ctrl:
        n_lines = live.slider("lines in the scene", 1, 3, 2, 1, key="hn_lines")
        n_pts = live.slider("points per line", 5, 120, 50, 5, key="hn_pts")
        noise = live.slider("position noise σ (px)", 0.0, 4.0, 0.8, 0.1, key="hn_noise")
        n_out = live.slider("outlier points", 0, 300, 60, 10, key="hn_out")
        n_theta = live.slider("θ bins (over 180°)", 18, 360, 180, 6, key="hn_ntheta")
        rho_res = live.slider("ρ bin size (px)", 0.5, 8.0, 1.0, 0.5, key="hn_rho")
    P = np.vstack([Hh.line_points(th, r, W, H, n_pts, noise, seed=i) for i, (th, r) in enumerate(LINES[:n_lines])]
                  + [Hh.clutter(n_out, W, H, seed=7)])
    A, ths, rhos = Hh.accumulator_rt(P, W, H, n_theta, rho_res)
    nms = (max(3, int(round(12 / rho_res)) | 1), max(3, int(round(n_theta / 18)) | 1))
    pk = Hh.find_peaks(A, n_lines, threshold=3, nms=nms, wrap_cols=True)
    found = [(float(ths[c]), float(rhos[r]), v) for r, c, v in pk]
    cv = Hh.cv_hough_lines(P, W, H, rho_res, n_theta, threshold=max(5, n_pts // 3), k=n_lines)
    fig = cui.two_panel("image space (y down)", "accumulator A[ρ, θ]", height=470, widths=(0.42, 0.58),
                        left_range=(-5, W + 5, -5, H + 5), left_ydown=True, right_range=(0, 180, -D, D),
                        left_title_x="x", left_title_y="y", right_title_x="θ (degrees)", right_title_y="ρ (pixels)")
    lx, ly = [], []
    for th, rho, _ in found:
        seg = Hh.clip_line(th, rho, W, H)
        if seg is not None:
            lx += [seg[0][0], seg[1][0], np.nan]
            ly += [seg[0][1], seg[1][1], np.nan]
    fig.add_traces([
        cui.heat_trace(np.sqrt(A), ths, rhos, xaxis="x2", yaxis="y2", hover="θ=%{x:.1f}°, ρ=%{y:.1f}: √votes %{z:.1f}"),
        cui.scatter([f[0] for f in found], [f[1] for f in found], theme.HIGHLIGHT, size=16, symbol="circle-open",
                    xaxis="x2", yaxis="y2").update(marker=dict(line=dict(width=2.5, color=theme.HIGHLIGHT))),
        cui.scatter(P[:, 0], P[:, 1], theme.ORIGINAL, size=4, xaxis="x", yaxis="y"),
        cui.scatter(lx, ly, theme.HIGHLIGHT, mode="lines", width=2.5, xaxis="x", yaxis="y"),
    ])
    rows = []
    for i, (th, r) in enumerate(LINES[:n_lines]):
        best = min(found, key=lambda f: Hh.angle_error(f[0], th) + abs(f[1] - r) / 10, default=None)
        best_cv = min(cv, key=lambda f: Hh.angle_error(f[1], th) + abs(f[0] - r) / 10, default=None)
        rows.append({"line": i + 1, "true (θ°, ρ)": f"({th:.0f}, {r:.0f})",
                     "our peak": f"({best[0]:.1f}, {best[1]:.1f}) · {best[2]} votes" if best else "-",
                     "angle error": f"{Hh.angle_error(best[0], th):.2f}°" if best else "-",
                     "cv2.HoughLines": f"({best_cv[1]:.1f}, {best_cv[0]:.1f})" if best_cv else "-"})
    with c_fig:
        st.plotly_chart(fig, key="hn_acc")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=143)   # fixed: 3 rows max
    st.caption("Heat = √votes (so weak structure stays visible). Each line is a bright crossing of many sinusoids. "
               "Coarse bins: blurry but robust peaks; fine bins: sharp but split peaks when the points are noisy. "
               f"Peak picking uses non-maximum suppression in a {nms[0]}×{nms[1]} window (θ wraps around).")


accumulator_section()


# ------------------------------------------------------------------ real images
@st.cache_resource(show_spinner=False, max_entries=12)
def edge_points(name: str, sigma: float):
    img = E.SAMPLES[name]()
    R = E.canny(img, sigma, 0.1, 0.25)
    ys, xs = np.nonzero(R["edges"])
    return img, R["edges"], np.c_[xs, ys].astype(float)


@live.section
def real_section():
    st.subheader("On a real image", divider="gray")
    c_ctrl, c1, c2 = st.columns([0.8, 1.5, 1.5], gap="large")
    names = [n for n in E.SAMPLES if n != "Coins (photo)"]
    with c_ctrl:
        name = st.selectbox("Image", names, index=names.index("Shapes (synthetic)"), key="hn_img")
        sigma = live.slider("Canny σ", 0.8, 4.0, 1.6, 0.1, key="hn_sigma")
        k = live.slider("lines to report (top-k peaks)", 1, 15, 6, 1, key="hn_k")
    img, Ed, P = edge_points(name, round(sigma, 1))
    h, w = img.shape
    A, ths, rhos = Hh.accumulator_rt(P, w, h, 180, 1.0)
    pk = Hh.find_peaks(A, k, threshold=10, nms=(15, 9), wrap_cols=True)
    lines = [(float(rhos[r]), float(ths[c])) for r, c, v in pk]
    vis = Hh.draw_lines_rgb(E.overlay(img, Ed, color=K.BLUE), lines, color=K.AMBER, thickness=1)
    with c1:
        st.image(ui.datauri(vis), caption=f"Canny edges (blue, {len(P):,} points) and the {len(lines)} strongest Hough "
                 "lines (amber)", width="stretch")
    with c2:
        fig = go.Figure(data=[cui.heat_trace(np.sqrt(A), ths, rhos, hover="θ=%{x:.0f}°, ρ=%{y:.0f}: √votes %{z:.1f}"),
                              cui.scatter([ths[c] for _, c, _ in pk], [rhos[r] for r, _, _ in pk], theme.HIGHLIGHT,
                                          size=13, symbol="circle-open").update(
                                  marker=dict(line=dict(width=2, color=theme.HIGHLIGHT)))],
                        layout=dict(height=360, margin=dict(l=8, r=8, t=10, b=8), showlegend=False,
                                    xaxis=dict(title="θ (degrees)"), yaxis=dict(title="ρ (pixels)")))
        st.plotly_chart(fig, key="hn_real_acc")
    st.caption("Lines are infinite: Hough finds the line, not where the segment starts and ends. "
               "cv2.HoughLinesP (probabilistic Hough) adds that step and returns segments.")


real_section()


# ------------------------------------------------------------------ measured: bin size vs accuracy
@st.cache_resource(show_spinner=False)
def bin_study():
    nts = [18, 30, 45, 60, 90, 120, 180, 270, 360, 540, 720]
    rows = []
    for noise in (0.0, 1.5):
        for nt in nts:
            errs, peaks = [], []
            for s in range(12):
                th = 20 + 11.3 * s
                P = Hh.line_points(th, 40 + 3 * s, W, H, 80, noise, seed=s)
                A, ths, rhos = Hh.accumulator_rt(P, W, H, nt, 1.0)
                r, c = np.unravel_index(int(np.argmax(A)), A.shape)
                errs.append(Hh.angle_error(ths[c], th))
                peaks.append(A.max() / 80)
            rows.append(dict(noise=noise, n_theta=nt, err=float(np.mean(errs)), peak=float(np.mean(peaks))))
    return pd.DataFrame(rows)


st.subheader("How fine should the θ bins be? (measured)", divider="gray")
bs = bin_study()
b1, b2 = st.columns(2, gap="large")
for col, key, ytitle in [(b1, "err", "mean angle error (°)"), (b2, "peak", "peak votes / points on the line")]:
    fig = go.Figure(layout=dict(height=300, margin=dict(l=8, r=8, t=30, b=8), hovermode="x unified",
                                legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0),
                                xaxis=dict(type="log", title="θ bins over 180°"), yaxis=dict(title=ytitle)))
    for noise, c in ((0.0, theme.TRANSFORMED), (1.5, theme.OUTLIER)):
        d = bs[bs.noise == noise]
        fig.add_trace(cui.scatter(d.n_theta, d[key], c, mode="lines+markers", size=6, width=2.5,
                                  name=f"noise σ = {noise} px", showlegend=True, hover=True))
    col.plotly_chart(fig, key=f"hn_bins_{key}")
st.caption("12 random lines of 80 points each. Left: finer θ bins → smaller angle error, until the noise is the limit. "
           "Right: finer bins also SPLIT the votes - with noise, the peak holds a smaller share of the points and gets "
           "harder to tell from clutter. The bin size is a trade-off between precision and robustness.")

ui.callout("intuition", "In (θ, ρ) every point draws a wave. Points on one line all draw waves that cross at one spot. "
                        "Many waves crossing = bright spot = line.")
ui.callout("pitfall", "θ ∈ [0°, 180°) and ρ can be NEGATIVE. The same line written with θ + 180° and −ρ is identical, "
                      "so a peak near θ = 0° may have a twin near θ = 180° on the other side of the accumulator - "
                      "peak picking must wrap around.")
ui.callout("takeaway", "Normal form: x cos θ + y sin θ = ρ. Point → sinusoid. Line → crossing of sinusoids → peak in "
                       "the accumulator A[ρ, θ]. Bounded parameters, vertical lines included.")

ui.whys([
    ("Why sinusoids?",
     "For a fixed point, ρ(θ) = x cos θ + y sin θ is a sum of a cosine and a sine with the same frequency, which is "
     "always ONE shifted cosine: √(x² + y²) cos(θ − atan2(y, x)). Its amplitude is the point's distance from the "
     "origin, its phase the point's direction."),
    ("Why is θ only from 0° to 180°?",
     "Because (θ + 180°, −ρ) describes the same line as (θ, ρ): cos and sin both flip sign. Allowing negative ρ with "
     "θ ∈ [0°, 180°) covers every line exactly once. (Alternatively θ ∈ [0°, 360°) with ρ ≥ 0.)"),
    ("Why are the peaks shaped like a butterfly?",
     "Near the crossing, the sinusoids of a line's points spread out at different slopes (points far apart along the "
     "line have different phases). Votes therefore spill into a bow-tie around the peak. NMS keeps only the top."),
    ("Does it matter where the origin is?",
     "The lines found are the same, but the ρ range is not: with the origin at the top-left ρ ∈ [−D, D]; with the "
     "origin at the image centre |ρ| ≤ D/2, so half the accumulator rows. OpenCV uses the top-left."),
    ("Why does noise lower the peak?",
     "A noisy point's sinusoid misses the exact crossing cell by a little, so the votes spread over neighbouring "
     "cells (measured above). Bigger bins collect them again - at the price of angular precision."),
    ("How does it find SEGMENTS, not infinite lines?",
     "Plain Hough only gives (θ, ρ). cv2.HoughLinesP votes with a random subset of points and then walks along each "
     "detected line in the edge image to find where the supporting pixels start and stop."),
])

st.divider()
ui.mcq("hn1", "In (θ, ρ) space, a single edge point becomes …",
       ["a straight line", "a sinusoid", "a circle", "a single cell"], 1,
       "ρ(θ) = x cos θ + y sin θ is a sinusoid of amplitude √(x² + y²).")
ui.mcq("hn2", "Which line has θ = 0° in the normal form?",
       ["A horizontal line", "A vertical line x = ρ", "The line through the origin at 45°", "None"], 1,
       "θ = 0 gives x cos 0 + y sin 0 = x = ρ: a vertical line.")

ui.lesson_nav(__file__)
