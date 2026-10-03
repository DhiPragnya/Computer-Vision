import cv2
import numpy as np
import streamlit as st

from core import anim, live, theme, ui
from core import contour_ui as cui
from core import contours as K
from core import hough as Hh

ui.lesson_header(
    __file__,
    "Contour tracing needs connected pixels. Real edges are broken by noise and occlusion. The <b>Hough transform</b> "
    "turns the question around: instead of following pixels, let every edge point <b>vote</b> for all the lines it "
    "could belong to, and look for the line with the most votes. One point in the image becomes one line in "
    "parameter space - and collinear points become lines that meet.",
)
MS = np.linspace(-3, 3, 121)
CS = np.linspace(-6, 14, 161)
MRANGE, CRANGE = (-3, 3), (-6, 14)


# ------------------------------------------------------------------ motivation: a broken line
@live.section
def broken_section():
    st.subheader("Why contours are not enough: a broken line", divider="gray")
    c_ctrl, c_img, c_txt = st.columns([0.8, 1.6, 1.1], gap="large")
    with c_ctrl:
        n_gaps = live.slider("number of gaps", 0, 5, 3, 1, key="hm_gaps")
        gap = live.slider("gap length (pixels)", 2, 30, 12, 1, key="hm_gaplen")
    h, w = 120, 220
    img = np.zeros((h, w), np.uint8)
    p0, p1 = np.array([12.0, 100.0]), np.array([208.0, 22.0])
    cv2.line(img, tuple(int(v) for v in p0), tuple(int(v) for v in p1), 1, 2)
    for k in range(n_gaps):
        t = (k + 1) / (n_gaps + 1)
        c = p0 + t * (p1 - p0)
        cv2.circle(img, (int(c[0]), int(c[1])), gap // 2, 0, -1)
    B = img > 0
    cs, _ = K.find_contours(B, cv2.RETR_EXTERNAL)
    ys, xs = np.nonzero(B)
    A, th, rh = Hh.accumulator_rt(np.c_[xs, ys], w, h, 180, 1.0)
    pk = Hh.find_peaks(A, 1, threshold=10)
    vis = K.draw_contours(K.binary_rgb(B), cs, K.PALETTE, scale=3, thickness=2)
    v = 0
    if pk:
        r, c, v = pk[0]
        seg = Hh.clip_line(th[c], rh[r], w - 1, h - 1)
        if seg is not None:
            (x0, y0), (x1, y1) = seg
            cv2.line(vis, (int((x0 + 0.5) * 3), int((y0 + 0.5) * 3)), (int((x1 + 0.5) * 3), int((y1 + 0.5) * 3)),
                     K.AMBER, 1, cv2.LINE_AA)
    with c_img:
        st.image(ui.datauri(vis), caption="each colour = one contour found by cv2.findContours · thin amber line = the "
                 "Hough peak", width="stretch")
    with c_txt:
        st.metric("contours (pieces)", len(cs), border=True)
        st.metric("Hough: strongest line", f"{v} votes", border=True,
                  help="votes = edge pixels that lie on that line")
        st.caption("Contour tracing sees one object per fragment. Voting does not care about gaps: every visible pixel "
                   "still lies on the same line, so all fragments vote for the same parameters.")


broken_section()
st.latex(r"\text{Image space} \;\longrightarrow\; \text{Parameter space} \;\longrightarrow\; \text{Voting} "
         r"\;\longrightarrow\; \text{Peaks}")

# ------------------------------------------------------------------ point -> line
st.subheader("One point, all the lines through it", divider="gray")
d1, d2 = st.columns([1, 1], gap="large")
with d1:
    st.latex(r"y = m x + c \qquad \Rightarrow \qquad \text{a line} \leftrightarrow (m, c)")
    st.latex(r"y_i = m x_i + c \;\;\Longleftrightarrow\;\; c = y_i - m x_i")
with d2:
    st.markdown("For a fixed edge point (x_i, y_i), **every line through it** has some slope m and intercept c with "
                "c = y_i − m x_i. As m varies, c changes **linearly** in m: in the (m, c) plane that is a straight line. "
                "**One point in image space ⟷ one line in parameter space.**")

PTS = np.array([[2.0, 3.0], [4.0, 4.0], [6.0, 5.0], [8.0, 6.0]])      # on y = 0.5 x + 2
OUT = np.array([[3.0, 7.5]])
M_TRUE, C_TRUE = 0.5, 2.0


@st.cache_resource(show_spinner=False)
@live.full_quality()
def voting_figure():
    allp = np.vstack([PTS, OUT])
    n = len(allp)
    ms_c = np.linspace(-3, 3, 61)
    cs_c = np.linspace(-6, 14, 81)
    fan_m = np.array([-1.5, -0.5, 0.5, 1.5, 2.5])

    def param_line(p, f):
        """c = y − m x for m from −3 to −3 + 6 f (grows from the left)."""
        m = np.array([-3.0, -3.0 + 6.0 * f])
        return m, p[1] - m * p[0]

    def state(sizes, line_f, fan_at=None, fan_on=0.0, peak=0.0, heat_k=0, lw=2.0):
        lx, ly = [], []
        for i in range(n):
            m, c = param_line(allp[i], line_f[i])
            lx += [m[0], m[1], np.nan]
            ly += [c[0], c[1], np.nan]
        fx, fy, dx, dy = [], [], [], []
        p = allp[fan_at if fan_at is not None else 0]
        for m in fan_m:
            x0, x1 = p[0] - fan_on * 3, p[0] + fan_on * 3
            fx += [x0, x1, np.nan]
            fy += [p[1] + m * (x0 - p[0]), p[1] + m * (x1 - p[0]), np.nan]
            dx.append(m)
            dy.append(p[1] - m * p[0])
        Acc = Hh.accumulator_mc(allp[:heat_k], ms_c, cs_c).astype(np.uint8) if heat_k else \
            np.zeros((len(cs_c), len(ms_c)), np.uint8)
        det = np.array([[0.0, C_TRUE], [10.0, M_TRUE * 10 + C_TRUE]])
        det = det[:1] + peak * (det - det[:1])
        return [
            cui.heat_trace(Acc, ms_c, cs_c, zmax=5, xaxis="x2", yaxis="y2", hover="m=%{x:.2f}, c=%{y:.2f}: %{z} votes"),
            cui.scatter(fx, fy, theme.ORIGINAL, mode="lines", width=1.5, xaxis="x", yaxis="y"),
            cui.scatter(det[:, 0], det[:, 1], theme.HIGHLIGHT, mode="lines", width=3, xaxis="x", yaxis="y"),
            cui.scatter(allp[:, 0], allp[:, 1], [theme.TRANSFORMED] * len(PTS) + [theme.OUTLIER], mode="markers",
                        xaxis="x", yaxis="y").update(marker=dict(size=sizes, color=[theme.TRANSFORMED] * len(PTS) +
                                                                 [theme.OUTLIER], line=dict(color="white", width=1))),
            cui.scatter(lx, ly, theme.TRANSFORMED, mode="lines", width=lw, xaxis="x2", yaxis="y2"),
            cui.scatter(dx, dy, theme.ORIGINAL, size=8 * fan_on, xaxis="x2", yaxis="y2"),
            cui.scatter([M_TRUE], [C_TRUE], theme.HIGHLIGHT, size=18 * peak, symbol="star", xaxis="x2", yaxis="y2"),
        ]

    z = np.zeros(n)
    big = 13.0

    def sz(k):
        return np.array([big if i < k else 0.0 for i in range(n)])

    def lf(k, last=1.0):
        return np.array([1.0 if i < k - 1 else (last if i == k - 1 else 0.0) for i in range(n)])

    steps = [
        anim.Step("One edge point", f"(x₁, y₁) = ({PTS[0][0]:g}, {PTS[0][1]:g})",
                  "We don't know which line it lies on. So consider ALL of them: here are five.",
                  state(sz(1), z, 0, 1.0)),
        anim.Step("Each line through it is one dot in (m, c)", "c = y₁ − m x₁   for every slope m",
                  "Five lines → five dots. All of them lie on ONE straight line in parameter space.",
                  state(sz(1), lf(1), 0, 1.0), between=lambda t: state(sz(1), lf(1, t), 0, 1.0)),
    ]
    for k in range(2, len(PTS) + 1):
        steps.append(anim.Step(f"Point {k} draws its own line", f"c = y{k} − m x{k}  =  {PTS[k - 1][1]:g} − "
                               f"{PTS[k - 1][0]:g} m", "Every point of the image becomes a line in (m, c).",
                               state(sz(k), lf(k), k - 1, 0.0), between=lambda t, k=k: state(sz(k), lf(k, t), k - 1, 0.0)))
    steps.append(anim.Step("Collinear points → lines that meet", f"(m*, c*) = ({M_TRUE:g}, {C_TRUE:g}):  "
                           f"{anim.hl('y = 0.5 x + 2')}", "The meeting point IS the line through all four points.",
                           state(sz(len(PTS)), lf(len(PTS)), None, 0.0, 1.0)))
    steps.append(anim.Step("An outlier", f"c = 7.5 − 3 m", "Its line passes elsewhere: it does not disturb the meeting "
                           "point, it just adds one vote somewhere else.", state(sz(n), lf(n), None, 0.0, 1.0),
                           between=lambda t: state(sz(n), lf(n, t), None, 0.0, 1.0)))
    steps.append(anim.Step("Voting in a grid of (m, c) cells", "A(m, c) ← A(m, c) + 1  along every line",
                           f"Brightest cell = most lines through it = {anim.hl('4 votes')} at (0.5, 2). The outlier "
                           "cell has 1.", state(sz(n), lf(n), None, 0.0, 0.0, heat_k=n, lw=0.6)))
    base = cui.two_panel("", "", height=470, left_range=(0, 10, 0, 10), right_range=(-3, 3, -6, 14),
                         left_title_x="x   ·   IMAGE SPACE", left_title_y="y",
                         right_title_x="slope m   ·   PARAMETER SPACE", right_title_y="intercept c")
    return anim.step_player(base, steps, n_sub=12, hold=16, wrap_at=105)


ui.anim_intro("Step by step: points become lines, collinear points become a meeting point")
st.plotly_chart(voting_figure(), key="hm_vote")


# ------------------------------------------------------------------ live explorer
@live.section
def explore_section():
    st.subheader("Accumulator voting, live", divider="gray")
    st.latex(r"A(m, c) = 0;\qquad \text{for every edge point: } A(m, y_i - m x_i) \leftarrow A(m, y_i - m x_i) + 1;"
             r"\qquad (m^*, c^*) = \arg\max_{m, c} A(m, c)")
    c_ctrl, c_fig = st.columns([0.8, 3], gap="large")
    with c_ctrl:
        m_true = live.slider("true slope m", -2.0, 2.0, 0.6, 0.05, key="hm_m")
        c_true = live.slider("true intercept c", 0.0, 8.0, 2.0, 0.1, key="hm_c")
        n = live.slider("points on the line", 2, 30, 12, 1, key="hm_n")
        noise = live.slider("noise σ", 0.0, 0.5, 0.05, 0.01, key="hm_noise")
        n_out = live.slider("outliers", 0, 20, 5, 1, key="hm_out")
    rng = np.random.default_rng(1)
    x = np.linspace(0.8, 9.2, n)
    P = np.c_[x, m_true * x + c_true] + rng.normal(0, noise, (n, 2)) * (noise > 0)
    P = P[(P[:, 1] >= 0) & (P[:, 1] <= 10)]
    O = rng.uniform(0.5, 9.5, (n_out, 2))
    allp = np.vstack([P, O]) if len(O) else P
    A = Hh.accumulator_mc(allp, MS, CS)
    i, j = np.unravel_index(int(np.argmax(A)), A.shape)
    m_s, c_s = MS[j], CS[i]
    fig = cui.two_panel("image space (x, y)", "parameter space A(m, c)", height=480, left_range=(0, 10, 0, 10),
                        right_range=(-3, 3, -6, 14), left_title_x="x", left_title_y="y", right_title_x="slope m",
                        right_title_y="intercept c")
    lx, ly = [], []
    for p in allp:
        lx += [MS[0], MS[-1], np.nan]
        ly += [p[1] - MS[0] * p[0], p[1] - MS[-1] * p[0], np.nan]
    xx = np.array([0, 10.0])
    fig.add_traces([
        cui.heat_trace(A, MS, CS, xaxis="x2", yaxis="y2", hover="m=%{x:.2f}, c=%{y:.2f}: %{z} votes"),
        cui.scatter(lx, ly, theme.rgba(theme.TRANSFORMED, 0.35), mode="lines", width=1, xaxis="x2", yaxis="y2"),
        cui.scatter([m_s], [c_s], theme.HIGHLIGHT, size=16, symbol="star", xaxis="x2", yaxis="y2"),
        cui.scatter(xx, m_true * xx + c_true, theme.ORIGINAL, mode="lines", dash="dash", width=1.5, xaxis="x", yaxis="y"),
        cui.scatter(xx, m_s * xx + c_s, theme.HIGHLIGHT, mode="lines", width=3, xaxis="x", yaxis="y"),
        cui.scatter(P[:, 0], P[:, 1], theme.TRANSFORMED, size=8, xaxis="x", yaxis="y"),
        cui.scatter(O[:, 0], O[:, 1], theme.OUTLIER, size=8, symbol="x", xaxis="x", yaxis="y"),
    ])
    with c_fig:
        st.plotly_chart(fig, key="hm_fig")
        a, b, c = st.columns(3)
        a.metric("peak (m*, c*)", f"({m_s:.2f}, {c_s:.2f})", border=True)
        b.metric("true (m, c)", f"({m_true:.2f}, {c_true:.2f})", border=True)
        c.metric("votes at the peak", int(A.max()), f"of {len(allp)} points", delta_color="off", border=True)
    st.caption("Blue dots = points on the line, red × = outliers, dashed = the true line, amber = the line with the most "
               "votes. Raise the outliers: the peak does not move - outliers spread their votes everywhere else.")


explore_section()


# ------------------------------------------------------------------ the problem: vertical lines
@st.cache_resource(show_spinner=False)
@live.full_quality()
def vertical_figure():
    cx, cy = 5.0, 5.0
    s = np.linspace(-4, 4, 6)
    phis = list(np.r_[np.linspace(0, 80, 17), np.linspace(82, 89.5, 8)])
    base = cui.two_panel("", "", height=470, left_range=(0, 10, 0, 10), right_range=(-3, 3, -6, 14),
                         left_title_x="x   ·   IMAGE SPACE: a line through (5, 5)", left_title_y="y",
                         right_title_x="slope m   ·   PARAMETER SPACE (dotted box = accumulator)",
                         right_title_y="intercept c")
    base.add_shape(type="rect", x0=-3, x1=3, y0=-6, y1=14, xref="x2", yref="y2",
                   line=dict(color=theme.HIGHLIGHT, width=2, dash="dot"))
    dm = MS[1] - MS[0]

    def make_state(phi):
        t = np.deg2rad(phi)
        d = np.array([np.cos(t), np.sin(t)])
        P = np.array([cx, cy]) + s[:, None] * d
        m = np.tan(t)
        c = cy - m * cx
        inside = abs(m) <= 3 and -6 <= c <= 14
        lx, ly = [], []
        for p in P:
            lx += [-3, 3, np.nan]
            ly += [p[1] + 3 * p[0], p[1] - 3 * p[0], np.nan]
        mk = (m, c) if inside else (np.clip(m, -3, 3), np.clip(c, -6, 14))
        dtheta = np.degrees(np.arctan(m + dm) - np.arctan(m))
        cap = (f"angle {phi:.1f}°  ·  m = tan(angle) = <b>{m:.2f}</b>, c = {c:.1f}  ·  " +
               (f"one m-bin of width {dm:.2f} spans {dtheta:.2f}° here" if inside else
                f"<b style='color:{theme.OUTLIER}'>the peak is OFF the accumulator: these votes are lost</b>"))
        L = np.array([[cx, cy] - 8 * d, [cx, cy] + 8 * d])
        return [cui.scatter(L[:, 0], L[:, 1], theme.ORIGINAL, mode="lines", dash="dash", width=1.5, xaxis="x", yaxis="y"),
                cui.scatter(P[:, 0], P[:, 1], theme.TRANSFORMED, size=10, xaxis="x", yaxis="y"),
                cui.scatter(lx, ly, theme.rgba(theme.TRANSFORMED, 0.5), mode="lines", width=1.5, xaxis="x2", yaxis="y2"),
                cui.scatter([mk[0]], [mk[1]], theme.HIGHLIGHT if inside else theme.OUTLIER, size=18,
                            symbol="star" if inside else "x", xaxis="x2", yaxis="y2")], cap

    return anim.scrubber(base, phis, make_state, active=4, label="angle", fmt="{:.1f}°", frame_ms=150, wrap_at=110)


st.subheader("The problem with y = m x + c: vertical lines", divider="gray")
ui.anim_intro("Sweep: tilt the line towards vertical",
              "▶ Play or drag the angle. Watch the meeting point (★) run off the accumulator (dotted box).")
st.plotly_chart(vertical_figure(), key="hm_vert")
st.latex(r"x = c \;\;(\text{vertical line}) \quad\Rightarrow\quad m \to \infty")
st.markdown("A vertical line has no slope-intercept form at all, and nearly vertical lines need huge |m| and |c|. An "
            "accumulator is a finite grid, so those lines are simply **lost**. And equal steps in m are very unequal "
            "steps in angle. Lesson 8 fixes both with the **normal form** x cos θ + y sin θ = ρ, where every parameter "
            "lives in a bounded range.")

ui.callout("intuition", "Each edge point says 'I could be on any of these lines' and puts one vote on each. Noise points "
                        "vote all over the place; the points of a real line all vote for the same cell. The pile of "
                        "votes is the detection.")
ui.callout("pitfall", "Least-squares line fitting lets every point pull on the answer - one outlier can tilt the line. "
                      "Voting lets outliers vote somewhere else, and it can find several lines at once (several peaks).")
ui.callout("takeaway", "Point (x_i, y_i) in the image ⟷ line c = y_i − m x_i in (m, c). Collinear points ⟷ lines "
                       "through one point. The accumulator turns 'lines meeting' into 'a cell with many votes'.")

ui.whys([
    ("Why vote instead of fitting a line by least squares?",
     "Least squares needs to know WHICH points belong to the line and gives each one a pull on the answer; outliers "
     "tilt it. Voting needs no grouping: every point votes, and only the agreement of many points makes a peak. "
     "Several lines simply give several peaks."),
    ("Why does a point become a LINE in (m, c)?",
     "Because x_i and y_i are fixed numbers and c = y_i − m x_i is linear in the unknowns m and c. The roles of "
     "'variables' and 'constants' swap - this swap is called duality."),
    ("Why is an unbounded m a problem for a discrete accumulator?",
     "The accumulator is an array: it needs a finite range and a cell size. Near-vertical lines have arbitrarily large "
     "m (and c), so no finite array holds them all. And equal steps in m mean wildly unequal steps in angle (see the "
     "sweep), so the resolution is poor exactly where m is small."),
    ("What happens to a vote that falls outside the accumulator?",
     "It is dropped. With slope-intercept form that happens for every steep line - which is why OpenCV and the notes "
     "use the normal form (next lesson)."),
    ("Could I use x = m′ y + c′ for steep lines instead?",
     "Yes - two accumulators, one for 'mostly horizontal' and one for 'mostly vertical' lines, works. The normal form "
     "does the same job more elegantly with a single bounded accumulator."),
    ("How many points does it take to define a peak?",
     "Two points already make two lines that meet. In practice set a threshold (e.g. 20 votes for a line of at least "
     "20 edge pixels) so that random coincidences of a few noise points do not count."),
])

st.divider()
ui.mcq("hm1", "An edge point (x_i, y_i) corresponds in (m, c) space to …",
       ["a point", "the line c = y_i − m x_i", "a circle", "a sinusoid"], 1,
       "All lines through the point satisfy c = y_i − m x_i: a straight line in (m, c).")
ui.mcq("hm2", "Why does the slope-intercept accumulator fail for vertical lines?",
       ["They have no edge points", "Their slope m is infinite, outside any finite accumulator",
        "They vote twice", "Their intercept is zero"], 1, "x = c cannot be written as y = m x + c; m → ∞.")

ui.lesson_nav(__file__)
