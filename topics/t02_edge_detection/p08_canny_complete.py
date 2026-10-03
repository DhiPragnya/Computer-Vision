import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core import edge_ui as eui
from core import edges as E
from core import anim, live, theme, ui

ui.lesson_header(
    __file__,
    "Put the five stages together and you have the Canny detector. This lesson runs the whole pipeline, shows "
    "<b>why</b> a derivative of a Gaussian is the right filter (Canny's three criteria, simulated), and compares "
    "Canny with Sobel and the LoG on the same image.",
)

@st.cache_resource(show_spinner=False, max_entries=24)
def canny_both(img: np.ndarray, sg: float, lo: float, hi: float, nms: str):
    R = E.canny(img, sg, lo, hi, nms=nms)
    cvE = E.cv_canny(img, sg, lo, hi)
    return R, cvE, E.match_f1(R["edges"], cvE, 1)


@st.cache_resource(show_spinner=False)
@live.full_quality()          # cached: never store a coarse mid-drag version
def stages_figure():
    """The five stages on one crop, morphing from one to the next."""
    img = E.SAMPLES["Coffee (photo)"]()
    R = E.canny(img, 1.6, 0.08, 0.2)
    h, w = img.shape
    y0, x0 = max(0, h // 2 - 70), max(0, w // 2 - 70)
    cut = lambda A: A[y0:y0 + 120, x0:x0 + 120]
    u8 = lambda A, hi_: np.clip(np.round(cut(A) / (hi_ + 1e-12) * 255), 0, 255).astype(np.uint8)
    cls = cut(R["classes"]).astype(float)
    layers = [
        u8(img, 1.0), u8(R["smoothed"], 1.0), u8(R["M"], np.percentile(R["M"], 99.5)),
        u8(R["Mn"], np.percentile(R["Mn"][R["Mn"] > 0], 99)),
        np.where(cls == E.STRONG, 255, np.where(cls == E.WEAK, 140, 0)).astype(np.uint8),
        np.where(cut(R["edges"]), 255, 0).astype(np.uint8),
    ]
    info = [("Input", "I", "A 120 × 120 crop of the coffee photo."),
            ("1 · Gaussian smoothing", "I_s = G_σ * I,  σ = 1.6", "Noise and fine texture are averaged away."),
            ("2 · Gradient magnitude", "M = √(I_x² + I_y²)", "Bright = steep change. Every edge is a thick band."),
            ("3 · Non-maximum suppression", "keep M only where it is a maximum across the edge",
             "The bands shrink to one-pixel ridges."),
            ("4 · Double threshold", "strong (white) ≥ T_high,  weak (grey) ≥ T_low",
             "Ridges are sorted into sure, maybe and no."),
            ("5 · Hysteresis", "keep weak pixels connected to strong ones",
             "The final edge map: thin, connected contours, specks removed.")]
    steps = [anim.Step(t_, e_, n_, [go.Heatmap(z=L, zmin=0, zmax=255, colorscale=eui.GREYS, showscale=False,
                                               hoverinfo="skip")])
             for L, (t_, e_, n_) in zip(layers, info)]
    base = go.Figure()
    base.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1, visible=False, constrain="domain")
    base.update_xaxes(visible=False, constrain="domain")
    base.update_layout(height=560, margin=dict(l=0, r=0))
    return anim.step_player(base, steps, n_sub=8, hold=22, wrap_at=70)


# ------------------------------------------------------------------ full pipeline
@live.section
def pipeline_section():
    st.subheader("The complete pipeline", divider="gray")
    st.latex(r"I \xrightarrow{G_\sigma *} I_s \xrightarrow{\partial_x,\,\partial_y} (I_x, I_y) \rightarrow (M, \theta)"
             r"\xrightarrow{\text{NMS}} M_N \xrightarrow{\text{double threshold}} \{S, W, N\} \xrightarrow{\text{hysteresis}}"
             r" E_{\text{final}}")
    c_ctrl, c_main = st.columns([0.8, 3], gap="large")
    with c_ctrl:
        img, clean, _ = eui.pick_image("cc", "Coffee (photo)", noise_default=0.0)
        sg = live.slider("σ", 0.5, 5.0, 1.6, 0.1, key="cc_s")
        lo, hi = live.slider("T_low, T_high (fraction of max)", 0.0, 1.0, (0.08, 0.2), 0.01, key="cc_t")
        nms = st.radio("NMS", ["quantized", "interpolated"], horizontal=True, key="cc_nms")
    R, cvE, (p_, r_, f_) = canny_both(img, sg, lo, hi, nms)
    with c_main:
        top = st.columns(3)
        top[0].image(ui.datauri(np.clip(R["smoothed"], 0, 1)), caption="1 · smoothed", width="stretch")
        top[1].image(ui.datauri(E.to_u8(R["M"], 0, np.percentile(R["M"], 99.7))), caption="2 · magnitude M",
                     width="stretch")
        top[2].image(ui.datauri(E.to_u8(R["Mn"], 0, np.percentile(R["Mn"][R["Mn"] > 0], 99) if (R["Mn"] > 0).any()
                                         else 1)), caption="3 · after NMS", width="stretch")
        bot = st.columns(3)
        bot[0].image(ui.datauri(E.classes_rgb(R["classes"])), caption="4 · strong (white) / weak (amber)",
                     width="stretch")
        bot[1].image(ui.datauri(E.edges_rgb(R["edges"])), caption="5 · after hysteresis = Canny edges", width="stretch")
        bot[2].image(ui.datauri(E.edges_rgb(cvE)), caption=f"cv2.Canny, same settings · agreement F1 = {f_:.2f}",
                     width="stretch")
    st.caption("The from-scratch pipeline on this site and OpenCV's optimised C++ version agree closely; small "
               "differences come from rounding to 8-bit and tie-breaking in NMS.")

    st.subheader("σ is the scale of the edges you want", divider="gray")
    rows = [(E.edges_rgb(canny_both(img, s_, lo, hi, nms)[0]["edges"]), f"Canny, σ = {s_:g}") for s_ in (1.0, 2.0, 4.0)]
    eui.show_row(rows)


pipeline_section()

ui.anim_intro("Step by step: the five stages on one picture",
              "A fixed crop with the default settings (σ = 1.6, T_low = 0.08, T_high = 0.2). Each step morphs into the "
              "next stage, so you can see exactly what that stage adds or removes.")
s1, s2 = st.columns([1.3, 1], gap="large")
with s1:
    st.plotly_chart(stages_figure(), key="cc_stages")
with s2:
    ui.callout("intuition",
               "Two stages *remove* things (smoothing removes noise, NMS removes the shoulders of every edge), two "
               "*decide* (the thresholds), and only one stage *creates* new information - the gradient. Scrub between "
               "steps 2 and 3 to see how much NMS throws away.")


# ------------------------------------------------------------------ the three criteria
def dog_filter(s):
    r = int(np.ceil(4 * s))
    t = np.arange(-r, r + 1)
    f = t * np.exp(-t ** 2 / (2 * s ** 2))                   # derivative of Gaussian (correlation form)
    return f / np.abs(f).sum()


@st.cache_data(show_spinner=False)
def simulate(contrast_over_noise: float, crowded: bool, seed: int = 0):
    rng = np.random.default_rng(seed)
    xs = np.arange(-60, 61)
    sig = (xs >= 0).astype(float)
    if crowded:
        sig = sig - (xs >= 8).astype(float)                  # a bright bar 8 px wide: edges at 0 and 8
    sig = sig * contrast_over_noise
    widths = np.round(np.arange(0.5, 8.01, 0.5), 2)
    out = {"s": widths, "snr": [], "loc": [], "extra": []}
    noise = rng.normal(0, 1, (200, xs.size))
    lo_, hi_ = 60 - 12, 60 + (4 if crowded else 12)          # search window around the edge at x = 0
    pos = xs[lo_ + 1:hi_ - 1]
    for s in widths:
        f = dog_filter(s)
        clean = np.convolve(sig, f[::-1], mode="same")
        noise_resp = np.array([np.convolve(n, f[::-1], mode="same") for n in noise])
        out["snr"].append(clean[60] / noise_resp[:, 5:35].std())
        resp = clean + noise_resp
        seg = resp[:, lo_:hi_]
        lm = (seg[:, 1:-1] > seg[:, :-2]) & (seg[:, 1:-1] >= seg[:, 2:])
        vals = np.where(lm, seg[:, 1:-1], -np.inf)
        best = pos[np.argmax(vals, axis=1)]
        out["loc"].append(np.mean(np.abs(best + 0.5)))       # true edge lies between x = −1 and x = 0
        strong = lm & (seg[:, 1:-1] > 0.5 * clean[60])
        out["extra"].append(np.mean(np.maximum(strong.sum(1) - 1, 0)))
    return {k: np.array(v) for k, v in out.items()}, xs, sig, noise[0]


@live.section
def criteria_section():
    st.subheader("Why a derivative of Gaussian? Canny's three criteria, simulated", divider="gray")
    st.markdown("Take a 1D step edge buried in noise and filter it with a derivative of Gaussian of width s. "
                "Repeat 200 times with fresh noise and measure the three things Canny wanted:")
    c_ctrl, c_fig = st.columns([0.8, 2.6], gap="large")
    with c_ctrl:
        snr_in = live.slider("edge contrast / noise level", 0.5, 5.0, 3.0, 0.1, key="cc_snr")
        crowd = st.toggle("a second edge 8 px away (real images are crowded)", value=True, key="cc_crowd")
        st.caption("Drag the **filter width s** with the slider under the picture →")
    sim, xs, sig0, n0 = simulate(snr_in, crowd)
    with c_fig:
        base = make_subplots(rows=2, cols=3, specs=[[{"colspan": 3}, None, None], [{}, {}, {}]],
                             row_heights=[0.48, 0.52], vertical_spacing=0.2, horizontal_spacing=0.08,
                             subplot_titles=["one noisy trial and the filter response",
                                             "1 · detection: SNR ↑", "2 · localisation error ↓", "3 · extra responses ↓"])
        sig = sig0 + n0
        base.add_trace(go.Scatter(x=xs, y=sig, mode="lines", line=dict(color=theme.ORIGINAL, width=1.4),
                                  hoverinfo="skip"), row=1, col=1)
        for c_, key, col in [(1, "snr", theme.E2), (2, "loc", theme.OUTLIER), (3, "extra", theme.E1)]:
            base.add_trace(go.Scatter(x=sim["s"], y=sim[key], mode="lines+markers", line=dict(color=col, width=2.2),
                                      marker=dict(size=5), hovertemplate="s=%{x}<br>%{y:.3f}<extra></extra>"),
                           row=2, col=c_)
            base.update_xaxes(title="filter width s", row=2, col=c_)
        for xv in [-0.5] + ([7.5] if crowd else []):
            base.add_vline(x=xv, line=dict(color=theme.HIGHLIGHT, dash="dot", width=1.2), row=1, col=1)
        base.update_layout(height=640, showlegend=False)
        base.update_annotations(yanchor="top", font_size=12)
        base.update_xaxes(range=[-40, 40], row=1, col=1)
        amp = float(np.abs(sig).max())
        base.update_yaxes(range=[-amp * 1.1, amp * 1.1], row=1, col=1)

        def make(s):
            f = dog_filter(s)
            resp = np.convolve(sig, f[::-1], mode="same")
            resp = resp / (np.abs(resp).max() + 1e-9) * amp
            i = int(np.argmin(np.abs(sim["s"] - s)))
            trs = [go.Scatter(x=xs, y=resp, mode="lines", line=dict(color=theme.TRANSFORMED, width=2.6),
                              xaxis="x", yaxis="y", hoverinfo="skip")]
            for c_, key, col in [(2, "x2", "y2"), (3, "x3", "y3"), (4, "x4", "y4")]:
                k_ = ["snr", "loc", "extra"][c_ - 2]
                trs.append(go.Scatter(x=[s], y=[sim[k_][i]], mode="markers", xaxis=key, yaxis=col,
                                      marker=dict(size=14, color=theme.HIGHLIGHT, symbol="diamond",
                                                  line=dict(color="white", width=1.5)), hoverinfo="skip"))
            cap = (f"s = <b>{s:g}</b>   ·   SNR {sim['snr'][i]:.2f}   ·   localisation error {sim['loc'][i]:.2f} px   ·   "
                   f"extra peaks {sim['extra'][i]:.2f}")
            return trs, cap

        st.plotly_chart(anim.scrubber(base, list(sim["s"]), make, active=3, label="s", fmt="{:g}", frame_ms=140),
                        key="cc_crit_scrub")
    ui.callout("intuition",
               "Widen the filter and it averages more noise away: **detection** improves and **multiple responses** "
               "vanish (with a second edge nearby, detection eventually drops again - the two edges start to cancel). "
               "Localisation first improves (fewer noise peaks) and then gets worse: a wide filter mixes in "
               "neighbouring structure and the peak drifts away from the true edge (switch the second edge off and the "
               "drift becomes much milder, but never disappears). Detection and "
               "localisation pull in opposite directions (Canny called this an uncertainty principle). Solving the "
               "trade-off for a step edge gives a filter very close to the **derivative of a Gaussian** - and σ "
               "is the knob that sets where on the trade-off you sit.")


criteria_section()


# ------------------------------------------------------------------ comparison
@st.cache_resource(show_spinner=False)
def ground_truth():
    clean_s = E.shapes_image(256)
    gtx, gty = E.gradients(clean_s, "Central difference")
    gt = E.nms_quantized(E.magnitude(gtx, gty), E.orientation(gtx, gty)) > 0.02
    return clean_s, gt


@live.section
def comparison_section():
    st.subheader("Sobel vs LoG vs Canny - measured on a scene with known edges", divider="gray")
    c_ctrl, c_main = st.columns([0.8, 3], gap="large")
    with c_ctrl:
        nz = live.slider("noise σₙ", 0.0, 0.2, 0.1, 0.01, key="cc_cmpn")
        s_cmp = live.slider("σ (for LoG and Canny)", 0.8, 4.0, 1.6, 0.1, key="cc_cmps")
        t_sob = live.slider("Sobel threshold (fraction of max)", 0.05, 0.8, 0.25, 0.01, key="cc_cmpt")
    clean_s, gt = ground_truth()
    noisy_s = E.add_noise(clean_s, nz, 9)
    sx, sy = E.gradients(noisy_s, "Sobel")
    Ms = E.magnitude(sx, sy)
    det_sobel = Ms >= t_sob * Ms.max()
    Lr = E.log_filter(noisy_s, s_cmp)
    det_log = E.zero_crossings(Lr, 0.15 * np.abs(Lr).max())
    det_canny = E.canny(noisy_s, s_cmp, 0.1, 0.25)["edges"]
    scores = {n: E.match_f1(d, gt, 2) for n, d in [("Sobel + threshold", det_sobel), ("LoG zero crossings", det_log),
                                                    ("Canny", det_canny)]}
    with c_main:
        eui.show_row([(np.clip(noisy_s, 0, 1), "input (true edges known)"),
                      (E.edges_rgb(det_sobel), f"Sobel + threshold · F1 {scores['Sobel + threshold'][2]:.2f}"),
                      (E.edges_rgb(det_log), f"LoG zero crossings · F1 {scores['LoG zero crossings'][2]:.2f}"),
                      (E.edges_rgb(det_canny), f"Canny · F1 {scores['Canny'][2]:.2f}")])
        fig = go.Figure()
        for i, (lab, col) in enumerate([("precision", theme.TRANSFORMED), ("recall", theme.E1), ("F1", theme.E2)]):
            fig.add_trace(go.Bar(name=lab, x=list(scores), y=[v[i] for v in scores.values()], marker_color=col,
                                 hovertemplate=f"{lab}: %{{y:.2f}}<extra></extra>"))
        fig.update_layout(barmode="group", height=260, margin=dict(l=8, r=8, t=10, b=8), yaxis=dict(range=[0, 1]),
                          legend=dict(orientation="h", y=1.12, x=0))
        st.plotly_chart(fig, key="cc_bars")
        st.caption("Precision = how many detected pixels are real edges; recall = how many real edge pixels were found "
                   "(within 2 px). With almost no noise all three do well. Raise the noise: Sobel's threshold lets "
                   "noise through and LoG's zero crossings turn into spaghetti (precision collapses), while Canny's "
                   "smoothing, NMS and hysteresis keep its precision high. None of them finds the faint square at these "
                   "settings - its contrast is only 0.07.")


comparison_section()

st.markdown("| | LoG | Canny |\n|---|---|---|\n| Smoothing | Gaussian | Gaussian |\n"
            "| Derivative | second | first |\n| Detection criterion | zero crossing | local gradient maximum |\n"
            "| Edge thinning | not explicit | non-maximum suppression |\n| Thresholding | zero-crossing threshold | double threshold |\n"
            "| Connectivity | not explicit | hysteresis |\n| Output | often thicker | thin, connected edges |")
st.markdown("**The conceptual progression:** derivative → smooth + derivative → thin + connect.")

ui.whys([
    ("Why is the derivative of a Gaussian (nearly) optimal?",
     "Canny wrote the three criteria as formulas for a noisy step edge and searched for the best filter. The "
     "optimum is a slightly complicated function, but it is within about 20 % of the derivative of a Gaussian - "
     "which is far simpler, separable and easy to compute. So that is what everyone uses."),
    ("Why does Canny use the first derivative while LoG uses the second?",
     "A first-derivative maximum gives both a location and a direction, which NMS needs to thin edges along the "
     "right line. The second derivative is isotropic and direction-free, more sensitive to noise, and its zero "
     "crossings form closed loops even where there is no real edge."),
    ("Why is there no single best σ?",
     "Because of the detection/localisation trade-off and because images contain edges at many scales: hair, "
     "texture and object outlines need different σ. Multi-scale methods (and the learned detectors in the next "
     "lesson) combine several scales instead of choosing one."),
    ("Why do OpenCV examples use thresholds like 100 and 200?",
     "cv2.Canny works on 0-255 images with the unnormalised Sobel kernel (≈ 8 × the true slope) and, by default, "
     "the L1 magnitude |Iₓ| + |I_y|. Those numbers are in that unit. This site uses fractions of the maximum "
     "instead, which transfer better between images."),
    ("Why are corners and T-junctions often broken in Canny's output?",
     "At a junction the gradient direction is ill-defined (several edges meet), the smoothed magnitude dips, and NMS "
     "compares along the 'wrong' direction for one of the branches. The branches end a pixel or two before they "
     "meet. Corner detectors (a later topic) are designed for exactly those points."),
    ("Why is Canny still used today?",
     "It needs no training data, runs in milliseconds, has interpretable parameters and behaves predictably - ideal "
     "for industrial inspection, robot calibration targets, document scanning and as a pre-processing step."),
])

st.divider()
ui.mcq("cc1", "Increasing σ in Canny generally …",
       ["finds more fine texture edges", "improves localisation", "suppresses noise but blurs the edge location",
        "has no effect after NMS"], 2, "Larger σ = better detection (less noise) but worse localisation; fine details vanish.")
ui.mcq("cc2", "Which stage of Canny guarantees that edges are one pixel thin?",
       ["Gaussian smoothing", "Gradient computation", "Non-maximum suppression", "Hysteresis"], 2,
       "NMS keeps only the ridge of the magnitude; hysteresis only decides which thin candidates to keep.")

ui.lesson_nav(__file__)
