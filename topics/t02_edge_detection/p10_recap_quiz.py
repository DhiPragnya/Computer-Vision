import numpy as np
import streamlit as st

from core import edges as E
from core import live, ui

ui.lesson_header(
    __file__,
    "The whole topic on one page, then practise: a <b>tuning challenge</b> (match a hidden Canny result), a "
    "<b>spot-the-detector</b> drill, and a concept check.",
)

# ------------------------------------------------------------------ progression + equations
st.subheader("The progression", divider="gray")
steps = ["Image", "Intensity changes", "First derivative", "Sobel / Prewitt", "Second derivative", "LoG", "Canny",
         "Learning-based"]
with st.container(horizontal=True, gap="small"):
    for i, s_ in enumerate(steps):
        st.badge(s_, color="blue" if i < 6 else ("green" if s_ == "Canny" else "violet"))
        if i < len(steps) - 1:
            st.markdown("→")
st.markdown("The major shift: from **designing the filter** to **learning the representation** - and increasingly "
            "towards learning **multi-scale features and global context**.")

st.subheader("Key equations", divider="gray")
e1, e2 = st.columns(2, gap="medium")
with e1:
    with st.container(border=True):
        st.markdown("**Gradient, magnitude, direction**")
        st.latex(r"\nabla I = \begin{bmatrix} I_x \\ I_y \end{bmatrix},\quad |\nabla I| = \sqrt{I_x^2 + I_y^2},\quad "
                 r"\theta = \operatorname{atan2}(I_y, I_x)")
    with st.container(border=True):
        st.markdown("**Sobel = smoothing × derivative**")
        st.latex(r"G_x = \begin{bmatrix} 1 \\ 2 \\ 1 \end{bmatrix}\begin{bmatrix} -1 & 0 & 1 \end{bmatrix} = "
                 r"\begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}")
    with st.container(border=True):
        st.markdown("**Canny**")
        st.markdown("Gaussian → Gradient → NMS → Double threshold → Hysteresis")
with e2:
    with st.container(border=True):
        st.markdown("**Laplacian**")
        st.latex(r"\nabla^2 I = I_{xx} + I_{yy} \approx \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix} \star I")
    with st.container(border=True):
        st.markdown("**Gaussian and LoG**")
        st.latex(r"G = \tfrac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}},\qquad \nabla^2(G * I) = (\nabla^2 G) * I")
        st.latex(r"\mathrm{LoG} = -\tfrac{1}{\pi\sigma^4}\left(1 - \tfrac{r^2}{2\sigma^2}\right)e^{-\frac{r^2}{2\sigma^2}}")

st.markdown("| | Sobel / Prewitt | LoG | Canny | HED-style (learned) |\n|---|---|---|---|---|\n"
            "| Derivative | 1st | 2nd | 1st (of Gaussian) | learned |\n| Noise handling | 3-tap smoothing | Gaussian σ | Gaussian σ | learned |\n"
            "| Edge location | threshold (thick) | zero crossing | NMS (1 px) | NMS on predicted map |\n"
            "| Connectivity | none | none | hysteresis | learned + post-processing |\n"
            "| Parameters | threshold | σ, threshold | σ, T_low, T_high | trained weights |\n"
            "| Training data | no | no | no | yes |")

# ------------------------------------------------------------------ challenge
st.subheader("Challenge: match the hidden Canny settings", divider="gray")
ss = st.session_state
ss.setdefault("ch_round", 0)


def target_params(r: int):
    rng = np.random.default_rng(500 + r)
    s = float(np.round(rng.uniform(0.8, 3.5), 1))
    lo = float(np.round(rng.uniform(0.04, 0.15), 2))
    hi = float(np.round(lo * rng.uniform(2.0, 3.0), 2))
    img = ["Cameraman (photo)", "Coins (photo)", "Coffee (photo)", "Cat (photo)", "Astronaut (photo)"][r % 5]
    return img, s, lo, hi


@st.cache_resource(show_spinner=False, max_entries=8)
def target_edges(r: int):
    tname, ts, tlo, thi = target_params(r)
    I = E.SAMPLES[tname]()
    return I, E.canny(I, ts, tlo, thi)["edges"]


@live.section
def challenge_section():
    tname, ts, tlo, thi = target_params(ss.ch_round)
    I, target = target_edges(ss.ch_round)
    c_ctrl, c1, c2, c3 = st.columns([0.85, 1, 1, 1], gap="medium")
    with c_ctrl:
        st.markdown(f"**Round {ss.ch_round + 1}** · {tname}")
        ys = live.slider("your σ", 0.5, 4.0, 1.5, 0.1, key=f"ch_s{ss.ch_round}")
        ylo, yhi = live.slider("your T_low, T_high", 0.0, 0.6, (0.1, 0.25), 0.01, key=f"ch_t{ss.ch_round}")
        if st.button("New target", icon=":material/refresh:", key="ch_new"):
            ss.ch_round += 1
            live.rerun()
    mine = E.canny(I, ys, ylo, yhi)["edges"]
    p_, r_, f_ = E.match_f1(mine, target, 1)
    diff = np.zeros(I.shape + (3,), np.uint8)
    diff[...] = (26, 26, 25)
    diff[target & mine] = (27, 175, 122)
    diff[target & ~mine] = (227, 73, 72)
    diff[mine & ~target] = (42, 120, 214)
    c1.image(ui.datauri(E.edges_rgb(target)), caption="target (hidden settings)", width="stretch")
    c2.image(ui.datauri(E.edges_rgb(mine)), caption="yours", width="stretch")
    c3.image(ui.datauri(diff), caption="green = both · red = you missed · blue = extra", width="stretch")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("F1 match", f"{f_:.2f}", border=True)
    m2.metric("precision", f"{p_:.2f}", border=True, help="Of your edge pixels, how many are in the target")
    m3.metric("recall", f"{r_:.2f}", border=True, help="Of the target's edge pixels, how many you found")
    m4.metric("goal", "F1 ≥ 0.90", border=True)
    if f_ >= 0.9:          # always one status line (same height), so the page does not jump while you drag
        st.success("Excellent match! Open the reveal to compare your settings.", icon=":material/emoji_events:")
    else:
        st.info(f"Keep tuning: F1 = {f_:.2f}, the goal is 0.90. The colours in the right image tell you what to change.",
                icon=":material/tune:")
    with st.expander("Hints and reveal", icon=":material/lightbulb:"):
        st.markdown("- Lots of **blue** fine texture → raise σ or T_high.\n- Contours **broken** (red gaps) → lower T_low.\n"
                    "- Contours **rounded/shifted** → σ is too large.\n- Isolated blue specks → T_high too low.")
        if st.toggle("Reveal the hidden settings", key=f"ch_rev{ss.ch_round}"):
            st.markdown(f"σ = **{ts}**, T_low = **{tlo}**, T_high = **{thi}** (ratio {thi / tlo:.1f})")


challenge_section()

# ------------------------------------------------------------------ which detector
st.subheader("Drill: which detector produced this?", divider="gray")
ss.setdefault("wd_round", 0)
ss.setdefault("wd_score", 0)
ss.setdefault("wd_total", 0)
ss.setdefault("wd_done", False)
DET = ["Sobel + threshold", "LoG zero crossings", "Canny"]


def make_case(r: int):
    rng = np.random.default_rng(900 + r)
    kind = DET[rng.integers(0, 3)]
    nm = ["Coins (photo)", "Cameraman (photo)", "Shapes (synthetic)", "Coffee (photo)"][rng.integers(0, 4)]
    img = E.add_noise(E.SAMPLES[nm](), float(rng.choice([0.0, 0.02])), 3)
    if kind == "Sobel + threshold":
        gx, gy = E.gradients(img, "Sobel")
        M = E.magnitude(gx, gy)
        out = M >= rng.uniform(0.15, 0.3) * M.max()
    elif kind == "LoG zero crossings":
        R = E.log_filter(img, rng.uniform(1.5, 2.5))
        out = E.zero_crossings(R, rng.uniform(0.03, 0.1) * np.abs(R).max())
    else:
        out = E.canny(img, rng.uniform(1.0, 2.0), 0.08, 0.2)["edges"]
    return kind, nm, out


def next_case():
    ss.wd_round += 1
    ss.wd_done = False
    ss.pop("wd_choice", None)


make_case_cached = st.cache_resource(show_spinner=False, max_entries=8)(make_case)


@live.section
def drill_section():
    kind, nm, out = make_case_cached(ss.wd_round)
    a, b = st.columns([1.3, 1], gap="large")
    a.image(ui.datauri(E.edges_rgb(out)), caption=nm, width="stretch")
    with b:
        st.markdown(f"**Case {ss.wd_round + 1}** · score **{ss.wd_score} / {ss.wd_total}**")
        ch = st.radio("Which detector?", DET, index=None, key="wd_choice", disabled=ss.wd_done)
        if ch is not None and not ss.wd_done:
            ss.wd_done = True
            ss.wd_total += 1
            ss.wd_ok = ch == kind
            ss.wd_score += int(ss.wd_ok)
            live.rerun()
        if ss.wd_done:
            tell = {"Sobel + threshold": "thick, double-width edges that get fatter where the contrast is high; no thinning.",
                    "LoG zero crossings": "closed loops ('spaghetti'), even around smooth shading; edges never end abruptly.",
                    "Canny": "thin one-pixel lines that end cleanly, with faint parts kept only when they continue a strong edge."}
            (st.success if ss.get("wd_ok") else st.error)(
                f"**{'Correct' if ss.get('wd_ok') else 'It was ' + kind}.** Tell-tale sign: {tell[kind]}",
                icon=":material/check_circle:" if ss.get("wd_ok") else ":material/cancel:")
        st.button("Next case", on_click=next_case, icon=":material/skip_next:", type="primary")


drill_section()

# ------------------------------------------------------------------ concept check
st.subheader("Concept check", divider="gray")
qs = [
    ("e1", "Why is the gradient direction used in non-maximum suppression?",
     ["It points along the edge", "It points across the edge, where the magnitude peaks", "It decides the threshold",
      "It is needed to compute the magnitude"], 1,
     "The magnitude is maximal across the edge (along the gradient), so that is where we look for the local maximum."),
    ("e2", "Why does the Laplacian need Gaussian smoothing more than the gradient does?",
     ["It is not linear", "Two derivatives amplify high-frequency noise roughly by ω²", "It has no zero crossings",
      "It only works on colour images"], 1, "Each derivative multiplies a frequency ω by ω; the Laplacian applies two."),
    ("e3", "Sobel's (1, 2, 1) column is …",
     ["a derivative", "a small Gaussian-like smoothing", "a threshold", "the Laplacian"], 1,
     "It is the binomial (1,1)⊛(1,1), the smallest discrete Gaussian-like average."),
    ("e4", "Which Canny stage connects faint parts of a contour to its strong parts?",
     ["Gaussian smoothing", "NMS", "Double threshold", "Hysteresis"], 3,
     "Hysteresis keeps weak pixels connected to strong ones and discards the rest."),
    ("e5", "Increasing σ in LoG or Canny …",
     ["finds finer details", "detects broader structures and resists noise, but localises worse",
      "has no effect on noise", "makes edges thinner"], 1, "σ sets the scale: more smoothing, coarser edges."),
    ("e6", "What lets learned detectors (HED and later) ignore texture edges?",
     ["A higher threshold", "Multi-scale learned features and context from large receptive fields",
      "Using the Laplacian", "Removing NMS"], 1,
     "Deep features know what pattern is present over a large region; the fusion learns which scale to trust."),
]
@live.section
def concept_section():
    score = answered = 0
    cols = st.columns(2, gap="medium")
    for i, (k, q, opts, ans, ex) in enumerate(qs):
        with cols[i % 2]:
            r = ui.mcq(k, q, opts, ans, ex, scored=True)
            if r is not None:
                answered += 1
                score += int(r)
    if answered:
        st.progress(score / len(qs), text=f"Concept check: {score} / {len(qs)} correct ({answered} answered)")
    if answered == len(qs) and score == len(qs):
        st.success("Full marks - edge detection mastered.", icon=":material/military_tech:")


concept_section()

ui.lesson_nav(__file__)
