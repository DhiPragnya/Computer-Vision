import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core import edge_ui as eui
from core import edges as E
from core import anim, live, theme, ui

ui.lesson_header(
    __file__,
    "The Laplacian is great at <i>locating</i> edges and terrible with noise. The cure is obvious once said: "
    "<b>smooth first, then take the Laplacian</b>. Because both steps are linear, they can be merged into one "
    "kernel - the <b>Laplacian of Gaussian (LoG)</b>, the famous 'Mexican hat'.",
)

# ------------------------------------------------------------------ the Gaussian
@live.section
def gaussian_section():
    st.subheader("The 2D Gaussian: a weighted average that forgets slowly with distance", divider="gray")
    st.latex(r"G(x, y; \sigma) = \frac{1}{2\pi\sigma^2}\exp\!\left(-\frac{x^2 + y^2}{2\sigma^2}\right),"
             r"\qquad I_s = G * I")
    c_ctrl, c_fig, c_txt = st.columns([0.75, 1.5, 1.2], gap="large")
    with c_ctrl:
        sg = live.slider("σ (spread)", 0.5, 4.0, 1.0, 0.1, key="gl_s")
        show_k = st.toggle("kernel numbers", value=True, key="gl_showk")
        nz = live.slider("noise on the image", 0.0, 0.15, 0.06, 0.01, key="gl_nz")
    r = int(np.ceil(3 * sg))
    xx = np.linspace(-8, 8, 61)
    X, Y = np.meshgrid(xx, xx)
    with c_fig:
        fig = go.Figure(go.Surface(x=xx, y=xx, z=E.gaussian_formula(X, Y, sg), colorscale="Blues", showscale=False,
                                   cmin=0, cmax=0.17, hovertemplate="x=%{x:.1f}, y=%{y:.1f}<br>G=%{z:.4f}<extra></extra>"))
        fig.update_layout(height=360, margin=dict(l=0, r=0, t=0, b=0), uirevision="keep",
                          scene=dict(zaxis=dict(range=[0, 0.17]), xaxis_title="x", yaxis_title="y", zaxis_title="G",
                                     aspectratio=dict(x=1, y=1, z=0.6), camera=dict(eye=dict(x=1.5, y=-1.4, z=0.9))))
        st.plotly_chart(fig, key="gl_surf")
    with c_txt:
        st.markdown(f"Kernel radius ≈ 3σ = **{r} px** → size **{2 * r + 1} × {2 * r + 1}** "
                    f"(99.7 % of the weight). The weights add up to {E.gaussian_2d(sg).sum():.3f}.")
        if show_k:
            K = E.gaussian_2d(sg)
            n = K.shape[0]
            c = n // 2
            crop = K[max(0, c - 4):c + 5, max(0, c - 4):c + 5]     # at most the central 9 x 9 (same figure size)
            st.plotly_chart(eui.kernel_figure(crop, "", fmt="{:.3f}", height=280, diverging=False), key="gl_kern")
            st.caption("Peak height 1/(2πσ²): small σ = a tall needle, large σ = a low, wide hill." if n <= 9 else
                       f"The central 9 × 9 of the {n} × {n} kernel: lower, flatter weights spread wider.")
        ui.callout("intuition",
                   "Every output pixel is an average of its neighbourhood, with close neighbours trusted more. "
                   "σ is the 'radius of trust': small σ = only immediate neighbours, large σ = a wide area.")
    noisy = E.add_noise(E.photo("cameraman.png"), nz, 3)
    eui.show_row([(np.clip(noisy, 0, 1), f"noisy (σₙ = {nz})"),
                  (np.clip(E.smooth(noisy, 1.0), 0, 1), "G * I, σ = 1"),
                  (np.clip(E.smooth(noisy, sg), 0, 1), f"G * I, σ = {sg}"),
                  (np.clip(E.smooth(noisy, 3.0), 0, 1), "G * I, σ = 3")])


gaussian_section()


# ------------------------------------------------------------------ two equivalent paths
@st.cache_resource(show_spinner=False)
def two_paths():
    I0 = E.photo("coins.png")
    sg2 = 2.0
    G = E.gaussian_2d(sg2, 8)
    pathA = E.laplacian(E.correlate(I0, G))                                   # ∇²(G * I)
    LoGk = E.convolve(np.pad(G, 2), E.K["laplace4"])                          # ∇²G as a kernel
    pathB = E.correlate(I0, LoGk)                                             # (∇²G) * I
    diff = np.abs(pathA - pathB)[20:-20, 20:-20].max()
    return I0, pathA, pathB, diff


def u8(img, lo, hi):
    return np.clip(np.round((np.asarray(img, float) - lo) / (hi - lo + 1e-12) * 255), 0, 255).astype(np.uint8)


@st.cache_resource(show_spinner=False)
@live.full_quality()          # cached: never store a coarse mid-drag version
def pipeline_figure():
    """Path A as pictures: noisy image → smoothed → Laplacian → zero crossings (shown as one morphing heatmap)."""
    img = E.add_noise(E.photo("cameraman.png"), 0.04, 3)
    img = img[40:168, 60:188]                                               # a 128 × 128 crop (face + coat)
    raw_lap = E.laplacian(img)
    smoothed = E.smooth(img, 2.0)
    lap = E.laplacian(smoothed)
    zc = E.zero_crossings(lap, 0.08 * np.abs(lap).max())
    v1 = np.percentile(np.abs(raw_lap), 99)
    v2 = np.percentile(np.abs(lap), 99)
    signed = lambda R, v: u8(R, -v, v)
    states = [u8(img, 0, 1), signed(raw_lap, v1), u8(smoothed, 0, 1), signed(lap, v2),
              np.where(zc, 255, (u8(smoothed, 0, 1) * 0.35)).astype(np.uint8)]
    cs_grey = [[0, "#111111"], [1, "#f5f5f5"]]
    cs_div = [[0, "#2a78d6"], [0.5, "#f0efec"], [1, "#e34948"]]

    def hm(Z, cs):
        return go.Heatmap(z=Z, zmin=0, zmax=255, colorscale=cs, showscale=False, hoverinfo="skip")

    H = anim.hl
    steps = [
        anim.Step("A noisy photo", "I", "A crop of the cameraman with a little noise (σₙ = 0.04).", [hm(states[0], cs_grey)]),
        anim.Step("Laplacian straight away", "∇²I", "Speckle everywhere: two derivatives amplify the noise by ω² - the "
                  "edges drown.", [hm(states[1], cs_div)]),
        anim.Step("Instead: smooth first", f"G * I,  σ = {H('2')}", "The Gaussian averages the noise away (and blurs the "
                  "edges a little).", [hm(states[2], cs_grey)]),
        anim.Step("Then the Laplacian", "∇²(G * I) = (∇²G) * I", "Now clean red/blue bands hug every real edge - "
                  "this IS the LoG response.", [hm(states[3], cs_div)]),
        anim.Step("Keep the zero crossings", "edges = sign changes of ∇²(G * I) with a big enough jump",
                  "Thin, connected contours exactly between the red and blue bands.", [hm(states[4], cs_grey)]),
    ]
    base = go.Figure()
    base.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1, visible=False, constrain="domain")
    base.update_xaxes(visible=False, constrain="domain")
    base.update_layout(height=560, margin=dict(l=0, r=0))
    return anim.step_player(base, steps, n_sub=8, hold=22, wrap_at=70)


st.subheader("Smooth-then-differentiate = differentiate the smoother", divider="gray")
c1, c2 = st.columns([1.2, 1], gap="large")
I0, pathA, pathB, diff = two_paths()
with c1:
    st.latex(r"\nabla^2 (G * I) = (\nabla^2 G) * I")
    eui.show_row([(E.signed_rgb(pathA), "path A: smooth with G, then Laplacian"),
                  (E.signed_rgb(pathB), "path B: one pass with the kernel ∇²G")])
with c2:
    st.metric("largest difference between the two paths", f"{diff:.1e}", border=True,
              help="Pure floating-point rounding - the two are mathematically identical.")
    ui.callout("intuition",
               "Convolution is a weighted **sum of shifted copies** of the image, and the derivative of a sum is "
               "the sum of the derivatives. So it does not matter whether you differentiate the image after "
               "blurring or differentiate the blur kernel once - and the second way builds a single kernel "
               "you can reuse on every image.")
    st.caption("The same trick with the first derivative gives the *derivative of Gaussian* - the filter Canny uses.")
ui.anim_intro("Step by step: why we smooth BEFORE the Laplacian",
              "The picture morphs from stage to stage. Compare step 2 (no smoothing) with step 4 (smoothed first).")
p1, p2 = st.columns([1.3, 1], gap="large")
with p1:
    st.plotly_chart(pipeline_figure(), key="gl_pipeline")
with p2:
    ui.callout("note",
               "Steps 3 → 4 → 5 are exactly the LoG detector: smooth with G, take ∇², keep the zero crossings. "
               "Steps 3 and 4 can be merged into ONE convolution with the kernel ∇²G - that is the point of the "
               "identity above.", title="LoG = these steps, merged")


# ------------------------------------------------------------------ the LoG kernel
def log_derivation_steps(s):
    """The cross-section y = 0 through each stage of the derivation (each curve scaled to the same height)."""
    x = np.linspace(-4 * s - 2, 4 * s + 2, 241)
    G = np.exp(-x ** 2 / (2 * s ** 2))
    curves = [
        G,
        -(x / s ** 2) * G,
        (x ** 2 / s ** 4 - 1 / s ** 2) * G,
        (x ** 2 / s ** 4 - 2 / s ** 2) * G,
        -(x ** 2 / s ** 4 - 2 / s ** 2) * G,
    ]
    curves = [c / np.abs(c).max() for c in curves]
    zeros = [[], [0.0], [-s, s], [-np.sqrt(2) * s, np.sqrt(2) * s], [-np.sqrt(2) * s, np.sqrt(2) * s]]

    def state(i):
        z = np.full(2, np.nan)
        z[:len(zeros[i])] = zeros[i]
        return [go.Scatter(x=x, y=curves[i], mode="lines", line=dict(color=theme.TRANSFORMED, width=3),
                           fill="tozeroy", fillcolor=theme.rgba(theme.TRANSFORMED, 0.12), hoverinfo="skip"),
                go.Scatter(x=z, y=[0, 0], mode="markers", marker=dict(size=13, color=theme.HIGHLIGHT, symbol="diamond"),
                           hoverinfo="skip")]

    H = anim.hl
    sg = f"{s:g}"
    return [
        anim.Step("Start: the Gaussian", "G = (1/2πσ²)·e<sup>−r²/2σ²</sup>,   r² = x² + y²",
                  f"Cross-section through the centre (y = 0), σ = {sg}. Each curve below is rescaled to the same height.",
                  state(0)),
        anim.Step("Chain rule: ∂G/∂x", f"∂G/∂x = {H('−(x/σ²)')}·G",
                  "The derivative of the exponent −x²/2σ² is −x/σ². The bell becomes an S: zero at the top of the "
                  "bell (amber), largest where the bell is steepest.", state(1)),
        anim.Step("Product rule: ∂²G/∂x²", f"∂²G/∂x² = ({H('x²/σ⁴ − 1/σ²')})·G",
                  f"Differentiate (−x/σ²)·G: the factor gives −1/σ², the G gives another −x/σ². Zero crossings at "
                  f"x = ±σ = ±{sg}.", state(2)),
        anim.Step("Add the y-part: ∇²G", f"∇²G = ∂²G/∂x² + ∂²G/∂y² = ({H('r²/σ⁴ − 2/σ²')})·G",
                  "∂²G/∂y² has the same form with y. On the line y = 0 it adds another −G/σ², so the centre dips "
                  f"deeper and the zero crossings move out to r = √2·σ = {np.sqrt(2) * s:.2f}.", state(3)),
        anim.Step("Flip the sign: the 'Mexican hat'", "−∇²G = (1/πσ⁴)(1 − r²/2σ²)·e<sup>−r²/2σ²</sup>",
                  "Plotted upside down it is a hat: a positive centre and a negative ring. Rotate it about the vertical "
                  "axis and you get the 2D kernel on the left.", state(4)),
    ], (x.min(), x.max())


@live.section
def log_kernel_section():
    st.subheader("The LoG kernel - the 'Mexican hat'", divider="gray")
    st.latex(r"\mathrm{LoG}(x, y) = \nabla^2 G = -\frac{1}{\pi\sigma^4}\left(1 - \frac{x^2 + y^2}{2\sigma^2}\right)"
             r"\exp\!\left(-\frac{x^2 + y^2}{2\sigma^2}\right)")
    c_ctrl, c_fig, c_prof = st.columns([0.75, 1.4, 1.3], gap="large")
    with c_ctrl:
        sl = live.slider("σ", 0.8, 4.0, 2.0, 0.1, key="gl_sl")
        hat = st.toggle("show −LoG (a 'hat')", value=True, key="gl_hat")
    sgn = -1 if hat else 1
    xx = np.linspace(-12, 12, 61)
    X, Y = np.meshgrid(xx, xx)
    Z = sgn * E.log_formula(X, Y, sl)
    with c_fig:
        fig = go.Figure(go.Surface(x=xx, y=xx, z=Z, colorscale=eui.DIVERGING, cmid=0, showscale=False,
                                   hovertemplate="x=%{x:.1f}, y=%{y:.1f}<br>%{z:.4f}<extra></extra>"))
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=0, b=0), uirevision="keep",
                          scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="−LoG" if hat else "LoG",
                                     aspectratio=dict(x=1, y=1, z=0.6), camera=dict(eye=dict(x=1.5, y=-1.4, z=0.8))))
        st.plotly_chart(fig, key="gl_log")
    with c_prof:
        rr = np.linspace(-12, 12, 400)
        fig = eui.signal_figure([dict(x=rr, y=sgn * E.log_formula(rr, 0 * rr, sl), name="cross-section through the centre",
                                      color=theme.TRANSFORMED)], height=300, x_title="x (with y = 0)",
                                vlines=[(-np.sqrt(2) * sl, "−√2σ"), (np.sqrt(2) * sl, "√2σ")])
        fig.add_hline(y=0, line=dict(color=theme.AXIS, width=1))
        st.plotly_chart(fig, key="gl_prof")
        st.caption(f"The kernel changes sign at r = √2·σ = {np.sqrt(2) * sl:.2f} px: a centre of one sign, a ring of "
                   "the other. It sums to zero, so flat regions give 0.")

    ui.anim_intro("Step by step: derive the LoG (as in the notes)",
                  f"Uses your σ = {sl:g}. Each step applies one rule of calculus to the curve before it.")
    steps, (x0, x1) = log_derivation_steps(sl)
    base = go.Figure()
    base.update_layout(height=470, showlegend=False, margin=dict(l=10, r=10))
    base.update_xaxes(range=[x0, x1], title="x  (cross-section y = 0)", gridcolor=theme.GRID, zeroline=True,
                      zerolinecolor=theme.AXIS)
    base.update_yaxes(range=[-1.15, 1.15], title="(scaled)", gridcolor=theme.GRID, zerolinecolor=theme.AXIS)
    d1, d2 = st.columns([1.5, 1], gap="large")
    with d1:
        st.plotly_chart(anim.step_player(base, steps, n_sub=14, wrap_at=80), key="gl_derive")
    with d2:
        st.latex(r"\frac{\partial G}{\partial x} = -\frac{x}{\sigma^2}\,G")
        st.latex(r"\frac{\partial^2 G}{\partial x^2} = \left(\frac{x^2}{\sigma^4} - \frac{1}{\sigma^2}\right) G")
        st.latex(r"\nabla^2 G = \left(\frac{x^2 + y^2}{\sigma^4} - \frac{2}{\sigma^2}\right) G")
        px_, py_, h_ = 0.7, -0.4, 1e-3
        Gf = lambda a, b: E.gaussian_formula(a, b, sl)
        num_lap = (Gf(px_ + h_, py_) + Gf(px_ - h_, py_) + Gf(px_, py_ + h_) + Gf(px_, py_ - h_)
                   - 4 * Gf(px_, py_)) / h_ ** 2
        st.markdown(f"**Check numerically** at (x, y) = ({px_}, {py_}), σ = {sl}: finite differences give "
                    f"{num_lap:.6f}, the formula gives {E.log_formula(px_, py_, sl):.6f}.")


log_kernel_section()


# ------------------------------------------------------------------ LoG edges + scale
@live.section
def log_edges_section():
    st.subheader("LoG edge detection: zero crossings, and σ as the scale", divider="gray")
    c_ctrl, c_main = st.columns([0.8, 3], gap="large")
    with c_ctrl:
        img, clean, _ = eui.pick_image("gl", "Cameraman (photo)", noise_default=0.03)
        s_e = live.slider("σ of the LoG", 0.8, 5.0, 2.0, 0.1, key="gl_se")
        zt = live.slider("zero-crossing threshold (fraction of max jump)", 0.0, 0.4, 0.05, 0.01, key="gl_zt")
    R = E.log_filter(img, s_e)
    Zc = E.zero_crossings(R, zt * np.abs(R).max())
    with c_main:
        eui.show_row([(np.clip(img, 0, 1), "input"), (E.signed_rgb(R), f"LoG response, σ = {s_e}"),
                      (E.edges_rgb(Zc), "zero crossings"), (E.overlay(np.clip(img, 0, 1), Zc), "overlay")])
        rows = []
        for s_ in (1.0, 2.0, 4.0):
            Rs = E.log_filter(img, s_)
            rows.append((E.edges_rgb(E.zero_crossings(Rs, zt * np.abs(Rs).max())), f"σ = {s_:g}"))
        st.markdown("**Same image, three scales:**")
        eui.show_row(rows)


log_edges_section()
st.markdown("| | small σ | large σ |\n|---|---|---|\n| smoothing | little | strong |\n"
            "| detects | fine details | broad structures |\n| noise | sensitive | robust |\n"
            "| localisation | precise | edges drift and round off corners |")


# ------------------------------------------------------------------ DoG
@live.section
def dog_section():
    with st.expander("Going further: the Difference of Gaussians (DoG) ≈ LoG", icon=":material/difference:"):
        c_ctrl, c_fig = st.columns([0.8, 2], gap="large")
        with c_ctrl:
            kk = live.slider("ratio k = σ₂ / σ₁", 1.05, 3.0, 1.6, 0.05, key="gl_k")
        s1 = 2.0
        rr = np.linspace(-10, 10, 400)
        g = lambda s: np.exp(-rr ** 2 / (2 * s ** 2)) / (2 * np.pi * s ** 2)
        dog = g(kk * s1) - g(s1)
        logp = E.log_formula(rr, 0 * rr, s1) * (kk - 1) * s1 ** 2
        with c_fig:
            fig = eui.signal_figure([dict(x=rr, y=logp, name="(k − 1)σ² · LoG", color=theme.TRANSFORMED),
                                     dict(x=rr, y=dog, name="G(kσ) − G(σ)", color=theme.E1, dash="dash")], height=280,
                                    x_title="x")
            st.plotly_chart(fig, key="gl_dog")
        st.caption("Subtracting two blurred images approximates the LoG (exactly in the limit k → 1). It needs only "
                   "two Gaussian blurs, which are cheap and separable - which is why SIFT's keypoint detector uses DoG.")


dog_section()

ui.whys([
    ("Why a Gaussian and not a simple box average?",
     "A box average has hard edges, so in frequency it ripples (it lets some fine detail through with the wrong "
     "sign) and makes blocky artefacts. The Gaussian falls off smoothly, is the same in every direction, can be "
     "applied as two cheap 1D passes, and - uniquely - blurring more never creates *new* bumps or edges, so "
     "coarser scales are truly simplified versions of finer ones."),
    ("Why the factor 1/(2πσ²)?",
     "It makes the kernel's total weight exactly 1, so smoothing averages but never brightens or darkens the "
     "image. Without it a wide Gaussian would multiply all intensities by a large number."),
    ("Why can we swap the derivative and the convolution?",
     "Both are linear and shift-invariant. G * I is a weighted sum of shifted copies of I; differentiating a sum "
     "term by term gives the same as summing the differentiated terms - which is (∇²G) * I."),
    ("Why does the LoG kernel change sign at r = √2σ?",
     "The factor (1 − r²/2σ²) is zero when r² = 2σ². Inside, the kernel has one sign; outside, the other. That "
     "centre–surround shape compares a pixel's neighbourhood with a ring around it - like the receptive fields "
     "of retinal cells."),
    ("Why do LoG zero crossings form closed loops ('spaghetti')?",
     "The zero crossings are the zero level set of a smooth 2D function, and level sets of a smooth function are "
     "closed curves (or run off the image). So even noise and smooth shading create closed contours - one reason "
     "a threshold on the jump across the crossing is needed."),
    ("How big must the kernel be?",
     "Large enough to contain the hat's ring: about ±4σ for the LoG (±3σ for a Gaussian). A truncated kernel no "
     "longer sums to zero and gives a false response on flat regions - this code subtracts the tiny leftover "
     "mean to fix that."),
])

st.divider()
ui.mcq("gl1", "Which statement is true for the LoG detector?",
       ["Edges are local maxima of the response", "Edges are zero crossings of the response",
        "It needs separate x and y kernels", "Larger σ finds finer details"], 1,
       "LoG is a (smoothed) second derivative, so edges are where it crosses zero. It is isotropic, and larger σ "
       "picks out broader structures.")
ui.mcq("gl2", "Why is (∇²G) * I equal to ∇²(G * I)?",
       ["Because G is symmetric", "Because convolution and differentiation are both linear and shift-invariant",
        "Only approximately, for small σ", "Because ∇² of an image is always zero"], 1,
       "Differentiating a weighted sum of shifted images term by term gives the same result in either order.")

ui.lesson_nav(__file__)
