import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core import edge_ui as eui
from core import edges as E
from core import anim, live, theme, ui

ui.lesson_header(
    __file__,
    "The Sobel kernel is not a magic table of numbers. It is a <b>derivative</b> in one direction multiplied by "
    "a <b>smoothing</b> in the perpendicular direction - so it estimates the slope while averaging away some "
    "noise. Prewitt is the same idea with plain, equal smoothing weights.",
)

# ------------------------------------------------------------------ derive Sobel
def outer_product_steps(wmid):
    """Gₓ = S · D built row by row (each row = smoothing weight × derivative), then G_y = the same turned 90°."""
    S = np.array([1.0, wmid, 1.0])
    D = np.array([-1.0, 0.0, 1.0])
    Gx = np.outer(S, D)
    Gy = Gx.T
    m = max(np.abs(Gx).max(), 1e-9)

    def hm(Z):
        return go.Heatmap(z=Z, zmin=-m, zmax=m, colorscale=eui.DIVERGING, showscale=False, xgap=4, ygap=4,
                          text=[[f"{v:g}" for v in r] for r in Z], texttemplate="%{text}",
                          textfont=dict(size=18, color="#1a1a19"), hoverinfo="skip")

    def rows(k):
        Z = np.zeros((3, 3))
        Z[:k] = Gx[:k]
        return Z

    H, w = anim.hl, f"{wmid:g}"
    return [
        anim.Step("Start: an empty 3×3 kernel", "Gₓ = S · D,   S = (1, " + w + ", 1)ᵀ,  D = (−1, 0, 1)",
                  "Column S smooths down the rows, row D differentiates across the columns.", [hm(rows(0))]),
        anim.Step("Row 1 = S₁ × D", f"{H('1')} × (−1, 0, 1) = (−1, 0, 1)",
                  "The row above the pixel: a plain 'right minus left'.", [hm(rows(1))]),
        anim.Step("Row 2 = S₂ × D", f"{H(w)} × (−1, 0, 1) = ({-wmid:g}, 0, {wmid:g})",
                  "The pixel's own row counts " + w + " times as much - it is the most relevant one.", [hm(rows(2))]),
        anim.Step("Row 3 = S₃ × D", f"{H('1')} × (−1, 0, 1) = (−1, 0, 1)",
                  "The row below. Gₓ = three derivatives, averaged with weights (1, " + w + ", 1).", [hm(rows(3))]),
        anim.Step("Turn by 90°: G_y = Gₓᵀ", "G_y = (−1, 0, 1)ᵀ · (1, " + w + ", 1)",
                  "Swapping the roles of rows and columns gives the vertical-derivative kernel.", [hm(Gy)]),
    ]


@live.section
def derive_section():
    st.subheader("Derive the kernel: smoothing (column) × derivative (row)", divider="gray")
    pc1, pc2 = st.columns([1.4, 1])
    with pc1:
        preset = st.segmented_control("smoothing weights", ["Prewitt (1, 1, 1)", "Sobel (1, 2, 1)", "custom (1, w, 1)"],
                                      default="Sobel (1, 2, 1)", key="sb_preset", required=True)
    wmid = {"Prewitt (1, 1, 1)": 1.0, "Sobel (1, 2, 1)": 2.0}.get(preset)
    with pc2:
        if wmid is None:
            wmid = live.slider("centre weight w", 0.0, 6.0, 3.0, 0.5, key="sb_w")
    c1, c2, c3, c4, c5 = st.columns([0.5, 0.15, 1.0, 0.15, 1.1], gap="small", vertical_alignment="center")
    Gx, Gy = E.derivative_smoothing_kernel(wmid)
    with c1:
        st.plotly_chart(eui.kernel_figure(np.array([[1.0], [wmid], [1.0]]), "S (smooth)", height=230, diverging=False,
                                          zmax=None), key="sb_S")
    c2.markdown("<div style='font-size:2rem;text-align:center'>×</div>", unsafe_allow_html=True)
    with c3:
        st.plotly_chart(eui.kernel_figure(np.array([[-1.0, 0, 1]]), "D (derivative)", height=120), key="sb_D")
    c4.markdown("<div style='font-size:2rem;text-align:center'>=</div>", unsafe_allow_html=True)
    with c5:
        st.plotly_chart(eui.kernel_figure(Gx, "Gₓ = S D", height=230), key="sb_Gx")
    a, b = st.columns([1, 1.3], gap="large")
    with a:
        st.latex(r"G_x = \begin{bmatrix} 1 \\ " + f"{wmid:g}" + r" \\ 1 \end{bmatrix}\begin{bmatrix} -1 & 0 & 1 \end{bmatrix}"
                 r" = " + ui.tex_matrix(Gx, 1))
        st.latex(r"G_y = \begin{bmatrix} -1 \\ 0 \\ 1 \end{bmatrix}\begin{bmatrix} 1 & " + f"{wmid:g}" +
                 r" & 1 \end{bmatrix} = " + ui.tex_matrix(Gy, 1))
        st.markdown("**Sobel ≈ Derivative × Smoothing.** Each column of Gₓ is the derivative (−1, 0, 1) scaled by the "
                    "smoothing weight of its row; Gᵧ is the same idea turned by 90°.")
        ui.callout("intuition",
                   "Gₓ measures 'right minus left' **three times** - in the row above, the row itself and the row "
                   "below - and averages the three answers, trusting the middle row most. Averaging three noisy "
                   "measurements of the same slope is much more reliable than trusting one.")
    with b:
        ui.anim_intro("Step by step: fill the kernel one row at a time")
        base = go.Figure()
        base.update_yaxes(autorange="reversed", visible=False, scaleanchor="x", scaleratio=1)
        base.update_xaxes(visible=False)
        base.update_layout(height=430, margin=dict(l=0, r=0))
        st.plotly_chart(anim.step_player(base, outer_product_steps(wmid), n_sub=12, wrap_at=60), key="sb_outer")


derive_section()


# ------------------------------------------------------------------ run it
@live.section
def run_section():
    st.subheader("Run it on an image", divider="gray")
    c_ctrl, c_main = st.columns([0.8, 3], gap="large")
    with c_ctrl:
        img, clean, _ = eui.pick_image("sb", "Coins (photo)", noise_default=0.0)
        op = st.radio("operator", ["Forward difference", "Central difference", "Roberts", "Prewitt", "Sobel"], index=4,
                      key="sb_op")
        norm = st.radio("magnitude", ["L2 = √(Iₓ² + I_y²)", "L1 ≈ |Iₓ| + |I_y|"], key="sb_norm")
        thr = live.slider("threshold on ‖∇I‖ (fraction of max)", 0.02, 0.6, 0.2, 0.01, key="sb_thr")
    gx, gy = E.gradients(img, op)
    M = E.magnitude(gx, gy, "L2" if norm.startswith("L2") else "L1")
    edges = M >= thr * M.max()
    with c_main:
        vmax = np.percentile(np.abs(np.r_[gx.ravel(), gy.ravel()]), 99.5)
        eui.show_row([(E.signed_rgb(gx, vmax), "Iₓ = Gₓ ⋆ I  (blue < 0 < red)"),
                      (E.signed_rgb(gy, vmax), "I_y = G_y ⋆ I"),
                      (E.to_u8(M, 0, np.percentile(M, 99.7)), f"‖∇I‖ ({norm[:2]})"),
                      (E.edges_rgb(edges), f"‖∇I‖ ≥ {thr:.2f}·max  →  edge map")])
        st.caption("Iₓ lights up on vertical edges (red = getting brighter to the right, blue = darker), I_y on "
                   "horizontal ones. Look at the binary map: edges are **several pixels thick**, and moving the "
                   "threshold trades missing faint edges against thick, noisy ones.")


run_section()


# ------------------------------------------------------------------ why 1 2 1
def pascal_steps():
    """(1, 1) ⊛ (1, 1): slide one box over the other and add the overlaps → (1, 2, 1)."""
    sig = np.array([1.0, 1.0])

    def state(offset, outs):
        kx = np.array([0.0, 1.0]) + offset
        out = np.full(3, np.nan)
        out[:len(outs)] = outs
        return [go.Bar(x=kx, y=[1.0, 1.0], width=0.55, marker_color=theme.rgba(theme.HIGHLIGHT, 0.75),
                       xaxis="x", yaxis="y", hoverinfo="skip"),
                go.Bar(x=[-1.0, 0.0, 1.0], y=out, width=0.55, marker_color=theme.TRANSFORMED, xaxis="x2", yaxis="y2",
                       text=[("" if np.isnan(v) else f"{v:g}") for v in out], textposition="outside", hoverinfo="skip")]

    H = anim.hl
    return [
        anim.Step("Two copies of (1, 1)", "grey = (1, 1) fixed,  amber = (1, 1) sliding",
                  "Convolution: slide one over the other; at every shift multiply the overlapping bars and add.",
                  state(-2.0, [])),
        anim.Step("Shift −1: one bar overlaps", f"1·1 = {H('1')}", "Only the right amber bar lies on the left grey bar.",
                  state(-1.0, [1.0])),
        anim.Step("Shift 0: both overlap", f"1·1 + 1·1 = {H('2')}", "Perfect overlap: two products to add - the "
                  "centre weight becomes 2.", state(0.0, [1.0, 2.0])),
        anim.Step("Shift +1: one bar overlaps", f"1·1 = {H('1')}   →   (1, 2, 1)",
                  "Averaging two neighbours, twice, gives Sobel's smoothing (1, 2, 1). Do it again and you get "
                  "(1, 3, 3, 1), then (1, 4, 6, 4, 1) …", state(1.0, [1.0, 2.0, 1.0])),
    ]


@live.section
def pascal_section():
    st.subheader("Why 1, 2, 1? Pascal's triangle walks towards the Gaussian", divider="gray")
    c_ctrl, c_fig, c_txt = st.columns([0.8, 1.6, 1.1], gap="large")
    with c_ctrl:
        n = live.slider("convolve (1, 1) with itself n times", 1, 14, 2, key="sb_n")
    bk = E.binomial(n)
    xk = np.arange(len(bk)) - n / 2
    with c_fig:
        xf = np.linspace(xk.min() - 1, xk.max() + 1, 300)
        sd = np.sqrt(n / 4)
        fig = go.Figure(go.Bar(x=xk, y=bk, marker_color=theme.TRANSFORMED, name=f"binomial row n = {n}",
                               hovertemplate="%{y:.4f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=xf, y=np.exp(-xf ** 2 / (2 * sd ** 2)) / (np.sqrt(2 * np.pi) * sd), mode="lines",
                                 line=dict(color=theme.E1, width=2.5), name=f"Gaussian with σ² = n/4 = {n / 4:g}"))
        fig.update_layout(height=320, margin=dict(l=8, r=8, t=20, b=8), bargap=0.15,
                          legend=dict(orientation="h", y=1.08, x=0))
        st.plotly_chart(fig, key="sb_binom")
    with c_txt:
        rows = [E.binomial(k) * 2 ** k for k in range(1, 6)]
        st.markdown("\n".join(f"- n = {k + 1}: ({', '.join(str(int(v)) for v in r)})" for k, r in enumerate(rows)))
        ui.callout("intuition",
                   "(1, 2, 1) is (1, 1) ⊛ (1, 1): averaging two neighbours, twice. Keep doing it and the weights become "
                   "a bell curve (the central limit theorem). So Sobel's (1, 2, 1) is the **smallest Gaussian-like "
                   "smoother** there is; Prewitt's (1, 1, 1) is a flat box.")

    ui.anim_intro("Step by step: where the 2 in (1, 2, 1) comes from")
    from plotly.subplots import make_subplots

    base = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.16,
                         subplot_titles=["the fixed (1, 1) and the sliding (1, 1)", "result of each shift"])
    base.add_trace(go.Bar(x=[-0.0, 1.0], y=[1.0, 1.0], width=0.8, marker_color=theme.rgba(theme.ORIGINAL, 0.55),
                          hoverinfo="skip"), row=1, col=1)
    base.update_layout(height=470, showlegend=False, barmode="overlay")
    base.update_xaxes(range=[-2.8, 2.8], dtick=1)
    base.update_yaxes(range=[0, 1.5], row=1, col=1)
    base.update_yaxes(range=[0, 2.7], row=2, col=1)
    base.update_annotations(yanchor="top", font_size=12)
    left, right = st.columns([1.4, 1], gap="large")
    with left:
        st.plotly_chart(anim.step_player(base, pascal_steps(), n_sub=14, wrap_at=75), key="sb_pascal_steps")
    with right:
        ui.callout("note",
                   "Each output position collects the products of the bars that overlap. The middle position has the "
                   "most overlap, so it gets the biggest weight - repeat the process and the middle keeps winning: "
                   "that is how the bell shape emerges.", title="Why the middle grows")


pascal_section()


# ------------------------------------------------------------------ measure Prewitt vs Sobel
@st.cache_data(show_spinner="Measuring direction errors …")
def direction_errors():
    angs = np.arange(0, 91, 5)
    errs = {}
    for opn in ["Central difference", "Prewitt", "Sobel"]:
        e = []
        for a_ in angs:
            errs_off = []
            for off_ in (0.0, 0.25, 0.5, 0.75):              # average over sub-pixel edge positions
                S = E.step_image(65, a_, 1.0, offset=off_)   # a slightly soft (anti-aliased) edge
                ex, ey = E.gradients(S, opn)
                mm = E.magnitude(ex, ey)
                sel = mm > 0.5 * mm.max()
                sel[:8] = sel[-8:] = False
                sel[:, :8] = sel[:, -8:] = False
                th = np.degrees(np.arctan2(ey, ex))[sel]
                errs_off.append(np.sqrt(np.mean((th + a_) ** 2)))
            e.append(np.mean(errs_off))
        errs[opn] = np.array(e)
    return angs, errs


st.subheader("Prewitt vs Sobel - measured, not just claimed", divider="gray")
m1, m2 = st.columns(2, gap="large")
with m1:
    angs, errs = direction_errors()
    fig = eui.signal_figure([dict(x=angs, y=errs["Central difference"], name="central difference", color=theme.ORIGINAL),
                             dict(x=angs, y=errs["Prewitt"], name="Prewitt", color=theme.E1),
                             dict(x=angs, y=errs["Sobel"], name="Sobel", color=theme.TRANSFORMED)],
                            height=320, title="Error in the estimated edge DIRECTION",
                            x_title="edge orientation (degrees)", y_title="RMS θ error (degrees)")
    st.plotly_chart(fig, key="sb_iso")
    st.caption(f"Mean direction error: Prewitt {errs['Prewitt'].mean():.2f}°, Sobel {errs['Sobel'].mean():.2f}°. "
               "Sobel's centre-weighted smoothing treats all orientations more evenly (it is closer to a round, "
               "Gaussian-shaped smoother), so its gradient directions are more trustworthy - which Canny's "
               "non-maximum suppression relies on.")
with m2:
    om = np.linspace(0, np.pi, 200)
    fig = eui.signal_figure([dict(x=om, y=(1 + 2 * np.cos(om)) / 3, name="Prewitt smoothing ⅓(1, 1, 1)", color=theme.E1),
                             dict(x=om, y=np.cos(om / 2) ** 2, name="Sobel smoothing ¼(1, 2, 1)", color=theme.TRANSFORMED)],
                            height=320, title="How much of each ripple frequency survives the smoothing",
                            x_title="frequency ω (π = pixel-to-pixel flicker)", y_title="gain")
    fig.add_hline(y=0, line=dict(color=theme.AXIS, width=1))
    st.plotly_chart(fig, key="sb_freq")
    st.caption("Sobel's weights remove the fastest flicker completely (gain 0 at ω = π). The flat Prewitt box lets "
               "a third of it through - with its sign flipped. For plain white noise the two reduce the overall "
               "noise level about equally (Prewitt even slightly more), so 'Sobel is more robust' really means: "
               "smoother, more isotropic, fewer high-frequency artefacts.")
st.markdown(
    "| | Prewitt | Sobel |\n|---|---|---|\n| Derivative | yes | yes |\n| Smoothing | uniform (1, 1, 1) | weighted (1, 2, 1) |\n"
    "| Kernel size | 3×3 | 3×3 |\n| Noise robustness (notes) | moderate | generally better |\n"
    "| Direction accuracy (measured above) | lower | higher |")


# ------------------------------------------------------------------ limitations
@live.section
def limitations_section():
    st.subheader("What first-order operators still get wrong", divider="gray")
    c_ctrl, c_fig, c_txt = st.columns([0.8, 1.6, 1.1], gap="large")
    with c_ctrl:
        blur_l = live.slider("edge blur", 0.5, 4.0, 2.0, 0.25, key="sb_lb")
        thr_l = live.slider("threshold", 0.05, 0.9, 0.3, 0.05, key="sb_lt")
        nz_l = live.slider("noise ", 0.0, 0.08, 0.02, 0.01, key="sb_ln")
    S1 = E.add_noise(E.step_image(64, 0, blur_l), nz_l, 4)
    ex, ey = E.gradients(S1, "Sobel")
    prof = E.magnitude(ex, ey)[32]
    with c_fig:
        fig = eui.signal_figure([dict(x=np.arange(64), y=prof / prof.max(), name="‖∇I‖ across the edge (normalised)",
                                      color=theme.TRANSFORMED, mode="lines+markers", size=6)],
                                height=320, hlines=[(thr_l, "threshold")], y_title="‖∇I‖ / max")
        above = np.where(prof / prof.max() >= thr_l)[0]
        for x_ in above:
            fig.add_vrect(x0=x_ - 0.5, x1=x_ + 0.5, fillcolor=theme.rgba(theme.HIGHLIGHT, 0.25), line_width=0)
        st.plotly_chart(fig, key="sb_lim")
    with c_txt:
        st.metric("pixels marked as 'edge' on this one row", len(above), border=True,
                  help="An ideal detector would mark exactly 1.")
        st.markdown("1. **Sensitive to noise** - spikes cross the threshold.\n"
                    "2. **Thick edges** - every pixel on the hump's shoulders passes.\n"
                    "3. **A threshold must be chosen** by hand.\n"
                    "4. **No explicit localisation** - which of those pixels *is* the edge?\n"
                    "5. **Several nearby responses** for one true edge.")
        st.caption("The fixes: smooth first (lesson 5), keep only the hump's peak (non-maximum suppression, lesson 6) "
                   "and use two thresholds (lesson 7). That is Canny.")


limitations_section()

ui.whys([
    ("Why does Sobel give the middle row weight 2?",
     "The middle row passes through the pixel we are evaluating, so its slope is the most relevant; the rows above "
     "and below are a little further away. Weighting (1, 2, 1) is the smallest symmetric, Gaussian-like average - "
     "and it makes the operator more isotropic (equal treatment of all edge directions)."),
    ("Why are the kernels 3×3 and not 2×2?",
     "A 3×3 kernel has a centre pixel, so the result belongs to that pixel and the derivative is the symmetric "
     "central difference. A 2×2 kernel (Roberts) sits between four pixels, so its output is shifted by half a "
     "pixel and it is more sensitive to noise."),
    ("Why approximate ‖∇I‖ by |Iₓ| + |I_y|?",
     "It avoids a square root per pixel - a real saving on old hardware or tiny chips. The price: for a diagonal "
     "edge |cos 45°| + |sin 45°| = 1.41, so diagonal edges look 41 % stronger than vertical ones of equal "
     "contrast. Try the L1 option above on the coins."),
    ("Why do we need both Gₓ and G_y?",
     "Gₓ only responds to brightness changing left↔right (vertical edges), G_y only to changes up↕down. An edge at "
     "any other angle shows up in both, and together (Iₓ, I_y) give its strength and direction."),
    ("Why is the raw Sobel response 8 times too big (and why care)?",
     "The weights add up to 1 + 2 + 1 = 4 rows-worth of smoothing and the derivative spans 2 pixels, so Gₓ ⋆ I ≈ "
     "8·∂I/∂x. For finding edges only the relative size matters, but if you need real slope units divide by 8. "
     "It is also why OpenCV's Canny thresholds look like 100 and 200 on 0-255 images."),
    ("Why does smoothing make edges thicker?",
     "Smoothing spreads the sudden jump over several pixels, so the derivative is non-zero on several pixels. "
     "That trade-off - less noise, but blurrier location - is exactly what Canny formalised."),
])

st.divider()
ui.mcq("sb1", "Which product gives the Sobel kernel Gₓ?",
       ["(−1, 0, 1)ᵀ (1, 2, 1)", "(1, 2, 1)ᵀ (−1, 0, 1)", "(1, 1, 1)ᵀ (−1, 0, 1)", "(1, 2, 1)ᵀ (1, 2, 1)"], 1,
       "Column = smoothing down the rows, row = derivative across the columns: S D with S = (1, 2, 1)ᵀ, D = (−1, 0, 1). "
       "(1, 1, 1)ᵀ gives Prewitt.")
ui.mcq("sb2", "A pure vertical edge filtered with Gₓ and G_y gives …",
       ["|Iₓ| ≫ |I_y|", "|I_y| ≫ |Iₓ|", "Iₓ = I_y", "both zero"], 0,
       "Brightness changes left↔right only, which Gₓ measures; G_y sees nothing changing up↕down.")

ui.lesson_nav(__file__)
