import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core import edge_ui as eui
from core import edges as E
from core import anim, live, theme, ui

ui.lesson_header(
    __file__,
    "A computer image is not a smooth function - it is a grid of numbers I[i, j]. There is no limit h → 0 to "
    "take, so derivatives become <b>finite differences</b> between neighbouring pixels, and finite differences "
    "become small <b>kernels</b> that slide over the image.",
)

# ------------------------------------------------------------------ finite differences
f = lambda x: 0.5 + 0.4 * np.tanh((x - 10.0) / 2.5)
fp = lambda x: 0.4 / 2.5 / np.cosh((x - 10.0) / 2.5) ** 2


@live.section
def differences_section():
    st.subheader("From the limit to a difference of neighbours", divider="gray")
    st.latex(r"\frac{\partial I}{\partial x} = \lim_{h\to 0}\frac{I(x+h, y) - I(x, y)}{h}"
             r"\;\;\xrightarrow{\;h\,=\,1\text{ pixel}\;}\;\; I(x+1, y) - I(x, y)"
             r"\quad\Longleftrightarrow\quad K_x = \begin{bmatrix} -1 & 1 \end{bmatrix},\;"
             r"K_y = \begin{bmatrix} -1 \\ 1 \end{bmatrix}")
    c_ctrl, c_fig, c_txt = st.columns([0.75, 1.6, 1.05], gap="large")
    with c_ctrl:
        xp = live.slider("pixel x", 1, 19, 8, key="dv_x")
        show = st.multiselect("show", ["true tangent", "forward difference", "central difference"],
                              default=["true tangent", "forward difference", "central difference"], key="dv_show")
    xs_fine = np.linspace(0, 20, 400)
    xs = np.arange(0, 21)
    fwd = f(xp + 1) - f(xp)
    cen = (f(xp + 1) - f(xp - 1)) / 2
    tru = fp(xp)
    with c_fig:
        fig = eui.signal_figure([dict(x=xs_fine, y=f(xs_fine), name="the smooth intensity the camera saw",
                                      color=theme.ORIGINAL, dash="dot"),
                                 dict(x=xs, y=f(xs), name="pixels I[x]", color=theme.ORIGINAL, mode="markers", size=8)],
                                height=380, y_title="I")
        span = np.array([xp - 2.5, xp + 2.5])
        if "true tangent" in show:
            fig.add_trace(go.Scatter(x=span, y=f(xp) + tru * (span - xp), mode="lines",
                                     name=f"tangent: slope {tru:.4f}", line=dict(color=theme.E2, width=2.5)))
        if "forward difference" in show:
            fig.add_trace(go.Scatter(x=span, y=f(xp) + fwd * (span - xp), mode="lines",
                                     name=f"forward (x, x+1): {fwd:.4f}", line=dict(color=theme.E1, width=2, dash="dash")))
        if "central difference" in show:
            fig.add_trace(go.Scatter(x=span, y=f(xp) + cen * (span - xp), mode="lines",
                                     name=f"central (x−1, x+1): {cen:.4f}", line=dict(color=theme.TRANSFORMED, width=2)))
        fig.add_trace(go.Scatter(x=[xp], y=[f(xp)], mode="markers", marker=dict(size=13, color=theme.HIGHLIGHT),
                                 name="x", showlegend=False))
        st.plotly_chart(fig, key="dv_fd")
    with c_txt:
        st.dataframe({"estimate": ["forward I(x+1) − I(x)", "backward I(x) − I(x−1)", "central ½[I(x+1) − I(x−1)]"],
                      "value": [f"{fwd:.4f}", f"{f(xp) - f(xp - 1):.4f}", f"{cen:.4f}"],
                      "error vs true slope": [f"{abs(fwd - tru):.4f}", f"{abs(f(xp) - f(xp - 1) - tru):.4f}",
                                              f"{abs(cen - tru):.4f}"]}, hide_index=True, width="stretch")
        st.caption(f"True slope at x = {xp}: {tru:.4f}")
        ui.callout("intuition",
                   "The forward difference measures the slope *between* x and x+1 - half a pixel to the right of "
                   "where you asked. The central difference looks one pixel each way, is centred exactly on x, and is "
                   "usually far closer to the tangent. Kernel: ½ [−1 0 1].")

    # ---- live: the limit h → 0
    ui.anim_intro("Live: shrink h and watch the secant turn into the tangent",
                  f"At x = {xp}. Drag the slider under the picture from h = 4 down towards 0. On a continuous curve "
                  "the secant slope converges to the tangent. On pixels we must STOP at h = 1 (amber) - there is no "
                  "smaller step.")
    base = eui.signal_figure([dict(x=xs_fine, y=f(xs_fine), name="I(x)", color=theme.ORIGINAL, dash="dot"),
                              dict(x=xs, y=f(xs), name="pixels", color=theme.ORIGINAL, mode="markers", size=7)],
                             height=420, y_title="I", legend=False)
    span2 = np.array([xp - 4.5, xp + 6.0])
    base.add_trace(go.Scatter(x=span2, y=f(xp) + tru * (span2 - xp), mode="lines", showlegend=False,
                              line=dict(color=theme.E2, width=2), hoverinfo="skip"))
    base.update_layout(hovermode=False)
    base.update_yaxes(range=[0.0, 1.0])
    hs = np.round(np.r_[np.arange(4.0, 1.0, -0.25), np.arange(1.0, 0.04, -0.05)], 3)

    def make(h):
        sl = (f(xp + h) - f(xp)) / h
        trs = [go.Scatter(x=span2, y=f(xp) + sl * (span2 - xp), mode="lines",
                          line=dict(color=theme.HIGHLIGHT if abs(h - 1) < 1e-9 else theme.E1, width=2.6),
                          showlegend=False, hoverinfo="skip"),
               go.Scatter(x=[xp, xp + h], y=[f(xp), f(xp + h)], mode="markers",
                          marker=dict(size=12, color=theme.E1, line=dict(color="white", width=1.5)),
                          showlegend=False, hoverinfo="skip")]
        tag = "   ← <b>h = 1 pixel: the finite difference</b>" if abs(h - 1) < 1e-9 else ""
        cap = (f"h = <b>{h:.2f}</b>   →   (I(x + h) − I(x)) / h = <b>{sl:.4f}</b>      tangent slope = "
               f"<span style='color:{theme.E2}'><b>{tru:.4f}</b></span>{tag}")
        return trs, cap

    st.plotly_chart(anim.scrubber(base, hs, make, active=0, label="h", fmt="{:.2f}", frame_ms=70), key="dv_limit")


differences_section()


# ------------------------------------------------------------------ sliding kernel 1D
KER = {"forward  [−1, 1]": np.array([-1.0, 1.0]), "central  ½[−1, 0, 1]": np.array([-0.5, 0.0, 0.5]),
       "smoothing  ¼[1, 2, 1]": np.array([0.25, 0.5, 0.25]), "second derivative  [1, −2, 1]": np.array([1.0, -2.0, 1.0])}


def slide_scrubber(sig, k, anchor, out):
    """Drag the kernel along the signal; the output appears one bar at a time behind it."""
    from plotly.subplots import make_subplots

    base = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.14,
                         subplot_titles=["input I[x]", "output (I ⋆ k)[x], built up as the kernel slides"])
    base.update_layout(height=520, showlegend=False, bargap=0.15)
    base.update_xaxes(dtick=1, range=[-0.6, 23.6])
    lo = min(0.0, float(out.min())) - 0.1
    base.update_yaxes(range=[min(0, sig.min()) - 0.05, max(1.0, sig.max()) + 0.25], row=1, col=1)
    base.update_yaxes(range=[lo, max(0.1, float(out.max())) + 0.15], row=2, col=1)
    base.update_annotations(yanchor="top", font_size=12)      # subplot titles inside: the caption sits above

    def make(pos):
        idx = [int(np.clip(pos + j - anchor, 0, 23)) for j in range(len(k))]
        col_in = [theme.HIGHLIGHT if i in idx else theme.ORIGINAL for i in range(24)]
        shown = np.where(np.arange(24) <= pos, out, np.nan)
        col_out = [theme.TRANSFORMED if i == pos else theme.rgba(theme.TRANSFORMED, 0.35) for i in range(24)]
        wx = np.array(idx, float)
        trs = [go.Bar(x=np.arange(24), y=sig, marker_color=col_in, xaxis="x", yaxis="y", hoverinfo="skip"),
               go.Scatter(x=wx, y=sig[idx] + 0.12, mode="text", text=[f"×{w:g}" for w in k], xaxis="x", yaxis="y",
                          textfont=dict(size=13), hoverinfo="skip"),
               go.Bar(x=np.arange(24), y=shown, marker_color=col_out, xaxis="x2", yaxis="y2", hoverinfo="skip")]
        terms = " + ".join(f"({k[j]:g})·{sig[ii]:.2f}" for j, ii in enumerate(idx))
        return trs, f"position i = <b>{pos}</b>:   out[{pos}] = {terms} = <b>{out[pos]:.3f}</b>"

    return anim.scrubber(base, list(range(1, 23)), make, active=0, label="i", fmt="{:d}", frame_ms=120)


def multiply_add_steps(sig, k, anchor, pos, out):
    """At one position: lay the kernel down, multiply each sample by its weight, add the products."""
    idx = [int(np.clip(pos + j - anchor, 0, 23)) for j in range(len(k))]
    vals = sig[idx]
    prods = k * vals
    n = len(k)
    xs_in = np.arange(n, dtype=float)              # the window, drawn at x = 0..n-1

    def state(heights, xs, widths, colors_alpha, show_sum):
        bars = go.Bar(x=xs, y=heights, width=widths, marker_color=[theme.rgba(theme.HIGHLIGHT, a) for a in colors_alpha],
                      text=[f"{h:.2f}" for h in heights], textposition="outside", hoverinfo="skip")
        res = go.Bar(x=[n + 1.0], y=[out[pos] * show_sum], width=[0.7], marker_color=theme.TRANSFORMED,
                     text=[f"{out[pos]:.3f}" if show_sum > 0.5 else ""], textposition="outside", hoverinfo="skip")
        return [bars, res]

    H = anim.hl
    stack_x = np.full(n, n + 1.0)
    return [
        anim.Step("The samples under the kernel", " , ".join(f"I[{i}] = {v:.2f}" for i, v in zip(idx, vals)),
                  f"Kernel position i = {pos}: the window covers {n} samples.",
                  state(vals, xs_in, np.full(n, 0.7), np.ones(n), 0.0)),
        anim.Step("Multiply each by its weight", " , ".join(f"{H(f'{w:g}')}·{v:.2f} = {p:.2f}"
                                                         for w, v, p in zip(k, vals, prods)),
                  "Negative weights flip a bar below zero; a zero weight wipes it out.",
                  state(prods, xs_in, np.full(n, 0.7), np.ones(n), 0.0)),
        anim.Step("Add the products", f"out[{pos}] = " + " + ".join(f"{p:.2f}" for p in prods) + f" = {H(f'{out[pos]:.3f}')}",
                  "The products slide together into ONE number - the output at this position (blue).",
                  state(prods, stack_x, np.full(n, 0.25), np.full(n, 0.25), 1.0)),
    ]


@live.section
def slide_section():
    st.subheader("A kernel slides along the signal: multiply, then add", divider="gray")
    c_ctrl, c_fig = st.columns([0.75, 2.6], gap="large")
    with c_ctrl:
        kname = st.radio("kernel", list(KER), index=1, key="dv_k")
        nz1 = live.slider("noise on the signal", 0.0, 0.1, 0.0, 0.01, key="dv_nz1")
        pos = live.slider("kernel position i (for the step-by-step view)", 1, 22, 10, key="dv_pos")
    k = KER[kname]
    sig = np.r_[np.full(12, 0.2), np.full(12, 0.8)] + np.random.default_rng(5).normal(0, nz1, 24)
    anchor = 0 if len(k) == 2 else 1
    out = np.array([sum(k[j] * sig[np.clip(i + j - anchor, 0, 23)] for j in range(len(k))) for i in range(24)])
    with c_fig:
        ui.anim_intro("Live: slide the kernel",
                      "Drag the slider under the picture (or press ▶ Play). The amber bars are the samples under the "
                      "kernel; each position produces one output bar.")
        st.plotly_chart(slide_scrubber(sig, k, anchor, out), key="dv_slide")
    ui.anim_intro(f"Step by step: what happens at position i = {pos}")
    base = go.Figure()
    base.update_layout(height=430, showlegend=False, bargap=0.2, barmode="overlay", margin=dict(l=10, r=10))
    lo = min(0.0, float(np.min(k * sig.max())), float(out[pos])) - 0.25
    hi = max(1.0, float(out[pos])) + 0.3
    base.update_yaxes(range=[lo, hi], gridcolor=theme.GRID, zerolinecolor=theme.AXIS)
    base.update_xaxes(range=[-0.7, len(k) + 1.7], tickvals=list(range(len(k))) + [len(k) + 1],
                      ticktext=[f"under weight {w:g}" for w in k] + ["output"])
    left, right = st.columns([1.4, 1], gap="large")
    with left:
        st.plotly_chart(anim.step_player(base, multiply_add_steps(sig, k, anchor, pos, out), n_sub=14, wrap_at=75),
                        key="dv_muladd")
    with right:
        terms = " + ".join(f"({k[j]:g})·{sig[int(np.clip(pos + j - anchor, 0, 23))]:.2f}" for j in range(len(k)))
        st.latex(rf"\text{{out}}[{pos}] = {terms} = \mathbf{{{out[pos]:.3f}}}")
        st.caption("This 'lay the kernel down, multiply, add' operation is called correlation. Libraries such as "
                   "OpenCV call it filtering; mathematicians call the flipped version convolution (∗).")


slide_section()


# ------------------------------------------------------------------ 2D neighbourhood
PATCH = E.step_image(9, 30, 0.8)
K2 = {"forward x  [−1 1]": np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0.0]]),
      "central x  ½[−1 0 1]": np.array([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0.0]]),
      "Sobel x": E.K["sobel_x"], "Sobel y": E.K["sobel_y"], "Laplacian": E.K["laplace4"]}


def neighbourhood_steps(nb, K):
    prod = nb * K
    total = prod.sum()
    m = max(np.abs(prod).max(), np.abs(total), 1e-9, np.abs(nb).max())

    def hm(Z, fmt="{:.2f}"):
        return go.Heatmap(z=Z, zmin=-m, zmax=m, colorscale=eui.DIVERGING, showscale=False, xgap=3, ygap=3,
                          text=[[fmt.format(v) for v in r] for r in Z], texttemplate="%{text}",
                          textfont=dict(size=15, color="#1a1a19"), hoverinfo="skip")

    collapsed = np.zeros((3, 3))
    collapsed[1, 1] = total
    H = anim.hl
    return [
        anim.Step("The 3×3 neighbourhood", "I[r−1 … r+1, c−1 … c+1]", "Nine pixel values around the chosen pixel.",
                  [hm(nb)]),
        anim.Step("Multiply element by element", "product = I ⊙ K",
                  "Every pixel is multiplied by the kernel weight lying on top of it. Watch zeros, sign flips and "
                  "the doubled centre row of Sobel.", [hm(prod)]),
        anim.Step("Sum everything into the centre", f"response = Σ product = {H(f'{total:.3f}')}",
                  "The nine products collapse into one number: the filter output at this pixel.", [hm(collapsed)]),
    ]


@live.section
def neighbourhood_section():
    st.subheader("In 2D: a 3×3 neighbourhood times a 3×3 kernel", divider="gray")
    c_ctrl, c_a, c_b, c_c = st.columns([0.8, 1, 1, 1], gap="medium")
    with c_ctrl:
        kn2 = st.selectbox("kernel", list(K2), index=2, key="dv_k2")
        r = live.slider("pixel row", 1, 7, 4, key="dv_r")
        c = live.slider("pixel column", 1, 7, 4, key="dv_c")
    K = K2[kn2]
    nb = PATCH[r - 1:r + 2, c - 1:c + 2]
    prod = nb * K
    with c_a:
        st.plotly_chart(eui.patch_figure(nb, "neighbourhood I", height=230, zmin=0, zmax=1), key="dv_nb")
    with c_b:
        st.plotly_chart(eui.kernel_figure(K, "kernel", height=230), key="dv_ker")
    with c_c:
        st.plotly_chart(eui.kernel_figure(prod, "element-wise product", fmt="{:.2f}", height=230), key="dv_prod")
    resp = E.correlate(PATCH, K)
    cc1, cc2 = st.columns([1, 1.4], gap="large")
    with cc1:
        st.latex(rf"\text{{response}} = \sum \text{{product}} = \mathbf{{{prod.sum():.3f}}}")
        ui.callout("note", "Every output pixel repeats this with its own neighbourhood. A flat neighbourhood gives 0 "
                           "for any derivative kernel, because their weights add up to 0.",
                   title="Why the response is 0 on flat areas")
        ui.anim_intro("Step by step: multiply, then sum")
        base = go.Figure()
        base.update_yaxes(autorange="reversed", visible=False, scaleanchor="x", scaleratio=1)
        base.update_xaxes(visible=False)
        base.update_layout(height=420, margin=dict(l=0, r=0))
        st.plotly_chart(anim.step_player(base, neighbourhood_steps(nb, K), n_sub=12, wrap_at=50), key="dv_nb_steps")
    with cc2:
        fig = eui.patch_figure(resp, "response of the whole 9×9 patch (selected pixel boxed)", height=300,
                               colorscale=eui.DIVERGING, zmin=-np.abs(resp).max(), zmax=np.abs(resp).max(), fmt="{:.2f}")
        fig.add_shape(type="rect", x0=c - 0.5, x1=c + 0.5, y0=r - 0.5, y1=r + 0.5,
                      line=dict(color=theme.HIGHLIGHT, width=3))
        st.plotly_chart(fig, key="dv_resp")


neighbourhood_section()


# ------------------------------------------------------------------ noise
@live.section
def noise_gain_section():
    st.subheader("Why derivatives hate noise", divider="gray")
    c_ctrl, c_fig, c_txt = st.columns([0.8, 1.6, 1.05], gap="large")
    with c_ctrl:
        w = live.slider("frequency ω of a ripple (radians / pixel)", 0.05, 3.0, 1.5, 0.05, key="dv_w")
        sig_g = live.slider("Gaussian σ for the smoothed derivative", 0.5, 3.0, 1.0, 0.1, key="dv_sg")
    om = np.linspace(0.001, np.pi, 300)
    with c_fig:
        gains = [dict(x=om, y=om, name="ideal derivative: gain ω", color=theme.ORIGINAL, dash="dot"),
                 dict(x=om, y=np.abs(2 * np.sin(om / 2)), name="forward difference", color=theme.E1),
                 dict(x=om, y=np.abs(np.sin(om)), name="central difference", color=theme.TRANSFORMED),
                 dict(x=om, y=np.abs(np.sin(om)) * np.cos(om / 2) ** 2, name="central diff. + ¼[1 2 1] smoothing",
                      color=theme.E2),
                 dict(x=om, y=om * np.exp(-(sig_g * om) ** 2 / 2), name=f"derivative of Gaussian (σ = {sig_g})",
                      color=theme.MAGENTA)]
        fig = eui.signal_figure(gains, height=380, x_title="frequency ω (0 = flat, π = alternates every pixel)",
                                y_title="amplification of a ripple", vlines=[(w, "your ω")])
        fig.add_vrect(x0=0, x1=0.4, fillcolor=theme.rgba(theme.E2, 0.08), line_width=0,
                      annotation_text="edges live here", annotation_position="top left")
        fig.add_vrect(x0=1.6, x1=np.pi, fillcolor=theme.rgba(theme.OUTLIER, 0.07), line_width=0,
                      annotation_text="pixel noise lives here (too)", annotation_position="top right")
        st.plotly_chart(fig, key="dv_gain")
    with c_txt:
        st.latex(r"\frac{d}{dx}\,A\sin(\omega x) = A\,\omega\cos(\omega x)")
        st.markdown(f"A ripple of amplitude A = 0.02 at ω = {w:.2f} becomes a derivative ripple of amplitude "
                    f"**{0.02 * w:.4f}** - multiplied by ω. Pixel noise contains fast ripples (ω up to π), which an "
                    f"ideal derivative multiplies by up to ≈ 3 and the forward difference by up to 2, while a gentle "
                    f"edge (small ω) is multiplied by only a fraction.")
        ui.callout("intuition",
                   "Differentiating is a **high-pass filter**: the faster something wiggles, the more it is amplified. "
                   "Edges are big, slow-ish changes; noise is small but fast. A derivative alone boosts the wrong "
                   "thing. Adding smoothing (green, pink) bends the curve back down at high ω - that is the whole "
                   "reason for larger kernels like Sobel, which smooth while they differentiate.")

    # ---- live: a ripple and its difference, sweeping ω
    ui.anim_intro("Live: sweep the ripple frequency ω",
                  "Top: a small ripple 0.1·sin(ωx) sampled at the pixels. Bottom: its forward difference. Drag the "
                  "slider under the picture: slow ripples barely survive the derivative, fast ones come out BIGGER "
                  "than they went in.")
    from plotly.subplots import make_subplots

    base = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                         subplot_titles=["ripple I[x] = 0.1·sin(ωx)", "forward difference I[x+1] − I[x]"])
    base.update_layout(height=520, showlegend=False)
    base.update_yaxes(range=[-0.23, 0.23], gridcolor=theme.GRID, zerolinecolor=theme.AXIS)
    base.update_xaxes(range=[-0.5, 40.5], gridcolor=theme.GRID)
    base.update_annotations(yanchor="top", font_size=12)      # subplot titles inside: the caption sits above
    xs = np.arange(41, dtype=float)
    xf = np.linspace(0, 40, 400)
    oms = np.round(np.arange(0.1, 3.1001, 0.1), 2)

    def make(wv):
        I = 0.1 * np.sin(wv * xs)
        d = np.r_[np.diff(I), np.nan]
        trs = [go.Scatter(x=xf, y=0.1 * np.sin(wv * xf), mode="lines", line=dict(color=theme.ORIGINAL, dash="dot", width=1),
                          xaxis="x", yaxis="y", hoverinfo="skip"),
               go.Scatter(x=xs, y=I, mode="markers+lines", line=dict(color=theme.ORIGINAL, width=1.5),
                          marker=dict(size=6, color=theme.ORIGINAL), xaxis="x", yaxis="y", hoverinfo="skip"),
               go.Scatter(x=xs + 0.5, y=d, mode="markers+lines", line=dict(color=theme.E1, width=2),
                          marker=dict(size=6, color=theme.E1), xaxis="x2", yaxis="y2", hoverinfo="skip")]
        gain = 2 * abs(np.sin(wv / 2))
        cap = (f"ω = <b>{wv:.1f}</b> rad/pixel   →   difference amplitude = {gain:.2f} × input amplitude   "
               + ("(<b>amplified</b>)" if gain > 1 else "(shrunk)"))
        return trs, cap

    st.plotly_chart(anim.scrubber(base, oms, make, active=int(np.argmin(np.abs(oms - w))), label="ω", fmt="{:.1f}",
                                  frame_ms=110), key="dv_omega")


noise_gain_section()


@live.section
def noise_image_section():
    nz = live.slider("noise added to the image", 0.0, 0.15, 0.04, 0.01, key="dv_nz")
    img = E.add_noise(E.photo("cameraman.png"), nz, 2)
    gx, gy = E.gradients(img, "Forward difference")
    gs_x, gs_y = E.gradients(E.smooth(img, 1.2), "Central difference")
    m1 = E.magnitude(gx, gy)
    m2 = E.magnitude(gs_x, gs_y)
    eui.show_row([(np.clip(img, 0, 1), "noisy image"),
                  (E.to_u8(m1, 0, np.percentile(m1, 99.5)), "forward difference ‖∇I‖ - noise everywhere"),
                  (E.to_u8(m2, 0, np.percentile(m2, 99.5)), "smooth (σ = 1.2) then central difference")])


noise_image_section()

ui.whys([
    ("Why is the central difference more accurate than the forward difference?",
     "Taylor: I(x+1) = I + I′ + ½I″ + …, I(x−1) = I − I′ + ½I″ − …. The forward difference keeps the ½I″ error "
     "term; subtracting the two expansions cancels it, so the central error is only of order I‴/6. It is also "
     "centred on the pixel, so it does not shift the edge by half a pixel."),
    ("Why do all derivative kernels add up to zero?",
     "On a perfectly flat region every pixel has the same value c, and the response is c × (sum of weights). "
     "A derivative must be 0 there, so the weights must sum to 0. Smoothing kernels instead sum to 1, so they "
     "keep the brightness."),
    ("Correlation or convolution - which one is the 'real' filtering?",
     "Convolution flips the kernel before sliding; correlation does not. For symmetric kernels (Gaussian, "
     "Laplacian) they are identical. For derivative kernels the flip only changes the sign. OpenCV's filter2D, "
     "most deep-learning 'convolution' layers and this site all use correlation."),
    ("Why does differentiation amplify noise so much?",
     "Because it multiplies each frequency by ω (see the chart). Pixel noise is strongest at the highest "
     "frequencies, where the gain is largest. Small wiggles of ±0.02 between neighbouring pixels produce "
     "derivatives as large as a faint real edge."),
    ("What happens at the image border?",
     "The kernel would stick out of the image. We must invent values outside: zeros, copies of the edge pixel "
     "(replicate), or a mirror image (reflect - used here). Zeros create fake edges along the border; reflect "
     "does not."),
    ("Why can the response be negative, and how do I display it?",
     "A derivative is negative where the image gets darker. We show signed responses with a diverging map "
     "(blue < 0 < red, grey = 0), or take |·| / the magnitude when only the strength matters."),
])

st.divider()
ui.mcq("dv1", "The kernel [−1, 1] applied to the pixels (…, 0.2, 0.2, 0.8, 0.8, …) gives at the jump …",
       ["0", "0.6", "−0.6", "1.6"], 1, "(−1)·0.2 + 1·0.8 = 0.6 - the size of the jump.")
ui.mcq("dv2", "Why are larger derivative kernels (like Sobel) used instead of [−1, 1]?",
       ["They are faster", "They smooth while differentiating, so noise is amplified less",
        "They detect colour", "They make edges thinner"], 1,
       "A derivative alone is a high-pass filter. Adding smoothing in the same kernel suppresses the high "
       "frequencies where noise lives.")

ui.lesson_nav(__file__)
