import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core import edge_ui as eui
from core import edges as E
from core import anim, live, plotting, theme, ui

ui.lesson_header(
    __file__,
    "The first derivative <b>peaks</b> at an edge. Where a function peaks, its own derivative passes through "
    "<b>zero</b>. So the second derivative marks an edge with a <b>zero crossing</b> - a sharp, single location. "
    "In 2D the second derivative becomes the Laplacian ∇²I = I_xx + I_yy.",
)

# ------------------------------------------------------------------ 1D picture
def zero_crossings_1d(d2):
    """Sign changes of d2 (an exact 0 between a + and a − counts once), ignoring tiny wiggles."""
    zc = []
    big = 0.25 * np.abs(d2).max()
    i = 1
    while i < len(d2) - 2:
        a_, b_, c_ = d2[i], d2[i + 1], d2[i + 2]
        if a_ * b_ < 0 and abs(a_ - b_) > big:
            zc.append(i + a_ / (a_ - b_))
            i += 2
            continue
        if b_ == 0 and a_ * c_ < 0 and abs(a_ - c_) > big:
            zc.append(i + 1.0)
            i += 3
            continue
        i += 1
    return zc


@live.section
def profile_section():
    st.subheader("Peak of the first derivative = zero crossing of the second", divider="gray")
    c_ctrl, c_fig = st.columns([0.8, 2.6], gap="large")
    with c_ctrl:
        blur = live.slider("edge blur", 0.5, 6.0, 2.0, 0.25, key="lp_blur")
        nz = live.slider("noise", 0.0, 0.03, 0.0, 0.002, key="lp_nz", format="%.3f")
        x0 = live.slider("edge position", 30, 70, 50, key="lp_x0")
    from math import erf
    xs = np.arange(100, dtype=float)
    I = np.array([0.2 + 0.6 * 0.5 * (1 + erf((x - x0) / (np.sqrt(2) * blur))) for x in xs])
    I = I + np.random.default_rng(7).normal(0, nz, xs.size)
    d1 = np.gradient(I)
    d2 = np.r_[0, I[2:] - 2 * I[1:-1] + I[:-2], 0]
    peak = int(np.argmax(np.abs(d1)))
    zc = zero_crossings_1d(d2)
    with c_fig:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                            subplot_titles=["I(x) - the edge", "I′(x) - first derivative: a PEAK",
                                            "I″(x) - second derivative: + then −, crossing ZERO"])
        fig.add_trace(go.Scatter(x=xs, y=I, line=dict(color=theme.ORIGINAL, width=2.4), name="I"), row=1, col=1)
        fig.add_trace(go.Scatter(x=xs, y=d1, line=dict(color=theme.TRANSFORMED, width=2.4), name="I′"), row=2, col=1)
        fig.add_trace(go.Scatter(x=xs, y=d2, line=dict(color=theme.E1, width=2.4), name="I″"), row=3, col=1)
        fig.add_hline(y=0, line=dict(color=theme.AXIS, width=1), row=3, col=1)
        for r in (1, 2, 3):
            fig.add_vline(x=peak, line=dict(color=theme.HIGHLIGHT, dash="dot", width=1.5), row=r, col=1)
        for z in zc:
            fig.add_trace(go.Scatter(x=[z], y=[0], mode="markers",
                                     marker=dict(size=12, color=theme.HIGHLIGHT, symbol="diamond"),
                                     name="zero crossing", showlegend=False), row=3, col=1)
        fig.update_layout(height=560, margin=dict(l=8, r=8, t=40, b=8), showlegend=False, hovermode="x unified")
        fig.update_xaxes(title="x (pixels)", row=3, col=1)
        st.plotly_chart(fig, key="lp_1d")
    a, b = st.columns(2, gap="large")
    with a:
        ui.callout("intuition",
                   "Climbing a hill: the slope (I′) is largest exactly half-way up - that is the edge. Before that "
                   "point the slope is *increasing* (I″ > 0), after it the slope is *decreasing* (I″ < 0). The moment "
                   "it switches is the zero crossing - the inflection point of I.")
    with b:
        st.metric("first-derivative peak at", f"x = {peak}", border=True)
        st.metric("second-derivative zero crossing(s)", ", ".join(f"{z:.1f}" for z in zc) or "none found",
                  border=True, help="With noise, extra crossings appear - watch the noise slider.")


profile_section()


# ------------------------------------------------------------------ derive the kernel
def diff_of_diff_steps(v_m, v_0, v_p):
    fwd, bwd = v_p - v_0, v_0 - v_m
    d2 = fwd - bwd

    def state(ahead, behind, result):
        return [
            plotting.arrows_trace([[1.0, v_0]], [[1.0 + ahead * 1.0, v_0 + ahead * fwd]], theme.E1, width=4, size=15),
            plotting.arrows_trace([[0.0, v_m]], [[0.0 + behind * 1.0, v_m + behind * bwd]], theme.E2, width=4, size=15),
            go.Bar(x=[3.1], y=[result], width=[0.5], marker_color=theme.HIGHLIGHT, text=[f"{result:+.2f}" if result else ""],
                   textposition="outside", hoverinfo="skip"),
        ]

    H = anim.hl
    return [
        anim.Step("Three neighbouring pixels", f"I(x−1) = {v_m:.2f},  I(x) = {v_0:.2f},  I(x+1) = {v_p:.2f}",
                  "The second derivative asks: is the slope CHANGING here?", state(0, 0, 0.0)),
        anim.Step("The slope behind", f"I(x) − I(x−1) = {H(f'{bwd:+.2f}', theme.E2)}",
                  "Green arrow: how much we climbed going from x − 1 to x.", state(0, 1, 0.0)),
        anim.Step("The slope ahead", f"I(x+1) − I(x) = {H(f'{fwd:+.2f}', theme.E1)}",
                  "Orange arrow: how much we climb from x to x + 1.", state(1, 1, 0.0)),
        anim.Step("Difference of the two slopes", f"I″ ≈ ({fwd:+.2f}) − ({bwd:+.2f}) = {H(f'{d2:+.2f}')}",
                  "= I(x+1) − 2·I(x) + I(x−1): the kernel [1, −2, 1]. Positive = the slope is increasing (bending "
                  "up), negative = decreasing, zero = a straight ramp.", state(1, 1, d2)),
    ]


def kernel_sum_steps():
    Kx = np.array([[0, 0, 0], [1.0, -2, 1], [0, 0, 0]])
    Ky = Kx.T
    L4 = Kx + Ky
    L8 = E.K["laplace8"] if "laplace8" in E.K else np.array([[1.0, 1, 1], [1, -8, 1], [1, 1, 1]])

    def hm(Z):
        return go.Heatmap(z=Z, zmin=-8, zmax=8, colorscale=eui.DIVERGING, showscale=False, xgap=4, ygap=4,
                          text=[[f"{v:g}" for v in r] for r in Z], texttemplate="%{text}",
                          textfont=dict(size=18, color="#1a1a19"), hoverinfo="skip")

    H = anim.hl
    return [
        anim.Step("∂²/∂x²: [1, −2, 1] along the row", "I(x+1,y) − 2I(x,y) + I(x−1,y)",
                  "The 1D kernel, laid across the middle row.", [hm(Kx)]),
        anim.Step("+ ∂²/∂y²: the same down the column", "I(x,y+1) − 2I(x,y) + I(x,y−1)",
                  "The vertical version on its own.", [hm(Ky)]),
        anim.Step("Add them: the 4-neighbour Laplacian", f"centre: −2 + (−2) = {H('−4')}",
                  "∇²I = I_xx + I_yy. The weights still add up to 0 (flat regions give 0).", [hm(L4)]),
        anim.Step("Include the diagonals: 8-neighbour version", f"centre {H('−8')}, every neighbour 1",
                  "Closer to the same response for every edge direction (more isotropic).", [hm(L8)]),
    ]


@live.section
def derive_section():
    st.subheader("Derive the discrete Laplacian: a difference of differences", divider="gray")
    c_ctrl, c_fig, c_txt = st.columns([0.8, 1.3, 1.3], gap="large")
    with c_ctrl:
        v_m = live.slider("I(x − 1)", 0.0, 1.0, 0.2, 0.05, key="lp_vm")
        v_0 = live.slider("I(x)", 0.0, 1.0, 0.35, 0.05, key="lp_v0")
        v_p = live.slider("I(x + 1)", 0.0, 1.0, 0.8, 0.05, key="lp_vp")
    fwd, bwd = v_p - v_0, v_0 - v_m
    with c_fig:
        fig = go.Figure(go.Bar(x=["x − 1", "x", "x + 1"], y=[v_m, v_0, v_p], marker_color=theme.ORIGINAL, width=0.5,
                               hovertemplate="%{y:.2f}<extra></extra>"))
        fig.add_annotation(x=1.5, y=max(v_0, v_p) + 0.12, text=f"slope ahead = {fwd:+.2f}", showarrow=False,
                           font=dict(color=theme.E1, size=13))
        fig.add_annotation(x=0.5, y=max(v_m, v_0) + 0.12, text=f"slope behind = {bwd:+.2f}", showarrow=False,
                           font=dict(color=theme.E2, size=13))
        fig.add_trace(go.Scatter(x=["x − 1", "x", "x + 1"], y=[v_m, v_0, v_p], mode="lines+markers",
                                 line=dict(color=theme.TRANSFORMED, width=2), marker=dict(size=10)))
        fig.update_layout(height=320, margin=dict(l=8, r=8, t=10, b=8), showlegend=False, yaxis=dict(range=[0, 1.25]))
        st.plotly_chart(fig, key="lp_bars")
    with c_txt:
        st.latex(r"\frac{\partial^2 I}{\partial x^2} \approx \underbrace{[I(x+1) - I(x)]}_{\text{slope ahead}} - "
                 r"\underbrace{[I(x) - I(x-1)]}_{\text{slope behind}}")
        st.latex(r"= I(x+1) - 2I(x) + I(x-1) \;\Longleftrightarrow\; \begin{bmatrix} 1 & -2 & 1 \end{bmatrix}")
        st.latex(rf"= {fwd:+.2f} - ({bwd:+.2f}) = \mathbf{{{fwd - bwd:+.2f}}}")
        st.caption("Positive: the slope is getting steeper (curving up, like the bottom of a valley). Negative: "
                   "getting flatter (the top of a hill). Zero: a straight ramp - or the exact middle of an edge.")

    s1, s2 = st.columns(2, gap="large")
    with s1:
        ui.anim_intro("Step by step: a difference of differences", "Uses the three values you set above.")
        base = go.Figure()
        base.add_trace(go.Bar(x=[0, 1, 2], y=[v_m, v_0, v_p], width=0.5, marker_color=theme.rgba(theme.ORIGINAL, 0.6),
                              hoverinfo="skip"))
        base.add_trace(go.Scatter(x=[0, 1, 2], y=[v_m, v_0, v_p], mode="markers", marker=dict(size=11,
                                  color=theme.ORIGINAL), hoverinfo="skip"))
        base.update_layout(height=470, showlegend=False, barmode="overlay", margin=dict(l=10, r=10))
        lo = min(0.0, fwd - bwd) - 0.2
        base.update_yaxes(range=[lo, 1.3], gridcolor=theme.GRID, zerolinecolor=theme.AXIS)
        base.update_xaxes(range=[-0.6, 3.7], tickvals=[0, 1, 2, 3.1], ticktext=["x − 1", "x", "x + 1", "I″(x)"])
        st.plotly_chart(anim.step_player(base, diff_of_diff_steps(v_m, v_0, v_p), n_sub=14, wrap_at=62),
                        key="lp_dd_steps")
    with s2:
        ui.anim_intro("Step by step: from [1, −2, 1] to the 2D Laplacian")
        base = go.Figure()
        base.update_yaxes(autorange="reversed", visible=False, scaleanchor="x", scaleratio=1)
        base.update_xaxes(visible=False)
        base.update_layout(height=470, margin=dict(l=0, r=0))
        st.plotly_chart(anim.step_player(base, kernel_sum_steps(), n_sub=12, wrap_at=62), key="lp_ksum")
    lc1, lc2 = st.columns([1.6, 1])
    lc1.latex(r"\nabla^2 I \approx I(x+1,y) + I(x-1,y) + I(x,y+1) + I(x,y-1) - 4I(x,y)")
    lc2.latex(r"\text{with diagonals: } L_8 = \begin{bmatrix} 1 & 1 & 1 \\ 1 & -8 & 1 \\ 1 & 1 & 1 \end{bmatrix}")


derive_section()


# ------------------------------------------------------------------ apply
@live.section
def apply_section():
    st.subheader("Apply it - and meet its weakness", divider="gray")
    c_ctrl, c_main = st.columns([0.8, 3], gap="large")
    with c_ctrl:
        img, clean, _ = eui.pick_image("lp", "Shapes (synthetic)", noise_default=0.0)
        kind = st.radio("kernel", ["4-neighbour (−4)", "8-neighbour (−8)"], key="lp_kind")
        zt = live.slider("zero-crossing threshold (fraction of max |∇²I|)", 0.0, 0.5, 0.1, 0.01, key="lp_zt")
    L = E.laplacian(img, 4 if kind.startswith("4") else 8)
    Z = E.zero_crossings(L, zt * np.abs(L).max())
    with c_main:
        eui.show_row([(np.clip(img, 0, 1), "input"),
                      (E.signed_rgb(L), "∇²I  (blue < 0 < red)"),
                      (E.edges_rgb(Z), f"zero crossings with jump > {zt:.2f}·max"),
                      (E.overlay(np.clip(img, 0, 1), Z), "overlay")])
        st.caption("On the clean shapes every edge becomes a red/blue double line with the zero crossing in between. "
                   "Now add even a little noise (σₙ = 0.02): the Laplacian explodes into speckle, and zero crossings "
                   "appear everywhere. Raising the threshold helps only a little - the next lesson shows the real fix.")


apply_section()

ui.whys([
    ("Why does the second derivative cross zero exactly at the edge?",
     "The edge is where the first derivative is largest. At a maximum a function stops increasing and starts "
     "decreasing, so its own derivative - the second derivative of I - changes sign there. That sign change is "
     "the zero crossing."),
    ("Why look for zero crossings instead of large Laplacian values?",
     "|∇²I| is large on *both sides* of an edge (the red and blue bands), not on it. The edge itself sits where "
     "the response passes through zero - a thin, well-localised line between the two bands."),
    ("Why is the centre weight −4 (or −8)?",
     "So the weights sum to zero: a flat region must give 0. The −4 is the two −2's of the x and y second "
     "differences added together; with diagonals there are 8 neighbours to balance."),
    ("Why is the Laplacian even more sensitive to noise than the gradient?",
     "Each derivative multiplies a ripple of frequency ω by ω. Two derivatives multiply it by ω². Pixel noise "
     "lives at high ω, so the second derivative amplifies it much more - the speckle you see above."),
    ("Why is the Laplacian called isotropic, and why is that useful?",
     "∇² = ∂²/∂x² + ∂²/∂y² gives the same value if you rotate the coordinate axes. One kernel therefore responds "
     "to edges of every orientation - no need for separate x and y kernels. The 8-neighbour version is a bit "
     "closer to that ideal than the 4-neighbour one."),
    ("What does the Laplacian NOT tell us?",
     "Its output is a single number per pixel: it says *where* (zero crossing) but not *which way* an edge runs, "
     "and its value says little about edge strength. The gradient gives both strength and direction."),
])

st.divider()
ui.mcq("lp1", "Apply the kernel [1, −2, 1] to the pixels (0.2, 0.2, 0.2). The response is …",
       ["0.2", "0", "−0.2", "0.6"], 1, "0.2 − 0.4 + 0.2 = 0: flat regions give no response, as for any derivative.")
ui.mcq("lp2", "At the exact centre of a blurred step edge, the second derivative is …",
       ["at its maximum", "zero (changing sign)", "equal to the first derivative", "negative and large"], 1,
       "The first derivative peaks there, so the second derivative crosses zero.")

ui.lesson_nav(__file__)
