import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core import edge_ui as eui
from core import edges as E
from core import anim, live, plotting, theme, ui

ui.lesson_header(
    __file__,
    "Think of an image as a <b>landscape</b>: I(x, y) is the height at every pixel. Objects are separated by "
    "<b>cliffs</b> - places where the intensity changes quickly. Edge detection is simply: <i>find the places "
    "where the image changes rapidly</i>. The derivative measures exactly that.",
)

# ------------------------------------------------------------------ the landscape
@live.section
def landscape_section():
    st.subheader("An image is a function I(x, y)", divider="gray")
    c_ctrl, c_img, c_surf = st.columns([0.8, 1.0, 1.6], gap="large")
    with c_ctrl:
        name = st.selectbox("Image", ["Cameraman (photo)", "Coins (photo)", "Shapes (synthetic)", "Cat (photo)"],
                            key="we_img")
        img = E.SAMPLES[name]()
        h, w = img.shape
        win = live.slider("window size (pixels)", 24, 96, 56, 8, key="we_win")
        # start the window on the strongest edge region of the chosen image
        gxx, gyy = E.gradients(E.smooth(img, 2.0), "Central difference")
        score = E.box_blur(E.magnitude(gxx, gyy), 31)
        score[:win // 2, :] = score[-win // 2:, :] = 0
        score[:, :win // 2] = score[:, -win // 2:] = 0
        by, bx = np.unravel_index(np.argmax(score), score.shape)
        cx = live.slider("window centre x", win // 2, w - win // 2 - 1, int(np.clip(bx, win // 2, w - win // 2 - 1)),
                         key=f"we_cx_{name}")
        cy = live.slider("window centre y", win // 2, h - win // 2 - 1, int(np.clip(by, win // 2, h - win // 2 - 1)),
                         key=f"we_cy_{name}")
    x0, y0 = cx - win // 2, cy - win // 2
    patch = img[y0:y0 + win, x0:x0 + win]
    with c_img:
        box = E.overlay(img, np.zeros_like(img, bool), dim=1.0)
        box[y0:y0 + win, [x0, min(x0 + win - 1, w - 1)]] = (235, 104, 52)
        box[[y0, min(y0 + win - 1, h - 1)], x0:x0 + win] = (235, 104, 52)
        st.image(ui.datauri(box), caption="The orange window is shown as a landscape →", width="stretch")
    with c_surf:
        fig = go.Figure(go.Surface(z=patch, colorscale=eui.GREYS, cmin=0, cmax=1, showscale=False,
                                   hovertemplate="x=%{x}, y=%{y}<br>I=%{z:.2f}<extra></extra>"))
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=0, b=0), uirevision="keep",
                          scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="I(x, y)",
                                     yaxis=dict(autorange="reversed"), zaxis=dict(range=[0, 1]),
                                     aspectratio=dict(x=1, y=1, z=0.55), camera=dict(eye=dict(x=1.4, y=-1.5, z=1.0))))
        st.plotly_chart(fig, key="we_surf")
    ui.callout("intuition",
               "Flat plains are regions of nearly constant brightness. **Cliffs** are edges. Rough, bumpy ground is "
               "texture or noise. Drag the window over a boundary (the cameraman's coat against the sky, a coin's "
               "rim) and watch a cliff appear. An edge detector is a cliff detector.")


landscape_section()


# ------------------------------------------------------------------ 1D: the step and its derivative
def step_profile(x0e, I1, I2, blur, nz):
    xs = np.arange(0, 100, dtype=float)
    if blur > 0:
        from math import erf
        prof = np.array([0.5 * (1 + erf((x - x0e + 0.5) / (np.sqrt(2) * blur))) for x in xs])
    else:
        prof = (xs >= x0e).astype(float)
    return xs, I1 + (I2 - I1) * prof + np.random.default_rng(3).normal(0, nz, xs.size)


def difference_steps(xs, I, x0e):
    """dI/dx ≈ I(x+1) − I(x), drawn as the little rises between samples that then drop to a baseline."""
    lo, hi = int(max(0, x0e - 10)), int(min(len(xs) - 1, x0e + 10))
    x = xs[lo:hi + 1]
    v = I[lo:hi + 1]
    d = v[1:] - v[:-1]
    xm = x[:-1] + 0.5
    k = int(np.argmax(np.abs(d)))

    def rises(base_lo, base_hi, width=4.0):
        X, Y = [], []
        for xi, a, b in zip(xm, base_lo, base_hi):
            X += [xi, xi, np.nan]
            Y += [a, b, np.nan]
        return go.Scatter(x=np.array(X), y=np.array(Y), mode="lines", line=dict(color=theme.TRANSFORMED, width=width),
                          showlegend=False, hoverinfo="skip")

    def state(stage):
        # stage 0: nothing; 1: rises drawn between neighbours; 2: rises dropped to the baseline; 3: + highlight
        if stage == 0:
            r = rises(v[:-1], v[:-1])
        elif stage == 1:
            r = rises(v[:-1], v[1:])
        else:
            r = rises(np.zeros_like(d), d)
        pk = go.Scatter(x=[xm[k]], y=[d[k] if stage >= 2 else v[k + 1]], mode="markers+text",
                        text=["max |dI/dx| → edge"], textposition="middle right",
                        marker=dict(size=14 if stage == 3 else 0.1, color=theme.HIGHLIGHT, symbol="diamond"),
                        textfont=dict(size=13 if stage == 3 else 1, color=theme.HIGHLIGHT),
                        showlegend=False, hoverinfo="skip")
        return [r, pk]

    H = anim.hl
    steps = [
        anim.Step("The sampled signal", "I(x) at every pixel", "Pixels are samples: the derivative must be built "
                  "from neighbouring samples.", state(0)),
        anim.Step("Take differences of neighbours", "dI/dx ≈ " + H("I(x + 1) − I(x)"),
                  "Each blue bar is the rise from one pixel to the next. Flat region → tiny rises; at the cliff → one "
                  "big rise.", state(1)),
        anim.Step("Collect the rises as a new signal", "the derivative signal",
                  "Drop every rise to the baseline: what you get is the derivative. It is ≈ 0 on the plains and peaks "
                  "on the cliff.", state(2)),
        anim.Step("The peak marks the edge", f"max |dI/dx| = {H(f'{abs(d[k]):.3f}')} at x = {xm[k]:.1f}",
                  "Edge detection in 1D = find the peaks of |dI/dx|.", state(3)),
    ]
    return steps, (x[0] - 0.8, x[-1] + 0.8), x, v


@live.section
def profile_section():
    st.subheader("In 1D: a step, and the derivative that finds it", divider="gray")
    c_ctrl, c_plot = st.columns([0.8, 2.4], gap="large")
    with c_ctrl:
        x0e = live.slider("edge position x₀", 20, 80, 50, key="we_x0")
        I1 = live.slider("left intensity I₁", 0.0, 1.0, 0.2, 0.05, key="we_I1")
        I2 = live.slider("right intensity I₂", 0.0, 1.0, 0.8, 0.05, key="we_I2")
        blur = live.slider("edge blur (how gradual)", 0.0, 8.0, 1.5, 0.5, key="we_blur")
        nz = live.slider("noise", 0.0, 0.08, 0.0, 0.005, key="we_nz", format="%.3f")
    xs, I = step_profile(x0e, I1, I2, blur, nz)
    dI = np.r_[np.diff(I), 0.0]                    # forward difference I(x+1) - I(x)
    with c_plot:
        f1 = eui.signal_figure([dict(x=xs, y=I, name="I(x)", color=theme.ORIGINAL)], height=230,
                               title="Intensity profile I(x)", y_title="I", vlines=[(x0e - 0.5, "x₀")])
        st.plotly_chart(f1, key="we_I")
        f2 = eui.signal_figure([dict(x=xs + 0.5, y=dI, name="dI/dx ≈ I(x+1) − I(x)", color=theme.TRANSFORMED)],
                               height=230, title="Its derivative", y_title="dI/dx", vlines=[(x0e - 0.5, "peak")])
        st.plotly_chart(f2, key="we_dI")
    k1, k2, k3 = st.columns(3)
    k1.metric("flat region: |dI/dx|", f"{np.abs(dI[5:15]).mean():.3f}", help="≈ 0 where intensity changes slowly",
              border=True)
    k2.metric("at the edge: max |dI/dx|", f"{np.abs(dI).max():.3f}", border=True)
    k3.metric("edge strength (contrast)", f"{abs(I2 - I1):.2f}", border=True,
              help="The total area under the derivative peak equals the jump I₂ − I₁")
    ui.callout("intuition",
               "Where the intensity barely changes the derivative is ≈ 0; at the cliff it spikes (dI/dx ≫ 0). "
               "Make the edge more gradual: the spike gets **lower and wider** - its area always equals the jump "
               "I₂ − I₁. Add noise: the flat parts start to wiggle. Both effects will matter a lot later.")

    ui.anim_intro("Step by step: how 'differences of neighbours' become the derivative",
                  "A zoom on the 20 pixels around your edge (same sliders). Grey = the samples I(x).")
    steps, xr, xz, vz = difference_steps(xs, I, x0e)
    base = go.Figure()
    base.add_trace(go.Scatter(x=xz, y=vz, mode="lines+markers", line=dict(color=theme.ORIGINAL, width=1.5, shape="hvh"),
                              marker=dict(size=8, color=theme.ORIGINAL), showlegend=False, hoverinfo="skip"))
    base.update_layout(height=470, showlegend=False, margin=dict(l=10, r=10))
    base.update_xaxes(range=list(xr), title="x (pixels)", gridcolor=theme.GRID, zeroline=False, dtick=2)
    lo_y = min(0.0, float(np.min(np.diff(vz)))) - 0.08
    base.update_yaxes(range=[lo_y, max(1.02, float(vz.max()) + 0.05)], gridcolor=theme.GRID, zerolinecolor=theme.AXIS)
    st.plotly_chart(anim.step_player(base, steps, n_sub=14), key="we_diff_steps")


profile_section()


# ------------------------------------------------------------------ 2D: gradient
def gradient_steps(gxc, gyc, ang_deg, centre=(31.5, 31.5), scale=1.0):
    """∇I = (Iₓ, I_y): first the x-part, then the y-part, then the resultant, then its relation to the edge."""
    c = np.array(centre)
    gx, gy = gxc * scale, gyc * scale
    a = np.deg2rad(ang_deg)
    edge = np.array([np.sin(a), np.cos(a)]) * 14

    def state(px, py, res, perp):
        return [
            plotting.arrows_trace([c], [c + [px * gx, 0]], theme.E1, width=4, size=15),
            plotting.arrows_trace([c + [px * gx, 0]], [c + [px * gx, py * gy]], theme.E2, width=4, size=15),
            plotting.arrows_trace([c], [c + res * np.array([gx, gy])], theme.HIGHLIGHT, width=5, size=17),
            go.Scatter(x=[c[0] - perp * edge[0], c[0] + perp * edge[0]], y=[c[1] - perp * edge[1], c[1] + perp * edge[1]],
                       mode="lines", line=dict(color=theme.MAGENTA, width=3, dash="dash"), showlegend=False,
                       hoverinfo="skip"),
        ]

    H = anim.hl
    mag = float(np.hypot(gxc, gyc))
    th = float(np.degrees(np.arctan2(gyc, gxc)))
    return [
        anim.Step("Centre pixel", "∇I = (Iₓ, I_y) = ?", "We build the gradient at the centre from its two parts.",
                  state(0, 0, 0, 0)),
        anim.Step("x-part: Iₓ", f"Iₓ = ∂I/∂x = {H(f'{gxc:.3f}', theme.E1)}",
                  "How fast brightness changes going RIGHT (orange arrow).", state(1, 0, 0, 0)),
        anim.Step("y-part: I_y", f"I_y = ∂I/∂y = {H(f'{gyc:.3f}', theme.E2)}",
                  "How fast it changes going DOWN (image y grows downwards), stacked on the x-part (green).",
                  state(1, 1, 0, 0)),
        anim.Step("Add them: the gradient vector", f"‖∇I‖ = √(Iₓ² + I_y²) = {H(f'{mag:.3f}')},   θ = atan2(I_y, Iₓ) = {H(f'{th:.1f}°')}",
                  "The amber resultant is the direction of steepest climb; its length is how steep the cliff is.",
                  state(1, 1, 1, 0)),
        anim.Step("Compare with the edge", "∇I ⟂ edge",
                  "The pink dashed line runs ALONG the edge. The gradient crosses it at a right angle - along the edge "
                  "the brightness does not change.", state(1, 1, 1, 1)),
    ]


@live.section
def gradient_section():
    st.subheader("In 2D: the gradient ∇I points uphill - across the edge", divider="gray")
    st.latex(r"\nabla I = \begin{bmatrix} \partial I/\partial x \\ \partial I/\partial y \end{bmatrix} = "
             r"\begin{bmatrix} I_x \\ I_y \end{bmatrix},\qquad \|\nabla I\| = \sqrt{I_x^2 + I_y^2},\qquad "
             r"\theta = \operatorname{atan2}(I_y,\ I_x)")
    c_ctrl, c_fig, c_txt = st.columns([0.8, 1.5, 1.1], gap="large")
    with c_ctrl:
        ang = live.slider("edge orientation (degrees from vertical)", -90, 90, 25, 5, key="we_ang", format="%d°")
        eb = live.slider("edge blur", 0.5, 4.0, 1.5, 0.5, key="we_eb")
        dens = live.slider("arrow spacing", 4, 12, 6, key="we_dens")
        flip = st.toggle("Swap dark / bright sides", value=False, key="we_flip")
    S = E.step_image(64, ang, eb, *((0.85, 0.15) if flip else (0.15, 0.85)))
    Ix, Iy = E.gradients(S, "Central difference")
    with c_fig:
        fig = eui.gray_heatmap(S, height=460)
        yy, xx = np.mgrid[dens // 2:64:dens, dens // 2:64:dens]
        gx, gy = Ix[yy, xx], Iy[yy, xx]
        sc = 3.0 / (np.hypot(Ix, Iy).max() + 1e-9) * dens
        # centre each arrow on its pixel so it visibly straddles the edge
        fig.add_trace(eui.arrows_trace(xx - gx * sc / 2, yy - gy * sc / 2, gx * sc, gy * sc, theme.E1,
                                       "∇I (length = magnitude)"))
        fig.add_trace(eui.arrow_heads(xx - gx * sc / 2, yy - gy * sc / 2, gx * sc, gy * sc, theme.E1))
        a = np.deg2rad(ang)
        t = np.linspace(-40, 40, 2)
        fig.add_trace(go.Scatter(x=31.5 + t * np.sin(a), y=31.5 + t * np.cos(a), mode="lines",
                                 line=dict(color=theme.E2, width=3.5, dash="dash"),
                                 name="edge direction (along the cliff)"))
        fig.update_xaxes(range=[-0.5, 63.5])
        fig.update_yaxes(range=[63.5, -0.5])
        st.plotly_chart(fig, key="we_grad")
    gxc, gyc = Ix[32, 32], Iy[32, 32]
    with c_txt:
        mag = np.hypot(gxc, gyc)
        th = np.degrees(np.arctan2(gyc, gxc))
        st.markdown("At the centre pixel:")
        st.latex(rf"I_x = {gxc:.3f},\quad I_y = {gyc:.3f}")
        st.latex(rf"\|\nabla I\| = {mag:.3f},\quad \theta = {th:.1f}^\circ")
        edge_dir = np.array([np.sin(a), np.cos(a)])
        dot = (gxc * edge_dir[0] + gyc * edge_dir[1]) / (mag + 1e-12)
        st.latex(rf"\cos\angle(\nabla I,\ \text{{edge}}) = {dot:.3f}\;\Rightarrow\; \perp")
        if abs(ang) <= 15:
            st.info("A nearly **vertical** edge: brightness changes left-to-right, so |Iₓ| ≫ |I_y|.",
                    icon=":material/swap_horiz:")
        elif abs(ang) >= 75:
            st.info("A nearly **horizontal** edge: brightness changes top-to-bottom, so |I_y| ≫ |Iₓ|.",
                    icon=":material/swap_vert:")
        ui.callout("intuition",
                   "The orange arrows always cross the green dashed edge at a right angle and point towards the "
                   "**brighter** side. Their length is the steepness of the cliff. Swap the sides and the arrows "
                   "flip: θ changes by 180° but the magnitude does not.")
        st.caption("Image convention: y grows **downwards** (row index), so θ is measured clockwise on screen.")

    ui.anim_intro("Step by step: building ∇I from Iₓ and I_y at the centre pixel")
    base = eui.gray_heatmap(S, height=560)
    base.update_xaxes(range=[11.5, 51.5])
    base.update_yaxes(range=[51.5, 11.5])
    base.update_layout(showlegend=False)
    scale = 16.0 / (np.hypot(gxc, gyc) + 1e-9)
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        st.plotly_chart(anim.step_player(base, gradient_steps(gxc, gyc, ang, scale=scale), n_sub=14, wrap_at=70),
                        key="we_grad_steps")
    with right:
        ui.callout("note",
                   "The arrows are drawn longer than life (×" + f"{scale:.0f}" + ") so you can see them; only their "
                   "direction and relative size matter. Change the orientation slider above and replay: the x- and "
                   "y-parts trade places as the edge turns from vertical to horizontal.", title="Scale of the arrows")


gradient_section()


# ------------------------------------------------------------------ real image: magnitude & direction
@live.section
def real_image_section():
    st.subheader("On a real image: how strong, and which way?", divider="gray")
    name2 = st.selectbox("Image ", ["Coins (photo)", "Cameraman (photo)", "Shapes (synthetic)", "Coffee (photo)"],
                         key="we_img2")
    P = E.SAMPLES[name2]()
    gx, gy = E.gradients(E.smooth(P, 1.0), "Central difference")
    M = E.magnitude(gx, gy)
    yy, xx = np.mgrid[0:120, 0:120] - 59.5
    rr = np.hypot(yy, xx)
    wheel = E.orientation_rgb(np.arctan2(yy, xx), rr * (rr < 58))
    eui.show_row([(P, "I(x, y)"), (E.to_u8(np.sqrt(M / M.max())), "‖∇I‖ - bright = strong change (√ for contrast)"),
                  (E.orientation_rgb(E.orientation(gx, gy), M), "direction θ as colour, brightness = ‖∇I‖"),
                  (wheel, "colour key: hue = direction θ")], cols=[1, 1, 1, 0.55])
    st.caption("Opposite sides of a coin have opposite colours: the gradient points outwards on one side and inwards "
               "on the other, always across the rim. The next lessons show *how* to compute Iₓ and I_y robustly.")


real_image_section()

ui.whys([
    ("Why use the derivative instead of the intensity itself?",
     "An edge is a **change**, not a particular brightness. A dark object on a darker background and a bright "
     "object on a brighter background both have edges, and a shadow can make the same surface brighter or "
     "darker. Thresholding intensity confuses 'dark' with 'edge'; the derivative ignores the absolute level and "
     "responds only to how fast it changes."),
    ("Why is the gradient perpendicular to the edge?",
     "Walk **along** an edge and the brightness stays the same, so the derivative in that direction is 0. "
     "The directional derivative in direction **t** is ∇I · **t**, and ∇I · **t** = 0 means ∇I ⟂ **t**. "
     "So the gradient must point straight across the edge - the direction of steepest climb."),
    ("Why √(Iₓ² + I_y²) and not just |Iₓ|?",
     "|Iₓ| only sees changes from left to right; a horizontal edge would be invisible. The length of the vector "
     "(Iₓ, I_y) is the slope in the steepest direction and does not depend on how the edge is rotated."),
    ("Why atan2(I_y, Iₓ) instead of tan⁻¹(I_y / Iₓ)?",
     "tan⁻¹ only returns angles between −90° and 90° and fails when Iₓ = 0. It cannot tell 'brighter to the "
     "right' (θ = 0°) from 'brighter to the left' (θ = 180°). atan2 looks at both signs and returns the full "
     "−180°…180° range. The notes write tan⁻¹(I_y/Iₓ); every implementation uses atan2."),
    ("Is every edge the boundary of an object?",
     "No. Intensity also jumps at shadows, at surface markings (a printed letter), at creases where the surface "
     "turns away from the light, and inside texture. Classical detectors find *intensity* edges; deciding which "
     "ones are object *boundaries* needs context - the topic of the last lesson."),
    ("Why does a blurry edge give a lower but wider derivative peak?",
     "The total rise I₂ − I₁ is fixed. Spread it over more pixels and each pixel only climbs a little: the "
     "derivative's area (= the jump) stays the same while its height drops. Real camera edges are always a "
     "little blurred by the lens and the pixel size."),
])

st.divider()
ui.mcq("we1", "A vertical edge has dark on the left and bright on the right. Which way does ∇I point?",
       ["Up", "Down", "Right", "Left"], 2,
       "The gradient points towards increasing intensity and is perpendicular to the edge: to the right (Iₓ > 0, I_y ≈ 0).")
ui.mcq("we2", "Along an edge (moving parallel to it) the derivative of intensity is …",
       ["maximal", "about zero", "negative", "equal to the gradient magnitude"], 1,
       "Brightness is constant along the edge, so the directional derivative along it is ≈ 0 - that is why ∇I is "
       "perpendicular to the edge.")

ui.lesson_nav(__file__)
