import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core import edge_ui as eui
from core import edges as E
from core import anim, live, plotting, theme, ui

ui.lesson_header(
    __file__,
    "Canny treats edge detection as a design problem: <b>detect</b> real edges, <b>localise</b> them precisely, "
    "and give <b>one response per edge</b>. Its pipeline has five stages; this lesson covers the first three - "
    "smoothing, the gradient, and the clever one: <b>non-maximum suppression</b>, which thins thick bands into "
    "one-pixel ridges.",
)

# ------------------------------------------------------------------ pipeline diagram
STAGES = [("Gaussian smoothing", "reduce noise"), ("Gradient", "detect intensity changes"),
          ("Non-maximum suppression", "thin edges"), ("Double threshold", "strong / weak"),
          ("Hysteresis", "connect valid weak edges")]
fig = go.Figure()
for i, (nm, why_) in enumerate(STAGES):
    x0 = i * 2.1
    active = i <= 2
    fig.add_shape(type="rect", x0=x0, x1=x0 + 1.8, y0=0, y1=1, line=dict(color=theme.TRANSFORMED, width=2),
                  fillcolor=theme.rgba(theme.TRANSFORMED, 0.18 if active else 0.04))
    fig.add_annotation(x=x0 + 0.9, y=0.66, text=f"<b>{i + 1}. {nm}</b>", showarrow=False, font=dict(size=12))
    fig.add_annotation(x=x0 + 0.9, y=0.3, text=why_, showarrow=False, font=dict(size=11), opacity=0.75)
    if i < 4:
        fig.add_annotation(x=x0 + 2.08, y=0.5, ax=x0 + 1.82, ay=0.5, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=3, arrowwidth=1.5, arrowcolor=theme.AXIS, text="")
fig.update_xaxes(visible=False, range=[-0.1, 10.2])
fig.update_yaxes(visible=False, range=[-0.05, 1.05])
fig.update_layout(height=120, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
st.plotly_chart(fig, key="cn_pipe", config=dict(staticPlot=True))
st.caption("Highlighted: the stages in this lesson. Thresholds and hysteresis follow in the next lesson.")

@st.cache_resource(show_spinner=False, max_entries=24)
def canny_stages(img: np.ndarray, sg: float) -> dict:
    """Smoothing → gradient → both kinds of NMS (cached: the row / zoom sliders only re-plot)."""
    R = E.canny(img, sg, 0.1, 0.2)
    R["Mi"] = E.nms_interpolated(R["M"], R["theta"])
    return R


# ------------------------------------------------------------------ steps 1 & 2 + what NMS keeps
@live.section
def image_section():
    st.subheader("Steps 1 & 2 · smooth, then take the gradient", divider="gray")
    st.latex(r"I_s = G_\sigma * I,\qquad I_x = \frac{\partial I_s}{\partial x},\; I_y = \frac{\partial I_s}{\partial y},"
             r"\qquad M = \sqrt{I_x^2 + I_y^2},\qquad \theta = \operatorname{atan2}(I_y, I_x)")
    c_ctrl, c_main = st.columns([0.8, 3], gap="large")
    with c_ctrl:
        img, clean, _ = eui.pick_image("cn", "Coins (photo)", noise_default=0.02)
        sg = live.slider("σ of the Gaussian", 0.5, 4.0, 1.4, 0.1, key="cn_s")
    R = canny_stages(img, sg)
    M, th, Mn, Mi = R["M"], R["theta"], R["Mn"], R["Mi"]
    with c_main:
        eui.show_row([(np.clip(R["smoothed"], 0, 1), f"1 · smoothed (σ = {sg})"),
                      (E.to_u8(M, 0, np.percentile(M, 99.7)), "2 · gradient magnitude M - thick bands"),
                      (E.orientation_rgb(th, M), "2 · direction θ (colour)")])

    st.subheader("Why the magnitude is thick - and what NMS keeps", divider="gray")
    h, w = img.shape
    c_ctrl, c_img, c_plot = st.columns([0.8, 1.0, 1.8], gap="large")
    with c_ctrl:
        row = live.slider("row to inspect", 0, h - 1, h // 2, key="cn_row")
        c_from, c_to = live.slider("columns", 0, w - 1, (0, min(w - 1, 120)), key="cn_cols")
        if c_to - c_from < 4:
            c_to = min(w - 1, c_from + 4)
    with c_img:
        vis = E.overlay(np.clip(img, 0, 1), Mn > 0.15 * Mn.max(), color=(42, 120, 214), dim=0.9)
        vis[max(0, row - 1):row + 1, c_from:c_to + 1] = (235, 104, 52)
        st.image(ui.datauri(vis), caption="orange line = the inspected row; blue = NMS ridges", width="stretch")
    with c_plot:
        xs = np.arange(c_from, c_to + 1)
        seg_n = Mn[row, c_from:c_to + 1]
        fig = eui.signal_figure([dict(x=xs, y=M[row, c_from:c_to + 1], name="gradient magnitude M", color=theme.ORIGINAL),
                                 dict(x=xs[seg_n > 0], y=seg_n[seg_n > 0], name="kept by NMS (local maxima across the edge)",
                                      color=theme.TRANSFORMED, mode="markers", size=9)],
                                height=330, x_title="column x", y_title="magnitude")
        st.plotly_chart(fig, key="cn_prof")
        st.caption("Each edge is a **hump** several pixels wide (smoothing spread it out). A threshold would keep every "
                   "pixel on the hump. NMS keeps just the summit of each hump - zeros elsewhere.")

    st.subheader("Before and after NMS", divider="gray")
    cz1, cz2 = st.columns(2)
    with cz1:
        zx = live.slider("zoom x", 0, w - 40, min(w // 2, w - 40), key="cn_zx")
    with cz2:
        zy = live.slider("zoom y", 0, h - 40, min(h // 3, h - 40), key="cn_zy")
    top = np.percentile(M, 99.7)
    crop = lambda A: E.enlarge(E.to_u8(A[zy:zy + 40, zx:zx + 40], 0, top), 6)
    eui.show_row([(crop(M), "magnitude M (zoom)"), (crop(Mn), "NMS - quantised directions"),
                  (crop(Mi), "NMS - interpolated (exact θ)")])
    st.caption(f"Non-zero pixels: M has {(M > 0.05 * M.max()).sum():,}, NMS keeps {(Mn > 0.05 * M.max()).sum():,} - "
               "thin, one-pixel ridges, ready for thresholding.")


image_section()


# ------------------------------------------------------------------ NMS geometry
def nms_steps(Mz, tz, cx, cy, p1, p2, n1, n2, keep, u):
    """At one pixel: look at the magnitudes, the gradient direction, the two neighbours, decide - then do it everywhere."""
    win = Mz[cy - 4:cy + 5, cx - 4:cx + 5] / Mz.max()
    nms_win = E.nms_quantized(Mz, tz)[cy - 4:cy + 5, cx - 4:cx + 5] / Mz.max()
    xs, ys = np.arange(cx - 4, cx + 5), np.arange(cy - 4, cy + 5)
    sc = 2.6

    def state(Z, arrow, cmp_, verdict):
        txt = [[f"{v:.2f}" for v in r] for r in Z]
        a0 = np.array([cx, cy]) - u * sc * arrow
        a1 = np.array([cx, cy]) + u * sc * arrow
        return [
            go.Heatmap(z=Z, x=xs, y=ys, zmin=0, zmax=1, colorscale=eui.LIGHT_BLUES, showscale=False, xgap=1, ygap=1,
                       text=txt, texttemplate="%{text}", textfont=dict(size=11), hoverinfo="skip"),
            plotting.arrows_trace([a0], [a1], theme.E1, width=3.5, size=15),
            go.Scatter(x=[p1[0], p2[0]], y=[p1[1], p2[1]], mode="markers+text", text=["p⁺", "p⁻"],
                       textposition="top center", textfont=dict(size=1 + 13 * cmp_, color=theme.MAGENTA),
                       marker=dict(size=18 * cmp_ + 0.1, symbol="diamond-open", color=theme.MAGENTA, line=dict(width=3)),
                       hoverinfo="skip"),
            go.Scatter(x=[cx], y=[cy - 0.62], mode="text", text=["KEEP" if keep else "SUPPRESS"],
                       textfont=dict(size=1 + 14 * verdict, color=theme.E2 if keep else theme.OUTLIER),
                       hoverinfo="skip"),
        ]

    H = anim.hl
    mx = Mz.max()
    return [
        anim.Step("The magnitude around the pixel", f"M(pixel) = {Mz[cy, cx] / mx:.3f}  (amber square)",
                  "A 9 × 9 window of M (scaled to the strongest value). The edge is the darker band.",
                  state(win, 0.0, 0.0, 0.0)),
        anim.Step("Which way is 'across'?", f"θ = {np.degrees(tz[cy, cx]):.1f}°",
                  "The gradient direction (orange) points across the edge - towards brighter pixels.",
                  state(win, 1.0, 0.0, 0.0)),
        anim.Step("Look one step forward and back", f"M(p⁺) = {n1 / mx:.3f},   M(p⁻) = {n2 / mx:.3f}",
                  "Only the two neighbours ALONG the gradient matter (pink diamonds).", state(win, 1.0, 1.0, 0.0)),
        anim.Step("Decide", f"{Mz[cy, cx] / mx:.3f} ≥ both?  →  {H('keep' if keep else 'set to 0', theme.E2 if keep else theme.OUTLIER)}",
                  "A pixel survives only if it is the top of the hill across the edge.", state(win, 1.0, 1.0, 1.0)),
        anim.Step("Do it for every pixel", "M → M_NMS", "Every shoulder pixel fades to 0; a one-pixel-wide ridge "
                  "remains along the edge.", state(nms_win, 0.0, 0.0, 0.0)),
    ]


@live.section
def nms_geometry_section():
    st.subheader("Step 3 · non-maximum suppression, up close", divider="gray")
    st.latex(r"M_{\text{NMS}}(x, y) = \begin{cases} M(x, y) & \text{if } M(x,y) \ge M(\mathbf{p}^+) \text{ and } "
             r"M(x,y) \ge M(\mathbf{p}^-) \\ 0 & \text{otherwise} \end{cases},\qquad "
             r"\mathbf{p}^{\pm} = (x, y) \pm (\cos\theta, \sin\theta)")
    c_ctrl, c_fig, c_txt = st.columns([0.8, 1.6, 1.1], gap="large")
    with c_ctrl:
        ang = live.slider("edge orientation (degrees from vertical)", -80, 80, 30, 5, key="cn_ang", format="%d°")
        off = live.slider("move the inspected pixel across the edge", -3, 3, 0, key="cn_off")
        mode = st.radio("compare with", ["quantised neighbours (4 directions)", "interpolated points (exact θ)"],
                        key="cn_mode")
    S = E.smooth(E.step_image(41, ang, 0.0), 1.2)
    Rs = E.canny(S, 0.0, 0.1, 0.2)
    Mz, tz = Rs["M"], Rs["theta"]
    a = np.deg2rad(ang)
    normal = np.array([np.cos(a), -np.sin(a)])                       # (dcol, drow) across the edge
    pc = np.round(np.array([20, 20]) + off * normal).astype(int)
    cx, cy = int(pc[0]), int(pc[1])
    t0 = tz[cy, cx]
    u = np.array([np.cos(t0), np.sin(t0)])
    q = int(E.quantize(np.array([t0]))[0])
    (dc1, dr1), (dc2, dr2) = E.NMS_OFFSETS[q]
    if mode.startswith("quant"):
        n1, n2 = Mz[cy + dr1, cx + dc1], Mz[cy + dr2, cx + dc2]
        p1, p2 = (cx + dc1, cy + dr1), (cx + dc2, cy + dr2)
    else:
        import cv2
        pts = np.array([[cx + u[0], cy + u[1]], [cx - u[0], cy - u[1]]], np.float32)
        vals = cv2.remap(Mz.astype(np.float32), pts[:, 0:1], pts[:, 1:2], cv2.INTER_LINEAR)
        n1, n2 = float(vals[0, 0]), float(vals[1, 0])
        p1, p2 = tuple(pts[0]), tuple(pts[1])
    keep = Mz[cy, cx] >= n1 and Mz[cy, cx] >= n2
    with c_fig:
        win = Mz[cy - 4:cy + 5, cx - 4:cx + 5]
        fig = eui.patch_figure(win / Mz.max(), "", fmt="{:.2f}", height=470, colorscale=eui.LIGHT_BLUES, zmin=0, zmax=1)
        fig.data[0].x = np.arange(cx - 4, cx + 5)
        fig.data[0].y = np.arange(cy - 4, cy + 5)
        fig.add_shape(type="rect", x0=cx - 0.5, x1=cx + 0.5, y0=cy - 0.5, y1=cy + 0.5,
                      line=dict(color=theme.HIGHLIGHT, width=4))
        sc = 2.6
        fig.add_trace(eui.arrows_trace([cx - u[0] * sc], [cy - u[1] * sc], [2 * u[0] * sc], [2 * u[1] * sc],
                                       theme.E1, "gradient direction θ", width=3))
        fig.add_trace(eui.arrow_heads([cx - u[0] * sc], [cy - u[1] * sc], [2 * u[0] * sc], [2 * u[1] * sc], theme.E1, 13))
        fig.add_trace(go.Scatter(x=[p1[0], p2[0]], y=[p1[1], p2[1]], mode="markers+text", text=["p⁺", "p⁻"],
                                 textposition="top center", textfont=dict(size=14, color=theme.MAGENTA),
                                 marker=dict(size=16, symbol="diamond-open", color=theme.MAGENTA, line=dict(width=3)),
                                 name="compared with"))
        t_ = np.linspace(-6, 6, 2)
        fig.add_trace(go.Scatter(x=20 + t_ * np.sin(a), y=20 + t_ * np.cos(a), mode="lines",
                                 line=dict(color=theme.E2, dash="dash", width=2.5), name="true edge line"))
        fig.update_xaxes(range=[cx - 4.5, cx + 4.5])
        fig.update_yaxes(range=[cy + 4.5, cy - 4.5])
        fig.update_layout(legend=dict(orientation="h", y=1.06, x=0))
        st.plotly_chart(fig, key="cn_zoom")
    with c_txt:
        st.markdown(f"Gradient direction θ = **{np.degrees(t0):.1f}°** → quantised to **{q}°**")
        st.latex(rf"M(\text{{pixel}}) = {Mz[cy, cx] / Mz.max():.3f}")
        st.latex(rf"M(p^+) = {n1 / Mz.max():.3f},\qquad M(p^-) = {n2 / Mz.max():.3f}")
        if keep:
            st.success("**Kept** - it is at least as large as both neighbours along the gradient: a ridge pixel.",
                       icon=":material/check_circle:")
        else:
            st.error("**Suppressed** (set to 0) - a neighbour along the gradient is larger, so this pixel is on the "
                     "shoulder of the hump.", icon=":material/cancel:")
        ui.callout("intuition",
                   "Walk **across** the edge (along the orange arrow) and the magnitude rises then falls - a hill. "
                   "Keep only the top of the hill. Walking **along** the edge (green line) the magnitude stays high, "
                   "which is why we never compare in that direction.")

    ui.anim_intro("Step by step: NMS at the inspected pixel, then everywhere",
                  "Uses your orientation and pixel position. Move the pixel off the ridge (slider above) and replay to "
                  "see a 'suppress' decision.")
    base = go.Figure()
    base.add_shape(type="rect", x0=cx - 0.5, x1=cx + 0.5, y0=cy - 0.5, y1=cy + 0.5,
                   line=dict(color=theme.HIGHLIGHT, width=4))
    base.update_xaxes(range=[cx - 4.5, cx + 4.5], showgrid=False, zeroline=False)
    base.update_yaxes(range=[cy + 4.5, cy - 4.5], scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False)
    base.update_layout(height=560, showlegend=False, margin=dict(l=10, r=10))
    left, right = st.columns([1.4, 1], gap="large")
    with left:
        st.plotly_chart(anim.step_player(base, nms_steps(Mz, tz, cx, cy, p1, p2, n1, n2, keep, u), n_sub=12,
                                         wrap_at=72), key="cn_nms_steps")
    with right:
        ui.callout("note",
                   "Step 5 applies exactly the same test to every pixel of the window at once. Compare it with step 1: "
                   "the dark band shrinks to a single line of pixels - the summits.", title="From one pixel to all")


nms_geometry_section()


# ------------------------------------------------------------------ quantisation wheel
@live.section
def quantisation_section():
    st.subheader("Gradient direction quantisation: 4 ways to look across a pixel", divider="gray")
    c_ctrl, c_wheel, c_grid, c_tab = st.columns([0.7, 1.1, 0.8, 1.3], gap="medium")
    with c_ctrl:
        tq = live.slider("gradient direction θ (degrees)", -180, 180, 30, 5, key="cn_tq", format="%d°")
    qq = int(E.quantize(np.radians(np.array([tq])))[0])
    COL = {0: theme.E1, 45: theme.E2, 90: theme.TRANSFORMED, 135: theme.MAGENTA}
    with c_wheel:
        fig = go.Figure()
        for base in [0, 45, 90, 135, 180, 225, 270, 315]:
            img_theta = (-base) % 360                 # polar angle is counter-clockwise; image θ (y down) is clockwise
            b_ = img_theta % 180
            fig.add_trace(go.Barpolar(r=[1], theta=[base], width=[45], marker_color=COL[b_],
                                      opacity=0.85 if b_ == qq else 0.25, showlegend=False,
                                      hovertemplate=f"θ ≈ {img_theta if img_theta <= 180 else img_theta - 360}° "
                                                    f"→ bin {b_}°<extra></extra>"))
        fig.add_trace(go.Scatterpolar(r=[0, 1.1], theta=[0, -tq], mode="lines", line=dict(color="black", width=3),
                                      showlegend=False))
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20),
                          polar=dict(radialaxis=dict(visible=False),
                                     angularaxis=dict(direction="counterclockwise", rotation=0, tickfont=dict(size=10),
                                                      tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                                                      ticktext=["0°", "−45°", "−90°", "−135°", "180°", "135°", "90°",
                                                                "45°"])))
        st.plotly_chart(fig, key="cn_wheel")
        st.caption("Labels are image angles θ (y down): +90° points DOWN the screen. Opposite sectors share a bin.")
    with c_grid:
        G3 = np.zeros((3, 3))
        (a1, b1), (a2, b2) = E.NMS_OFFSETS[qq]
        G3[1, 1] = 2
        G3[1 + b1, 1 + a1] = G3[1 + b2, 1 + a2] = 1
        figg = go.Figure(go.Heatmap(z=G3, colorscale=[[0, "rgba(128,128,128,0.15)"], [0.5, COL[qq]], [1, theme.HIGHLIGHT]],
                                    showscale=False, xgap=4, ygap=4, hoverinfo="skip",
                                    text=[["", "", ""], ["", "pixel", ""], ["", "", ""]], texttemplate="%{text}"))
        figg.update_yaxes(autorange="reversed", visible=False, scaleanchor="x")
        figg.update_xaxes(visible=False)
        figg.update_layout(height=230, margin=dict(l=0, r=0, t=26, b=0),
                           title=dict(text=f"bin {qq}°: compare with", x=0.5, xanchor="center", font=dict(size=13)))
        st.plotly_chart(figg, key="cn_grid")
    with c_tab:
        st.markdown("| θ bin | neighbours (image, y down) | neighbours in the notes (y up) |\n|---|---|---|\n"
                    "| 0° | (x−1, y), (x+1, y) | (x−1, y), (x+1, y) |\n"
                    "| 45° | (x−1, y−1), (x+1, y+1) | (x−1, y+1), (x+1, y−1) |\n"
                    "| 90° | (x, y−1), (x, y+1) | (x, y−1), (x, y+1) |\n"
                    "| 135° | (x+1, y−1), (x−1, y+1) | (x−1, y−1), (x+1, y+1) |")
        ui.callout("pitfall",
                   "The two diagonal rows swap depending on whether y points up (maths, the notes) or down (image "
                   "rows, OpenCV, this site). Use the convention in which you computed I_y - mixing them compares the "
                   "pixel *along* the edge and silently breaks NMS on diagonal edges.")


quantisation_section()

ui.whys([
    ("Why compare along the gradient direction and not along the edge?",
     "Along the edge every pixel is an edge pixel with a similar magnitude - comparing there would suppress real "
     "edge pixels at random. Across the edge (the gradient direction) the magnitude forms a hill whose top is "
     "the true edge location. The notes' appendix says it precisely: an edge pixel's magnitude is a local "
     "maximum in the direction θ."),
    ("Why not just raise the threshold until edges become thin?",
     "Edges have different contrasts. A threshold high enough to thin a strong edge erases weak edges entirely, "
     "and a hump never shrinks to exactly one pixel. NMS thins every edge to one pixel *regardless* of its "
     "strength, and thresholding then decides which ones to keep."),
    ("Why quantise θ into just four directions?",
     "A pixel has 8 neighbours, which lie on exactly 4 lines through it (horizontal, vertical, two diagonals). "
     "Rounding θ to the nearest line means NMS only reads existing pixel values - very fast. Interpolating at "
     "(x ± cos θ, y ± sin θ) is more accurate, especially for edges between the four directions."),
    ("Why do directions only matter modulo 180°?",
     "We compare with the pixel in front *and* the pixel behind. θ and θ + 180° point to the same pair, so "
     "−45° and 135° land in the same bin."),
    ("Why smooth first if it makes edges wider?",
     "Smoothing suppresses noise so that real edges win (good detection), at the cost of spreading them (worse "
     "localisation). NMS recovers a thin location afterwards - Canny gets the best of both, within limits."),
    ("Can NMS leave double pixels or tiny gaps?",
     "Yes. On plateaus two neighbours can be exactly equal, and quantised directions can mis-compare near 22.5°. "
     "Implementations break ties (here: '>' on one side, '≥' on the other) and hysteresis later bridges small gaps."),
])

st.divider()
ui.mcq("cn1", "For a pixel whose gradient points straight to the right (θ = 0°), NMS compares it with …",
       ["the pixels above and below", "the pixels to its left and right", "all 8 neighbours", "the diagonal neighbours"],
       1, "θ = 0° means the edge is vertical; across it means left and right.")
ui.mcq("cn2", "What does non-maximum suppression NOT do?",
       ["Thin thick ridges", "Use the gradient direction", "Decide which edges are strong enough", "Set non-ridge pixels to 0"],
       2, "Deciding which edges are strong enough is the job of the double threshold and hysteresis.")

ui.lesson_nav(__file__)
