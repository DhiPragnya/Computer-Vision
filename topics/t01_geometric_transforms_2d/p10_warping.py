import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core import anim, images, live, plotting, theme, ui
from core import transforms as tf

ui.lesson_header(
    __file__,
    "So far we moved shapes. Moving an <b>image</b> raises a practical question: an output pixel sits on an "
    "integer grid, but the matrix gives fractional positions. How do we decide each output pixel's colour? "
    "The answer - <i>inverse mapping + interpolation</i> - is inside every warp function you will ever call.",
)

# ------------------------------------------------------------------ the image warp
@live.section
def warp_section():
    c_ctrl, c_main = st.columns([0.85, 3], gap="large")
    with c_ctrl:
        src_name = st.selectbox("Image", list(images.SAMPLES) + ["Upload my own"], key="wp_img")
        img = None
        if src_name == "Upload my own":
            up = st.file_uploader("PNG / JPG", type=["png", "jpg", "jpeg"], key="wp_up")
            if up is not None:
                try:
                    img = images.load_upload(up, 320)
                except Exception:
                    st.error("Could not read that file - using the poster instead.", icon=":material/error:")
        if img is None:
            img = images.SAMPLES.get(src_name, images.poster)(300)
        st.markdown("**Transform (about the image centre)**")
        s = live.slider("scale", 0.5, 3.0, 1.8, 0.05, key="wp_s", help="Scale > 1 is where forward mapping breaks.")
        th = live.slider("rotation θ°", -180, 180, 20, 1, key="wp_th", format="%d°")
        k = live.slider("shear", -1.0, 1.0, 0.0, 0.05, key="wp_k")
        p = live.slider("perspective h₃₁ (×10⁻³)", -3.0, 3.0, 0.0, 0.1, key="wp_p")
        method = st.radio("Interpolation (inverse mapping)", ["nearest", "bilinear", "bicubic"], index=1,
                          key="wp_m", horizontal=True)
        pink = st.toggle("Paint holes pink", value=True, key="wp_pink",
                         help="Off: holes look as they really would - black pixels.")

    M = tf.rotation(th) @ tf.shear(k) @ tf.scaling(s)
    M[2, 0] = p * 1e-3
    H = images.about_center(M, img.shape)
    fw, holes = images.forward_warp(img, H)
    iw = images.inverse_warp(img, H, method=method)
    coverage = images.inverse_warp(np.full_like(img, 255), H, method="nearest")[..., 0] > 0
    hole_frac = (holes & coverage).sum() / max(coverage.sum(), 1)
    fw_show = fw.copy()
    if pink:
        fw_show[holes & coverage] = (213, 81, 129)      # paint holes magenta so they cannot be missed

    with c_main:
        a, b, c = st.columns(3)
        a.image(ui.datauri(img), caption="Source image", width="stretch")
        b.image(ui.datauri(fw_show), caption=f"Forward mapping - holes: {hole_frac:.0%} of the covered area",
                width="stretch")
        c.image(ui.datauri(iw), caption=f"Inverse mapping + {method} interpolation - no holes", width="stretch")
        h, w = iw.shape[:2]
        z1, z2 = st.columns(2)
        with z1:
            zx = live.slider("zoom window x", 0, w - 1, w // 2, key="wp_zx")
        with z2:
            zy = live.slider("zoom window y", 0, h - 1, h // 2, key="wp_zy")
        a2, b2, c2 = st.columns(3)
        sx_, sy_ = tf.apply(np.linalg.inv(H), [[zx, zy]])[0]
        if not np.isfinite([sx_, sy_]).all():
            sx_, sy_ = w / 2, h / 2
        a2.image(ui.datauri(images.zoom(img, sx_, sy_, 12, 10)), caption="zoom: source (the same spot)",
                 width="stretch")
        b2.image(ui.datauri(images.zoom(fw_show, zx, zy, 24, 5)), caption="zoom: forward", width="stretch")
        c2.image(ui.datauri(images.zoom(iw, zx, zy, 24, 5)), caption=f"zoom: inverse ({method})", width="stretch")


warp_section()


# ------------------------------------------------------------------ why holes appear
def cell_trace(cells, fill, line="rgba(128,128,128,0.35)", opacity=1.0):
    xs, ys = [], []
    for X, Y in cells:
        xs += [X - 0.5, X + 0.5, X + 0.5, X - 0.5, X - 0.5, None]
        ys += [Y - 0.5, Y - 0.5, Y + 0.5, Y + 0.5, Y - 0.5, None]
    if not xs:
        xs, ys = [None], [None]
    return go.Scatter(x=xs, y=ys, mode="lines", fill="toself", fillcolor=fill, opacity=opacity,
                      line=dict(color=line, width=0.8), showlegend=False, hoverinfo="skip")


def diagram_geometry(sd):
    Md = tf.rotation(25) @ tf.scaling(sd)
    gy, gx = np.mgrid[0:6, 0:6]
    src = np.c_[gx.ravel(), gy.ravel()].astype(float) - 2.5          # source pixel centres, centred
    fwd = tf.apply(Md, src)
    nearest = np.rint(fwd)
    hit = {tuple(v) for v in nearest.astype(int)}
    lo = np.floor(np.nanmin(fwd, axis=0)).astype(int) - 1
    hi = np.ceil(np.nanmax(fwd, axis=0)).astype(int) + 1
    inv = np.linalg.inv(Md)
    cells_hit, cells_hole = [], []
    for X in range(lo[0], hi[0] + 1):
        for Y in range(lo[1], hi[1] + 1):
            back = tf.apply(inv, [[X, Y]])[0]
            if -3.0 <= back[0] <= 3.0 and -3.0 <= back[1] <= 3.0:
                (cells_hit if (X, Y) in hit else cells_hole).append((X, Y))
    return Md, src, fwd, nearest, cells_hit, cells_hole, (lo, hi)


def forward_steps(sd):
    Md, src, fwd, nearest, hit, hole, _ = diagram_geometry(sd)

    def dots(P, col=theme.TRANSFORMED):
        return go.Scatter(x=P[:, 0], y=P[:, 1], mode="markers", showlegend=False, hoverinfo="skip",
                          marker=dict(size=9, color=col, line=dict(color="white", width=1)))

    def state(P, o_hit, o_hole):
        return [cell_trace(hit, theme.rgba(theme.TRANSFORMED, 0.25), opacity=o_hit),
                cell_trace(hole, theme.rgba(theme.MAGENTA, 0.55), opacity=o_hole), dots(P)]

    def partial(t):
        Mt = tf.rotation(25 * t) @ tf.scaling(1 + t * (sd - 1))
        return state(tf.apply(Mt, src), 0.0, 0.0)

    n_cov = len(hit) + len(hole)
    return [
        anim.Step("36 source pixels", "a 6 × 6 image",
                  "Forward mapping takes each SOURCE pixel and asks: where do you go?", state(src, 0.0, 0.0)),
        anim.Step("Push every pixel: x′ = H x", f"scale {ui.num(sd)}, rotate 25°",
                  "The pixels spread out (scale > 1) and land at fractional positions.", state(fwd, 0.0, 0.0),
                  between=partial),
        anim.Step("Round to the nearest output pixel", "x′ → round(x′)",
                  "Each source pixel writes its colour into ONE output cell (blue).", state(nearest, 1.0, 0.0)),
        anim.Step("Look at the cells nobody wrote to", f"{len(hole)} of {n_cov} covered cells are {anim.hl('holes', theme.MAGENTA)}",
                  "36 source pixels cannot fill more than 36 output cells - the pink ones stay empty."
                  if len(hole) else "At this scale every cell got a value - drag the scale above 1 to see holes.",
                  state(nearest, 1.0, 1.0)),
    ]


def inverse_steps(sd):
    Md, src, fwd, nearest, hit, hole, _ = diagram_geometry(sd)
    C = np.array(hit + hole, float)
    back = tf.apply(np.linalg.inv(Md), C)
    # one example: the output pixel whose pre-image is closest to the source centre
    j = int(np.argmin(np.linalg.norm(back - [0.2, 0.3], axis=1)))
    q = back[j]
    lo_ = np.floor(q - 0.5) + 0.5           # source pixel centres sit on half-integers
    nbrs = np.array([lo_, lo_ + [1, 0], lo_ + [0, 1], lo_ + [1, 1]])

    def state(P, o_lines):
        segs = []
        for n_ in nbrs:
            segs += [P[j], n_, [np.nan, np.nan]]
        S = np.array(segs)
        return [go.Scatter(x=P[:, 0], y=P[:, 1], mode="markers", showlegend=False, hoverinfo="skip",
                           marker=dict(size=7, color=theme.HIGHLIGHT, line=dict(color="white", width=1))),
                go.Scatter(x=S[:, 0], y=S[:, 1], mode="lines", opacity=o_lines, showlegend=False, hoverinfo="skip",
                           line=dict(color=theme.E1, width=2.5)),
                go.Scatter(x=nbrs[:, 0], y=nbrs[:, 1], mode="markers", opacity=o_lines, showlegend=False,
                           hoverinfo="skip", marker=dict(size=13, color=theme.E1, symbol="square-open",
                                                          line=dict(width=2.5)))]

    def partial(t):
        Mt_inv = tf.scaling(1 / (1 + t * (sd - 1))) @ tf.rotation(-25 * t)
        return state(tf.apply(Mt_inv, C), 0.0)

    return [
        anim.Step("Every OUTPUT pixel", f"{len(C)} output pixels inside the warped image",
                  "Inverse mapping starts from the other side: each output pixel asks 'where did I come from?'",
                  state(C, 0.0)),
        anim.Step("Pull back: x = H⁻¹ x′", "rotate −25°, scale 1/" + ui.num(sd),
                  "All of them land INSIDE the source image (grey squares) - densely when we magnify. Nobody is "
                  "forgotten, so there can be no holes.", state(back, 0.0), between=partial),
        anim.Step("Interpolate from the 4 neighbours", "f(x) = weighted mix of the 4 surrounding source pixels",
                  "The pulled-back position is between pixel centres; bilinear interpolation blends the four "
                  "orange-framed pixels (next section).", state(back, 1.0)),
    ]


@live.section
def holes_section():
    st.subheader("Why forward mapping leaves holes", divider="gray")
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        sd = live.slider("scale in this diagram", 0.6, 2.5, 1.7, 0.1, key="wp_sd")
        Md, src, fwd, nearest, cells_hit, cells_hole, (lo, hi) = diagram_geometry(sd)
        fig = plotting.base_figure((lo[0] - 0.5, hi[0] + 0.5, lo[1] - 0.5, hi[1] + 0.5), height=470)
        for cells, col, name in [(cells_hit, theme.rgba(theme.TRANSFORMED, 0.18), "output pixel that received a value"),
                                 (cells_hole, theme.rgba(theme.MAGENTA, 0.45), "output pixel left EMPTY (hole)")]:
            tr = cell_trace(cells, col)
            tr.update(name=name, showlegend=True)
            fig.add_trace(tr)
        fig.add_trace(plotting.points_trace(fwd, theme.TRANSFORMED, "where each source pixel lands  (H x)", size=8))
        fig.update_xaxes(dtick=1)
        fig.update_yaxes(dtick=1)
        st.plotly_chart(fig, key="wp_diag")
    with right:
        ui.callout("intuition",
                   "**Forward** = 'for every source pixel, *push* it to where H sends it.' When the image is magnified "
                   "(scale > 1), 36 source pixels have to cover more than 36 output pixels - some output cells simply "
                   "receive nobody (pink). When it is shrunk, several source pixels collide in one cell.")
        ui.callout("takeaway",
                   "**Inverse** = 'for every output pixel x′, *pull* the colour from x = H⁻¹x′.' Every output pixel "
                   "asks exactly one question, so there are **no holes and no collisions**. The price: x is usually "
                   "between pixels, so we must interpolate.")
        st.code(
            "# inverse mapping, the heart of every warp\n"
            "Hinv = np.linalg.inv(H)\n"
            "for each output pixel (u, v):\n"
            "    x, y, w = Hinv @ [u, v, 1]\n"
            "    out[v, u] = interpolate(img, x / w, y / w)",
            language="python",
        )

    ui.anim_intro("Step by step: forward mapping (left) vs inverse mapping (right)",
                  "Both use the scale you chose above. Play them one after the other and compare who gets left out.")
    ext = (lo[0] - 0.5, hi[0] + 0.5, lo[1] - 0.5, hi[1] + 0.5)
    all_cells = cells_hit + cells_hole
    l2, r2 = st.columns(2, gap="large")
    with l2:
        base = plotting.base_figure(ext, height=560, legend=False)
        base.add_trace(cell_trace(all_cells, "rgba(0,0,0,0)", "rgba(128,128,128,0.25)"))
        base.update_xaxes(dtick=1)
        base.update_yaxes(dtick=1)
        st.plotly_chart(anim.step_player(base, forward_steps(sd), n_sub=14, wrap_at=62), key="wp_fwd_steps")
    with r2:
        base = plotting.base_figure(ext, height=560, legend=False)
        src_cells = [tuple(v) for v in (np.c_[np.mgrid[0:6, 0:6][1].ravel(), np.mgrid[0:6, 0:6][0].ravel()] - 2.5)]
        base.add_trace(cell_trace(src_cells, theme.rgba(theme.ORIGINAL, 0.18), "rgba(128,128,128,0.6)"))
        base.add_trace(cell_trace(all_cells, "rgba(0,0,0,0)", "rgba(128,128,128,0.2)"))
        base.update_xaxes(dtick=1)
        base.update_yaxes(dtick=1)
        st.plotly_chart(anim.step_player(base, inverse_steps(sd), n_sub=14, wrap_at=62), key="wp_inv_steps")


holes_section()


# ------------------------------------------------------------------ bilinear interpolation
def bilinear_steps(ax, by, f):
    f00, f10, f01, f11 = f
    fa0 = (1 - ax) * f00 + ax * f10
    fa1 = (1 - ax) * f01 + ax * f11
    val = (1 - by) * fa0 + by * fa1

    def build(p):
        u, v, w = p["u"], p["v"], p["w"]        # progress of the three lerps
        x0 = u * ax
        z0 = (1 - x0) * f00 + x0 * f10
        x1 = v * ax
        z1 = (1 - x1) * f01 + x1 * f11
        yb = w * by
        zb = (1 - yb) * fa0 + yb * fa1
        return [
            go.Scatter3d(x=[x0], y=[0], z=[z0], mode="markers", marker=dict(size=7, color=theme.E1),
                         showlegend=False, hoverinfo="skip"),
            go.Scatter3d(x=[x1], y=[1], z=[z1], mode="markers", marker=dict(size=7, color=theme.E2),
                         showlegend=False, hoverinfo="skip"),
            go.Scatter3d(x=[ax, ax], y=[0, yb], z=[fa0, zb], mode="lines", line=dict(color=theme.HIGHLIGHT, width=7),
                         opacity=1.0 if w > 0 else 0.0, showlegend=False, hoverinfo="skip"),
            go.Scatter3d(x=[ax], y=[yb], z=[zb], mode="markers", marker=dict(size=9 * p["show"] + 0.1,
                         color=theme.HIGHLIGHT, symbol="diamond"), showlegend=False, hoverinfo="skip"),
        ]

    P = [dict(u=0.0, v=0.0, w=0.0, show=0.0), dict(u=1.0, v=0.0, w=0.0, show=0.0),
         dict(u=1.0, v=1.0, w=0.0, show=0.0), dict(u=1.0, v=1.0, w=1.0, show=1.0)]
    f_, H = ui.num, anim.hl
    info = [
        ("Four known pixels", f"f₀₀ = {f00}, f₁₀ = {f10}, f₀₁ = {f01}, f₁₁ = {f11}",
         "Heights = intensities. We want the value at (a, b) = "
         f"({f_(ax)}, {f_(by)}) - somewhere in between."),
        ("Lerp along the bottom edge (b = 0)",
         f"f(a, 0) = (1 − a)·f₀₀ + a·f₁₀ = {H(f'{fa0:.1f}', theme.E1)}",
         "The orange point slides from f₀₀ towards f₁₀, a of the way."),
        ("Lerp along the top edge (b = 1)",
         f"f(a, 1) = (1 − a)·f₀₁ + a·f₁₁ = {H(f'{fa1:.1f}', theme.E2)}",
         "The same a, on the other row: the green point."),
        ("Lerp between the two results",
         f"f(a, b) = (1 − b)·f(a, 0) + b·f(a, 1) = {H(f'{val:.1f}')}",
         "Two lerps in x, one in y. Multiply out and you get the four weights (1−a)(1−b), a(1−b), (1−a)b, ab."),
    ]
    return [anim.Step(t_, e_, n_, build(P[i]), between=None if i == 0 else anim.morph(build, P[i - 1], P[i]))
            for i, (t_, e_, n_) in enumerate(info)]


@live.section
def bilinear_section():
    st.subheader("Bilinear interpolation: a weighted vote of 4 neighbours", divider="gray")
    c_ctrl, c_fig, c_txt = st.columns([0.75, 1.15, 1.3], gap="large")
    with c_ctrl:
        ax = live.slider("fractional x offset  a", 0.0, 1.0, 0.3, 0.01, key="wp_a")
        by = live.slider("fractional y offset  b", 0.0, 1.0, 0.65, 0.01, key="wp_b")
        st.markdown("Neighbour intensities")
        r1 = st.columns(2)
        f00 = r1[0].number_input("f(0,0)", 0, 255, 40, key="wp_f00")
        f10 = r1[1].number_input("f(1,0)", 0, 255, 200, key="wp_f10")
        r2 = st.columns(2)
        f01 = r2[0].number_input("f(0,1)", 0, 255, 120, key="wp_f01")
        f11 = r2[1].number_input("f(1,1)", 0, 255, 250, key="wp_f11")
    w00, w10, w01, w11 = (1 - ax) * (1 - by), ax * (1 - by), (1 - ax) * by, ax * by
    val = w00 * f00 + w10 * f10 + w01 * f01 + w11 * f11
    with c_fig:
        fig = plotting.base_figure((-0.35, 1.35, -0.35, 1.35), height=440, legend=False)
        # the weight of each corner is the area of the rectangle OPPOSITE to it
        rects = [((ax, by, 1, 1), w00, f00, (0, 0)), ((0, by, ax, 1), w10, f10, (1, 0)),
                 ((ax, 0, 1, by), w01, f01, (0, 1)), ((0, 0, ax, by), w11, f11, (1, 1))]
        for (x0, y0, x1, y1), wgt, fv, (cx, cy) in rects:
            g = int(fv)
            fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color="rgba(128,128,128,0.6)", width=1),
                          fillcolor=f"rgba({g},{g},{g},0.55)", layer="below")
            fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=f"w={wgt:.2f}", showarrow=False,
                               font=dict(size=12, color="white" if g < 140 else "black"))
            fig.add_trace(go.Scatter(x=[cx], y=[cy], mode="markers+text", text=[f"f({cx},{cy}) = {fv}"],
                                     textposition="top center" if cy == 0 else "bottom center",
                                     marker=dict(size=26, color=f"rgb({g},{g},{g})", symbol="square",
                                                 line=dict(color=theme.ORIGINAL, width=1.5)),
                                     hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[ax], y=[by], mode="markers", marker=dict(size=14, color=theme.HIGHLIGHT,
                                 symbol="diamond", line=dict(color="white", width=1.5)), hoverinfo="skip"))
        fig.update_yaxes(range=[1.35, -0.35])          # image convention: row index grows downwards
        st.plotly_chart(fig, key="wp_bilin")
    with c_txt:
        st.latex(r"\begin{aligned} f(a,b) =\ & (1-a)(1-b)\,f_{00} \\ +\ & a(1-b)\,f_{10} \\ "
                 r"+\ & (1-a)\,b\,f_{01} \\ +\ & a\,b\,f_{11} \end{aligned}"
                 rf"\;=\;\begin{{aligned}} & {w00:.2f}\cdot{f00} \\ +\ & {w10:.2f}\cdot{f10} \\ "
                 rf"+\ & {w01:.2f}\cdot{f01} \\ +\ & {w11:.2f}\cdot{f11} \end{{aligned}}")
        st.latex(rf"f({ax:.2f},\ {by:.2f}) = \mathbf{{{val:.1f}}}")
        ui.callout("intuition",
                   "Each corner's weight is the **area of the rectangle on the opposite side** of the amber point. "
                   "Move the point towards a corner and that corner's opposite rectangle grows - so its vote counts "
                   "more. The four weights always add up to 1.")
        st.markdown("- **nearest**: copy the closest pixel - fast, blocky, jagged edges.\n"
                    "- **bilinear**: 2×2 neighbours, weights above - smooth, a little blurry.\n"
                    "- **bicubic**: 4×4 neighbours with a cubic kernel - sharper, may overshoot (ringing).")

    ui.anim_intro("Step by step: bilinear = three straight-line blends",
                  "The surface is the bilinear interpolant of the four pixel values (heights). Uses your a, b and the "
                  "four intensities. Drag the 3D view to look at it from any side.")
    g = np.linspace(0, 1, 21)
    X, Y = np.meshgrid(g, g)
    Z = (1 - X) * (1 - Y) * f00 + X * (1 - Y) * f10 + (1 - X) * Y * f01 + X * Y * f11
    base = go.Figure()
    base.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale=[[0, "#3a3a3a"], [1, "#f2f2f2"]], cmin=0, cmax=255,
                              opacity=0.55, showscale=False, hoverinfo="skip"))
    corners = [(0, 0, f00), (1, 0, f10), (0, 1, f01), (1, 1, f11)]
    for x_, y_, z_ in corners:
        base.add_trace(go.Scatter3d(x=[x_, x_], y=[y_, y_], z=[0, z_], mode="lines+markers+text",
                                    text=["", f"{z_}"], textposition="top center",
                                    line=dict(color=theme.ORIGINAL, width=5), marker=dict(size=[0, 6], color=theme.ORIGINAL),
                                    showlegend=False, hoverinfo="skip"))
    base.add_trace(go.Scatter3d(x=[0, 1], y=[0, 0], z=[f00, f10], mode="lines", line=dict(color=theme.E1, width=4,
                                dash="dash"), showlegend=False, hoverinfo="skip"))
    base.add_trace(go.Scatter3d(x=[0, 1], y=[1, 1], z=[f01, f11], mode="lines", line=dict(color=theme.E2, width=4,
                                dash="dash"), showlegend=False, hoverinfo="skip"))
    base.update_layout(height=640, showlegend=False, uirevision="keep",
                       scene=dict(xaxis=dict(range=[-0.05, 1.05], title="a (x)"),
                                  yaxis=dict(range=[-0.05, 1.05], title="b (y)"),
                                  zaxis=dict(range=[0, 270], title="intensity"),
                                  aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.7),
                                  camera=dict(eye=dict(x=1.45, y=-1.45, z=0.85))))
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        st.plotly_chart(anim.step_player(base, bilinear_steps(ax, by, (f00, f10, f01, f11)), n_sub=14, wrap_at=80),
                        key="wp_bil_steps")
    with right:
        ui.callout("note",
                   "Why is it called *bi*-linear? Along any line parallel to an axis the surface is a straight line "
                   "(linear in a, or linear in b) - but it is NOT a plane: the edges twist. That is why the "
                   "formula has the a·b term.", title="Bi-linear, not linear")


bilinear_section()

c1, c2 = st.columns(2, gap="large")
with c1:
    ui.callout("practice",
               "`cv2.warpAffine` / `cv2.warpPerspective` take the forward matrix H (source → destination) and "
               "internally invert it to do inverse mapping (pass `cv2.WARP_INVERSE_MAP` if you already have H⁻¹). "
               "The `flags` argument picks the interpolation; `borderMode` decides what fills pixels whose "
               "pre-image falls outside the source.")
with c2:
    ui.callout("pitfall",
               "Image coordinates are (column, row) with **y pointing down**, and the origin is the **top-left "
               "pixel**. That is why this page conjugates every matrix by the image centre, T(c)·M·T(−c), and why "
               "a positive rotation here looks *clockwise*.")

ui.whys([
    ('Why not use forward mapping and simply fill the holes afterwards?',
     "You can ('splatting' spreads each source pixel over several output pixels), but you must also decide what to do when several source pixels land on the same output pixel, and the result depends on the scale. Inverse mapping avoids both problems by asking one clean question per output pixel."),
    ('Why must H be invertible for warping?',
     'Inverse mapping needs H⁻¹ to find where each output pixel came from. A singular H squashes the image onto a line; there is no way back, so most output pixels would have no source at all.'),
    ('Why are bilinear weights the areas of the opposite rectangles?',
     "Bilinear interpolation is 'linear in x, then linear in y'. The weight of f₀₀ is (1−a)(1−b): how far the point is from the *other* side in x times in y - which is precisely the width × height of the rectangle opposite f₀₀. The closer you are to a corner, the bigger its opposite rectangle."),
    ('Why does bicubic look sharper but sometimes create halos?',
     'Bicubic fits a smooth cubic through 4×4 neighbours. Cubics can overshoot near sudden changes, so a dark/bright edge gets a thin brighter/darker rim (ringing) - the price of sharper detail.'),
    ('Why does the warped image get black corners?',
     "Those output pixels map back (through H⁻¹) to positions outside the source image - there is simply no data there. OpenCV's borderMode decides whether they become black, a copy of the edge, or a reflection."),
])

st.divider()
ui.mcq("wp1", "You enlarge an image 3× with forward mapping. What do you expect?",
       ["A perfect result", "Holes between the copied pixels", "The image gets darker", "The image becomes blurry"],
       1, "Each source pixel lands on one output pixel, but the output has ~9× more pixels - most stay empty.")
ui.mcq("wp2", "In inverse mapping, which matrix maps the output pixel back to the source?",
       ["H", "Hᵀ", "H⁻¹", "−H"], 2, "x = H⁻¹ x′, then divide by w and interpolate.")

ui.lesson_nav(__file__)
