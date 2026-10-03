import cv2
import numpy as np
import pandas as pd
import streamlit as st

from core import anim, images, live, plotting, shapes, theme, ui
from core import transforms as tf

ui.lesson_header(
    __file__,
    "Type any 3×3 matrix and the page tells you which family it belongs to, what it preserves, and how it "
    "decomposes into simple moves - while you watch it act on a shape and on an image.",
)

PRESETS = {
    "Identity": np.eye(3),
    "Rotate 30°": tf.rotation(30),
    "Rigid: rotate 45° + shift": tf.euclidean(45, 1.0, 0.5),
    "Similarity: zoom 1.5 + rotate": tf.similarity(1.5, -20, -0.5, 0.3),
    "Shear": tf.shear(0.7),
    "Non-uniform scale": tf.scaling(1.6, 0.6),
    "Mirror (reflection)": tf.reflection_x() @ tf.translation(0, -0.5),
    "Perspective tilt": np.array([[1.0, 0.1, 0.0], [0.0, 0.9, 0.0], [0.2, 0.12, 1.0]]),
    "Singular (collapse to a line)": np.array([[1.0, 0.5, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    "Random affine": None,
    "Random homography": None,
}

HOUSE = shapes.house(0.9, offset=(-0.9, -1.1))


def decomp_H(p):
    P = np.array([[1.0, 0, 0], [0, 1.0, 0], [p["v1"], p["v2"], 1.0]])
    return (tf.translation(p["tx"], p["ty"]) @ tf.rotation(p["th"]) @ tf.shear(p["sh"]) @ tf.scaling(p["sx"], p["sy"])
            @ P)


def decomp_steps(Hn, info):
    """H = T · R · Sh · S · P  (P = perspective part, applied first; the rest is the affine part)."""
    v1, v2 = Hn[2, 0], Hn[2, 1]
    Pinv = np.array([[1.0, 0, 0], [0, 1.0, 0], [-v1, -v2, 1.0]])
    d = tf.decompose_affine(Hn @ Pinv)
    projective = abs(v1) + abs(v2) > 1e-9
    n = 24 if projective else 2

    def build(p):
        return plotting.layer_traces(HOUSE, decomp_H(p), grid_extent=3.0, n=n)

    P = [dict(v1=0.0, v2=0.0, sx=1.0, sy=1.0, sh=0.0, th=0.0, tx=0.0, ty=0.0)]
    f, H = ui.num, anim.hl
    info_steps = [("Start", "H = I", "The house before H.")]
    if projective:
        P.append(dict(P[-1], v1=v1, v2=v2))
        info_steps.append(("Perspective part P", f"P = [1 0 0 ; 0 1 0 ; {H(f(v1, 3))} {H(f(v2, 3))} 1]",
                           "Only the bottom row: lines start to converge. Everything that follows is affine."))
    if not np.isfinite(d["shear"]):
        d["shear"] = 0.0
    P.append(dict(P[-1], sx=d["sx"], sy=d["sy"]))
    info_steps.append(("Scale S", f"S({H(f(d['sx']))}, {H(f(d['sy']))})",
                       "Stretch along the axes" + (" - a negative factor is a mirror: the house passes through a line."
                                                   if d["sy"] < 0 or d["sx"] < 0 else ".")))
    P.append(dict(P[-1], sh=d["shear"]))
    info_steps.append(("Shear Sh", f"Sh({H(f(d['shear']))}):  x′ = x + k·y", "The step that tilts right angles."))
    P.append(dict(P[-1], th=d["theta"]))
    info_steps.append(("Rotate R", f"R({H(f(d['theta'], 1) + '°')})", "Turning keeps the shape."))
    P.append(dict(P[-1], tx=d["tx"], ty=d["ty"]))
    info_steps.append(("Translate T", f"T({H(f(d['tx']))}, {H(f(d['ty']))})   →   T·R·Sh·S" + ("·P" if projective else "")
                       + " = H", "And we have rebuilt your matrix from simple moves (read the product right to left)."))
    return [anim.Step(t_, e_, n_, build(P[i]), between=None if i == 0 else anim.morph(build, P[i - 1], P[i]))
            for i, (t_, e_, n_) in enumerate(info_steps)]


def decomp_base():
    base = plotting.base_figure(4.0, height=600, legend=False)
    for tr in plotting.static_layers(HOUSE, grid_extent=3.0):
        base.add_trace(tr)
    return base


@live.section
def playground_section():
    c_in, c_fig, c_out = st.columns([1.15, 1.5, 1.05], gap="large")
    with c_in:
        preset = st.selectbox("Start from a preset", list(PRESETS), index=7, key="pg_preset")
        st.session_state.setdefault("pg_rand", 0)
        if preset.startswith("Random"):
            if st.button("Roll again", icon=":material/casino:", key="pg_roll"):
                st.session_state.pg_rand += 1
            rng = np.random.default_rng(st.session_state.pg_rand)
            base = tf.affine(*rng.uniform(-1.3, 1.3, 4), *rng.uniform(-1, 1, 2))
            base[:2, :2] += np.eye(2) * 0.6
            if preset == "Random homography":
                base[2, :2] = rng.uniform(-0.25, 0.25, 2)
            M0 = base
        else:
            M0 = PRESETS[preset]
        st.markdown("**Edit the matrix** (double-click a cell)")
        df = pd.DataFrame(np.round(M0, 3), columns=["× x", "× y", "× 1"], index=["x̃'", "ỹ'", "w̃'"])
        edited = st.data_editor(df, key=f"pg_ed_{preset}_{st.session_state.pg_rand}", width="stretch",
                                column_config={c: st.column_config.NumberColumn(c, format="%.2f", step=0.05,
                                                                                width="small")
                                               for c in df.columns})
        st.caption("Row i, column j is hᵢⱼ. Columns multiply x, y and 1; rows produce x̃', ỹ', w̃'.")
        H = edited.to_numpy(dtype=float)
        if not np.all(np.isfinite(H)):
            st.error("All 9 entries must be numbers.")
            st.stop()
        show_img = st.toggle("Also warp an image", value=True, key="pg_img")

    info = tf.classify(H)
    Hn = tf.normalize_h(H)
    with c_fig:
        extent = 4.0
        fig = plotting.transform_figure(HOUSE, Hn if not info["singular"] else H,
                                        extent=extent, height=500, grid_step=0.5, grid_extent=3)
        st.plotly_chart(fig, key="pg_fig")

    with c_out:
        if info["singular"]:
            st.error(f"**{info['name']}** - det H = 0, so the plane collapses and there is no inverse.",
                     icon=":material/block:")
        else:
            st.success(f"**{info['name']}**  ·  {info['dof']} DOF", icon=":material/verified:")
        ui.show_matrix(r"H / h_{33}" if abs(H[2, 2]) > 1e-9 else "H", Hn, 3,
                       {(2, 0): theme.MAGENTA, (2, 1): theme.MAGENTA, **ui.column_highlight()})
        if not info["singular"] and info["key"] != "projective":
            d = tf.decompose_affine(Hn)
            st.markdown("**Reads as** (right to left: scale → shear → rotate → shift)")
            st.latex(rf"H = T({ui.num(d['tx'])}, {ui.num(d['ty'])})\;R({ui.num(d['theta'], 1)}^\circ)\;"
                     rf"Sh({ui.num(d['shear'])})\;S({ui.num(d['sx'])}, {ui.num(d['sy'])})")
            U, S, Vt = tf.svd_2x2(Hn)
            st.caption(f"det A = {info['det']:.3f} (area × {abs(info['det']):.3f}{', mirrored' if info['det'] < 0 else ''}) · "
                       f"singular values σ₁ = {S[0]:.3f}, σ₂ = {S[1]:.3f}")
        elif info["key"] == "projective":
            lv = np.linalg.inv(Hn).T @ np.array([0, 0, 1.0])
            st.markdown(f"Perspective row: h₃₁ = {Hn[2, 0]:.3f}, h₃₂ = {Hn[2, 1]:.3f} → the horizon in the output is "
                        f"the line {lv[0]:.2f}x + {lv[1]:.2f}y + {lv[2]:.2f} = 0.")
            vx, vy = Hn @ [1, 0, 0], Hn @ [0, 1, 0]
            fmt = lambda v: f"({v[0] / v[2]:.2f}, {v[1] / v[2]:.2f})" if abs(v[2]) > 1e-9 else "at infinity"
            st.caption(f"Vanishing point of horizontal lines: {fmt(vx)} · of vertical lines: {fmt(vy)}")
        if not info["singular"]:
            ui.preserved_badges(info["key"])

    if show_img and not info["singular"]:
        W = 360
        P = np.array([[W / 5, 0, W / 2], [0, -W / 5, W / 2], [0, 0, 1.0]])      # math [-2.5,2.5]² (y up) → pixels (y down)
        canvas = np.full((W, W, 3), 245, np.uint8)
        # place the poster on the math square [-1, 1]²
        side = int(W / 5 * 2)
        pic = cv2.resize(images.poster(240), (side, side))
        canvas[W // 2 - side // 2: W // 2 - side // 2 + side, W // 2 - side // 2: W // 2 - side // 2 + side] = pic
        Hpx = P @ Hn @ np.linalg.inv(P)
        warped = cv2.warpPerspective(canvas, Hpx, (W, W), flags=cv2.INTER_LINEAR, borderValue=(245, 245, 245))
        a, b, c = st.columns([1, 1, 1.2], gap="medium")
        a.image(ui.datauri(canvas), caption="Image placed on the square [−1, 1]²", width="stretch")
        b.image(ui.datauri(warped), caption="After H (same axes as the plot)", width="stretch")
        with c:
            ui.callout("note",
                       "To warp a picture with a matrix written in maths coordinates (y up, origin in the middle) we "
                       "change coordinates first: H_pixels = P · H · P⁻¹, where P maps maths units to pixel rows and "
                       "columns. This conjugation trick is the same one used for rotating about a pivot.",
                       title="Two coordinate systems")


    # ------------------------------------------------------------------ decomposition, step by step
    if info["singular"] or abs(Hn[2, 2] - 1) > 1e-6:      # h₃₃ = 0: no 'perspective × affine' split
        return
    st.subheader("Take the matrix apart, one simple move at a time", divider="gray")
    st.plotly_chart(anim.step_player(decomp_base(), decomp_steps(Hn, info), n_sub=14), key="pg_decomp")


playground_section()

st.subheader("Experiments worth doing", divider="gray")
a, b, c = st.columns(3, gap="medium")
with a:
    ui.callout("try", "Start from **Rotate 30°** and multiply the whole matrix by 2 (every cell, including the "
                      "bottom-right 1). Nothing changes! H and k·H are the same homography.")
with b:
    ui.callout("try", "Start from **Shear** and change the 2×2 block to [[1, −0.7], [0.7, 1]]: the two columns are "
                      "now perpendicular and equally long, so the class jumps from *Affine* to *Similarity* "
                      "(rotation 35°, scale 1.22).")
with c:
    ui.callout("try", "Start from **Identity** and put 0.3 in row 2, col 0. A single number turns the map "
                      "projective and creates a vanishing point at (1/0.3, 0) ≈ (3.33, 0).")

ui.lesson_nav(__file__)
