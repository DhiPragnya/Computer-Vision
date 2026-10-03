import numpy as np
import streamlit as st

from core import anim, live, plotting, shapes, theme, ui
from core import transforms as tf

ui.lesson_header(
    __file__,
    "Because every transform is a 3×3 matrix, doing one transform after another is just "
    "<b>matrix multiplication</b>. That single fact explains why order matters, how to undo a chain of "
    "moves, and why each family is closed under composition.",
)

OPS = ["Translate", "Rotate", "Scale", "Shear", "Perspective", "Do nothing"]
HOUSE = shapes.house(0.8, offset=(0.2, 0.2))


def op_controls(i: int, default_op: str, defaults: dict):
    """Returns (name, matrix, latex label, partial(t) -> matrix of the op done t of the way, plain label)."""
    with st.container(border=True):
        st.markdown(f"**Step {i}**")
        op = st.selectbox("operation", OPS, index=OPS.index(default_op), key=f"cp_op{i}",
                          label_visibility="collapsed")
        if op == "Translate":
            a, b = st.columns(2)
            with a:
                tx = live.slider("tₓ", -3.0, 3.0, defaults.get("tx", 2.0), 0.1, key=f"cp_tx{i}")
            with b:
                ty = live.slider("t_y", -3.0, 3.0, defaults.get("ty", 0.0), 0.1, key=f"cp_ty{i}")
            return (op, tf.translation(tx, ty), rf"T({ui.num(tx, 1)},{ui.num(ty, 1)})",
                    lambda t: tf.translation(t * tx, t * ty), f"T({ui.num(tx, 1)}, {ui.num(ty, 1)})")
        if op == "Rotate":
            th = live.slider("θ°", -180, 180, defaults.get("th", 90), 5, key=f"cp_th{i}", format="%d°")
            return op, tf.rotation(th), rf"R({th}^\circ)", lambda t: tf.rotation(t * th), f"R({th}°)"
        if op == "Scale":
            a, b = st.columns(2)
            with a:
                sx = live.slider("sₓ", 0.2, 2.5, defaults.get("sx", 1.5), 0.1, key=f"cp_sx{i}")
            with b:
                sy = live.slider("s_y", 0.2, 2.5, defaults.get("sy", 1.5), 0.1, key=f"cp_sy{i}")
            return (op, tf.scaling(sx, sy), rf"S({ui.num(sx, 1)},{ui.num(sy, 1)})",
                    lambda t: tf.scaling(1 + t * (sx - 1), 1 + t * (sy - 1)), f"S({ui.num(sx, 1)}, {ui.num(sy, 1)})")
        if op == "Shear":
            k = live.slider("k", -1.5, 1.5, defaults.get("k", 0.6), 0.1, key=f"cp_k{i}")
            return op, tf.shear(k), rf"Sh({ui.num(k, 1)})", lambda t: tf.shear(t * k), f"Sh({ui.num(k, 1)})"
        if op == "Perspective":
            p = live.slider("h₃₁", -0.4, 0.4, defaults.get("p", 0.15), 0.01, key=f"cp_p{i}")
            P = lambda t: np.array([[1, 0, 0], [0, 1, 0], [t * p, 0, 1.0]])
            return op, P(1.0), rf"P({ui.num(p, 2)})", P, f"P({ui.num(p, 2)})"
        st.caption("identity")
        return op, np.eye(3), "I", lambda t: np.eye(3), "I"


def chain_layers(H, projective, axes=None):
    trs = plotting.layer_traces(HOUSE, H, grid_extent=5.0, n=24 if projective else 2)
    if axes:
        for tr in trs:
            tr.update(xaxis=axes[0], yaxis=axes[1])
    return trs


def chain_steps(ops, projective):
    """Apply the chain one operation at a time; each morph performs exactly one factor."""
    mats = [o[1] for o in ops]
    labels = [o[4] for o in ops]
    acc = [np.eye(3)]
    for M in mats:
        acc.append(M @ acc[-1])
    H = anim.hl
    steps = [anim.Step("Start", "p", "The house before anything happens.", chain_layers(acc[0], projective))]
    done = []
    for k, (name, M, _, part, lab) in enumerate(ops):
        eq = " · ".join([H(lab)] + done[::-1] + ["p"])
        done.append(lab)
        prev = acc[k]
        steps.append(anim.Step(
            f"Step {k + 1}: {name}", eq,
            ["The rightmost matrix acts first - it touches the original point.",
             "This matrix acts on the RESULT of step 1 (it does not know where the house started).",
             "And this one acts on everything done so far. The whole chain is ONE matrix."][k],
            chain_layers(acc[k + 1], projective),
            between=(lambda t, part=part, prev=prev: chain_layers(part(t) @ prev, projective))))
    return steps


def order_steps(ops, projective):
    """Your order (left) vs the reversed order (right), one factor at a time, side by side."""
    fwd, rev = list(ops), list(reversed(ops))
    accs = []
    for seq in (fwd, rev):
        a = [np.eye(3)]
        for o in seq:
            a.append(o[1] @ a[-1])
        accs.append(a)

    def both(Hl, Hr):
        return chain_layers(Hl, projective, ("x", "y")) + chain_layers(Hr, projective, ("x2", "y2"))

    steps = [anim.Step("Start", "left: your order   ·   right: reversed order", "Both start from the same house.",
                       both(np.eye(3), np.eye(3)))]
    for k in range(3):
        lf, rf = fwd[k], rev[k]
        pl, pr = accs[0][k], accs[1][k]
        steps.append(anim.Step(
            f"Step {k + 1}", f"left applies {anim.hl(lf[4])}   ·   right applies {anim.hl(rf[4], theme.MAGENTA)}",
            "Same three matrices, different order." + (" Compare the two houses now." if k == 2 else ""),
            both(accs[0][k + 1], accs[1][k + 1]),
            between=(lambda t, lf=lf, rf=rf, pl=pl, pr=pr: both(lf[3](t) @ pl, rf[3](t) @ pr))))
    return steps


def undo_steps(ops, M, projective):
    """Take the chain apart again: last on, first off."""
    inv = [np.linalg.inv(o[1]) for o in ops]
    states = [M]
    for k in (2, 1, 0):
        states.append(inv[k] @ states[-1])
    H = anim.hl
    labs = [o[4] for o in ops]
    steps = [anim.Step("The finished chain", f"M = {labs[2]} · {labs[1]} · {labs[0]}",
                       "To get back we must undo the LAST move first.", chain_layers(M, projective))]
    for j, k in enumerate((2, 1, 0)):
        part = ops[k][3]
        cur = states[j]
        # undoing op k = running op k backwards: part(1 - t) · part(1)⁻¹
        steps.append(anim.Step(
            f"Undo step {k + 1}", f"apply {H(labs[k] + '⁻¹')}",
            ["Shoes come off first: the last move is undone first.", "Then the middle move.",
             "Finally the first move - and we are back at the start: M⁻¹ = M₁⁻¹ M₂⁻¹ M₃⁻¹."][j],
            chain_layers(states[j + 1], projective),
            between=(lambda t, part=part, k=k, cur=cur: chain_layers(part(1 - t) @ inv[k] @ cur, projective))))
    return steps


def two_panel_base(titles):
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2, subplot_titles=titles, horizontal_spacing=0.06)
    for c in (1, 2):
        for tr in plotting.static_layers(HOUSE, grid_extent=5.0):
            fig.add_trace(tr, row=1, col=c)
    fig.update_xaxes(range=[-4.5, 4.5], zeroline=True, zerolinecolor=theme.AXIS, gridcolor=theme.GRID,
                     constrain="domain")
    fig.update_yaxes(range=[-4.5, 4.5], zeroline=True, zerolinecolor=theme.AXIS, gridcolor=theme.GRID,
                     constrain="domain")
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_yaxes(scaleanchor="x2", scaleratio=1, row=1, col=2)
    fig.update_layout(height=560, showlegend=False)
    fig.update_annotations(font_size=13, y=0.995, yanchor="top")    # titles inside, the caption sits above
    return fig


@live.section
def composition_section():
    # ------------------------------------------------------------------ build a pipeline
    st.subheader("Build a chain of three moves", divider="gray")
    st.markdown("The default chain is the lecture-notes example **p′ = T R S p**: first scale, then rotate, "
                "finally translate. Change any step.")
    cols = st.columns(3)
    with cols[0]:
        o1 = op_controls(1, "Scale", dict(sx=1.5, sy=0.8))
    with cols[1]:
        o2 = op_controls(2, "Rotate", dict(th=45))
    with cols[2]:
        o3 = op_controls(3, "Translate", dict(tx=2.0, ty=0.5))
    ops = [o1, o2, o3]
    (op1, M1, n1, _, _), (op2, M2, n2, _, _), (op3, M3, n3, _, _) = ops
    projective = "Perspective" in (op1, op2, op3)

    M = M3 @ M2 @ M1
    ui.anim_intro("Step by step: the chain acting one matrix at a time",
                  "▶ Play runs the chain; the numbered buttons replay one step; drag the slider to go back and forth. "
                  "Each morph performs exactly one factor of the product.")
    base = plotting.base_figure(4.5, height=560, legend=False)
    for tr in plotting.static_layers(HOUSE, grid_extent=5.0):
        base.add_trace(tr)
    left, right = st.columns([1.45, 1], gap="large")
    with left:
        st.plotly_chart(anim.step_player(base, chain_steps(ops, projective), n_sub=14, wrap_at=80), key="cp_steps")
    with right:
        st.latex(rf"M = \underbrace{{{n3}}}_{{\text{{3rd}}}}\;\underbrace{{{n2}}}_{{\text{{2nd}}}}\;"
                 rf"\underbrace{{{n1}}}_{{\text{{1st}}}}")
        st.latex("M = " + ui.tex_matrix(M3, 2) + ui.tex_matrix(M2, 2) + ui.tex_matrix(M1, 2))
        st.latex("= " + ui.tex_matrix(tf.normalize_h(M), 2))
        ui.callout("intuition",
                   "A point is a column vector on the **right**, so the matrix nearest to it acts first: "
                   "$M\\mathbf{x} = M_3(M_2(M_1\\mathbf{x}))$. Read products **right to left**, like function "
                   "composition $f\\circ g$.")
        parts = [tf.classify(Mi)["key"] for Mi in (M1, M2, M3)]
        most = max(parts, key=lambda k: tf.CLASS_ORDER.index(k) if k in tf.CLASS_ORDER else -1)
        st.markdown(f"Class of the result: **{tf.classify(M)['name']}** "
                    f"(the most general step was *{tf.CLASS_BY_KEY.get(most, {'name': most})['name']}*).")

    # ------------------------------------------------------------------ order matters
    st.subheader("Does the order matter?", divider="gray")
    M_rev = M1 @ M2 @ M3
    same = np.allclose(tf.normalize_h(M), tf.normalize_h(M_rev), atol=1e-9)
    ui.anim_intro("Side by side: your order (left) and the reversed order (right)",
                  "Both panels play their steps at the same time, so you can see exactly where the two histories "
                  "part ways.")
    t_left = f"Your order:  {o3[4]} · {o2[4]} · {o1[4]}"
    t_right = f"Reversed:  {o1[4]} · {o2[4]} · {o3[4]}"
    st.plotly_chart(anim.step_player(two_panel_base([t_left, t_right]), order_steps(ops, projective), n_sub=14),
                    key="cp_order")
    if same:
        st.success("These particular steps **commute** - both orders give the same matrix. Try *Rotate* then "
                   "*Translate*, or *Scale* (non-uniform) then *Rotate*, to break it.", icon=":material/sync:")
    else:
        st.warning("Different results: matrix multiplication is **not commutative**. "
                   "*Rotate then translate* turns the house about the origin and then shifts it; "
                   "*translate then rotate* swings the already-shifted house around the origin like a ball on a "
                   "string.", icon=":material/swap_horiz:")
    with st.expander("Which pairs DO commute?", icon=":material/rule:"):
        st.markdown("- two translations: T(a)T(b) = T(a + b)\n"
                    "- two rotations about the **same** centre: angles add\n"
                    "- a rotation and a **uniform** scale about the same centre\n"
                    "- anything with the identity\n\n"
                    "Almost everything else - rotation & translation, rotation & non-uniform scale, shear & scale - "
                    "does **not** commute.")

    # ------------------------------------------------------------------ inverses
    st.subheader("Undoing a chain: socks and shoes", divider="gray")
    if abs(np.linalg.det(M)) < 1e-9:
        st.error("This chain is not invertible (det = 0).")
        return
    Minv = np.linalg.inv(M)
    ui.anim_intro("Step by step: undo the chain, last move first")
    left, right = st.columns([1.45, 1], gap="large")
    with left:
        st.plotly_chart(anim.step_player(base, undo_steps(ops, M, projective), n_sub=14, wrap_at=80), key="cp_undo")
    with right:
        st.latex(r"(M_3 M_2 M_1)^{-1} = M_1^{-1}\,M_2^{-1}\,M_3^{-1}")
        st.latex("M^{-1} = " + ui.tex_matrix(tf.normalize_h(Minv), 3))
        st.latex(r"M^{-1}M = " + ui.tex_matrix(Minv @ M, 2))
        st.markdown("Handy closed forms:")
        st.latex(r"T(\mathbf{t})^{-1} = T(-\mathbf{t}),\qquad R(\theta)^{-1} = R(-\theta) = R^T")
        st.latex(r"\begin{bmatrix} R & \mathbf{t} \\ 0 & 1 \end{bmatrix}^{-1} = "
                 r"\begin{bmatrix} R^T & -R^T\mathbf{t} \\ 0 & 1 \end{bmatrix}")
        ui.callout("intuition",
                   "You put on socks, then shoes. To undo, you take off the **shoes first**. "
                   "Inverting a chain reverses the order *and* inverts each step.")
    ui.callout("practice",
               "Robotics and vision are full of chains like  ᵂT_camera · ᶜT_object. "
               "The pose of the object in the world is a product of matrices, and the pose of the camera "
               "seen *from* the object is its inverse. Panorama stitching chains homographies "
               "H₁₃ = H₂₃ · H₁₂ image to image.")


composition_section()

st.subheader("Each family is a group", divider="gray")
ui.callout("note",
           "Compose two members of the same family and you stay in the family; every member has an inverse in "
           "the family; the identity belongs to all of them. So translations, rigid motions SE(2), similarities, "
           "affinities and homographies are each a **group** - and each is a subgroup of the next. Mixing families "
           "gives a result in the **most general** family involved (try Rotate + Perspective above).",
           title="Closure")

ui.whys([
    ("Why are matrix products read from right to left?",
     "The point is a column vector standing on the right: T R S p means T(R(S p)). S touches p first, then R acts "
     "on the result, then T. It is the same as function composition f(g(x)) - the inner function runs first."),
    ("Why is the usual order scale → rotate → translate (T R S)?",
     "Scaling and rotation act about the origin. If the object is still sitting at the origin when you scale and "
     "rotate it, it grows and turns 'in place'; translating last then puts it where you want. Translate first "
     "and the later rotation swings the object around the origin like a ball on a string."),
    ("Why do two translations commute but rotation and translation don't?",
     "Translations just add vectors, and a + b = b + a. Rotating a translation vector changes its direction, so "
     "'shift then turn' moves the object along a rotated path, while 'turn then shift' moves it along the "
     "original path: R T(t) = T(Rt) R, which equals T(t) R only if Rt = t."),
    ("Why does undoing a chain reverse the order?",
     "The last thing done must be the first thing undone (socks and shoes). Algebraically "
     "(ABC)(C⁻¹B⁻¹A⁻¹) = AB(CC⁻¹)B⁻¹A⁻¹ = … = I."),
    ("Why multiply the matrices once instead of applying them one by one?",
     "Applying a chain of k matrices to N pixels costs k·N matrix-vector products; multiplying the matrices "
     "first costs k−1 tiny 3×3 products and then N. For a million-pixel image that is the difference "
     "between doing the work once and doing it k times - and it avoids resampling (blurring) the image k times."),
    ("Why does a chain end up in the most general family it contains?",
     "Each family is closed under composition, and every family sits inside the next. A rotation is also an "
     "affine map, an affine map is also a homography… so the product of rigid and projective pieces is, in "
     "general, only guaranteed to be projective."),
])

st.divider()
ui.mcq("cp1", "You want to scale an image by 2 about its centre c. Which matrix?",
       ["S(2) · T(−c) · T(c)", "T(c) · S(2) · T(−c)", "T(−c) · S(2) · T(c)", "S(2) · T(c)"], 1,
       "Right to left: move c to the origin, scale, move back.")
ui.mcq("cp2", "If H = A·B, what is H⁻¹?", ["A⁻¹·B⁻¹", "B⁻¹·A⁻¹", "B·A", "−A·B"], 1,
       "Socks and shoes: (AB)⁻¹ = B⁻¹A⁻¹. Check: (AB)(B⁻¹A⁻¹) = A(BB⁻¹)A⁻¹ = I.")

ui.lesson_nav(__file__)
