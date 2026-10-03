import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core import anim, live, plotting, shapes, theme, ui
from core import transforms as tf

ui.lesson_header(
    __file__,
    "In real images nobody hands you the matrix. You detect features, match them between two images, and "
    "<b>solve</b> for the transform that best explains the matches. Noise blurs the answer, wrong matches "
    "(outliers) can wreck it - and RANSAC rescues it.",
)

KEYS = tf.CLASS_ORDER
NAMES = {k: tf.CLASS_BY_KEY[k]["name"] for k in KEYS}


def random_transform(kind: str, rng) -> np.ndarray:
    t = rng.uniform(-1.2, 1.2, 2)
    th = rng.uniform(-50, 50)
    if kind == "translation":
        return tf.translation(*t)
    if kind == "euclidean":
        return tf.euclidean(th, *t)
    if kind == "similarity":
        return tf.similarity(rng.uniform(0.6, 1.4), th, *t)
    A = tf.translation(*t) @ tf.rotation(th) @ tf.shear(rng.uniform(-0.5, 0.5)) @ tf.scaling(*rng.uniform(0.6, 1.4, 2))
    if kind == "projective":
        A[2, :2] = rng.uniform(-0.12, 0.12, 2)
    return A


HOUSE = shapes.house(0.9, offset=(-0.9, -1.1))


def ransac_steps(fit_kind, src, dst, is_out, thr, n_show=6, n_total=60, seed=0):
    """Replay RANSAC: a few individual iterations, then the best of many, then the refit."""
    fit, m = tf.FITTERS[fit_kind]
    rng = np.random.default_rng(seed)
    n = len(src)
    records = []
    best = (-1, None, None, None)
    for it in range(n_total):
        idx = rng.choice(n, m, replace=False)
        try:
            H = fit(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(H)):
            continue
        inl = tf.residuals(H, src, dst) < thr
        if inl.sum() > best[0]:
            best = (int(inl.sum()), H, inl, idx)
        if len(records) < n_show:
            records.append((it + 1, idx, H, inl, best[0]))
    if best[1] is None:
        return None
    H_ref = fit(src[best[2]], dst[best[2]]) if best[2].sum() >= m else best[1]
    inl_ref = tf.residuals(H_ref, src, dst) < thr

    def state(H, idx, inl):
        trs = plotting.layer_traces(HOUSE, H, grid=False, basis=False, fill_alpha=0.16)
        S = dst[idx]
        trs.append(go.Scatter(x=S[:, 0], y=S[:, 1], mode="markers", showlegend=False, hoverinfo="skip",
                              marker=dict(size=17, color="rgba(0,0,0,0)", symbol="square",
                                          line=dict(color=theme.HIGHLIGHT, width=3))))
        sizes = np.where(inl, 15.0, 0.0)
        trs.append(go.Scatter(x=dst[:, 0], y=dst[:, 1], mode="markers", showlegend=False, hoverinfo="skip",
                              marker=dict(size=sizes, color="rgba(0,0,0,0)", line=dict(color=theme.E2, width=2))))
        return trs

    H_ = anim.hl
    steps = []
    for k, (it, idx, H, inl, best_so_far) in enumerate(records):
        bad = int(is_out[idx].sum())
        note = ("This sample contains a WRONG match - the model it produces disagrees with most points."
                if bad else "All sample points are correct matches - the model fits many points.")
        steps.append(anim.Step(
            f"Iteration {it}: random sample of {m}",
            f"fit to the {m} amber-framed matches → {H_(str(int(inl.sum())))} of {n} within the threshold "
            f"(best so far: {best_so_far})",
            note, state(H, idx, inl)))
    steps.append(anim.Step(
        f"Best of {n_total} iterations", f"consensus = {H_(str(best[0]))} of {n} matches",
        "Keep the model whose sample collected the biggest consensus (green rings = its inliers).",
        state(best[1], best[3], best[2])))
    steps.append(anim.Step(
        "Refit with least squares on all its inliers", f"{H_(str(int(inl_ref.sum())))} inliers after refitting",
        "The minimal sample was noisy; using every inlier averages the noise out. Outliers never get a vote.",
        state(H_ref, best[3], inl_ref)))
    return steps


@live.section
def estimation_section():
    st.session_state.setdefault("es_seed", 7)
    c_ctrl, c_fig = st.columns([0.9, 2.4], gap="large")
    with c_ctrl:
        st.markdown("**1 · The world (unknown to the solver)**")
        true_kind = st.selectbox("True transform", KEYS, index=3, format_func=NAMES.get, key="es_true")
        n = live.slider("number of matches", 4, 80, 30, key="es_n")
        sigma = live.slider("noise σ (pixel jitter)", 0.0, 0.3, 0.04, 0.01, key="es_sigma")
        out_frac = live.slider("outlier fraction (wrong matches)", 0.0, 0.6, 0.0, 0.05, key="es_out")
        if st.button("New random scene", icon=":material/casino:", width="stretch", key="es_new"):
            st.session_state.es_seed += 1
        st.markdown("**2 · The solver**")
        fit_kind = st.selectbox("Model to fit", KEYS, index=3, format_func=NAMES.get, key="es_fit")
        method = st.segmented_control("Method", ["Least squares", "RANSAC"], default="Least squares",
                                      key="es_method", required=True)
        m_min = tf.FITTERS[fit_kind][1]
        thr = 0.15
        iters = 50
        if method == "RANSAC":
            thr = live.slider("inlier threshold", 0.02, 0.6, 0.15, 0.01, key="es_thr")
            suggested = tf.ransac_iterations(max(out_frac, 0.05), m_min, 0.99)
            iters = live.slider("iterations", 1, 500, int(min(max(suggested, 10), 500)), key="es_it",
                                help=f"For 99% confidence with {out_frac:.0%} outliers and samples of {m_min}: "
                                     f"N = {suggested}")

    rng = np.random.default_rng(st.session_state.es_seed)
    H_true = random_transform(true_kind, rng)
    src = rng.uniform(-2.4, 2.4, (n, 2))
    dst = tf.apply(H_true, src) + rng.normal(0, sigma, (n, 2))
    is_out = np.zeros(n, bool)
    n_out = int(round(out_frac * n))
    if n_out:
        idx = rng.choice(n, n_out, replace=False)
        is_out[idx] = True
        dst[idx] = rng.uniform(-3.5, 3.5, (n_out, 2))

    if n < m_min:
        c_fig.error(f"A {NAMES[fit_kind]} needs at least {m_min} matches.")
        return
    if method == "RANSAC":
        H_fit, inl, history = tf.ransac(fit_kind, src, dst, thr, iters, seed=st.session_state.es_seed)
    else:
        H_fit = tf.FITTERS[fit_kind][0](src, dst)
        inl = np.ones(n, bool)
    pred = tf.apply(H_fit, src)
    res = np.linalg.norm(pred - dst, axis=1)
    corner_err = np.nanmean(np.linalg.norm(tf.apply(H_fit, HOUSE["outline"]) - tf.apply(H_true, HOUSE["outline"]),
                                           axis=1))

    with c_fig:
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Image 1: detected points", "Image 2: matches vs model"],
                            horizontal_spacing=0.04)
        for tr in plotting.shape_traces(HOUSE, None, theme.ORIGINAL, "object", dash="dash", fill_alpha=0.05,
                                        width=1.5, showlegend=False, hover=False):
            fig.add_trace(tr, row=1, col=1)
        fig.add_trace(plotting.points_trace(src, theme.ORIGINAL, "source points", size=7, showlegend=False),
                      row=1, col=1)
        for tr in plotting.shape_traces(HOUSE, H_true, theme.ORIGINAL, "true transform", dash="dash", fill_alpha=0.04,
                                        width=1.6, hover=False):
            fig.add_trace(tr, row=1, col=2)
        for tr in plotting.shape_traces(HOUSE, H_fit, theme.TRANSFORMED, "estimated transform", fill_alpha=0.16,
                                        width=2.2, hover=False):
            fig.add_trace(tr, row=1, col=2)
        xs, ys = [], []
        for p_, d_ in zip(pred, dst):
            xs += [p_[0], d_[0], None]
            ys += [p_[1], d_[1], None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=theme.HIGHLIGHT, width=1.5),
                                 name="residual (prediction → observation)", hoverinfo="skip"), row=1, col=2)
        good = ~is_out
        fig.add_trace(plotting.points_trace(dst[good], theme.TRANSFORMED, "observed match (correct)", size=8),
                      row=1, col=2)
        if is_out.any():
            fig.add_trace(plotting.points_trace(dst[is_out], theme.OUTLIER, "observed match (WRONG - outlier)", size=10,
                                                symbol="x"), row=1, col=2)
        if method == "RANSAC":
            fig.add_trace(go.Scatter(x=dst[inl, 0], y=dst[inl, 1], mode="markers",
                                     marker=dict(size=15, color="rgba(0,0,0,0)", line=dict(color=theme.E2, width=2)),
                                     name="RANSAC kept as inlier", hoverinfo="skip"), row=1, col=2)
        for c in (1, 2):
            fig.update_xaxes(range=[-4, 4], row=1, col=c, zeroline=True, zerolinecolor=theme.AXIS, gridcolor=theme.GRID)
            fig.update_yaxes(range=[-4, 4], row=1, col=c, zeroline=True, zerolinecolor=theme.AXIS, gridcolor=theme.GRID)
        fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
        fig.update_yaxes(scaleanchor="x2", scaleratio=1, row=1, col=2)
        fig.update_layout(height=520, margin=dict(l=4, r=4, t=60, b=4),
                          legend=dict(orientation="h", y=-0.06, x=0, yanchor="top"))
        st.plotly_chart(fig, key="es_fig")

        m1, m2, m3, m4 = st.columns(4)
        rms = np.sqrt(np.mean(res[~is_out] ** 2)) if (~is_out).any() else float("nan")
        m1.metric("RMS residual", f"{rms:.3f}", border=True,
                  help="On the correct matches only. Compare with the noise σ you set: a good fit has RMS ≈ σ·√2.")
        m2.metric("Error vs truth", f"{corner_err:.3f}", border=True,
                  help="Mean distance between the estimated and the true house outline.")
        m3.metric("Unknowns / eqs", f"{tf.CLASS_BY_KEY[fit_kind]['dof']} / {2 * n}", border=True)
        m4.metric("Inliers kept", f"{int(inl.sum())} / {n}" if method == "RANSAC" else "all (LS)", border=True)

    # ------------------------------------------------------------------ RANSAC, iteration by iteration
    st.subheader("RANSAC, one iteration at a time", divider="gray")
    steps = ransac_steps(fit_kind, src, dst, is_out, thr, seed=int(st.session_state.es_seed))
    if steps is None:
        st.info("No usable sample in this scene - add more matches.")
        return
    ui.anim_intro("Step by step: sample → fit → count the consensus → keep the best → refit",
                  f"Uses the scene above and an inlier threshold of {thr:.2f}"
                  + (" (the RANSAC slider)." if method == "RANSAC" else " (switch the method to RANSAC to change it).")
                  + " Amber squares = the random minimal sample; green rings = matches that agree with that model; "
                    "red × = wrong matches.")
    base = plotting.base_figure(4.0, height=580, legend=False)
    for tr in plotting.shape_traces(HOUSE, H_true, theme.ORIGINAL, "true", dash="dash", fill_alpha=0.04, width=1.6,
                                    showlegend=False, hover=False):
        base.add_trace(tr)
    base.add_trace(plotting.points_trace(dst[~is_out], theme.TRANSFORMED, "correct", size=8, showlegend=False))
    if is_out.any():
        base.add_trace(plotting.points_trace(dst[is_out], theme.OUTLIER, "wrong", size=10, symbol="x",
                                             showlegend=False))
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        st.plotly_chart(anim.step_player(base, steps, n_sub=12, wrap_at=80), key="es_ransac_steps")
    with right:
        ui.callout("intuition",
                   "A sample that contains even one wrong match produces a model that most points disagree with - "
                   "so it collects a small consensus and loses the vote. RANSAC never has to know which matches are "
                   "wrong; it only needs ONE lucky all-correct sample among its tries.")
        if out_frac == 0:
            st.caption("Tip: raise the outlier fraction above to see bad samples lose.")


estimation_section()

# ------------------------------------------------------------------ explanation
st.subheader("What to try", divider="gray")
a, b, c = st.columns(3, gap="medium")
with a:
    ui.callout("try",
               "**Under-fitting.** Keep the true transform *Projective* and fit a *Similarity*. Even with zero noise "
               "the residuals stay large and show a systematic pattern: the model is too simple to explain the data. "
               "Now fit *Projective*: residuals collapse to the noise level.")
with b:
    ui.callout("try",
               "**Outliers.** Set the outlier fraction to 20 %. Least squares drags the blue house towards the red "
               "crosses - one bad match can pull the whole solution. Switch to **RANSAC**: the house snaps back.")
with c:
    ui.callout("try",
               "**Minimal data.** Lower the matches to the minimum (1, 2, 2, 3, 4 for the five models). With noise, "
               "the solution fits the matches perfectly but is wrong elsewhere - more matches average the noise out.")

left, right = st.columns(2, gap="large")
with left:
    with st.container(border=True):
        st.markdown(":material/function: **Least squares - affine example**")
        st.markdown("Each match (xᵢ, yᵢ) → (uᵢ, vᵢ) gives two **linear** equations in the 6 unknowns. Stack them:")
        st.latex(r"\underbrace{\begin{bmatrix} x_1 & y_1 & 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & x_1 & y_1 & 1 \\ "
                 r"\vdots & & & & & \vdots \end{bmatrix}}_{2N\times 6}"
                 r"\begin{bmatrix} a\\b\\t_x\\c\\d\\t_y \end{bmatrix} = "
                 r"\begin{bmatrix} u_1 \\ v_1 \\ \vdots \end{bmatrix}"
                 r"\quad\Rightarrow\quad \mathbf{p}^* = \arg\min_{\mathbf{p}} \|J\mathbf{p} - \mathbf{b}\|^2 = "
                 r"(J^TJ)^{-1}J^T\mathbf{b}")
        st.caption("Translation, similarity (a, b form) and affine are linear in their parameters → one "
                   "`np.linalg.lstsq` call. Rigid motion needs the SVD-based Procrustes / Umeyama solution "
                   "(to keep R a rotation); a homography uses the DLT from the previous lessons.")
        st.dataframe({"model": [NAMES[k] for k in KEYS], "DOF": [tf.CLASS_BY_KEY[k]["dof"] for k in KEYS],
                      "min. matches": [tf.FITTERS[k][1] for k in KEYS],
                      "OpenCV": ["-", "estimateAffinePartial2D ¹", "estimateAffinePartial2D",
                                 "estimateAffine2D", "findHomography"]},
                     hide_index=True, width="stretch")
        st.caption("¹ OpenCV's 'partial affine' is a similarity; it has no pure-rigid 2D estimator, so the scale is "
                   "simply left free. All of them accept `method=cv2.RANSAC`.")
with right:
    with st.container(border=True):
        st.markdown(":material/casino: **RANSAC - RANdom SAmple Consensus**")
        st.markdown("1. pick a **minimal** random sample (e.g. 4 matches for H)\n"
                    "2. fit the model to just those\n"
                    "3. count matches with residual < threshold (the *consensus*)\n"
                    "4. repeat N times, keep the model with the biggest consensus\n"
                    "5. re-fit with least squares on its inliers")
        st.latex(r"N = \frac{\log(1-p)}{\log\!\left(1-(1-e)^s\right)}")
        rows = {"outlier ratio e": [], "s = 2 (similarity)": [], "s = 3 (affine)": [], "s = 4 (homography)": []}
        for ee in [0.1, 0.3, 0.5, 0.7]:
            rows["outlier ratio e"].append(f"{ee:.0%}")
            for s_, kname in [(2, "s = 2 (similarity)"), (3, "s = 3 (affine)"), (4, "s = 4 (homography)")]:
                rows[kname].append(tf.ransac_iterations(ee, s_, 0.99))
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption("Iterations needed for p = 99 % confidence of drawing at least one all-inlier sample. "
                   "Small minimal samples are why we like simple models - and why 8-DOF models get expensive "
                   "when outliers are common.")

ui.callout("practice",
           "This exact pipeline - detect keypoints (ORB/SIFT), match descriptors, **RANSAC + homography** - is how "
           "panoramas are stitched, how a known planar marker or a box face is located for grasping, and how a "
           "simulated camera view can be registered to a real one when transferring a vision policy from "
           "simulation to the real robot.")

ui.whys([
    ('Why minimise the *squared* error?',
     'Squares make the problem a linear system with a closed-form solution, and they are the right choice when the noise is Gaussian (least squares = maximum likelihood). The downside: one large error squared dominates everything - hence RANSAC or robust losses when outliers exist.'),
    ('Why does RANSAC use the *smallest* possible sample?',
     'Every sample must be outlier-free to give a good model. The chance that s random matches are all inliers is (1 − e)ˢ, which drops quickly as s grows - so small samples need far fewer tries.'),
    ('Why refit on all inliers at the end of RANSAC?',
     'The winning model came from only a minimal sample, so its noise was not averaged out. Refitting with least squares on all its inliers uses every good match and gives a more accurate answer.'),
    ('Why not always fit the most flexible model (a homography)?',
     "A more flexible model always fits the matches at least as well, but it needs more matches, more RANSAC iterations, and it can 'explain' noise as fake perspective. Use the simplest family that the physics allows - e.g. a similarity for a camera looking straight down at a table."),
])

st.divider()
ui.mcq("es1", "Why does plain least squares fail when 20 % of the matches are wrong?",
       ["It needs more iterations", "Squared errors let a few large residuals dominate the solution",
        "It can only fit translations", "It ignores the noise"], 1,
       "Least squares minimises the SUM of SQUARED residuals: one outlier 3 units away costs as much as 900 "
       "inliers 0.1 away, so the fit bends towards it. RANSAC ignores points outside the threshold.")

ui.lesson_nav(__file__)
