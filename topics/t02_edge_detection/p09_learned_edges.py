import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core import edge_ui as eui
from core import edges as E
from core import anim, live, theme, ui

ui.lesson_header(
    __file__,
    "Sobel, LoG and Canny are <b>hand-designed</b>: fixed filters, hand-tuned thresholds, purely local decisions. "
    "Natural images break those assumptions - texture, lighting and weak object boundaries. The modern answer is "
    "to stop <i>designing the filter</i> and start <b>learning the representation</b>, at many scales and with "
    "global context. Everything below is an interactive <i>imitation</i> of the ideas - no trained network is run.",
)


@st.cache_data(show_spinner=False)
def scenes():
    train = E.texture_scene(256, return_gt=True)
    test = E.texture_scene(256, cx=0.4, cy=0.45, r=0.25, period=2.5, wobble=13, light_dir=-1, return_gt=True)
    return train, test


(T, GT), (T2, GT2) = scenes()


def best_f1(F, theta, gt):
    Mn = E.nms_quantized(F, theta)
    return max(E.match_f1(Mn >= q * Mn.max(), gt, 2)[2] for q in np.arange(0.05, 0.8, 0.05)) if Mn.max() > 0 else 0.0


# ------------------------------------------------------------------ where classical fails
@st.cache_data(show_spinner=False)
def canny_curve():
    ss = np.array([0.5, 1, 1.5, 2, 3, 4, 6, 8, 10, 12])
    return ss, np.array([E.match_f1(E.canny(T, s, 0.1, 0.3)["edges"], GT, 2)[2] for s in ss])


@st.cache_resource(show_spinner=False)
def photo_edges():
    return [(E.edges_rgb(E.canny(E.photo("bricks.png"), 1.4, 0.1, 0.25)["edges"]), "bricks: mortar AND surface grain"),
            (E.edges_rgb(E.canny(E.photo("cat.png"), 1.4, 0.1, 0.25)["edges"]), "cat: fur texture swamps the outline"),
            (E.edges_rgb(E.canny(E.photo("astronaut.png"), 1.4, 0.1, 0.25)["edges"]),
             "astronaut: fabric folds, flag stripes, text")]


st.subheader("Where classical detectors struggle", divider="gray")
@live.section
def classical_section():
    c_ctrl, c_main = st.columns([0.8, 3], gap="large")
    with c_ctrl:
        sg = live.slider("Canny σ", 0.5, 12.0, 1.5, 0.5, key="le_s")
        lo, hi = live.slider("T_low, T_high", 0.0, 1.0, (0.1, 0.3), 0.01, key="le_t")

    Rc = E.canny(T, sg, lo, hi)
    f_now = E.match_f1(Rc["edges"], GT, 2)[2]
    ss, fs = canny_curve()
    with c_main:
        a, b, c = st.columns([1, 1, 1.3])
        a.image(ui.datauri(np.clip(T, 0, 1)), caption="a disc on a striped, unevenly lit background", width="stretch")
        b.image(ui.datauri(E.overlay(T, Rc["edges"])), caption=f"Canny edges · F1 vs the true disc outline = {f_now:.2f}",
                width="stretch")
        with c:
            fig = eui.signal_figure([dict(x=ss, y=fs, name="F1 of Canny vs σ", color=theme.E1, mode="lines+markers")],
                                    height=300, x_title="σ", y_title="F1 (1 = perfect)", vlines=[(sg, "yours")])
            fig.update_yaxes(range=[0, 1])
            st.plotly_chart(fig, key="le_curve")


classical_section()

st.markdown("Small σ: every stripe is an edge. Large σ: the stripes fade but the disc's outline drifts and rounds "
            "off. **No setting of σ and thresholds separates 'texture' from 'object boundary'** - because a 3×3 or "
            "even 30×30 intensity derivative cannot know what a stripe pattern *is*.")
st.markdown("Limitations of classical edge detection: **(1)** fixed filters are not adaptive, **(2)** thresholds need "
            "manual tuning, **(3)** texture generates unwanted edges, **(4)** illumination changes produce false edges, "
            "**(5)** weak object boundaries are missed, **(6)** an *object boundary* is not a purely local property, "
            "**(7)** natural images contain edges at many scales.")
eui.show_row(photo_edges())

# ------------------------------------------------------------------ timeline
st.subheader("How the field evolved", divider="gray")
EV = [(1963, "Roberts cross", "2×2 diagonal differences", "classical"),
      (1968, "Sobel", "derivative × (1, 2, 1) smoothing", "classical"),
      (1970, "Prewitt", "derivative × (1, 1, 1) smoothing", "classical"),
      (1980, "Marr–Hildreth LoG / DoG", "second derivative of a Gaussian, zero crossings", "classical"),
      (1986, "Canny", "optimal criteria: smoothing + gradient + NMS + hysteresis", "classical"),
      (2013, "Structured Edges", "random forests predict whole edge patches", "learned (hand-crafted features)"),
      (2015, "HED", "fully convolutional, multi-scale side outputs, deep supervision", "deep"),
      (2017, "RCF", "richer features: every conv layer contributes", "deep"),
      (2019, "BDCN", "bi-directional cascade, scale-specific supervision", "deep"),
      (2020, "DexiNed", "dense extreme inception, thin edges, multiple outputs", "deep"),
      (2022, "Transformer-based (e.g. EDTER)", "attention: global context for local boundaries", "attention / transformer")]
colours = {"classical": theme.ORIGINAL, "learned (hand-crafted features)": theme.E2, "deep": theme.TRANSFORMED,
           "attention / transformer": theme.MAGENTA}
fig = go.Figure()
for i, (yr, nm, desc, kind) in enumerate(EV):
    y = 1 if i % 2 == 0 else -1
    fig.add_trace(go.Scatter(x=[yr, yr], y=[0, y * 0.8], mode="lines", line=dict(color=colours[kind], width=1.5),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[yr], y=[y * 0.8], mode="markers+text", text=[f"<b>{nm}</b><br>{yr}"],
                             textposition="top center" if y > 0 else "bottom center",
                             marker=dict(size=11, color=colours[kind]), name=kind, legendgroup=kind,
                             showlegend=kind not in [e[3] for e in EV[:i]],
                             hovertemplate=f"<b>{nm}</b> ({yr})<br>{desc}<extra></extra>"))
fig.add_hline(y=0, line=dict(color=theme.AXIS, width=2))
fig.update_yaxes(visible=False, range=[-2.1, 2.1])
fig.update_xaxes(range=[1958, 2026], title="year")
fig.update_layout(height=380, margin=dict(l=8, r=8, t=10, b=8), legend=dict(orientation="h", y=1.08, x=0))
st.plotly_chart(fig, key="le_time")
st.caption("Hover a point for one line on what changed. Classical → hand-crafted learning → deep CNNs → attention.")

# ------------------------------------------------------------------ what a learned layer computes
@st.cache_resource(show_spinner=False)
def feature_layers_T():
    return E.feature_layers(T)


@st.cache_resource(show_spinner=False)
def boundary_f1():
    Fl_ = feature_layers_T()
    return best_f1(Fl_["boundary"], Fl_["theta"], GT)


st.subheader("What a network layer computes - built by hand", divider="gray")
st.markdown("A CNN layer is **filter → non-linearity → pooling**. Stack that a few times and the network measures "
            "not just 'brightness changes' but 'which **pattern** is here'. Boundaries then show up where the "
            "*pattern* changes. Here is a 3-step version with filters chosen by hand:")
Fl = feature_layers_T()
eui.show_row([(E.signed_rgb(Fl["responses"][0]), "1 · filter: oriented derivative (0°)"),
              (E.signed_rgb(Fl["responses"][2]), "1 · filter: oriented derivative (90°)"),
              (E.to_u8(Fl["energies"][0]), "2+3 · square & pool: 'how much 0° texture is here'"),
              (E.to_u8(Fl["boundary"], 0, np.percentile(Fl["boundary"], 99.5)), "boundary = change of the PATTERN")])
f_feat = boundary_f1()
k1, k2, k3 = st.columns(3)
fs_best = canny_curve()[1].max()
k1.metric("best Canny F1 (any σ)", f"{fs_best:.2f}", border=True)
k2.metric("feature-boundary F1", f"{f_feat:.2f}", delta=f"+{f_feat - fs_best:.2f}", border=True)
k3.metric("filters learned from data", "0 (hand-made)", border=True,
          help="A real network learns thousands of filters from labelled images.")
ui.callout("intuition",
           "The stripes are a big *intensity* change but no *pattern* change - inside the background it is stripes "
           "everywhere. The disc's outline is where 'stripes' turns into 'smooth'. Measuring the pattern first and "
           "differentiating that is the step from hand-designed edge detection to learned representations.")

# ------------------------------------------------------------------ receptive field
def receptive_fields(L, pool_every):
    rf, jump, rfs, jumps = 1, 1, [], []
    for layer in range(1, L + 1):
        rf += 2 * jump
        if pool_every != "never" and layer % int(pool_every) == 0:
            rf += jump
            jump *= 2
        rfs.append(rf)
        jumps.append(jump)
    return rfs, jumps


def rf_steps(L, pool_every, c0=(128, 100)):
    """The square an output pixel 'sees', growing layer by layer."""
    rfs, jumps = receptive_fields(L, pool_every)
    show = list(range(min(L, 10)))

    def square(r):
        h = r / 2
        x0, x1, y0, y1 = c0[0] - h, c0[0] + h, c0[1] - h, c0[1] + h
        return go.Scatter(x=[x0, x1, x1, x0, x0], y=[y0, y0, y1, y1, y0], mode="lines", fill="toself",
                          fillcolor=theme.rgba(theme.E1, 0.18), line=dict(color=theme.E1, width=3), hoverinfo="skip")

    H = anim.hl
    steps = [anim.Step("One output pixel", "receptive field = 1 × 1", "Before any layer the pixel only knows itself.",
                       [square(1)])]
    prev_jump = 1
    for i in show:
        layer = i + 1
        pooled = jumps[i] != prev_jump
        grow = 2 * prev_jump + (prev_jump if pooled else 0)
        note = (f"A 3×3 layer adds one ring of neighbours on each side: +2 × stride ({prev_jump}) = +{2 * prev_jump} px."
                + (f" Then 2× pooling: +{prev_jump} px and the stride doubles to {jumps[i]}." if pooled else ""))
        steps.append(anim.Step(f"After layer {layer}", f"receptive field = {H(f'{rfs[i]} × {rfs[i]}')} px   (+{grow})",
                               note, [square(rfs[i])]))
        prev_jump = jumps[i]
    return steps


@st.cache_resource(show_spinner=False)
def texture_heatmap_base(height=560):
    base = go.Figure(go.Heatmap(z=np.round(T * 255).astype(np.uint8), zmin=0, zmax=255, colorscale=eui.GREYS,
                                showscale=False, hoverinfo="skip"))
    base.add_trace(go.Scatter(x=[128], y=[100], mode="markers", marker=dict(size=9, color=theme.HIGHLIGHT,
                                                                           symbol="square"), hoverinfo="skip"))
    base.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1, visible=False, constrain="domain")
    base.update_xaxes(visible=False, constrain="domain")
    base.update_layout(height=height, margin=dict(l=0, r=0), showlegend=False)
    return base


@live.section
def receptive_section():
    st.subheader("Why depth gives context: the receptive field", divider="gray")
    c_ctrl, c_img, c_plot = st.columns([0.8, 1.1, 1.4], gap="large")
    with c_ctrl:
        L = live.slider("number of 3×3 conv layers", 1, 16, 6, key="le_L")
        pool_every = live.select_slider("2× down-sampling (pooling) every … layers", ["never", 4, 3, 2], value=2,
                                        key="le_pool")
    rfs, _ = receptive_fields(L, pool_every)
    rf = rfs[-1]
    with c_img:
        vis = E.overlay(T, np.zeros_like(T, bool), dim=1.0)
        c0 = (128, 100)
        half = min(rf // 2, 127)
        y0, y1 = max(0, c0[1] - half), min(255, c0[1] + half)
        x0, x1 = max(0, c0[0] - half), min(255, c0[0] + half)
        vis[y0:y1 + 1, [x0, x1]] = (235, 104, 52)
        vis[[y0, y1], x0:x1 + 1] = (235, 104, 52)
        vis[c0[1] - 2:c0[1] + 3, c0[0] - 2:c0[0] + 3] = (237, 161, 0)
        st.image(ui.datauri(vis), caption=f"what one output pixel can 'see' after {L} layers: {rf}×{rf} px",
                 width="stretch")
    with c_plot:
        fig = eui.signal_figure([dict(x=list(range(1, L + 1)), y=rfs, name="receptive field (px)",
                                      color=theme.TRANSFORMED, mode="lines+markers")], height=300, x_title="layer",
                                y_title="receptive field (px)")
        st.plotly_chart(fig, key="le_rf")
        st.caption("Each 3×3 layer adds 2 pixels (× the current stride); pooling doubles the stride. Shallow layers see "
                   "a few pixels - precise but local. Deep layers see most of the image - context, but coarse.")
    ui.anim_intro("Step by step: watch the field of view grow, layer by layer",
                  f"Uses your settings (pooling every {pool_every} layers; at most the first 10 layers are animated).")
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        st.plotly_chart(anim.step_player(texture_heatmap_base(), rf_steps(L, pool_every), n_sub=12, wrap_at=72),
                        key="le_rf_steps")
    with right:
        st.latex(r"r_{l} = r_{l-1} + 2\,j_{l-1}, \qquad j_l = \begin{cases} 2\,j_{l-1} & \text{after pooling} \\ "
                 r"j_{l-1} & \text{otherwise} \end{cases}")
        ui.callout("intuition",
                   "Without pooling the square grows by the same 2 pixels every layer - slowly. Each pooling step "
                   "doubles the stride, so every LATER layer adds twice as much. That is how deep networks reach "
                   "image-sized context with only small 3×3 filters.")


receptive_section()


# ------------------------------------------------------------------ HED-style fusion with learned weights
@st.cache_data(show_spinner=False)
def side_outputs(which: int):
    img = T if which == 0 else T2
    outs = []
    for s in (1, 2, 4, 8):
        gx, gy = E.gradients(E.smooth(img, s), "Central difference")
        m = E.magnitude(gx, gy)
        outs.append(m / m.max())
    fl = E.feature_layers(img)
    outs.append(fl["boundary"])
    return outs, fl["theta"]


NAMES = ["stage 1 (fine, σ=1)", "stage 2 (σ=2)", "stage 3 (σ=4)", "stage 4 (coarse, σ=8)", "stage 5 (pattern)"]


@st.cache_data(show_spinner=False)
def learn_weights(n_trials: int = 70, seed: int = 0):
    """'Training' = search for fusion weights that maximise F1 on the TRAINING scene."""
    S_tr, th_tr = side_outputs(0)
    rng = np.random.default_rng(seed)
    best = (-1, None)
    history = []
    for i in range(n_trials):
        w = rng.dirichlet(np.ones(5) * 0.6) if i else np.ones(5) / 5
        F = sum(wi * si for wi, si in zip(w, S_tr))
        f = best_f1(F, th_tr, GT)
        if f > best[0]:
            best = (f, w)
        history.append(best[0])
    return best[1], best[0], history


@live.section
def fusion_section():
    st.subheader("HED's idea: side outputs at every depth, fused with learned weights", divider="gray")
    st.markdown("HED attaches a small edge predictor (a *side output*) to every stage of the network and supervises "
                "all of them (**deep supervision**); a final 1×1 layer **fuses** them with learned weights. Below, the "
                "five side outputs are stand-ins: intensity edges at four scales plus the pattern boundary from above.")
    S_tr, th_tr = side_outputs(0)
    S_te, th_te = side_outputs(1)
    eui.show_row([(E.to_u8(s_, 0, np.percentile(s_, 99.5)), n_) for s_, n_ in zip(S_tr, NAMES)])
    c_ctrl, c_res = st.columns([1, 2.2], gap="large")
    with c_ctrl:
        st.markdown("**Fusion weights** (you are the fusion layer):")
        w_user = np.array([live.slider(n_, 0.0, 1.0, 0.2, 0.05, key=f"le_w{i}") for i, n_ in enumerate(NAMES)])
        learn = st.toggle("Learn the weights (train)", value=False, key="le_learn",
                          help="Search for the weights that give the best F1 on the TRAINING scene, then test on a new "
                               "scene")
    if learn:
        w_use, f_train, hist = learn_weights()
    else:
        w_use = w_user / max(w_user.sum(), 1e-9)
    fuse = lambda S: sum(wi * si for wi, si in zip(w_use, S))
    F_tr, F_te = fuse(S_tr), fuse(S_te)
    f_tr, f_te = best_f1(F_tr, th_tr, GT), best_f1(F_te, th_te, GT2)
    with c_res:
        a, b = st.columns(2)
        a.image(ui.datauri(E.to_u8(F_tr, 0, np.percentile(F_tr, 99.5) + 1e-12)),
                caption=f"fused - TRAINING scene · F1 = {f_tr:.2f}", width="stretch")
        b.image(ui.datauri(E.to_u8(F_te, 0, np.percentile(F_te, 99.5) + 1e-12)),
                caption=f"fused - NEW test scene · F1 = {f_te:.2f}", width="stretch")
        wfig = go.Figure(go.Bar(x=NAMES, y=w_use, marker_color=theme.TRANSFORMED, hovertemplate="%{y:.2f}<extra></extra>"))
        wfig.update_layout(height=200, margin=dict(l=8, r=8, t=24, b=8), title=dict(text="fusion weights in use",
                                                                                     font=dict(size=13), x=0.01))
        st.plotly_chart(wfig, key="le_wbar")
        if learn:
            st.caption("Learned by random search over weight vectors on the training scene only, then applied unchanged "
                       "to a scene it has never seen (different disc, stripe frequency and lighting). Good weights "
                       "generalise - that is the point of learning.")
    ui.callout("intuition",
               "Shallow side outputs localise precisely but fire on texture; deep ones are coarse but know 'what is "
               "there'. The fusion layer learns how much to trust each depth. RCF pushed this further by using *every* "
               "convolution layer, not just the last of each stage; BDCN supervises each scale separately; DexiNed uses "
               "dense blocks and several outputs to keep edges thin.")


fusion_section()


# ------------------------------------------------------------------ architectures
@live.section
def architecture_section():
    st.subheader("Two architecture families", divider="gray")
    arch = st.segmented_control("Show", ["HED / RCF: side outputs + fusion", "Encoder–decoder with skip connections"],
                                default="HED / RCF: side outputs + fusion", key="le_arch", required=True)
    fig = go.Figure()
    sizes = [1.0, 0.8, 0.62, 0.46, 0.32]
    for i, sz in enumerate(sizes):
        x = i * 1.6
        fig.add_shape(type="rect", x0=x, x1=x + 1.0, y0=2 - sz, y1=2 + sz,
                      fillcolor=theme.rgba(theme.TRANSFORMED, 0.15 + 0.12 * i), line=dict(color=theme.TRANSFORMED))
        fig.add_annotation(x=x + 0.5, y=2, text=f"stage {i + 1}", showarrow=False, font=dict(size=11))
        if i < 4:
            fig.add_annotation(x=x + 1.55, y=2, ax=x + 1.05, ay=2, xref="x", yref="y", axref="x", ayref="y", text="",
                               showarrow=True, arrowhead=3, arrowcolor=theme.AXIS)
        if arch.startswith("HED"):
            fig.add_shape(type="rect", x0=x + 0.2, x1=x + 0.8, y0=0.15, y1=0.65, fillcolor=theme.rgba(theme.E1, 0.25),
                          line=dict(color=theme.E1))
            fig.add_annotation(x=x + 0.5, y=0.4, text="side out", showarrow=False, font=dict(size=10))
            fig.add_annotation(x=x + 0.5, y=0.68, ax=x + 0.5, ay=2 - sz - 0.02, xref="x", yref="y", axref="x",
                               ayref="y", text="", showarrow=True, arrowhead=3, arrowcolor=theme.E1)
            fig.add_annotation(x=8.6, y=0.4, ax=x + 0.82, ay=0.4, xref="x", yref="y", axref="x", ayref="y", text="",
                               showarrow=True, arrowhead=2, arrowcolor=theme.rgba(theme.E1, 0.5))
        else:
            fig.add_shape(type="rect", x0=i * 1.6, x1=i * 1.6 + 1.0, y0=-1 - sz, y1=-1 + sz,
                          fillcolor=theme.rgba(theme.E2, 0.15 + 0.12 * (4 - i)), line=dict(color=theme.E2))
            fig.add_annotation(x=i * 1.6 + 0.5, y=-1, text=f"decode {i + 1}", showarrow=False, font=dict(size=11))
            fig.add_annotation(x=i * 1.6 + 0.5, y=-1 + sz + 0.05, ax=i * 1.6 + 0.5, ay=2 - sz - 0.05, xref="x",
                               yref="y", axref="x", ayref="y", text="", showarrow=True, arrowhead=3,
                               arrowcolor=theme.HIGHLIGHT)
    if arch.startswith("HED"):
        fig.add_shape(type="rect", x0=8.6, x1=9.8, y0=0.0, y1=0.8, fillcolor=theme.rgba(theme.E2, 0.3),
                      line=dict(color=theme.E2))
        fig.add_annotation(x=9.2, y=0.4, text="<b>fuse</b><br>(1×1 conv)", showarrow=False, font=dict(size=11))
        cap = ("Each stage sees a larger receptive field. Every side output is compared with the ground truth during "
               "training (deep supervision), and the fusion layer learns how to weight them.")
    else:
        cap = ("The encoder compresses the image into abstract features I → F₁ → … → F_L; the decoder climbs back to "
               "full resolution, and skip connections (amber) hand over the precise low-level detail: high-level "
               "context + low-level localisation.")
    fig.update_xaxes(visible=False, range=[-0.3, 10])
    fig.update_yaxes(visible=False, range=[-2.2, 3.2])
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
    st.plotly_chart(fig, key="le_arch_fig", config=dict(staticPlot=True))
    st.caption(cap)


architecture_section()


# ------------------------------------------------------------------ attention
@st.cache_resource(show_spinner=False)
def attention_features():
    Fl_ = feature_layers_T()
    return np.stack(Fl_["energies"] + [E.smooth(T, 8) / E.smooth(T, 8).max()])


@live.section
def attention_section():
    st.subheader("Attention: local appearance + global context", divider="gray")
    st.markdown("A small window on the disc's outline looks just like a stripe. Self-attention lets every pixel compare "
                "itself with *all* other pixels and group the ones that 'look alike'. A toy version: click-free, pick a "
                "query pixel and see which pixels it attends to, using the pattern features from above.")
    c_ctrl, c_a, c_b = st.columns([0.8, 1.1, 1.1], gap="large")
    with c_ctrl:
        qx = live.slider("query x", 0, 255, 140, key="le_qx")
        qy = live.slider("query y", 0, 255, 128, key="le_qy")
        win = live.slider("local window (what a small filter sees)", 8, 128, 16, 8, key="le_win")
        temp = live.slider("attention sharpness", 2.0, 40.0, 12.0, 1.0, key="le_temp")
    feats = attention_features()
    q = feats[:, qy, qx][:, None, None]
    d2 = ((feats - q) ** 2).sum(0)
    att = np.exp(-temp * d2 / (d2.std() + 1e-9))
    att = att / att.sum()
    with c_a:
        hw = win // 2
        Tp = np.pad(T, hw, mode="reflect")              # near the border: mirror, like OpenCV's filters do
        crop = Tp[qy:qy + 2 * hw, qx:qx + 2 * hw]        # always win x win, so the picture never changes size
        st.image(ui.datauri(E.enlarge(E.to_u8(crop, 0, 1), max(1, 160 // max(crop.shape)))),
                 caption=f"the {win}×{win} local view - texture or boundary?", width="stretch")
    with c_b:
        vis = E.overlay(T, np.zeros_like(T, bool), dim=0.35)
        heat = E.to_u8(att, 0, np.percentile(att, 99.5)).astype(float) / 255
        vis = (vis * (1 - heat[..., None]) + np.array([235, 104, 52]) * heat[..., None]).astype(np.uint8)
        vis[max(0, qy - 3):qy + 4, max(0, qx - 3):qx + 4] = (42, 120, 214)
        st.image(ui.datauri(vis), caption="orange = pixels the query attends to (blue square = query)", width="stretch")
    ui.callout("intuition",
               "Put the query inside the disc: attention spreads over the whole disc. Put it on the stripes: it spreads "
               "over the background, across the lighting gradient. The global grouping makes the boundary obvious even "
               "where the local window is ambiguous - 'local appearance + global context → better boundary reasoning'.")


attention_section()

# ------------------------------------------------------------------ summary tables
st.subheader("Classical vs learning-based", divider="gray")
st.markdown("| | Classical | Learning-based |\n|---|---|---|\n| Examples | Sobel, Prewitt, LoG, Canny | HED, RCF, BDCN, DexiNed |\n"
            "| Features | hand-designed | learned |\n| Thresholds | usually hand-tuned | learned / data-driven |\n"
            "| Context | mostly local | local + global |\n| Multi-scale reasoning | explicitly designed | learned from the hierarchy |\n"
            "| Training data | not required | usually required |\n| Interpretability | high | lower |\n"
            "| Computation | usually low | usually higher |\n| Adaptability | limited | high |")
st.markdown("**The whole story as four problems:**")
p1, p2, p3, p4 = st.columns(4)
for col, (t_, sol, ex) in zip([p1, p2, p3, p4], [
        ("1 · Detect intensity changes", "gradient operators", "Roberts, Prewitt, Sobel"),
        ("2 · Suppress noise", "Gaussian smoothing", "LoG, Canny"),
        ("3 · Thin and connected edges", "NMS + hysteresis", "Canny"),
        ("4 · Which changes are meaningful?", "learned representations + context", "HED, RCF, transformers")]):
    with col:
        with st.container(border=True):
            st.markdown(f"**{t_}**")
            st.markdown(f"→ {sol}")
            st.caption(ex)

ui.whys([
    ("Why can't a 3×3 filter know where an object boundary is?",
     "It sees 9 pixels. A stripe of texture and the rim of an object can produce exactly the same 9 numbers. "
     "Whether a change is a *boundary* depends on what is on either side over a much larger area - context that "
     "only large receptive fields or attention provide."),
    ("Why do deeper layers have larger receptive fields?",
     "Each 3×3 layer looks at 3×3 outputs of the layer below, each of which already summarised a region. Regions "
     "grow by 2 pixels per layer - and doubling with every down-sampling step - so after 10-15 layers one output "
     "can depend on most of the image."),
    ("Why supervise the intermediate layers (deep supervision)?",
     "It gives every stage a direct training signal, so the error does not have to travel back through the whole "
     "network (which makes early layers learn slowly), and it forces each scale to produce a meaningful edge map "
     "on its own - which the fusion layer can then combine."),
    ("Why do learned detectors still need NMS?",
     "Networks output a soft edge *probability* that is a few pixels wide (like a blurred ridge). Benchmarks and "
     "most applications want one-pixel contours, so a thinning step - usually the same NMS idea from Canny - is "
     "applied to the predicted map."),
    ("Where does the training data come from?",
     "From people. Datasets such as BSDS500 contain natural photos with object boundaries traced by several human "
     "annotators each. The network learns to predict where humans agree there is a boundary - which is why texture "
     "edges get ignored."),
    ("Will classical detectors become obsolete?",
     "No. They need no data or GPU, run in microseconds, behave predictably and are easy to explain. In controlled "
     "settings - industrial inspection, calibration patterns, document edges, robot cells - they often remain the "
     "best tool, and their ideas (smoothing, gradients, NMS) live on inside the learned methods."),
])

st.divider()
ui.mcq("le1", "What is the main conceptual shift from Canny to HED?",
       ["Using colour instead of grey", "From designing the filter to learning the representation",
        "From thin to thick edges", "Removing the need for any training data"], 1,
       "HED learns multi-scale features end-to-end from annotated images instead of using a hand-designed operator.")
ui.mcq("le2", "Shallow layers vs deep layers in an edge network: which pairing is right?",
       ["shallow = semantic context, deep = fine localisation", "shallow = fine localisation, deep = semantic context",
        "both see the same region", "deep layers have smaller receptive fields"], 1,
       "Shallow features keep precise positions; deep features see large regions and know 'what' is there - fusing "
       "them gives both.")

ui.lesson_nav(__file__)
