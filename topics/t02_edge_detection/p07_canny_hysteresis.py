import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core import edge_ui as eui
from core import edges as E
from core import anim, live, theme, ui

ui.lesson_header(
    __file__,
    "After NMS we have thin candidate edges with different strengths. Some are real contours, some are noise or "
    "texture. Canny uses <b>two</b> thresholds: strong pixels are trusted at once, weak pixels are kept only if "
    "they are <b>connected</b> to a strong one. The high threshold buys reliability, the low one continuity.",
)

@st.cache_resource(show_spinner=False, max_entries=24)
def canny_cached(img: np.ndarray, sg: float, lo: float, hi: float) -> dict:
    R = E.canny(img, sg, lo, hi)
    R["steps"] = E.hysteresis_steps(R["classes"], max_steps=600)
    return R


@st.cache_resource(show_spinner=False)
@live.full_quality()          # cached: never store a coarse mid-drag version
def flood_figure():
    """Hysteresis on a tiny grid: paint flows from the strong pixels through weak ones, one ring per iteration."""
    H_, W_ = 13, 26
    M = np.zeros((H_, W_))
    # a contour that starts strong and fades out
    path = [(6, 1), (6, 2), (5, 3), (5, 4), (4, 5), (4, 6), (4, 7), (5, 8), (5, 9), (6, 10), (6, 11), (7, 12),
            (7, 13), (7, 14), (6, 15), (6, 16), (5, 17), (5, 18), (6, 19), (7, 20)]
    for i, (r, c) in enumerate(path):
        M[r, c] = 0.95 - 0.7 * i / (len(path) - 1)
    # a second, fainter branch hanging off the contour
    for i, (r, c) in enumerate([(7, 10), (8, 10), (9, 11), (10, 12)]):
        M[r, c] = 0.45 - 0.05 * i
    # isolated specks: weak ones (should vanish) and one strong one (survives - why T_high matters)
    for r, c in [(1, 22), (2, 23), (10, 3), (11, 18), (2, 12)]:
        M[r, c] = 0.4
    M[11, 24] = 0.8
    tl, th = 0.3, 0.6
    cls = E.double_threshold(M, tl, th)
    steps = E.hysteresis_steps(cls)
    weak = cls == E.WEAK
    rr, cc = np.nonzero(weak)

    def state(accepted_mask, discarded):
        acc = np.where(accepted_mask[rr, cc], 17.0, 0.0)
        dis = np.where((~steps[-1][rr, cc]) & discarded, 17.0, 0.0)
        return [go.Scatter(x=cc, y=rr, mode="markers", marker=dict(size=acc, symbol="square", color=theme.E1,
                                                                   line=dict(color="white", width=1)), hoverinfo="skip"),
                go.Scatter(x=cc, y=rr, mode="markers", marker=dict(size=dis, symbol="x", color=theme.OUTLIER,
                                                                   line=dict(width=0)), hoverinfo="skip")]

    H = anim.hl
    out = [anim.Step("Candidates after NMS and the double threshold",
                     f"white = strong (≥ T_high = {th}),  amber = weak ({tl} … {th}),  dark = discarded",
                     "Only the strong pixels are edges for sure. Every weak pixel is still undecided.",
                     state(np.zeros_like(weak), False))]
    for k in range(1, len(steps)):
        new = int(((steps[k] & ~steps[k - 1]) & weak).sum())
        out.append(anim.Step(f"Iteration {k}", f"E⁽{k}⁾ = E⁽{k - 1}⁾ ∪ {{weak pixels touching it}}   ·   "
                             f"{H(f'+{new}')} pixel(s)",
                             "The paint spreads one neighbour further along every weak chain that touches the edge.",
                             state(steps[k], False)))
    out.append(anim.Step("Converged: discard the rest", f"E⁽{len(steps)}⁾ = E⁽{len(steps) - 1}⁾  →  red × = never reached",
                         "The faint tail and the side branch were rescued; isolated weak specks are dropped. The strong "
                         "speck bottom-right survives - that is why T_high must not be too low.",
                         state(steps[-1], True)))
    base = go.Figure(go.Heatmap(z=cls, colorscale=[[0, "#1a1a19"], [0.5, "#6e5214"], [1.0, "#ffffff"]], zmin=0,
                                zmax=2, showscale=False, xgap=2, ygap=2, hoverinfo="skip"))
    base.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1, visible=False, constrain="domain")
    base.update_xaxes(visible=False, constrain="domain")
    base.update_layout(height=470, showlegend=False, margin=dict(l=0, r=0))
    return anim.step_player(base, out, n_sub=8, hold=14, wrap_at=95)


@live.section
def threshold_section():
    # ------------------------------------------------------------------ double threshold
    st.subheader("Step 4 · double thresholding: strong, weak, or nothing", divider="gray")
    st.latex(r"C(x, y) = \begin{cases} \text{strong}, & M_N(x,y) \ge T_{\text{high}} \\ "
             r"\text{weak}, & T_{\text{low}} \le M_N(x,y) < T_{\text{high}} \\ \text{non-edge}, & M_N(x,y) < T_{\text{low}} "
             r"\end{cases}")
    c_ctrl, c_main = st.columns([0.8, 3], gap="large")
    with c_ctrl:
        img, clean, _ = eui.pick_image("hy", "Cameraman (photo)", noise_default=0.02)
        sg = live.slider("σ", 0.5, 4.0, 1.4, 0.1, key="hy_s")
        lo, hi = live.slider("T_low and T_high (fraction of the strongest NMS response)", 0.0, 1.0, (0.08, 0.22), 0.01,
                             key="hy_t")
    R = canny_cached(img, sg, lo, hi)
    Mn, cls, scale = R["Mn"], R["classes"], R["scale"]
    with c_main:
        a, b = st.columns([1.15, 1], gap="medium")
        with a:
            st.image(ui.datauri(E.classes_rgb(cls)), caption="white = strong · amber = weak · dark = non-edge",
                     width="stretch")
        with b:
            vals = Mn[Mn > 0] / scale
            fig = go.Figure(go.Histogram(x=vals, nbinsx=80, marker_color=theme.ORIGINAL, name="NMS ridge pixels",
                                         hovertemplate="%{x:.2f}: %{y} px<extra></extra>"))
            fig.add_vrect(x0=lo, x1=hi, fillcolor=theme.rgba(theme.HIGHLIGHT, 0.2), line_width=0,
                          annotation_text="weak", annotation_position="top")
            fig.add_vrect(x0=hi, x1=1, fillcolor=theme.rgba(theme.TRANSFORMED, 0.12), line_width=0,
                          annotation_text="strong", annotation_position="top")
            fig.update_layout(height=300, margin=dict(l=8, r=8, t=24, b=8), showlegend=False,
                              xaxis_title="M_N / max", yaxis_title="pixels (log)", yaxis_type="log")
            st.plotly_chart(fig, key="hy_hist")
            st.markdown(f"**{(cls == E.STRONG).sum():,}** strong · **{(cls == E.WEAK).sum():,}** weak · "
                        f"**{(R['edges'] & (cls == E.WEAK)).sum():,}** weak pixels kept by hysteresis")
        st.caption("Most ridge pixels are faint (left of the histogram): noise, texture and gentle shading. "
                   "A few are strong. The weak band in between is where the hard decisions are.")

    # ------------------------------------------------------------------ hysteresis growth
    st.subheader("Step 5 · hysteresis: growing out from the strong pixels", divider="gray")
    ui.anim_intro("Step by step: hysteresis on a tiny grid",
                  "Each square is one pixel. ▶ Play and watch the 'paint' spread from the strong (white) pixels through "
                  "the weak (amber) ones - one neighbour per iteration.")
    f1, f2 = st.columns([1.6, 1], gap="large")
    with f1:
        st.plotly_chart(flood_figure(), key="hy_flood")
    with f2:
        st.latex(r"E^{(0)} = S")
        st.latex(r"E^{(k+1)} = E^{(k)} \cup \{\, p \in W : N_8(p) \cap E^{(k)} \neq \emptyset \,\}")
        ui.callout("intuition",
                   "Imagine pouring paint on every strong pixel. The paint can only flow through weak pixels, one "
                   "neighbour per step. Weak pixels that touch a painted contour get painted; weak specks floating on "
                   "their own are never reached and are discarded.")

    st.markdown("**The same on your image** - drag the iteration slider:")
    steps = R["steps"]
    c_ctrl, c_img, c_plot = st.columns([0.8, 1.5, 1.2], gap="large")
    with c_ctrl:
        if len(steps) > 1:
            k = live.slider("iteration k", 0, len(steps) - 1, min(3, len(steps) - 1), key="hy_k")
            st.caption(f"Converges after {len(steps) - 1} iterations: E(k+1) = E(k).")
        else:
            k = 0
            st.caption("Nothing to grow: with these thresholds no weak pixel touches a strong one.")
    Ek = steps[k]
    vis = np.zeros(cls.shape + (3,), np.uint8)
    vis[...] = (26, 26, 25)
    weak = cls == E.WEAK
    vis[weak] = (110, 82, 20)                                  # weak, not (yet) accepted - dim amber
    vis[Ek & weak] = (235, 104, 52)                            # weak, accepted so far - orange
    vis[cls == E.STRONG] = (255, 255, 255)
    if k == len(steps) - 1:
        vis[weak & ~Ek] = (227, 73, 72)                        # discarded at the end - red
    with c_img:
        st.image(ui.datauri(vis), caption="white = strong seeds · orange = weak pixels accepted so far · dim amber = "
                 "weak, still waiting" + (" · red = weak, never reached → discarded" if k == len(steps) - 1 else ""),
                 width="stretch")
    with c_plot:
        counts = [int((s_ & weak).sum()) for s_ in steps]
        fig = eui.signal_figure([dict(x=list(range(len(steps))), y=counts, name="weak pixels accepted",
                                      color=theme.E1, mode="lines+markers", size=5)],
                                height=300, x_title="iteration k", y_title="accepted weak pixels", vlines=[(k, "k")])
        st.plotly_chart(fig, key="hy_grow")

    # ------------------------------------------------------------------ why two thresholds
    st.subheader("Why two thresholds beat one", divider="gray")
    st.markdown("A real boundary often fades along its length: M₁ > M₂ > M₃ > M₄. Noise produces isolated responses "
                "of similar strength to the faint end of the boundary:")
    H_, W_ = 90, 180
    canvas = np.zeros((H_, W_))
    tt = np.linspace(0, 1, 400)
    cxs = (15 + 150 * tt).astype(int)
    cys = (45 + 28 * np.sin(tt * 2 * np.pi * 0.9)).astype(int)
    canvas[cys, cxs] = 1.0 - 0.78 * tt                         # strength fades from 1.0 to 0.22 along the contour
    rng = np.random.default_rng(11)
    for _ in range(70):
        canvas[rng.integers(0, H_), rng.integers(0, W_)] = rng.uniform(0.15, 0.5)
    c_ctrl, c1, c2, c3 = st.columns([0.7, 1, 1, 1], gap="medium")
    with c_ctrl:
        tl, th_ = live.slider("T_low, T_high", 0.0, 1.0, (0.18, 0.6), 0.01, key="hy_demo")
    single_hi = canvas >= th_
    single_lo = canvas >= tl
    hyst = E.hysteresis(E.double_threshold(canvas, tl, th_))
    show = lambda m: E.enlarge(E.edges_rgb(m), 2)
    c1.image(ui.datauri(show(single_hi)), caption=f"single threshold {th_:.2f}: reliable, but the faint part of the "
             "contour is LOST", width="stretch")
    c2.image(ui.datauri(show(single_lo)), caption=f"single threshold {tl:.2f}: continuous but NOISE gets in",
             width="stretch")
    c3.image(ui.datauri(show(hyst)), caption="hysteresis: whole contour, no noise", width="stretch")
    st.markdown("On the real image above:")
    eui.show_row([(E.edges_rgb(Mn >= hi * scale), "only T_high"),
                  (E.edges_rgb(Mn >= max(lo, 1e-9) * scale), "only T_low"),
                  (E.edges_rgb(R["edges"]), "hysteresis (T_low, T_high)")])
    ui.callout("takeaway", "**High threshold → reliability. Low threshold → continuity.** A strong response is an "
                           "anchor; weaker connected responses are allowed to continue the contour.")


threshold_section()

# ------------------------------------------------------------------ graph view
st.subheader("Hysteresis as a graph search", divider="gray")
g1, g2 = st.columns([1.2, 1], gap="large")
with g1:
    st.latex(r"S = \{p : M_N(p) \ge T_{\text{high}}\}")
    st.latex(r"W = \{p : T_{\text{low}} \le M_N(p) < T_{\text{high}}\}")
    st.latex(r"E_{\text{final}} = S \,\cup\, \{\, p \in W : \exists \text{ path } p \to S "
             r"\text{ inside } S \cup W \,\}")
    st.markdown("Every candidate pixel is a **node**; neighbouring candidates (8-connectivity) share an **edge** of the "
                "graph. Final edges = all nodes in a connected component that contains at least one strong pixel.")
    st.code("stack = list(strong_pixels)            # start from every strong pixel\n"
            "accepted = set(stack)\n"
            "while stack:\n"
            "    p = stack.pop()\n"
            "    for q in neighbours8(p):\n"
            "        if q in weak and q not in accepted:\n"
            "            accepted.add(q)\n"
            "            stack.append(q)            # keep growing from the new pixel", language="python")
with g2:
    N8 = np.ones((3, 3))
    N8[1, 1] = 2
    fig = go.Figure(go.Heatmap(z=N8, colorscale=[[0, theme.rgba(theme.E1, 0.6)], [1, theme.HIGHLIGHT]], showscale=False,
                               xgap=5, ygap=5, text=[["↖", "↑", "↗"], ["←", "p", "→"], ["↙", "↓", "↘"]],
                               texttemplate="%{text}", textfont=dict(size=22), hoverinfo="skip"))
    fig.update_yaxes(autorange="reversed", visible=False, scaleanchor="x")
    fig.update_xaxes(visible=False)
    fig.update_layout(height=260, margin=dict(l=40, r=40, t=30, b=0),
                      title=dict(text="N₈(p): the 8 neighbours", x=0.5, xanchor="center", font=dict(size=13)))
    st.plotly_chart(fig, key="hy_n8")

st.markdown("| | Non-maximum suppression | Hysteresis |\n|---|---|---|\n"
            "| Input | gradient magnitude | thresholded candidates |\n| Main purpose | thin edges | connect valid edges |\n"
            "| Uses gradient direction? | yes | no |\n| Uses connectivity? | local directional comparison | yes |\n"
            "| Main question | 'Is this pixel a local maximum?' | 'Is this weak pixel connected to a strong edge?' |\n"
            "| Result | thin candidate edges | final connected edge map |\n"
            "| Solves | spatial **localisation** | edge **connectivity** |")

ui.whys([
    ("Why is it called 'hysteresis'?",
     "In physics hysteresis means the output depends on history - like a thermostat that switches the heater on "
     "at 19 °C but off only at 21 °C. Here: you need a strong value (T_high) to *start* an edge, but once you are "
     "on one you can follow it down to the lower value (T_low) before it ends."),
    ("Why 8-connectivity and not 4?",
     "With 4-connectivity (up, down, left, right) a one-pixel-thin diagonal edge is a chain of pixels that touch "
     "only at their corners - it would fall apart into single pixels. NMS produces exactly such thin diagonal "
     "lines, so 8-connectivity is needed to follow them."),
    ("How should I choose the two thresholds?",
     "Canny suggested T_high ≈ 2-3 × T_low. In practice: pick T_high so that the strong pixels are clearly real "
     "edges (e.g. the top 10-20 % of ridge pixels), then lower T_low until contours close without noise creeping "
     "in. A popular automatic rule uses the image's median intensity: T_low ≈ 0.67·median, T_high ≈ 1.33·median."),
    ("Can hysteresis bring back pixels that NMS removed?",
     "No. It only decides about *weak candidates* that survived NMS. Suppressed pixels are zero and below any "
     "threshold, so the result stays thin."),
    ("Why do isolated noise specks disappear?",
     "A speck stronger than T_low but weaker than T_high is weak; with no path of candidate pixels leading to a "
     "strong pixel, the flood never reaches it. A speck stronger than T_high, however, survives - which is why "
     "T_high must not be too low."),
    ("Is the order of visiting pixels important?",
     "No. Whatever order the stack or queue uses, the final set is the same: all weak pixels in components that "
     "contain a strong pixel. Only the speed changes."),
])

st.divider()
ui.mcq("hy1", "A weak pixel is kept by hysteresis if …",
       ["its magnitude is above T_high", "it is connected through candidate pixels to a strong pixel",
        "its gradient direction matches its neighbours", "it is a local maximum"], 1,
       "Weak + connection to strong ⇒ valid edge; weak + no connection ⇒ discard.")
ui.mcq("hy2", "Raising T_low (keeping T_high fixed) mostly …",
       ["adds more noise", "breaks faint contours into pieces", "makes edges thicker", "changes nothing"], 1,
       "Fewer weak pixels qualify, so fading parts of contours can no longer be bridged.")

ui.lesson_nav(__file__)
