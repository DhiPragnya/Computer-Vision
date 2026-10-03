"""Copy this file to start a new lesson page (it is NOT listed in the sidebar).

1. copy it to  topics/tNN_<topic>/pNN_<lesson>.py
2. register it in topics/registry.py (title + Material icon)
3. keep the rhythm: header -> controls | figure | live math -> step-by-step derivation -> intuition/pitfall/
   takeaway -> why-questions -> quick check -> nav

Three building blocks keep every lesson smooth:
* live.slider(...)      drop-in for st.slider that updates the page WHILE it is dragged (not on release)
* @live.section         wraps one interactive block, so a slider only reruns its own block, not the page
* anim.step_player(...) a derivation as numbered steps with smooth morphs between them (no video: real figures)
* while a slider is held, players skip their in-between frames (they come on release); a CACHED builder
  must therefore be decorated with @live.full_quality() so a coarse copy never lands in the cache
"""
import numpy as np
import streamlit as st

from core import anim, live, plotting, shapes, theme, ui
from core import transforms as tf

ui.lesson_header(__file__, "One or two sentences: what will the student be able to <b>see</b> after this page?")


@live.section
def explore_section():
    c_ctrl, c_fig, c_math = st.columns([0.8, 1.6, 1.05], gap="large")
    with c_ctrl:
        theta = live.slider("θ (degrees)", -180, 180, 30, 1, key="tpl_theta")   # keys must be unique per page
    H = tf.rotation(theta)
    with c_fig:
        fig = plotting.transform_figure(shapes.house(), lambda t: tf.rotation(t * theta), extent=4)
        st.plotly_chart(fig, key="tpl_fig")
    with c_math:
        ui.show_matrix("H", H, 3, ui.column_highlight())


@st.cache_resource(show_spinner=False)
@live.full_quality()          # cached: never store a coarse mid-drag version
def derivation_figure():
    """Each Step is the picture AT that step; the player morphs smoothly from one step to the next."""
    house = shapes.house()
    base = plotting.base_figure(extent=4, height=440)
    for tr in plotting.shape_traces(house, color=theme.ORIGINAL, name="original", dash="dash", fill_alpha=0.0):
        base.add_trace(tr)

    def picture(H):
        return plotting.shape_traces(house, H, name="transformed")

    S, R = tf.scaling(1.5, 1.5), tf.rotation(40)
    steps = [
        anim.Step("Start with the original points", "p", "Nothing has happened yet.", picture(np.eye(3))),
        anim.Step("Scale first", f"S p  ({anim.hl('×1.5')})", "Every point moves away from the origin.",
                  picture(S)),
        anim.Step("Then rotate", f"R S p  ({anim.hl('40°')})", "The scaled shape turns about the origin.",
                  picture(R @ S),
                  between=lambda t: picture(tf.rotation(40 * t) @ S)),   # travel along an arc, not a straight line
    ]
    return anim.step_player(base, steps)


explore_section()

st.subheader("Step by step", divider="gray")
ui.anim_intro("Step by step: scale, then rotate")
st.plotly_chart(derivation_figure(), key="tpl_steps")

ui.callout("intuition", "The idea in plain words.")
ui.callout("pitfall", "The mistake students usually make.")
ui.callout("takeaway", "The one sentence to remember.")

ui.whys([("Why does the order matter?", "Because matrix products do not commute: R S ≠ S R in general.")])
ui.mcq("tpl_q1", "A question?", ["option A", "option B"], 0, "Why A is right.")
ui.lesson_nav(__file__)
