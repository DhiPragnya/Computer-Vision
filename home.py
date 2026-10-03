import numpy as np
import streamlit as st

from core import plotting, shapes, theme, ui
from topics.registry import TOPICS, page_path

st.markdown("<div class='kicker'>Interactive course notes</div>", unsafe_allow_html=True)
st.title("Computer Vision, visually")
st.markdown(
    "<p class='lead'>Every idea in these notes comes with a picture you can move. "
    "Drag a slider, press <b>Play</b>, watch what the matrix does - and only then read the equation. "
    "Seeing the geometry first is what makes the algebra stick.</p>",
    unsafe_allow_html=True,
)

left, right = st.columns([1.05, 1], gap="large")
with right:
    H = np.array([[1.05, 0.25, 0.4], [0.05, 0.9, 0.2], [0.12, 0.08, 1.0]])
    fig = plotting.transform_figure(shapes.house(offset=(-1, -1.2)), H, extent=3.2, height=440, n_frames=36, legend=False)
    st.plotly_chart(fig, key="hero")
    ui.colour_legend()

with left:
    st.subheader("How to learn with these notes")
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown(":material/tune: **Move**")
            st.caption("Every lesson has controls next to the figure. Change one thing at a time.")
    with c2:
        with st.container(border=True):
            st.markdown(":material/play_circle: **Watch**")
            st.caption("Press Play under a figure to morph from 'do nothing' (t = 0) to the transform (t = 1).")
    with c3:
        with st.container(border=True):
            st.markdown(":material/function: **Connect**")
            st.caption("The live matrix updates as you move. Match each number to what you see.")

    st.subheader("One colour language, everywhere")
    st.markdown(
        f"- <span style='color:{theme.ORIGINAL}'><b>Grey dashed</b></span> is the shape **before**.\n"
        f"- <span style='color:{theme.TRANSFORMED}'><b>Blue</b></span> is the shape **after**.\n"
        f"- <span style='color:{theme.E1}'><b>Orange</b></span> arrow: where **e₁ = (1, 0)** lands - it *is* the 1st matrix column.\n"
        f"- <span style='color:{theme.E2}'><b>Green</b></span> arrow: where **e₂ = (0, 1)** lands - it *is* the 2nd matrix column.",
        unsafe_allow_html=True,
    )

st.divider()
st.header("Topics")

for topic in TOPICS:
    with st.container(border=True):
        a, b = st.columns([1.2, 2], gap="large")
        with a:
            st.markdown(f"<div class='kicker'>Topic {topic['number']}</div>", unsafe_allow_html=True)
            st.subheader(topic["title"], anchor=False)
            st.markdown(topic["summary"])
            st.page_link(page_path(topic, topic["pages"][0]), label="Start this topic",
                         icon=":material/arrow_forward:")
        with b:
            cols = st.columns(2)
            for i, p in enumerate(topic["pages"]):
                cols[i % 2].page_link(page_path(topic, p), label=f"{i + 1}. {p['title']}", icon=p["icon"])

with st.container(border=True):
    st.markdown(":material/add_circle: **More topics are on the way.** "
                "Each new topic appears here and in the sidebar as soon as it is added.")
