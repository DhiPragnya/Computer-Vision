"""Computer Vision - Visual & Interactive Notes
Run with:   streamlit run app.py
"""
import time

import streamlit as st

from core import live, ui
from topics.registry import TOPICS, page_path, url_path

st.set_page_config(
    page_title="Computer Vision · Visual Notes",
    page_icon=":material/visibility:",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui.inject_css()

pages = {"Start here": [st.Page("home.py", title="Home", icon=":material/home:", default=True)]}
for topic in TOPICS:
    pages[f"{topic['number']} · {topic['title']}"] = [
        st.Page(page_path(topic, p), title=p["title"], icon=p["icon"], url_path=url_path(topic, p))
        for p in topic["pages"]
    ]

nav = st.navigation(pages, expanded=True)

with st.sidebar:
    st.caption("Visual notes for computer vision · built with Streamlit, NumPy, OpenCV and Plotly")
    live.page_guard()

t0 = time.perf_counter()
nav.run()
# how long a full page run takes (live sliders outside a live.section pace themselves with it)
st.session_state["_live_ms_page"] = (time.perf_counter() - t0) * 1000
