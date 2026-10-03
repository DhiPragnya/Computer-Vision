"""Reusable page furniture so every lesson has the same rhythm:

    header -> intuition -> interactive figure -> the math -> pitfalls -> takeaway -> quick check -> nav
"""
from __future__ import annotations

import base64

import cv2
import numpy as np
import streamlit as st

from . import transforms as tf

CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1380px;}
p.lead {font-size: 1.12rem; line-height: 1.6; opacity: 0.86; margin-top: -0.4rem; max-width: 62rem;}
.kicker {font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; opacity: 0.65; font-weight: 600;}
table.inv {border-collapse: collapse; width: 100%; font-size: 0.92rem;}
table.inv th, table.inv td {padding: 6px 10px; border-bottom: 1px solid rgba(128,128,128,0.22); text-align: center;}
table.inv th:first-child, table.inv td:first-child {text-align: left;}
table.inv th {font-weight: 600;}
table.inv td.yes {color: #1a9a6c; font-weight: 700;}
table.inv td.no {opacity: 0.35;}
table.inv .hl {background: rgba(42,120,214,0.12);}
table.inv th.hl {border-top: 3px solid #2a78d6;}
.legend-row {display:flex; gap:18px; flex-wrap:wrap; font-size:0.86rem; opacity:0.9; margin: 2px 0 6px 0;}
.legend-row span {display:inline-flex; align-items:center; gap:6px;}
.sw {display:inline-block; width:22px; height:0; border-top:3px solid; }
.sw.dash {border-top-style: dashed;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# images
# --------------------------------------------------------------------------

def datauri(img, fmt: str = "png", quality: int = 90) -> str:
    """Encode an image array as a data: URI for st.image.

    Why: st.image(array) stores the picture as a temporary media file and the browser fetches it by URL.
    With live sliders a new picture arrives several times a second and the previous file is deleted
    while the browser may still be loading it ("Image source error" flicker). A data: URI travels with
    the page update itself, so there is nothing to fetch and nothing to lose."""
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = (np.clip(np.nan_to_num(a.astype(float)), 0.0, 1.0) * 255 + 0.5).astype(np.uint8)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    if a.ndim == 3 and a.shape[2] == 3:
        a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
    elif a.ndim == 3 and a.shape[2] == 4:
        a = cv2.cvtColor(a, cv2.COLOR_RGBA2BGRA)
    jpeg = fmt.lower() in ("jpg", "jpeg")
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if jpeg else [cv2.IMWRITE_PNG_COMPRESSION, 1]
    ok, buf = cv2.imencode(".jpg" if jpeg else ".png", np.ascontiguousarray(a), params)
    if not ok:                                  # pragma: no cover
        raise ValueError("could not encode image")
    return f"data:image/{'jpeg' if jpeg else 'png'};base64," + base64.b64encode(buf.tobytes()).decode("ascii")


# --------------------------------------------------------------------------
# headers & navigation
# --------------------------------------------------------------------------

def lesson_header(file: str, lead: str) -> None:
    from topics.registry import locate

    topic, i, page = locate(file)
    if topic is not None:
        st.markdown(f"<div class='kicker'>Topic {topic['number']} · {topic['title']} · "
                    f"Lesson {i + 1} of {len(topic['pages'])}</div>", unsafe_allow_html=True)
        st.title(page["title"])
    st.markdown(f"<p class='lead'>{lead}</p>", unsafe_allow_html=True)


def lesson_nav(file: str) -> None:
    from topics.registry import locate, page_path

    topic, i, _ = locate(file)
    if topic is None:
        return
    st.divider()
    left, _, right = st.columns([1, 1, 1])
    pages = topic["pages"]
    if i > 0:
        p = pages[i - 1]
        left.page_link(page_path(topic, p), label=f"Previous: {p['title']}", icon=":material/arrow_back:")
    if i < len(pages) - 1:
        p = pages[i + 1]
        right.page_link(page_path(topic, p), label=f"Next: {p['title']}", icon=":material/arrow_forward:")


def colour_legend(items=("original", "transformed", "e1", "e2")) -> None:
    """A tiny legend explaining the colour language (shown under figures)."""
    from . import theme

    spec = {
        "original": (theme.ORIGINAL, "dash", "original (before)"),
        "transformed": (theme.TRANSFORMED, "", "transformed (after)"),
        "e1": (theme.E1, "", "e₁ = (1,0) lands here → 1st column"),
        "e2": (theme.E2, "", "e₂ = (0,1) lands here → 2nd column"),
        "highlight": (theme.HIGHLIGHT, "", "special point"),
    }
    html = "<div class='legend-row'>"
    for k in items:
        c, d, label = spec[k]
        html += f"<span><i class='sw {d}' style='border-color:{c}'></i>{label}</span>"
    st.markdown(html + "</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# callouts
# --------------------------------------------------------------------------

_CALLOUT = {
    "intuition": (st.info, ":material/lightbulb:", "Intuition"),
    "pitfall": (st.warning, ":material/warning:", "Common pitfall"),
    "takeaway": (st.success, ":material/push_pin:", "Key takeaway"),
    "note": (st.info, ":material/info:", "Note"),
}


def callout(kind: str, body: str, title: str | None = None) -> None:
    if kind in _CALLOUT:
        fn, icon, default = _CALLOUT[kind]
        fn(f"**{title or default}.** {body}", icon=icon)
    else:
        icons = {"practice": (":material/photo_camera:", "Where you meet this in computer vision"),
                 "try": (":material/touch_app:", "Try this"),
                 "math": (":material/function:", "The math")}
        icon, default = icons.get(kind, (":material/info:", ""))
        with st.container(border=True):
            st.markdown(f"{icon} **{title or default}**")
            st.markdown(body)


# --------------------------------------------------------------------------
# animated derivations
# --------------------------------------------------------------------------

def anim_intro(title: str, hint: str | None = None) -> None:
    """Header for a step-by-step animated derivation."""
    st.markdown(f":material/animation: **{title}**")
    st.caption(hint or "▶ Play runs every step; the numbered buttons play one step; drag the slider to move back "
                       "and forth between steps - the picture follows instantly.")


# --------------------------------------------------------------------------
# "why?" questions students ask
# --------------------------------------------------------------------------

def why(question: str, answer: str | None = None, expanded: bool = False):
    """One student 'why?' question answered in an expander.

    Returns the expander, so a page can add a small figure inside it:
        with ui.why("Why 1, 2, 1?"):
            st.plotly_chart(...)
    """
    exp = st.expander(question.strip(), icon=":material/psychology_alt:", expanded=expanded)
    if answer:
        exp.markdown(answer)
    return exp


def whys(items: list[tuple[str, str]], title: str = "Questions students ask", columns: int = 2) -> None:
    """A block of 'why?' questions - the ones that come up in class even if the notes don't ask them."""
    st.subheader(title, divider="gray")
    cols = st.columns(columns, gap="medium")
    for i, (q, a) in enumerate(items):
        with cols[i % columns]:
            why(q, a)


# --------------------------------------------------------------------------
# matrices & numbers
# --------------------------------------------------------------------------

def num(v: float, p: int = 2) -> str:
    if not np.isfinite(v):
        return r"\infty"
    s = f"{v:.{p}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def tex_matrix(M, p: int = 2, highlight: dict | None = None) -> str:
    """LaTeX bmatrix. `highlight` maps (i, j) -> colour to make entries stand out."""
    M = np.atleast_2d(np.asarray(M, float))
    rows = []
    for i, r in enumerate(M):
        cells = []
        for j, v in enumerate(r):
            s = num(v, p)
            if highlight and (i, j) in highlight:
                s = r"\color{" + highlight[(i, j)] + "}{" + s + "}"
            cells.append(s)
        rows.append(" & ".join(cells))
    return r"\begin{bmatrix}" + r" \\ ".join(rows) + r"\end{bmatrix}"


def tex_vec(v, p: int = 2) -> str:
    return tex_matrix(np.asarray(v, float).reshape(-1, 1), p)


def show_matrix(name: str, M, p: int = 2, highlight: dict | None = None, suffix: str = "") -> None:
    st.latex(f"{name} = {tex_matrix(M, p, highlight)}{suffix}")


def column_highlight(n_rows: int = 3, last_row: bool = True) -> dict:
    """Colour column 0 orange and column 1 green, matching the e1/e2 arrows."""
    from . import theme

    hl = {}
    for i in range(2):
        hl[(i, 0)] = theme.E1
        hl[(i, 1)] = theme.E2
    return hl


# --------------------------------------------------------------------------
# invariants
# --------------------------------------------------------------------------

def invariants_table(highlight: str | None = None) -> None:
    cols = tf.CLASSES
    html = "<table class='inv'><tr><th>Property</th>"
    for c in cols:
        cls = " class='hl'" if c["key"] == highlight else ""
        html += f"<th{cls}>{c['name'].split(' (')[0]}<br><span style='font-weight:400;opacity:.7'>{c['dof']} DOF</span></th>"
    html += "</tr>"
    for prop, keepers in tf.INVARIANTS:
        html += f"<tr><td>{prop}</td>"
        for c in cols:
            yes = c["key"] in keepers
            cls = ("yes" if yes else "no") + (" hl" if c["key"] == highlight else "")
            html += f"<td class='{cls}'>{'✓' if yes else '✗'}</td>"
        html += "</tr>"
    st.markdown(html + "</table>", unsafe_allow_html=True)


def preserved_badges(key: str) -> None:
    with st.container(horizontal=True, gap="small"):
        for prop, keepers in tf.INVARIANTS:
            if key in keepers:
                st.badge(prop, icon=":material/check:", color="green")
            else:
                st.badge(prop, icon=":material/close:", color="gray")


# --------------------------------------------------------------------------
# quick checks
# --------------------------------------------------------------------------

def _mcq_body(key: str, question: str, options: list[str], answer: int, explanation: str):
    with st.container(border=True):
        st.markdown(f":material/help: **{question}**")
        choice = st.radio(question, options, index=None, key=f"mcq_{key}", label_visibility="collapsed")
        if choice is None:
            return None
        ok = options.index(choice) == answer
        if ok:
            st.success(f"**Correct.** {explanation}", icon=":material/check_circle:")
        else:
            st.error(f"**Not quite.** {explanation}", icon=":material/cancel:")
        return ok


@st.fragment
def _mcq_fragment(key, question, options, answer, explanation):
    _mcq_body(key, question, options, answer, explanation)


def mcq(key: str, question: str, options: list[str], answer: int, explanation: str, scored: bool = False):
    """Multiple-choice question with instant feedback.

    scored=False (lesson pages): runs as its own fragment, so answering does not re-run the whole page.
    scored=True (quiz pages): returns True / False / None (unanswered) so the page can add up a score."""
    if scored:
        return _mcq_body(key, question, options, answer, explanation)
    _mcq_fragment(key, question, options, answer, explanation)
    return None
