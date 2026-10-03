"""Live sliders and fast sections.

Why this file exists
--------------------
`st.slider` only tells Python its new value when you RELEASE the mouse, so while you drag nothing
changes - the picture jumps once at the end. For learning, the *motion* is the point: you want to
see the shape stretch, the edge map thin out, the threshold eat the weak edges WHILE you move the knob.

`live.slider` is a drop-in replacement for `st.slider` (same arguments) built with Streamlit's own
component API (plain HTML + JavaScript, no framework). It sends its value continuously while you
drag - but never faster than the page can redraw, so updates never pile up and the figure follows
your hand. All the maths still runs in Python with NumPy / OpenCV.

`live.section` wraps a block of a lesson in `st.fragment`, so moving a slider re-runs ONLY that block
(not the whole page). It also measures how long the block takes, which the sliders use to pace
their updates.

Under `streamlit.testing.AppTest` (no browser) the sliders fall back to `st.slider`, so automated
tests can still drive every control.
"""
from __future__ import annotations

import contextlib
import contextvars
import functools
import re
import time

import streamlit as st

from .fastplotly import fast

_TIMING = bool(__import__("os").environ.get("CV_TIMING"))
_SECTION: contextvars.ContextVar[str | None] = contextvars.ContextVar("live_section", default=None)
_DRAGGING: contextvars.ContextVar[bool] = contextvars.ContextVar("live_dragging", default=False)


@contextlib.contextmanager
def full_quality():
    """Build at full quality even mid-drag. Use it (as decorator or `with`) on CACHED figure builders, so that a
    coarse mid-drag version can never end up in the cache."""
    token = _DRAGGING.set(False)
    try:
        yield
    finally:
        _DRAGGING.reset(token)


def dragging() -> bool:
    """True while this run was triggered by a slider whose thumb is still held down.

    Use it to skip work nobody can see mid-drag (e.g. the in-between frames of a step animation); the
    release sends one more update, and that run builds everything at full quality."""
    return _DRAGGING.get()

# --------------------------------------------------------------------------------------------- JS / CSS

_JS = r"""
const fmtNum = (v, f) => {
  let s = f.int ? String(Math.round(v)) : Number(v).toFixed(f.dec);
  if (/^-0(\.0*)?$/.test(s)) s = s.slice(1);
  return (f.pre || '') + s + (f.suf || '');
};

function current(S) {
  const d = S.data;
  const vals = S.inputs.slice(0, d.range ? 2 : 1).map(i => Number(i.value));
  if (d.labels) return vals.map(v => Math.round(v));
  const k = Math.pow(10, Math.min(8, (d.fmt.dec || 0) + 4));
  return vals.map(v => Math.round(v * k) / k);
}

function text(S, v) {
  const d = S.data;
  return d.labels ? (d.labels[Math.round(v)] ?? '') : fmtNum(v, d.fmt);
}

function paint(S) {
  const d = S.data, lo = Number(S.inputs[0].min), hi = Number(S.inputs[0].max);
  const span = (hi - lo) || 1;
  const vals = current(S);
  const pct = vals.map(v => Math.max(0, Math.min(1, (v - lo) / span)));
  vals.forEach((v, i) => {
    const b = S.bubbles[i];
    b.textContent = text(S, v);
    b.style.left = `calc(${pct[i] * 100}% + ${7 - pct[i] * 14}px)`;
    b.style.transform = `translateX(${-Math.max(0, Math.min(100, (pct[i] - 0.04) / 0.92 * 100))}%)`;
  });
  const a = d.range ? pct[0] : 0, z = d.range ? pct[1] : pct[0];
  S.fill.style.left = `calc(${a * 100}% + ${7 - a * 14}px)`;
  S.fill.style.width = `calc(${(z - a) * 100}% - ${(z - a) * 14}px)`;
}

function flush(S) {
  S.timer = null;
  if (S.pending === null) return;
  const p = S.pending; S.pending = null;
  const k = JSON.stringify(p);
  if (k === S.sentKey) return;
  S.sentKey = k;
  // d = 1 while the thumb is held: Python may then skip expensive extras (e.g. in-between animation frames)
  S.send('value', { v: p.v, d: p.d, rev: S.rev });
  S.timer = setTimeout(() => flush(S), S.data.iv || 80);
}

function queue(S, final) {
  S.pending = { v: current(S), d: (!final && S.pointer) ? 1 : 0 };
  if (final) {
    if (S.timer) { clearTimeout(S.timer); S.timer = null; }
    flush(S);
  } else if (!S.timer) {
    flush(S);
  }
}

function build(root) {
  const S = { rev: null, timer: null, pending: null, sentKey: null, pointer: false };
  const w = document.createElement('div');
  w.className = 'ls';
  w.innerHTML = `
    <div class="ls-label"><span class="ls-text"></span><span class="ls-help" hidden>?</span></div>
    <div class="ls-track">
      <div class="ls-rail"></div><div class="ls-fill"></div>
      <input type="range" class="ls-in"><input type="range" class="ls-in ls-b">
      <div class="ls-bubble"></div><div class="ls-bubble ls-b"></div>
    </div>
    <div class="ls-ends"><span class="ls-min"></span><span class="ls-max"></span></div>`;
  root.appendChild(w);
  S.root = w;
  S.inputs = Array.from(w.querySelectorAll('input'));
  S.bubbles = Array.from(w.querySelectorAll('.ls-bubble'));
  S.fill = w.querySelector('.ls-fill');
  S.inputs.forEach((inp, i) => {
    inp.addEventListener('input', () => {
      if (S.data.range) {           // keep the two thumbs in order
        const a = Number(S.inputs[0].value), b = Number(S.inputs[1].value);
        if (a > b) { if (i === 0) S.inputs[0].value = b; else S.inputs[1].value = a; }
      }
      paint(S); queue(S, false);
    });
    inp.addEventListener('change', () => { S.pointer = false; paint(S); queue(S, true); });
    inp.addEventListener('pointerdown', () => {
      S.pointer = true;
      S.inputs.forEach(o => o.style.zIndex = o === inp ? 3 : 2);
    });
  });
  // released anywhere (even outside the slider or the window): send the final value
  const release = () => { if (S.pointer) { S.pointer = false; paint(S); queue(S, true); } };
  window.addEventListener('pointerup', release);
  window.addEventListener('pointercancel', release);
  window.addEventListener('blur', release);
  return S;
}

function apply(S, d) {
  const w = S.root;
  w.querySelector('.ls-text').textContent = d.label;
  const help = w.querySelector('.ls-help');
  help.hidden = !d.help; help.title = d.help || '';
  w.querySelector('.ls-label').style.display = d.vis === 'collapsed' ? 'none' : '';
  w.querySelector('.ls-label').style.visibility = d.vis === 'hidden' ? 'hidden' : '';
  w.classList.toggle('ls-disabled', !!d.disabled);
  w.classList.toggle('ls-range', !!d.range);
  const lo = d.labels ? 0 : d.min, hi = d.labels ? d.labels.length - 1 : d.max, st = d.labels ? 1 : d.step;
  S.inputs.forEach((inp, i) => {
    inp.min = lo; inp.max = hi; inp.step = st; inp.disabled = !!d.disabled;
    inp.hidden = i === 1 && !d.range;
    inp.setAttribute('aria-label', d.label + (d.range ? (i ? ' (upper)' : ' (lower)') : ''));
  });
  S.bubbles[1].hidden = !d.range;
  w.querySelector('.ls-min').textContent = text(S, lo);
  w.querySelector('.ls-max').textContent = text(S, hi);
  if (S.rev !== d.rev) {            // Python set the value (first mount, reset button, clamping ...)
    S.rev = d.rev;
    d.value.forEach((v, i) => { S.inputs[i].value = v; });
    S.sentKey = JSON.stringify({ v: current(S), d: 0 });
  }
  paint(S);
}

export default function (component) {
  const { data, parentElement, setStateValue } = component;
  let S = parentElement.__liveSlider;
  if (!S) S = parentElement.__liveSlider = build(parentElement);
  S.send = setStateValue;
  S.data = data;
  apply(S, data);
  return () => { if (S.timer) clearTimeout(S.timer); };
}
"""

_CSS = r"""
.ls { font-family: var(--st-font, "Source Sans Pro", sans-serif); color: var(--st-text-color);
      font-size: var(--st-base-font-size, 16px); padding: 0 0 2px 0; user-select: none; }
.ls-label { font-size: 0.875rem; line-height: 1.4; margin-bottom: 1.55rem; display: flex; gap: 6px;
            align-items: center; min-height: 1.2rem; }
.ls-help { display: inline-flex; width: 14px; height: 14px; border-radius: 50%; font-size: 10px;
           align-items: center; justify-content: center; border: 1px solid currentColor; opacity: 0.55;
           cursor: help; }
.ls-help[hidden] { display: none; }
.ls-track { position: relative; height: 16px; margin: 0 2px; }
.ls-rail, .ls-fill { position: absolute; top: 6px; height: 4px; border-radius: 2px; pointer-events: none; }
.ls-rail { left: 0; right: 0; background: var(--st-secondary-background-color, #f0f2f6);
           box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--st-text-color) 10%, transparent); }
.ls-fill { background: var(--st-primary-color, #ff4b4b); }
.ls-in { position: absolute; left: 0; top: 0; width: 100%; height: 16px; margin: 0; background: none;
         -webkit-appearance: none; appearance: none; pointer-events: none; outline: none; z-index: 2; }
.ls-in[hidden] { display: none; }
.ls-in::-webkit-slider-runnable-track { background: none; height: 16px; }
.ls-in::-moz-range-track { background: none; height: 16px; }
.ls-in::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; pointer-events: auto; cursor: grab;
         width: 14px; height: 14px; margin-top: 1px; border-radius: 50%; background: var(--st-primary-color, #ff4b4b);
         border: none; box-shadow: 0 0 0 0 transparent; transition: box-shadow 0.12s; }
.ls-in::-moz-range-thumb { pointer-events: auto; cursor: grab; width: 14px; height: 14px; border-radius: 50%;
         background: var(--st-primary-color, #ff4b4b); border: none; }
.ls-in:active::-webkit-slider-thumb { cursor: grabbing;
         box-shadow: 0 0 0 6px color-mix(in srgb, var(--st-primary-color) 22%, transparent); }
.ls-in:focus-visible::-webkit-slider-thumb {
         box-shadow: 0 0 0 4px color-mix(in srgb, var(--st-primary-color) 35%, transparent); }
.ls:not(.ls-range) .ls-in { pointer-events: auto; }
.ls-bubble { position: absolute; bottom: 19px; white-space: nowrap;
             font-size: 0.875rem; color: var(--st-primary-color); font-variant-numeric: tabular-nums;
             pointer-events: none; }
.ls-bubble[hidden] { display: none; }
.ls-ends { display: flex; justify-content: space-between; font-size: 0.75rem; margin-top: 2px;
           color: color-mix(in srgb, var(--st-text-color) 55%, transparent); font-variant-numeric: tabular-nums; }
.ls-disabled { opacity: 0.45; }
.ls-disabled .ls-in { pointer-events: none !important; }
.ls-disabled .ls-in::-webkit-slider-thumb { cursor: not-allowed; }
"""

_COMPONENT = None


def _component():
    global _COMPONENT
    if _COMPONENT is None:
        _COMPONENT = st.components.v2.component("cv_live_slider", css=_CSS, js=_JS)
    return _COMPONENT


def _has_browser() -> bool:
    """False under streamlit.testing.AppTest (no browser: use plain widgets so tests can drive them)."""
    try:
        from streamlit.runtime import Runtime
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        ctx = get_script_run_ctx()
        if ctx is None or ctx.session_id == "test session id":     # the id AppTest gives its session
            return False
        return Runtime.exists()
    except Exception:       # pragma: no cover
        return False


def _noop():
    return None


def rerun() -> None:
    """Rerun only the current section when we are inside a fragment rerun, otherwise the whole page.

    ``st.rerun(scope="fragment")`` raises when a section happens to execute as part of a full page run
    (e.g. right after navigation), so pick the scope that is legal *now*.
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        ctx = get_script_run_ctx()
        in_fragment_run = bool(ctx is not None and ctx.fragment_ids_this_run)
    except Exception:       # pragma: no cover
        in_fragment_run = False
    st.rerun(scope="fragment" if in_fragment_run else "app")


# --------------------------------------------------------------------------------------------- formatting

_FMT = re.compile(r"%([-+0 #]*)(\d*)(?:\.(\d+))?([dfieEgGu])")


def _decimals(step) -> int:
    s = f"{step:.10f}".rstrip("0")
    return len(s.split(".")[1]) if "." in s else 0


def _fmt_spec(fmt: str | None, is_int: bool, step) -> dict:
    if fmt:
        m = _FMT.search(fmt)
        if m:
            conv = m.group(4)
            dec = int(m.group(3)) if m.group(3) is not None else (0 if conv in "diu" else 6)
            return dict(pre=fmt[:m.start()].replace("%%", "%"), suf=fmt[m.end():].replace("%%", "%"),
                        dec=dec, int=conv in "diu")
    if is_int:
        return dict(pre="", suf="", dec=0, int=True)
    return dict(pre="", suf="", dec=max(2, min(4, _decimals(step))), int=False)


# --------------------------------------------------------------------------------------------- sections


def section(fn=None, *, key: str | None = None):
    """Decorator: run this block as a Streamlit fragment (only it re-runs when its controls change)
    and remember how long it takes (live sliders use that to pace their updates)."""
    if fn is None:
        return functools.partial(section, key=key)
    name = key or f"{fn.__code__.co_filename}:{fn.__qualname__}"

    @st.fragment
    @functools.wraps(fn)
    def run(*args, **kwargs):
        token = _SECTION.set(name)
        drag_token = _DRAGGING.set(False)
        t0 = time.perf_counter()
        try:
            with fast():            # our own figures: skip Plotly's slow per-property validation
                return fn(*args, **kwargs)
        finally:
            _SECTION.reset(token)
            _DRAGGING.reset(drag_token)
            ms = (time.perf_counter() - t0) * 1000
            if _TIMING:
                print(f"[live] {name.rsplit('/', 1)[-1]}: {ms:.0f} ms", flush=True)
            prev = st.session_state.get(f"_live_ms_{name}")
            st.session_state[f"_live_ms_{name}"] = ms if prev is None else 0.5 * prev + 0.5 * ms

    return run


def _interval() -> int:
    """How often a slider may send while dragging (ms): a little longer than the block takes to run."""
    name = _SECTION.get()
    ms = st.session_state.get(f"_live_ms_{name}") if name else st.session_state.get("_live_ms_page")
    ms = 250.0 if ms is None else ms
    iv = 40 + 1.25 * ms
    return int(min(900, max(50, round(iv / 25) * 25)))


# --------------------------------------------------------------------------------------------- sliders


def _mount(key: str, label: str, value_list, rev: int, *, lo=None, hi=None, step=None, fmt=None,
           labels=None, rng=False, help=None, label_visibility="visible", disabled=False):
    data = dict(label=label, value=value_list, rev=rev, min=lo, max=hi, step=step, fmt=fmt or {},
                labels=labels, range=rng, help=help, vis=label_visibility, disabled=disabled,
                iv=_interval())
    res = _component()(key=f"{key}-live", data=data, on_value_change=_noop, height="content")
    state = res.get("value") if hasattr(res, "get") else getattr(res, "value", None)
    if isinstance(state, dict) and state.get("d") and state.get("rev") == rev:
        _DRAGGING.set(True)
    return state


def _resolve(key: str, initial, clamp):
    """Shared bookkeeping: the value lives in st.session_state[key] (so pages can read or set it)."""
    ss = st.session_state
    last_k, rev_k = f"_live_last_{key}", f"_live_rev_{key}"
    if key not in ss:
        ss[key] = initial
    if last_k in ss and ss[key] != ss[last_k]:          # someone set st.session_state[key] directly
        ss[rev_k] = ss.get(rev_k, 0) + 1
    clamped = clamp(ss[key])
    if clamped != ss[key]:
        ss[key] = clamped
        ss[rev_k] = ss.get(rev_k, 0) + 1
    return ss.get(rev_k, 0)


def _finish(key, value):
    st.session_state[key] = value
    st.session_state[f"_live_last_{key}"] = value
    return value


def slider(label: str, min_value=None, max_value=None, value=None, step=None, format: str | None = None,
           key: str | None = None, help: str | None = None, *, disabled: bool = False,
           label_visibility: str = "visible", **_ignored):
    """Like st.slider, but the value streams to Python while you drag. Supports single values and
    (low, high) ranges."""
    if not _has_browser():
        return st.slider(label, min_value, max_value, value, step, format=format, key=key, help=help,
                         disabled=disabled, label_visibility=label_visibility)
    rng = isinstance(value, (tuple, list))
    probe = [min_value, max_value, step] + (list(value) if rng else [value])
    is_int = all(isinstance(v, int) and not isinstance(v, bool) for v in probe if v is not None)
    lo = min_value if min_value is not None else (0 if is_int else 0.0)
    hi = max_value if max_value is not None else (100 if is_int else 1.0)
    st_ = step if step is not None else (1 if is_int else 0.01)
    if value is None:
        value = lo
    key = key or f"live_{label}"
    cast = int if is_int else float

    def snap(v):
        v = min(max(v, lo), hi)
        k = round((v - lo) / st_)
        out = lo + k * st_
        out = min(max(out, lo), hi)
        return cast(round(out, 10)) if not is_int else int(round(out))

    def clamp(v):
        if rng:
            a, b = (snap(cast(x)) for x in v)
            return (min(a, b), max(a, b))
        return snap(cast(v))

    initial = tuple(value) if rng else value
    rev = _resolve(key, clamp(initial), clamp)
    cur = st.session_state[key]
    state = _mount(key, label, list(cur) if rng else [cur], rev, lo=lo, hi=hi, step=st_,
                   fmt=_fmt_spec(format, is_int, st_), rng=rng, help=help,
                   label_visibility=label_visibility, disabled=disabled)
    if isinstance(state, dict) and state.get("rev") == rev and isinstance(state.get("v"), list):
        v = state["v"]
        try:
            cur = clamp(tuple(v[:2]) if rng else v[0])
        except (TypeError, ValueError, IndexError):
            pass
    return _finish(key, cur)


def select_slider(label: str, options=(), value=None, format_func=str, key: str | None = None,
                  help: str | None = None, *, disabled: bool = False, label_visibility: str = "visible",
                  **_ignored):
    """Like st.select_slider (single value), streaming while you drag."""
    options = list(options)
    if not _has_browser():
        return st.select_slider(label, options, value=value, format_func=format_func, key=key, help=help,
                                disabled=disabled, label_visibility=label_visibility)
    key = key or f"live_sel_{label}"
    labels = [str(format_func(o)) for o in options]

    def clamp(v):
        return v if v in options else options[0]

    rev = _resolve(key, value if value in options else options[0], clamp)
    cur = st.session_state[key]
    state = _mount(key, label, [options.index(cur)], rev, labels=labels, help=help,
                   label_visibility=label_visibility, disabled=disabled)
    if isinstance(state, dict) and state.get("rev") == rev and isinstance(state.get("v"), list):
        try:
            i = int(state["v"][0])
            if 0 <= i < len(options):
                cur = options[i]
        except (TypeError, ValueError, IndexError):
            pass
    return _finish(key, cur)


# --------------------------------------------------------------------------------------------- page guard

_GUARD_JS = r"""
export default function () {
  if (window.__cvGuard) return;
  window.__cvGuard = true;
  // Plotly rejects the promise of an animation that gets interrupted (e.g. you grab the slider while
  // "Play" is running) with an empty reason. That is expected behaviour, not an error - silence it.
  window.addEventListener('unhandledrejection', (e) => { if (e.reason === undefined) e.preventDefault(); });
}
"""
_GUARD = None


def page_guard() -> None:
    """Mount once per page (app.py does it in the sidebar)."""
    global _GUARD
    _DRAGGING.set(False)            # a full page run is never "mid-drag" until a slider says so
    if not _has_browser():
        return
    if _GUARD is None:
        _GUARD = st.components.v2.component("cv_page_guard", js=_GUARD_JS)
    _GUARD(key="cv-page-guard", height=0)
