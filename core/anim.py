"""Smooth, browser-side animations whose every state is computed in Python.

Streamlit widgets only send their value when you RELEASE them, and every change makes Python run
again. For the things a student wants to *feel* - "what happens while I turn this knob?", "what does
this step of the derivation do?" - we precompute all states with NumPy and hand them to Plotly as
animation frames. The browser then switches between them instantly while you drag: no server
round-trip, no delay.

Two tools
---------
step_player(base, steps)
    Walk through a derivation. Each Step has a title, the equation of that step (highlighted where
    it changed), a sentence on *why it matters*, and the picture for that step. Between two steps the
    picture MORPHS smoothly, so the student sees exactly what the algebra just did. Drag the slider to
    scrub back and forth, press a step number, or play everything.

scrubber(base, values, make_state)
    A live parameter slider inside the figure. Drag it and the picture (and the numbers in the
    caption) follow instantly.

All dynamic traces of all states must have the same structure (same number of traces, and the same
number of points per trace) so that they can be interpolated.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go

from . import theme
from .fastplotly import fast
from .live import dragging


_SHORT = {np.dtype(k): v for k, v in {"int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2", "int32": "i4",
                                        "uint32": "u4", "float32": "f4", "float64": "f8"}.items()}
_SKIP_KEYS = frozenset({"geojson", "layer", "layers", "range"})


def _typed(v: np.ndarray):
    """numpy array -> plotly.js typed-array spec {dtype, bdata[, shape]} (same format as plotly.py)."""
    if v.dtype.kind in "iu" and v.dtype.itemsize == 8:            # 64-bit ints are not supported by plotly.js
        lo, hi = int(v.min()), int(v.max())
        for t in ((np.int8, np.int16, np.int32) if v.dtype.kind == "i" else (np.uint8, np.uint16, np.uint32)):
            info = np.iinfo(t)
            if info.min <= lo and hi <= info.max:
                v = v.astype(t)
                break
        else:
            return v
    code = _SHORT.get(v.dtype)
    if code is None:
        return v
    spec = {"dtype": code, "bdata": base64.b64encode(np.ascontiguousarray(v)).decode("ascii")}
    if v.ndim > 1:
        spec["shape"] = str(v.shape)[1:-1]
    return spec


def _encode(obj) -> None:
    """In place: every numeric numpy array inside dicts / lists -> compact base64 typed array.

    A lean version of plotly's convert_to_base64 for our raw frames: it only looks at numpy arrays, which is
    all our frames contain, and skips plotly's generic (slow) array detection."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, np.ndarray):
                if v.size and v.dtype.kind in "iuf" and k not in _SKIP_KEYS:
                    obj[k] = _typed(v)
            elif isinstance(v, (dict, list, tuple)) and k not in _SKIP_KEYS:
                _encode(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            if isinstance(v, (dict, list, tuple)):
                _encode(v)


class AnimFigure(go.Figure):
    """A go.Figure whose animation frames (and the big slider / button menus) are stored as plain dicts.

    go.Frame objects validate and deep-copy every trace; with hundreds of interpolated frames that cost
    more than all the maths. The raw parts are added only when the figure is serialised for the browser."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._raw_frames = []
        self._raw_layout = {}

    def to_dict(self):
        d = super().to_dict()
        if self._raw_frames:
            _encode(self._raw_frames)          # numpy arrays -> compact typed arrays (in place, idempotent)
            d["frames"] = self._raw_frames
        if self._raw_layout:
            d.setdefault("layout", {}).update(self._raw_layout)
        return d

# ----------------------------------------------------------------------------- text helpers


def hl(text: str, color: str = theme.HIGHLIGHT, bold: bool = True) -> str:
    """Highlight the part of an equation that changed in this step."""
    t = f"<b>{text}</b>" if bold else text
    return f"<span style='color:{color}'>{t}</span>"


def faint(text: str) -> str:
    return f"<span style='color:#8f8d87'>{text}</span>"


@dataclass
class Step:
    """One step of a derivation.

    traces  : the picture AT this step (same structure in every step)
    between : optional function t -> traces for the morph INTO this step (t from 0 to 1). Use it when a
              straight-line blend would be misleading, e.g. a rotation should travel along an arc.
    """
    title: str
    eq: str = ""
    note: str = ""
    traces: list = field(default_factory=list)
    shapes: list | None = None          # optional layout shapes for this step (not interpolated)
    between: object = None


# ----------------------------------------------------------------------------- interpolation

_SCATTER_LIKE = {"scatter", "scattergl", "bar"}


def _ease(t: float) -> float:
    return 0.5 - 0.5 * np.cos(np.pi * t)


def _as_array(v):
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        return v.astype(float) if v.dtype.kind in "iuf" else None
    if isinstance(v, dict) and "bdata" in v:          # already-encoded typed array
        return None
    if isinstance(v, (list, tuple)):
        try:
            return np.array([np.nan if e is None else e for e in v], dtype=float) if (
                len(v) == 0 or not isinstance(v[0], (list, tuple, np.ndarray))) else np.array(v, dtype=float)
        except (TypeError, ValueError):
            return None
    if isinstance(v, (int, float, np.floating, np.integer)):
        return np.array(float(v))
    return None


def _is_u8(v) -> bool:
    return isinstance(v, np.ndarray) and v.dtype == np.uint8


def _compact(v):
    """float32 numpy arrays are sent to the browser as compact base64 typed arrays."""
    if isinstance(v, np.ndarray) and v.dtype.kind == "f":
        return v.astype(np.float32)
    if isinstance(v, np.ndarray) and v.dtype.kind in "iu" and v.size > 64:
        return v.astype(np.uint8) if v.min() >= 0 and v.max() < 256 else v.astype(np.int32)
    return v


def _get(d: dict, path):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def _set(d: dict, path, value):
    for p in path[:-1]:
        d = d.setdefault(p, {})
    d[path[-1]] = value


_NUMERIC_PATHS = [("x",), ("y",), ("z",), ("opacity",), ("marker", "size"), ("marker", "opacity"),
                  ("line", "width"), ("marker", "angle"), ("textfont", "size")]


def _copy2(d: dict) -> dict:
    """Copy a trace dict two levels deep (enough: we only ever set keys like marker.size, line.width)."""
    return {k: (dict(v) if isinstance(v, dict) else v) for k, v in d.items()}


def _lerp_trace(a: dict, b: dict, t: float) -> dict:
    if t >= 1.0 - 1e-9 or t <= 1e-9:                # exactly the key state (no 0·NaN surprises)
        out = _copy2(b if t >= 0.5 else a)
        for path in [("x",), ("y",), ("z",)]:
            v = _get(out, path)
            if isinstance(v, np.ndarray):
                _set(out, path, _compact(v))
        return out
    out = _copy2(b if t >= 0.5 else a)
    for path in _NUMERIC_PATHS:
        va, vb = _as_array(_get(a, path)), _as_array(_get(b, path))
        if va is None or vb is None or va.shape != vb.shape:
            continue
        with np.errstate(invalid="ignore"):
            mixed = (1 - t) * va + t * vb
        gaps = np.isnan(va) | np.isnan(vb)          # a gap that opens or closes: jump at half-time
        if gaps.any():
            mixed = np.where(gaps, vb if t >= 0.5 else va, mixed)
        if _is_u8(_get(a, path)) and _is_u8(_get(b, path)):     # images stored as uint8 stay uint8 (4x smaller)
            _set(out, path, np.clip(np.round(mixed), 0, 255).astype(np.uint8))
            continue
        _set(out, path, _compact(mixed) if mixed.ndim else float(mixed))
    for path in [("x",), ("y",), ("z",)]:
        v = _get(out, path)
        if isinstance(v, np.ndarray):
            _set(out, path, _compact(v))
    return out


def _trace_dicts(traces) -> list[dict]:
    out = []
    for tr in traces:
        if hasattr(tr, "to_plotly_json"):
            with fast():
                tr = tr.to_plotly_json()
        d = dict(tr)
        for path in [("x",), ("y",), ("z",)]:
            arr = _get(d, path)
            if isinstance(arr, (list, tuple)) and arr and not isinstance(arr[0], str):
                a = _as_array(arr)
                if a is not None:
                    _set(d, path, a)
            arr = _get(d, path)
            if isinstance(arr, np.ndarray):
                _set(d, path, _compact(arr))
        out.append(d)
    return out


# ----------------------------------------------------------------------------- caption


def _caption(text: str, height_px: int) -> dict:
    return dict(text=text, x=0, y=1.0, xref="paper", yref="paper", xanchor="left", yanchor="bottom",
                align="left", showarrow=False, font=dict(size=14), bgcolor="rgba(128,128,128,0.09)",
                bordercolor="rgba(128,128,128,0.35)", borderwidth=1, borderpad=8, yshift=10)


def wrap(text: str, width: int = 105) -> str:
    """Insert <br> so long sentences fit (Plotly annotations do not wrap by themselves).
    HTML tags are kept intact and do not count towards the width; existing <br> reset the count."""
    import re

    out, line_len, pending_space = [], 0, False
    for tok in re.split(r"(<[^>]+>|\s+)", text):
        if not tok:
            continue
        if tok.isspace():
            pending_space = line_len > 0
            continue
        if tok.startswith("<") and tok.endswith(">"):
            if tok.lower().startswith("<br"):
                out.append(tok)
                line_len, pending_space = 0, False
            else:
                if pending_space:
                    out.append(" ")
                    line_len += 1
                    pending_space = False
                out.append(tok)
            continue
        if pending_space:
            if line_len + 1 + len(tok) > width:
                out.append("<br>")
                line_len = 0
            else:
                out.append(" ")
                line_len += 1
            pending_space = False
        out.append(tok)
        line_len += len(tok)
    return "".join(out)


def _step_caption(i: int, n: int, s: Step, width: int = 105) -> str:
    parts = [f"<b>Step {i + 1} of {n} · {s.title}</b>"]
    if s.eq:
        parts.append(wrap(s.eq, width + 10))
    if s.note:
        parts.append(f"<i>{wrap(s.note, width)}</i>")
    return "<br>".join(parts)


# ----------------------------------------------------------------------------- core builder


def _frames_figure(base: go.Figure, states: list[list[dict]], captions: list[str], names: list[str],
                   slider_labels: list[str], play_sequence: list[str], extra_buttons: list[dict] | None,
                   active: int, top_margin: int, frame_ms: int, slider_prefix: str | None,
                   shapes: list | None = None, hide_ticks: bool = False) -> go.Figure:
    if top_margin is None:          # room for the tallest caption
        lines = max([c.count("<br>") + 1 for c in captions if c] or [0])
        top_margin = 30 + 19 * lines + (8 if lines else 0)
    with fast():
        fig = AnimFigure(base)
    lay_update = {}                              # applied in ONE update_layout call at the end (much faster)
    if fig.layout.showlegend is not False:      # keep the legend clear of the caption
        lay_update["legend"] = dict(orientation="v", x=1, xanchor="right", y=1, yanchor="top",
                                    bgcolor="rgba(128,128,128,0.07)")
    k0 = len(fig.data)
    with fast():
        fig.add_traces(list(states[active]))
    idx = list(range(k0, k0 + len(states[active])))
    # redraw=True so that the caption (a layout annotation) is repainted on every frame
    redraw = True
    base_ann = [a.to_plotly_json() for a in (fig.layout.annotations or [])]   # plain dicts: frames are raw
    frames = []
    for i, (st_, cap, nm) in enumerate(zip(states, captions, names)):
        lay = dict(annotations=base_ann + ([_caption(cap, top_margin)] if cap else []))
        if shapes is not None and shapes[i] is not None:
            lay["shapes"] = shapes[i]
        frames.append(dict(data=st_, traces=idx, name=nm, layout=lay))
    fig._raw_frames = frames
    lay_update["annotations"] = base_ann + ([_caption(captions[active], top_margin)] if captions[active] else [])
    if shapes is not None and shapes[active] is not None:
        lay_update["shapes"] = shapes[active]
    anim = dict(frame=dict(duration=frame_ms, redraw=redraw), transition=dict(duration=0), mode="immediate",
                fromcurrent=False)
    buttons = [dict(label="▶ Play", method="animate", args=[play_sequence, anim]),
               dict(label="⏮", method="animate", args=[[names[0]], dict(anim, frame=dict(duration=0, redraw=redraw))])]
    if extra_buttons:       # one row: [▶ Play] [⏮]  step: [1] [2] [3] ...
        buttons += [dict(label="  step:", method="skip", args=[None])] + extra_buttons
    # leave room under the plot for x-axis titles, so the controls never cover them
    titled = any(getattr(getattr(fig.layout[k], "title", None), "text", None)
                 for k in fig.layout if k.startswith("xaxis"))
    gap = 26 if titled else 0
    menus = [dict(type="buttons", direction="left", x=0, y=0, xanchor="left", yanchor="top", pad=dict(t=34 + gap, r=4),
                  showactive=False, buttons=buttons, font=dict(size=12))]
    fig._raw_layout = dict(      # plain dicts: validating hundreds of slider steps would cost more than the maths
        updatemenus=menus,
        sliders=[dict(active=active, x=0, len=1.0, y=0, yanchor="top", xanchor="left", pad=dict(t=72 + gap, b=0),
                      currentvalue=dict(visible=slider_prefix is not None, prefix=slider_prefix or "",
                                        xanchor="right", font=dict(size=13, color="#8f8d87")),
                      ticklen=0 if hide_ticks else 3, minorticklen=0, transition=dict(duration=0),
                      font=dict(size=1, color="rgba(0,0,0,0)") if hide_ticks else dict(size=11),
                      steps=[dict(method="animate", label=lab,
                                  args=[[nm], dict(mode="immediate", frame=dict(duration=0, redraw=redraw),
                                                   transition=dict(duration=0))])
                             for nm, lab in zip(names, slider_labels)])],
    )
    m = fig.layout.margin
    lay_update["margin"] = dict(l=m.l if m.l is not None else 10, r=m.r if m.r is not None else 10,
                                t=top_margin, b=120 + gap)
    fig.update_layout(**lay_update)
    return fig


def step_player(base: go.Figure, steps: list[Step], n_sub: int = 14, hold: int = 18, frame_ms: int = 40,
                top_margin: int | None = None, start: int = 0, wrap_at: int = 105) -> go.Figure:
    """A derivation, step by step, with smooth morphs between the steps.

    wrap_at: characters per caption line - use ~70 when the figure sits in a narrow column."""
    if dragging():          # mid-drag nobody can press Play: only the key states, the morphs come on release
        n_sub = 1
    n = len(steps)
    keys = [_trace_dicts(s.traces) for s in steps]
    caps = [_step_caption(i, n, s, wrap_at) for i, s in enumerate(steps)]
    states, captions, names, labels, shapes = [keys[0]], [caps[0]], ["s0"], ["1"], [steps[0].shapes]
    key_index = [0]
    for k in range(n - 1):
        for j in range(1, n_sub + 1):
            t = _ease(j / n_sub)
            if steps[k + 1].between is not None and j < n_sub:
                with fast():
                    trs = steps[k + 1].between(t)
                states.append(_trace_dicts(trs))
            else:
                states.append([_lerp_trace(a, b, t) for a, b in zip(keys[k], keys[k + 1])])
            captions.append(caps[k + 1])
            shapes.append(steps[k + 1].shapes)
            last = j == n_sub
            names.append(f"s{k + 1}" if last else f"m{k}_{j}")
            labels.append(str(k + 2) if last else "")
        key_index.append(len(states) - 1)
    # play everything, pausing on each step
    play = []
    for k in range(n):
        if k == 0:
            play += ["s0"] * hold
        else:
            play += [f"m{k - 1}_{j}" for j in range(1, n_sub)] + [f"s{k}"] * hold
    # one button per step: plays the morph INTO that step
    redraw = True
    step_btns = []
    for k in range(n):
        seq = ["s0"] if k == 0 else [f"m{k - 1}_{j}" for j in range(1, n_sub)] + [f"s{k}"]
        step_btns.append(dict(label=f"{k + 1}", method="animate",
                              args=[seq, dict(frame=dict(duration=frame_ms, redraw=redraw), transition=dict(duration=0),
                                              mode="immediate", fromcurrent=False)]))
    if not any(sh is not None for sh in shapes):
        shapes = None
    return _frames_figure(base, states, captions, names, labels, play, step_btns, key_index[start],
                          top_margin, frame_ms, None, shapes)


def scrubber(base: go.Figure, values, make_state, active: int | None = None, label: str = "value",
             fmt: str = "{:.2f}", frame_ms: int = 60, top_margin: int | None = None, bounce: bool = True,
             wrap_at: int = 110) -> go.Figure:
    """A live slider inside the figure. make_state(v) -> (list_of_traces, caption_html)."""
    values = list(values)
    active = len(values) // 2 if active is None else active
    if dragging() and len(values) > 12:     # mid-drag: a coarse sweep is enough, full resolution on release
        keep = sorted(set(range(0, len(values), 4)) | {active, len(values) - 1})
        values, active = [values[i] for i in keep], keep.index(active)
    states, captions = [], []
    for v in values:
        with fast():
            trs, cap = make_state(v)
        states.append(_trace_dicts(trs))
        captions.append(wrap(cap, wrap_at) if cap else cap)
    names = [f"v{i}" for i in range(len(values))]
    labels = [fmt.format(v) for v in values]
    play = names[active:] + names[::-1] + names[:active + 1] if bounce else names
    return _frames_figure(base, states, captions, names, labels, play, None, active, top_margin, frame_ms,
                          f"{label} = ", hide_ticks=True)


def morph(build, a, b):
    """A `between` function that blends PARAMETERS a -> b and redraws with build(params).

    a, b : numbers, arrays or dicts of numbers. Use it when blending the parameters is more honest than
    blending the drawn points (a rotation should sweep through the angles, not cut across the circle)."""
    def lerp(x, y, t):
        if isinstance(x, dict):
            return {k: lerp(x[k], y[k], t) for k in x}
        return (1 - t) * np.asarray(x, float) + t * np.asarray(y, float) if np.ndim(x) else (1 - t) * x + t * y

    return lambda t: build(lerp(a, b, t))


def nearest_index(values, v) -> int:
    values = np.asarray(list(values), float)
    return int(np.argmin(np.abs(values - v)))
