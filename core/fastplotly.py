"""Skip Plotly's per-property validation of TRACES inside trusted, hot code paths.

Building an animation means creating hundreds of trace objects. Plotly validates every property of every
object (≈0.4 ms per trace), which made a single slider update take a few hundred milliseconds. Our traces are
produced by our own tested helpers, so inside `with fast():` validation is switched off (≈0.03 ms per trace).
Outside the block Plotly behaves exactly as usual.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager

import plotly.basedatatypes as _bdt

# Streamlit runs every session in its own thread. The state lives on plotly's module (not this one), so that
# it is shared even if Streamlit re-imports this file after an edit while the app is running.
_STATE = getattr(_bdt, "_cv_fast_state", None)
if _STATE is None:
    _STATE = _bdt._cv_fast_state = threading.local()


_LAYOUT_TYPES = tuple(t for t in (getattr(_bdt, "BaseLayoutType", None), getattr(_bdt, "BaseLayoutHierarchyType", None))
                      if t is not None)


def _get(self):
    flag = self.__dict__.get("_validate_flag", True)
    if not flag or not getattr(_bdt._cv_fast_state, "on", False):
        return flag
    # only traces (and their markers, lines, ...) skip validation; layout objects keep it, because
    # make_subplots / update_xaxes rely on validated child objects to keep axes linked to their domains
    return isinstance(self, _LAYOUT_TYPES)


def _set(self, value):
    self.__dict__["_validate_flag"] = value


if not isinstance(getattr(_bdt.BasePlotlyType, "_validate", None), property):
    _bdt.BasePlotlyType._validate = property(_get, _set)


@contextmanager
def fast():
    prev = getattr(_STATE, "on", False)
    _STATE.on = True
    try:
        yield
    finally:
        _STATE.on = prev
