"""Headless checks with streamlit.testing.AppTest - no browser needed.

    python tools/apptest_checks.py smoke                      every page loads without an exception
    python tools/apptest_checks.py fuzz [page-substr ...]     drive EVERY control of every page through its extremes
    python tools/apptest_checks.py validate                   same pages with Plotly validation forced ON
    python tools/apptest_checks.py perf [page-substr ...]     median run time of every @live.section per slider
    python tools/apptest_checks.py profile <page-substr> <slider-key> <section-substr> [n] [--sort=tottime]

Under AppTest the live sliders fall back to plain st.slider (core/live.py detects the AppTest session id), so
they can be driven like normal widgets. The bar for "done" is: zero exceptions everywhere.
"""
from __future__ import annotations

import contextlib
import functools
import io
import sys
import time
import traceback

from _common import open_page, page_files, project_on_path, quiet

CATEGORICAL = ["selectbox", "radio", "segmented_control", "pills", "toggle", "checkbox", "select_slider",
               "multiselect"]
TAB_KEYS = {"t01_geometric_transforms_2d/p02_homogeneous.py": "h_tabs",
            "t01_geometric_transforms_2d/p08_projective.py": "pj_tabs"}
QUIZ_KEYS = ("dr_choice", "wd_choice", "sd_choice")      # drill radios on the recap pages


# ============================================================================================ smoke
def smoke(_args):
    bad = 0
    for page in page_files():
        t0 = time.time()
        at = open_page(page)
        errs = [e.message[:200] for e in at.exception]
        bad += bool(errs)
        print(f"{'ERR' if errs else 'OK '} {page:60s} {time.time() - t0:5.1f}s {errs or ''}", flush=True)
    print(f"pages with exceptions: {bad}")
    return bad


# ============================================================================================ fuzz
class Fuzz:
    def __init__(self, page):
        self.page, self.runs, self.errors, self.harness, self.t0, self.tab = page, 0, {}, [], time.time(), None

    def new(self):
        at = open_page(self.page)
        self.runs += 2
        self.check(at, "load")
        return at

    def check(self, at, ctx):
        for e in at.exception:
            self.errors.setdefault((e.message or "")[:300], (ctx, " | ".join((e.stack_trace or [])[-3:])[:600]))

    def run(self, at, ctx):
        try:
            if self.tab:                                  # AppTest forgets the open tab: re-select it
                at.session_state[self.tab[0]] = self.tab[1]
            at.run()
        except Exception as ex:                           # timeouts etc. (harness problems, not app errors)
            self.harness.append(f"{ctx}: {type(ex).__name__}: {ex}"[:300])
            return
        self.runs += 1
        self.check(at, ctx)

    @staticmethod
    def categorical_values(kind, w):
        if kind in ("toggle", "checkbox"):
            return [True, False]
        if kind == "multiselect":
            return [[], list(w.options)]
        opts = list(getattr(w, "options", []) or [])
        if kind == "select_slider":
            return [opts[0], opts[-1]] if opts else []
        return opts

    @staticmethod
    def set_categorical(kind, w, v, i):
        if kind == "selectbox":
            w.select_index(i)       # NB: with format_func, AppTest can fail here -> reported as (harness)
        elif kind in ("radio", "segmented_control", "pills"):
            w.set_value(w.options[i] if isinstance(v, str) else v)
        else:
            w.set_value(v)

    @staticmethod
    def slider_values(w):
        lo, hi = w.min, w.max
        if isinstance(w.value, (tuple, list)):
            mid = lo + (hi - lo) / 2 if not isinstance(lo, int) else (lo + hi) // 2
            return [(lo, lo), (lo, hi), (hi, hi), (mid, mid)]
        return [lo, hi]

    def sweep_sliders(self, at, ctx):
        for kind in ("slider", "number_input"):
            for key in [w.key for w in getattr(at, kind)]:
                if key is None:
                    continue
                try:
                    w = at.slider(key=key) if kind == "slider" else at.number_input(key=key)
                    orig = w.value
                except Exception:
                    continue
                vals = self.slider_values(w) if kind == "slider" else [w.min if w.min is not None else 0,
                                                                         w.max if w.max is not None else 255]
                for v in vals:
                    try:
                        ww = at.slider(key=key) if kind == "slider" else at.number_input(key=key)
                        ww.set_value(v)
                    except Exception as ex:
                        self.harness.append(f"{ctx} {kind} {key}={v}: {ex}"[:200])
                        continue
                    self.run(at, f"{ctx} · {kind} {key}={v}")
                try:
                    ww = at.slider(key=key) if kind == "slider" else at.number_input(key=key)
                    ww.set_value(orig)
                    self.run(at, f"{ctx} · {kind} {key} restore")
                except Exception:
                    pass

    def sweep(self, at, ctx="base"):
        self.sweep_sliders(at, ctx)
        for kind in CATEGORICAL:
            for key in [w.key for w in getattr(at, kind)]:
                if key is None or key.startswith("mcq_") or key.startswith("final_") or key in QUIZ_KEYS:
                    continue
                try:
                    w = getattr(at, kind)(key=key)
                    orig, vals = w.value, self.categorical_values(kind, w)
                except Exception as ex:
                    self.harness.append(f"{ctx} {kind} {key}: {ex}"[:200])
                    continue
                for i, v in enumerate(vals):
                    try:
                        self.set_categorical(kind, getattr(at, kind)(key=key), v, i if kind != "select_slider" else 0)
                    except Exception as ex:
                        self.harness.append(f"{ctx} {kind} {key}={v!r}: {ex}"[:200])
                        continue
                    sub = f"{ctx} · {kind} {key}={v!r}"
                    self.run(at, sub)
                    if kind in ("selectbox", "radio", "segmented_control", "pills", "toggle"):
                        self.sweep_sliders(at, sub)        # every slider at its extremes under this setting
                try:
                    ww = getattr(at, kind)(key=key)
                    if kind == "selectbox":
                        ww.select_index(list(ww.options).index(str(orig)) if str(orig) in list(ww.options) else 0)
                    else:
                        ww.set_value(orig)
                    self.run(at, f"{ctx} · {kind} {key} restore")
                except Exception:
                    pass
        for w in list(at.radio):                            # quizzes: answer every multiple-choice question
            if w.key and (w.key.startswith("mcq_") or w.key in QUIZ_KEYS):
                for o in list(w.options)[:4]:
                    try:
                        at.radio(key=w.key).set_value(o)
                    except Exception:
                        continue
                    self.run(at, f"{ctx} · quiz {w.key}={o!r}")
        for b in list(at.button):
            try:
                b.click()
            except Exception as ex:
                self.harness.append(f"{ctx} button {b.label}: {ex}"[:200])
                continue
            self.run(at, f"{ctx} · button {b.label!r}")


def fuzz_page(page):
    fz = Fuzz(page)
    at = fz.new()
    tab_key = next((k for p, k in TAB_KEYS.items() if page.endswith(p)), None)
    if tab_key:
        for lab in [t.label for t in at.tabs]:
            fz.tab = None
            at = fz.new()
            fz.tab = (tab_key, lab)
            fz.run(at, f"tab {lab!r}")
            fz.sweep(at, f"tab {lab!r}")
    else:
        fz.sweep(at)
    return fz


def fuzz(args):
    bad = 0
    for page in page_files(args or None):
        try:
            fz = fuzz_page(page)
        except Exception:
            print(f"HARNESS CRASH {page}\n{traceback.format_exc()[-1500:]}", flush=True)
            continue
        bad += bool(fz.errors)
        print(f"{'ERR' if fz.errors else 'OK '} {page}: {fz.runs} runs, {len(fz.errors)} distinct errors, "
              f"{time.time() - fz.t0:.0f}s", flush=True)
        for msg, (ctx, stack) in fz.errors.items():
            print(f"    ✗ {msg}\n      when: {ctx}\n      at: {stack}", flush=True)
        for h in fz.harness[:6]:
            print(f"    (harness) {h}", flush=True)
    return bad


# ============================================================================================ validate
def validate(args):
    """core.fastplotly skips Plotly's property validation inside live sections (speed). That can hide INVALID values
    (e.g. a font size < 1) which would crash if validation ever ran. Here validation is forced back on."""
    project_on_path()
    import core.anim
    import core.fastplotly
    import core.live
    import core.plotting

    @contextlib.contextmanager
    def nofast():
        yield

    for mod in (core.fastplotly, core.live, core.anim, core.plotting):
        mod.fast = nofast
    bad = 0
    for page in page_files(args or None):
        at = open_page(page)
        errs = {e.message[:160].replace("\n", " ") for e in at.exception}
        for key in [w.key for w in at.slider if w.key]:
            for which in ("min", "max"):
                try:
                    s = at.slider(key=key)
                    v = getattr(s, which)
                    s.set_value((v, v) if isinstance(s.value, (tuple, list)) else v)
                    at.run()
                    errs |= {e.message[:160].replace("\n", " ") + f"   [{key}={which}]" for e in at.exception}
                except Exception:
                    pass
        bad += bool(errs)
        print(f"{'ERR' if errs else 'OK '} {page} {sorted(errs)[:3] if errs else ''}", flush=True)
    return bad


# ============================================================================================ perf / profile
def _patch_sections(profile: bool):
    """Wrap every @live.section so its run time (and optionally a cProfile) is recorded per section."""
    import cProfile

    project_on_path()
    from core import live

    times, profs, current = {}, {}, [None]
    orig = live.section

    def section(fn=None, *, key=None):
        if fn is None:
            return functools.partial(section, key=key)
        name = fn.__code__.co_filename.replace("\\", "/").rsplit("/", 1)[-1] + ":" + fn.__qualname__

        @functools.wraps(fn)
        def inner(*a, **k):
            pr = profs.setdefault(name, cProfile.Profile()) if profile else None
            t0 = time.perf_counter()
            if pr:
                pr.enable()
            try:
                return fn(*a, **k)
            finally:
                if pr:
                    pr.disable()
                times.setdefault(name, {}).setdefault(current[0], []).append((time.perf_counter() - t0) * 1000)

        return orig(inner, key=key or f"{fn.__code__.co_filename}:{fn.__qualname__}")

    live.section = section
    return times, profs, current


def perf(args):
    times, _, current = _patch_sections(profile=False)
    for page in page_files(args or None, home=False):
        at = open_page(page)
        for key in [w.key for w in at.slider if w.key]:
            try:
                s = at.slider(key=key)
                lo, hi, v0 = s.min, s.max, s.value
            except Exception:
                continue
            current[0] = key
            for f in (0.3, 0.45, 0.6, 0.75):
                if isinstance(v0, (tuple, list)):
                    a, b = lo + (hi - lo) * 0.1, lo + (hi - lo) * f
                    val = (int(round(a)), int(round(b))) if isinstance(lo, int) else (a, b)
                else:
                    val = lo + (hi - lo) * f
                    val = int(round(val)) if isinstance(lo, int) else float(val)
                try:
                    at.slider(key=key).set_value(val)
                    at.run()
                except Exception as ex:
                    print("  harness", key, ex)
                    break
            current[0] = None
    rows = []
    for name, per in times.items():
        for key, v in per.items():
            if key is not None and len(v) >= 3:
                v = sorted(v[1:])
                rows.append((v[len(v) // 2], name, key))
    print("median ms  section                                             slider")
    for med, name, key in sorted(rows, reverse=True)[:40]:
        print(f"{med:6.0f} ms  {name:55s} {key}")
    return 0


def profile(args):
    import pstats

    import numpy as np

    sort = next((a.split("=", 1)[1] for a in args if a.startswith("--sort=")), "cumulative")
    args = [a for a in args if not a.startswith("--")]
    page_sub, skey, sec = args[0], args[1], args[2]
    n = int(args[3]) if len(args) > 3 else 8
    times, profs, _ = _patch_sections(profile=True)
    page = page_files([page_sub], home=False)[0]
    at = open_page(page)
    profs.clear()
    times.clear()
    print("sliders on this page:", [w.key for w in at.slider])
    s = at.slider(key=skey)
    lo, hi = s.min, s.max
    for v in np.linspace(lo, hi, n):
        v = int(round(v)) if isinstance(lo, int) else float(round(v, 3))
        at.slider(key=skey).set_value(v)
        at.run()
        assert not at.exception, at.exception
    for k, per in times.items():
        v = sorted(x for vals in per.values() for x in vals)
        print(f"{k}: median {v[len(v) // 2]:.0f} ms over {len(v)} runs (inflated by the profiler)")
    key = next(k for k in profs if sec in k)
    buf = io.StringIO()
    pstats.Stats(profs[key], stream=buf).sort_stats(sort).print_stats(30)
    print("\n".join(line for line in buf.getvalue().splitlines() if line.strip()))
    return 0


COMMANDS = dict(smoke=smoke, fuzz=fuzz, validate=validate, perf=perf, profile=profile)

if __name__ == "__main__":
    quiet()
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(2)
    sys.exit(1 if COMMANDS[sys.argv[1]](sys.argv[2:]) else 0)
