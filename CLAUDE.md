# CLAUDE.md: Computer Vision visual notes

> Handoff for Claude Code (VS Code). Claude Code reads this file automatically at the start of every session.
> It was written on **2026-10-03** at the end of the Claude Cowork session that built the project (session link:
> https://claude.ai/code/session_015f8aoq7vhnyojcUhsa9fXV).
> Background, decisions and the full history: `docs/PROJECT_HISTORY.md`. Raw transcript and digest:
> `docs/session-history/`. Read those when you need the "why" behind something here.

---

## 1. What this is and what the user wants

An **interactive study site for computer vision**, built **with Streamlit only** (no other web framework, so it
works directly with Python CV libraries). It teaches each concept through the **mathematical intuition, shown
visually**, so students understand and remember it. It is built **topic by topic**: for each topic the user drops
the lecture-notes PDF into this folder and asks for it to be added.

The user is a PhD scholar at BITS Pilani (Hyderabad). The site is for students.

Hard requirements the user stated (keep them for every new topic):

1. **Streamlit only** for the UI. Python libraries (NumPy, OpenCV, Plotly, Pillow, pandas) for everything else.
2. **Visual explanation of the maths first.** When given the choice the user picked "visual explanation".
3. **Answer the "why?" questions** students raise, even ones the notes don't mention. Every lesson ends with
   *Questions students ask*. Content beyond the notes is welcome ("always better to learn more").
4. **Sliders update live and smoothly while dragging**, not after release. This is built: `live.slider`.
5. **Every mathematical derivation step is animated**: a smooth, step-by-step *visualization* (not a video), so a
   student sees what each step changes. This is built: `anim.step_player`.
6. **Test everything before moving on.** Every sub-page works, with no error at any stage. Run `tools/` (section 9)
   before calling anything done.

## 2. Status (2026-10-03)

| Topic | Source notes | Lessons | State |
|---|---|---|---|
| 1 · 2D Geometric Transformations | `cv_notes.pdf` | 13 (`topics/t01_geometric_transforms_2d/p01…p13`) | done, fully tested |
| 2 · Edge Detection | `CV_lec_edge.pdf` | 10 (`topics/t02_edge_detection/p01…p10`) | done, fully tested |
| 3 · (contours?) | `CV_lec_contour.pdf` (already in the folder) | – | **not started**: the likely next request |

Last full regression (2026-10-03), all passing:

- AppTest fuzz: ≈ 4,400 page runs, 0 errors.
- Plotly validation sweep: 0 invalid values.
- Browser load, light and dark, every page and tab: 0 exceptions, 0 JS errors.
- Quiz and button sweep: pass.
- 56 step animations play and scrub.
- 165 live sliders change the page while dragged.
- Layout-jump check: clean.
- Live section reruns: median 55 ms, p90 111 ms (heaviest: `p10_warping.warp_section` ≈ 200 ms).

The Desktop folder `C:\Users\sanaj\OneDrive\Desktop\ComputerVision` holds exactly this tested version.

## 3. Run it

```bat
run_windows.bat                   :: creates .venv on first run, pip-installs requirements.txt, starts the app
:: or by hand
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py              :: http://localhost:8501
```

Tested with Python 3.13, **Streamlit 1.64** (the minimum: live sliders use `st.components.v2`), Plotly 7.1 (6.0 also
checked), NumPy 2.5, OpenCV 5.0 (`opencv-python-headless`). `.streamlit/config.toml`: blue primary colour, light
and dark themes follow the OS, `runOnSave = true`, and the message cache is switched off (rule 13).

## 4. Code layout

```
app.py                      st.navigation from topics/registry.py; live.page_guard() in the sidebar; times full runs
home.py                     landing page (topic cards come from the registry)
topics/registry.py          TABLE OF CONTENTS - TOPICS = [dict(id, number, title, summary, folder, pages=[...])]
topics/_lesson_template.py  copy this to start a lesson (shows live.slider, @live.section, step_player)
topics/t01_geometric_transforms_2d/p01..p13.py
topics/t02_edge_detection/p01..p10.py
core/                       shared by all topics (≈ 2,500 lines)
  live.py        live sliders, @live.section (fragment), rerun(), dragging(), full_quality(), page_guard()
  anim.py        Step, step_player, scrubber, morph, hl/faint/wrap, AnimFigure (raw frames + fast encoder)
  plotting.py    base_figure, transform_figure, layer_traces/static_layers, arrows, grid; TraceDict
  fastplotly.py  `with fast():` skips Plotly trace validation (layout objects stay validated)
  ui.py          lesson_header/nav, callout, anim_intro, why/whys, show_matrix/tex_matrix, mcq, datauri, css
  theme.py       colour roles (see §8)
  transforms.py  matrices, homogeneous coords, classify(), DLT, fitters, RANSAC   (topic 1 maths)
  shapes.py      house, letter F, squares, circle, grid_lines, densify
  images.py      sample images, forward/inverse warp, bilinear, cv_warp, zoom
  edges.py       kernels, correlate/convolve, gradients, LoG, zero crossings, NMS, hysteresis, canny, F1  (topic 2)
  edge_ui.py     pick_image (+upload/noise), kernel/patch/signal figures, gray_heatmap, arrows
assets/images/   small CC0 / public-domain photos (credits in LICENSES.md)
tools/           test suite (see §9 and tools/README.md)
docs/            PROJECT_HISTORY.md, session-history/ (transcript, digest, earlier summary)
cv_notes.pdf, CV_lec_edge.pdf, CV_lec_contour.pdf   the user's lecture notes (source material, don't edit)
```

URL of a page = `<topic id>-<file stem without pNN_>`, e.g. `t01-affine`, `t02-canny_nms`.

## 5. Anatomy of a lesson page

Rhythm (keep it): **header → controls | figure | live maths → step-by-step derivation → intuition / pitfall /
takeaway → why-questions → quick check → prev/next.**

```python
from core import anim, live, plotting, shapes, theme, ui
from core import transforms as tf

ui.lesson_header(__file__, "What the student will be able to SEE after this page.")

@live.section                       # a Streamlit fragment: its sliders rerun only this block
def explore_section():
    c_ctrl, c_fig, c_math = st.columns([0.8, 1.6, 1.05], gap="large")
    with c_ctrl:
        theta = live.slider("θ (degrees)", -180, 180, 30, 1, key="xx_theta")   # unique key per page (prefix!)
    with c_fig:
        st.plotly_chart(plotting.transform_figure(shapes.house(), lambda t: tf.rotation(t * theta)), key="xx_fig")
    with c_math:
        ui.show_matrix("H", tf.rotation(theta), 3, ui.column_highlight())

explore_section()                   # call it right after defining it

ui.anim_intro("Step by step: …")    # then a step player (see §6) for every derivation
ui.callout("intuition", "…"); ui.callout("pitfall", "…"); ui.callout("takeaway", "…")
ui.whys([("Why …?", "Because …"), ...])
ui.mcq("xx_q1", "Question?", ["A", "B", "C"], 0, "Why A.")
ui.lesson_nav(__file__)
```

To register a page, add `dict(file="pNN_name.py", title="…", icon=":material/<name>:")` to the topic's `pages`
in `topics/registry.py`. Sidebar, home cards and prev/next update automatically. Icons are Google Material
Symbols names.

## 6. Core API cheat-sheet

**live** (`core/live.py`)
- `live.slider(label, min, max, value, step, format=None, key=…, help=…)`: drop-in for `st.slider`. Single
  values and `(lo, hi)` ranges. Streams values while dragging, throttled adaptively to `40 + 1.25 × section ms`.
  The value lives in `st.session_state[key]`; assigning it from Python moves the slider (a revision counter
  handles it).
- `live.select_slider(label, options, value, format_func, key)`: single value only.
- `@live.section`: wraps `st.fragment`, turns on `fast()` mode, and records the run time (the throttle uses it).
- `live.rerun()`: use it instead of `st.rerun(scope="fragment")`, which crashes when the section runs inside a
  full-page run.
- `live.dragging()`: True while the slider thumb is held. Step players, scrubbers and `transform_figure`
  automatically drop their in-between frames then; release triggers a full-quality run.
- `@live.full_quality()`: put it under `@st.cache_resource` on any **cached** figure builder, so a coarse
  mid-drag copy never gets cached.
- Under AppTest the sliders fall back to `st.slider` (detected via session id `"test session id"`).

**anim** (`core/anim.py`)
- `anim.Step(title, eq="", note="", traces=[...], shapes=None, between=None)`: the picture AT that step. Every
  step must have the **same trace structure** (same number of traces and points per trace) so states can be
  interpolated. `between=lambda t: traces` overrides the straight blend, e.g. rotate along an arc.
  `anim.morph(build, a, b)` blends *parameters*.
- `anim.step_player(base_fig, steps, n_sub=14, hold=18, frame_ms=40, top_margin=None, start=0, wrap_at=105)`:
  ▶ Play, ⏮, one button per step, and a scrub slider. The caption shows step title, equation and note
  (`anim.hl(text)` highlights what changed). Use `wrap_at≈70–80` in narrow columns.
- `anim.scrubber(base, values, make_state, active=None, label="value", fmt="{:.2f}")`, where
  `make_state(v) -> (traces, caption_html)`. Sweeps one quantity live inside the figure.
- NaN in x/y marks a gap. Gaps that open or close jump at half time. uint8 arrays stay uint8 (images).
- `AnimFigure` stores frames and the slider/menus as raw dicts (no `go.Frame`) and encodes numpy arrays as base64
  typed arrays (`_encode`, the same format as plotly.py).

**plotting** (`core/plotting.py`)
- `base_figure(extent, height, title, legend)`: one-pass layout; y-up, equal aspect.
- `transform_figure(parts, H or t->H, extent, grid, basis, animate, n_frames=24, …)`: original vs transformed plus
  a Play/t-slider.
- `layer_traces(parts, H, grid_extent, grid_step, n, …)` gives the dynamic layer for step players with fixed
  point counts. `static_layers(parts)` gives the faint "before" background.
- `arrows_trace(starts, ends, color, …)` puts many arrows in one trace. `mapped(H, pts)` returns NaN behind the
  camera or at infinity.
- Helpers return `TraceDict` (a plain dict trace, much faster than `go.Scatter`).

**ui** (`core/ui.py`): `lesson_header`, `lesson_nav`, `callout(kind)` with kinds intuition/pitfall/takeaway/note,
`anim_intro`, `why`/`whys`, `show_matrix`/`tex_matrix`/`tex_vec`/`column_highlight`, `invariants_table`,
`preserved_badges`, `mcq(key, q, options, answer_idx, explanation, scored=False)` (its radio key is `mcq_<key>`),
`datauri(img)`.

**edge_ui**: `pick_image(prefix, default, noise, noise_default, upload)` returns `(noisy, clean, name)`. Also
`kernel_figure`, `patch_figure`, `signal_figure([dict(x, y, name, color, …)])`, `gray_heatmap`, `arrows_trace`,
`show_row([(img, caption), …])`.

## 7. Rules learned the hard way (do not regress)

1. **Every slider is `live.slider` inside an `@live.section` function.** Don't use plain `st.slider` (it updates
   only on release).
2. **Images go through `ui.datauri(img)`** in `st.image`. Streamlit's media-file URLs race during fast live reruns
   and show "Image source error".
3. **Keep the layout height stable while a slider moves.** No elements that appear or disappear with the slider
   value. Show a same-size placeholder instead: alert ↔ alert, figure ↔ static figure of the same height, crops
   padded to a constant size. `tools/browser_checks.py layout --all` checks this.
4. **No invalid Plotly values.** `fast()` skips validation, so an invalid value (e.g. `textfont.size < 1`) only
   crashes when validation runs. Run `tools/apptest_checks.py validate`.
5. **Plotly speed:**
   - Build traces as dicts (`TraceDict` / plain dicts) in hot paths.
   - Build layouts in one `go.Figure(layout=…)` / one `update_layout(**dict)`, since every `update_*` call costs
     milliseconds.
   - Never use `go.Frame`; use `AnimFigure._raw_frames`.
   - Layout objects must stay validated: skipping their validation broke `make_subplots` axis domains.
6. **Cached figure builders** (`@st.cache_resource`) that call `step_player`/`scrubber`/`transform_figure` need
   `@live.full_quality()` (rule from `live.dragging()`).
7. **Never `st.rerun(scope="fragment")`.** Use `live.rerun()`.
8. **Widget keys are unique per page and prefixed** (`af_`, `pj_`, `cn_` …, see §8). Component keys must not
   contain `"__"` (live sliders use `f"{key}-live"`).
9. **Streamlit re-imports edited modules while running.** Module-level state that must survive that lives on a
   stable module (`fastplotly` keeps its flag on `plotly.basedatatypes._cv_fast_state`).
10. **Plotly animation captions** are layout annotations, so frames use `redraw=True`. `page_guard` silences the
    harmless "animation interrupted" promise rejection.
11. **Tabs:** `st.tabs(..., key=…, on_change="rerun")` with lazy content via the tab's `.open` (p02 `h_tabs`, p08
    `pj_tabs`).
12. **Image coordinates are y-down**, kernels are applied by **correlation** like OpenCV (`edges.correlate`), and
    `atan2`, never `atan(Iy/Ix)`. The pages say so.
13. **Keep `[global] minCachedMessageSize = 1e12` in `.streamlit/config.toml`.** Streamlit's message cache can
    race with rapid live reruns ("Cached ForwardMsg MISS" → the page disconnects).
14. Show claims as **measured** where possible (e.g. Sobel vs Prewitt direction error, detector F1 on known edges).

## 8. Conventions

- **Notation follows the lecture notes:** affine `a, b, c, d, tₓ, t_y`; homography `h₁₁ … h₃₃`; Canny `T_low/T_high`.
- **Colour roles** (`core/theme.py`, consistent across all figures):
  - grey dashed = original / before;
  - blue = transformed / after;
  - orange = e₁ / 1st column;
  - green = e₂ / 2nd column;
  - amber = highlighted term / special points;
  - magenta = perspective / infinity;
  - red = outliers / discarded;
  - diverging blue–grey–red = signed responses.
- **Widget key prefixes:**
  - Topic 1: p01 `ov_`, p02 `h_`/`l_`, p03 `tr_`, p04 `sc_`, p05 `rg_`, p06 `sm_`, p07 `af_`/`af3_`, p08 `pj_`,
    p09 `cp_`, p10 `wp_`, p11 `es_`, p12 `pg_`, p13 `dr_`.
  - Topic 2: p01 `we_`, p02 `dv_`, p03 `sb_`, p04 `lp_`, p05 `gl_`, p06 `cn_`, p07 `hy_`, p08 `cc_`, p09 `le_`,
    p10 `ch_`/`wd_`.
  - Pick a new prefix per new page.
- Lessons are long single files (150–700 lines) with sections as `@live.section` functions and cached heavy
  builders. Write comments and copy for students: plain language, short sentences.

## 9. Testing: definition of done

Zero exceptions, zero JS errors, every animation plays, every slider is live, no layout jumps. Commands (see
`tools/README.md`; dev setup: `pip install -r requirements-dev.txt` and `python -m playwright install chromium`):

```bat
python tools\apptest_checks.py smoke          :: every page loads (seconds)
python tools\apptest_checks.py fuzz t03_      :: every control of the new topic's pages
python tools\apptest_checks.py validate
python tools\run_all.py                       :: everything incl. browser checks (starts its own server)
python tools\browser_checks.py drag t01-composition 1   :: one slider, mid-drag vs after release
```

## 10. Adding the next topic (the workflow the user liked)

1. Read the new PDF (e.g. `CV_lec_contour.pdf`) fully. Propose a lesson list that follows the notes' teaching
   order, adds the extra material a student needs, and plans the "why?" questions. Confirm with the user
   (short multiple-choice questions worked well).
2. Put reusable maths in `core/` (new module, e.g. `core/contours.py`, plus a `*_ui.py` for shared widgets if
   needed). Create `topics/t03_<name>/pNN_*.py` from the template and register the topic in `topics/registry.py`.
3. Each derivation → `anim.step_player`. Each parameter → `live.slider` in a `@live.section`. End with `ui.whys`
   and `ui.mcq`. Last page = recap & quiz (see the two existing recap pages).
4. Run the tests (§9) until clean. Then update `README.md` (lesson table) and `docs/PROJECT_HISTORY.md`.

## 11. More context

- `docs/PROJECT_HISTORY.md`: timeline, the user's requests (turn 3 verbatim), decisions, bug and fix log,
  performance work, test results.
- `docs/session-history/`: raw transcript `.jsonl` of the last Cowork session (git-ignored, has personal
  details), a readable digest of it, and the summary of the earlier conversation (Oct 1–2, whose raw transcript
  no longer exists).
- `tools/transcript_digest.py <file.jsonl>`: makes the same kind of digest from any Claude Code session
  (`~/.claude/projects/…/*.jsonl`).
- The claude.ai Project "Computer Vision" has a doc `claude/course-structure.md` with the same architecture notes
  (Claude Code cannot see it; everything in it is here).

## 12. Working with this user

- Wants things **working and tested** before new content. Report what was verified, briefly.
- Likes short clarifying choices before big work (topic scope, visual vs formal); otherwise just proceed.
- Prefers concise answers. Deliverables live in this folder (it is the Desktop project folder).
