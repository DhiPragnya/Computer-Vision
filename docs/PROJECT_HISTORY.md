# Project history: Computer Vision visual notes

How the project came to be, what was decided and why, what broke and how it was fixed. Read `CLAUDE.md` first
for the working rules. This file is the background.

- **Built in:** Claude Cowork (cloud session with the Desktop folder connected), 2026-10-01 → 2026-10-03.
- **Session link:** https://claude.ai/code/session_015f8aoq7vhnyojcUhsa9fXV
- **Raw transcript** (only the part after the last context compaction, 2026-10-03 01:26 UTC onward):
  `docs/session-history/2026-10-03_cowork_transcript.jsonl`. A readable digest of it is next to it.
- The earlier conversation (Oct 1–2) survives only as the summary written at compaction time:
  `docs/session-history/2026-10-01_to_03_earlier_conversation_summary.md`.

---

## 1. Timeline

| When | Request (user) | What was built / done |
|---|---|---|
| 2026-10-01 | Build an interactive web page (Streamlit only) to understand computer-vision concepts and the maths behind them visually; first topic **2D geometric transformations**; save it in Desktop\ComputerVision. Clarifying choice: **"Use standard syllabus (Recommended)"** | Project structure (`app.py`, registry, `core/`), shared maths/plotting/UI, 13 lessons, AppTest + browser tests, README, `run_windows.bat`; written to the Desktop folder |
| 2026-10-01/02 | Align topic 1 with the lecture notes `cv_notes.pdf`; answer the "why?" questions students ask | Lessons re-aligned with the notes (polar/angle-addition derivation of rotation, 3-point affine solve with det M ≠ 0, "why four points", step-by-step DLT + SVD, p′ = TRSp composition …); *Questions students ask* on every page |
| 2026-10-02 | Add the **edge detection** lecture (`CV_lec_edge.pdf`); keep the extra content; visual math intuition; address "why" questions. Clarifying choice: **"Visual explanation (Recommended)"**. Learned detectors explained with offline interactive imitations; the user chose **not** to download/run a real HED model | `core/edges.py`, `core/edge_ui.py`, 10 lessons, maths verified (e.g. ∇²(G∗I) = (∇²G)∗I numerically, cv2.Canny comparison), synced to the Desktop folder (2026-10-02 20:38) |
| 2026-10-02 → 03 | **Turn 3**: test everything, smooth live sliders, smooth step animations for every derivation step (verbatim text below) | Live-slider component + fragments, speed work, step-animation engine, animations on every derivation in both topics, widget fuzzer + browser test suite, many fixes (§4) |
| 2026-10-03 01:26 | (context compaction; work continued from its summary) | Recap-page crash fix, full regression, second performance pass, drag-aware animations, layout-jump fixes |
| 2026-10-03 ~04:00 | (computer offline) | Final version delivered as `ComputerVision.zip` in the chat |
| 2026-10-03 09:04 | "Try reaching the computer now and finish and complete the changes" | Checked that the Desktop folder had no user edits, copied the update in (41 changed files), verified all 54 files by hash |
| 2026-10-03 09:41 | Prepare a README so work can continue in **VS Code + Claude Code**, including the session file for context | `CLAUDE.md`, this file, `docs/session-history/`, `tools/` test suite, `requirements-dev.txt` |
| 2026-10-03 (Claude Code, VS Code) | Continue in Claude Code: summarise, check the environment, smoke test, compare CLAUDE.md with the code | Smoke test 24/24 pages OK; CLAUDE.md mismatches listed (all documentation-only) |
| 2026-10-03 (Claude Code) | **Topic 3** from `CV_lec_contour.pdf`. Clarifying choices: **11 lessons as proposed**, **all extras**, title **"Contours, Polygons & Hough"** | `core/contours.py`, `core/hough.py`, `core/contour_ui.py`, 11 lessons in `topics/t03_contours_hough/`, README + this file updated (§7) |

### Turn 3, verbatim

> Before moving to further lectures, there are few pointers I wanted to mention
> * Test the application such that everything is fine, every sub pages are woking fine, no error at any stage using
>   the application, etc
> * When i make any interactions, such as sliders, the changes are made after a delay. rather I want a smooth and
>   online change, so that I can see the effect when I increase and decrease the values.
> * Also for understanding the concepts, I feel it would be better to have animations to show the step changes in
>   every mathematical derivative steps. The animation should not be a video, rather than a visualization only, as
>   created earlier, but it should be smooth and it has to work work each mathematical step, as the student able
>   to know the importance of change when a mathematical step is carried out.

### Handoff request, verbatim

> Give me the complete context of this project integrations and progress made. Next I want to continue using the
> VSCode ClaudeCode extension. So, prepare a readme file, so that I an upload in VSCode Clause Code extension and
> continue working from there. If possible mention the file for this session in readme file, so that it will create
> more robust context, with history and many required details.

(The turn 1 and turn 2 requests above are paraphrased: their raw transcripts were lost when the context was
compacted.)

## 2. Lesson inventory

See the two tables in `README.md` (what each lesson lets you play with). Per page:

| Page | Live sliders | Step players / scrubbers | Lines |
|---|---|---|---|
| t01 p01 overview | 1 | 1 | 213 |
| t01 p02 homogeneous (3 tabs) | 12 | 4 | 510 |
| t01 p03 translation | 2 | 1 | 163 |
| t01 p04 scaling | 2 | 2 | 227 |
| t01 p05 rigid | 9 | 3 | 423 |
| t01 p06 similarity | 4 | 1 | 172 |
| t01 p07 affine | 18 | 4 | 526 |
| t01 p08 projective (5 tabs) | 9 | 4 | 719 |
| t01 p09 composition | 7 | 3 | 291 |
| t01 p10 warping | 9 | 3 | 406 |
| t01 p11 estimation | 5 | 1 | 308 |
| t01 p12 playground | 0 | 1 | 187 |
| t01 p13 recap & quiz | 0 | 0 | 175 |
| t02 p01 what is an edge | 11 | 2 | 334 |
| t02 p02 derivatives | 8 | 5 | 401 |
| t02 p03 Sobel & Prewitt | 6 | 2 | 327 |
| t02 p04 Laplacian | 7 | 2 | 246 |
| t02 p05 Gaussian & LoG | 6 | 2 | 334 |
| t02 p06 Canny NMS | 8 | 1 | 326 |
| t02 p07 Canny hysteresis | 4 | 1 | 266 |
| t02 p08 Canny complete | 6 | 2 | 299 |
| t02 p09 learned edges | 9 | 1 | 479 |
| t02 p10 recap & quiz | 2 | 0 | 221 |
| t03 p01 edges → contours | 1 | 1 | 189 |
| t03 p02 connectivity & boundary | 1 | 3 | 337 |
| t03 p03 tracing & hierarchy | 1 | 3 | 343 |
| t03 p04 length & area | 3 | 3 | 351 |
| t03 p05 RDP | 2 | 3 | 475 |
| t03 p06 ε & measurements | 3 | 2 | 349 |
| t03 p07 Hough (m, c) | 7 | 2 | 309 |
| t03 p08 Hough (ρ, θ) | 8 | 3 | 390 |
| t03 p09 Hough circles | 8 | 3 | 342 |
| t03 p10 general Hough | 6 | 2 | 331 |
| t03 p11 recap & quiz | 1 | 0 | 240 |

Animated derivations include, among others:
- **Topic 1:** lifting a point into homogeneous space and its ray; w̃ and line-angle scrubbers; dividing by w in
  3D; an h₃₁ scrubber; DLT rows → SVD heat-map steps; the TRS chain, order and undo players; forward / inverse /
  bilinear warping in 3D; a RANSAC replay; affine decomposition; the 3-point solve.
- **Topic 2:** finite difference → derivative; the kernel sliding along a signal; multiply-add; noise gain vs ω;
  Sobel = outer product; Pascal → Gaussian; second difference; zero-sum kernels; the LoG derivation; the NMS
  decision; hysteresis flooding; the Canny stages; Canny's three criteria; receptive fields.

## 3. Key engineering decisions (and why)

1. **Live sliders = a custom bidirectional component** (`st.components.v2.component`, inline JS/CSS in
   `core/live.py`). It sends values *while dragging*, which Streamlit's own slider never does. Throttle:
   `40 + 1.25 × (EMA of the section's run time)`, clamped to 50–900 ms, so a slow block is never flooded.
   Programmatic value changes bump a revision counter so the browser accepts them. The look matches Streamlit
   through its theme CSS variables (`--st-primary-color` etc.).
2. **`@live.section` = `st.fragment`**, so a slider reruns only its own block, not the page.
3. **Step animations are Plotly animations computed in Python.** Every state is real data; the browser
   interpolates nothing. The engine (`core/anim.py`) builds key states, eases between them (`n_sub` in-between
   frames), puts the caption (title, equation with the changed term highlighted, one-line "why it matters") in a
   layout annotation, and puts all buttons in one row (▶ Play, ⏮, step 1…n) plus a scrub slider.
4. **Speed came from avoiding Plotly's object model in hot paths:**
   - dict traces instead of `go.Scatter` (validation ≈ 0.4 ms per trace);
   - raw frame dicts instead of `go.Frame` (deep copies);
   - a lean base64 typed-array encoder instead of `convert_to_base64` (which went through narwhals for every
     array);
   - raw slider/menu dicts;
   - single-pass `base_figure` (11 ms → 1.3 ms);
   - a cached homogeneous grid with one matrix product per state;
   - a faster `mapped()`.
   `fastplotly.fast()` turns trace validation off only inside sections.
5. **Mid-drag frames are coarse.** While a thumb is held (`live.dragging()`), players keep only key states,
   scrubbers keep every 4th value, and `transform_figure` keeps 2 frames. Release sends one more message (`d: 0`)
   and the page rebuilds at full quality. Cached builders are protected with `@live.full_quality()`.
6. **Images as data URIs** (`ui.datauri`): no media-file races during rapid reruns.
7. **AppTest compatibility.** `live.slider` falls back to `st.slider` under AppTest, so the fuzzer can drive every
   control.

## 4. Bug and fix log (what testing found)

| Symptom | Cause | Fix |
|---|---|---|
| Slider changes appeared only after a delay | `st.slider` commits on release, and every change reran the whole page | live component + fragments + speed work (§3) |
| Component key rejected | keys may not contain `"__"` | `f"{key}-live"` |
| "?" help icon always visible; bubble clipped at the edges | CSS | `.ls-help[hidden]{display:none}`; bubble clamped with translateX |
| Player buttons overlapping; captions overflowing; legend over the caption | layout | one button row, `wrap()` + automatic top margin, legend top-right |
| `PAGEERROR undefined` in the console | Plotly rejects a promise when an animation is interrupted | `live.page_guard()` swallows that rejection |
| "Image source error" while dragging | Streamlit media-file race | `ui.datauri` everywhere |
| Step players took ~450 ms | Plotly validation, `go.Frame` deep copies | dict traces + raw frames → ~54 ms |
| Subplot axes lost domain/anchor | fast mode also skipped layout validation | layout types always validated |
| `ValueError: Invalid value` in frames | live `Annotation` objects inside raw frames | annotations converted to dicts |
| Bars missing at key frames | NaN × 0 in interpolation | exact key states; NaN gaps jump at half time |
| AppTest showed no sliders | `Runtime.exists()` is true inside AppTest | detect the AppTest session id |
| p07 flood figure crashed; side branch disconnected | tuple unpacking; coordinates | fixed |
| p10 recap `UnboundLocalError` | a function name was reassigned inside a function | module-level cached helper |
| **Recap drills crashed on answer** | `st.rerun(scope="fragment")` during a full run | `live.rerun()` picks the legal scope |
| **NMS page crashed after a code edit while running** | Streamlit re-imported `fastplotly` and its fast flag was lost; `textfont.size = 0.1` (< 1) then failed validation | flag stored on `plotly.basedatatypes`; text sizes start at 1 |
| Page height jumped while dragging (affine 3-point solve at det M = 0; Gaussian kernel; attention crop at the border; Canny challenge) | elements appearing/disappearing | constant-size placeholders, 9×9 kernel crop, reflect-padded crop, always-on status line |
| **Page disconnected once during a long drag test** ("Cached ForwardMsg MISS", Websocket fatal error) | Streamlit's browser-side message cache: with many quick reruns the browser can evict a cached figure just before the server refers to it | `.streamlit/config.toml` → `[global] minCachedMessageSize = 1e12` (cache off, so no references are sent; free on localhost) |
| Selectbox sweep "None is not in list" (fuzzer only) | AppTest + `format_func` | harness limitation; p11 swept by raw values instead: 225 runs clean |

## 5. Final test results (2026-10-03)

- **AppTest fuzz** (`tools/apptest_checks.py fuzz`): every page, ≈ 4,400 runs. **0 errors.**
- **Validation sweep** (Plotly validation forced on, every slider at min/max): **0 invalid values.**
- **Browser load**, light and dark, every page and every tab: **0 exceptions, 0 JS errors.**
- **Quiz/drill/button sweep:** all pass. Tested on Plotly 6.0.0 too: pages render and animations play.
- **Animations:** 56 animated figures play, step and scrub.
- **Live sliders:** 165 tested.
  - 151 show 5–6 different page states during a 1.3 s drag.
  - Range sliders show 1 under that test (it can't grab their thumbs) and were checked separately: live.
  - The rest have only a few values.
- **Drag check:** the composition page shows 11 states in 11 samples while dragging; 4 frames per player
  mid-drag, 43 after release.
- **Layout-jump check:** clean.
- **Live section rerun times** in the browser, on a 2-core cloud box while Chromium was rendering:

  |  | median | p90 | p99 |
  |---|---|---|---|
  | before the second speed pass | 90 ms | 249 ms | 460 ms |
  | after | **55 ms** | **111 ms** | **233 ms** |

  Composition 424 → 105 ms, affine 354 → < 110 ms. Heaviest now: `p10_warping.warp_section` ≈ 200 ms (image
  warps + images).

**Re-run with the `tools/` suite at handoff (2026-10-03):**
- smoke, validate, load (light and dark): pass.
- quiz: pass.
- anim: 56/56 plots OK.
- One page disconnected during the concurrent anim + quiz run (the message-cache race in §4). It was fixed in
  `config.toml`. Re-tested under the same concurrent load: no errors, every live slider live.

## 6. Open items / ideas

- **Topic 4**: the next lecture PDF, same workflow (CLAUDE.md §10).
- CLAUDE.md §2/§4/§8 still describe two topics; add topic 3 (folder, core modules, key prefixes below) when convenient.
- `p10_warping.warp_section` is the slowest section (~200 ms in the sandbox). If it feels slow, shrink the sample
  image or cache per parameter.
- The README lesson tables must be extended with each new topic.

## 7. Topic 3 · Contours, Polygons & Hough (Claude Code, 2026-10-03)

Built in VS Code + Claude Code from `CV_lec_contour.pdf` (26 pages, 42 short sections, text and equations only).
The user confirmed the proposed plan: 11 lessons in the notes' order, all extras, title "Contours, Polygons & Hough".

**New shared code.**
- `core/contours.py`: scenes, binarize, N₄/N₈, `label_layers` (flood fill written out), `boundary` (= Ω minus its
  erosion), `moore_trace` (Jacob's stopping rule, optional naive stop), Freeman chain code, perimeter, shoelace,
  `line_coeffs`, `point_line_distance`, `rdp` (iterative, with a trace of every segment), `rdp_closed`,
  `optimal_vertices` (shortest path through allowed shortcuts), `max_deviation`, `measure`, `classify`,
  `find_contours` (OpenCV wrapper with hierarchy re-indexing), `depths`, `disk`, `staircase`, display helpers.
- `core/hough.py`: synthetic edge points, (m, c) and (ρ, θ) accumulators, `find_peaks` (NMS, θ wrap-around),
  `cv_hough_lines`, `ransac_line`, fixed-r / 3D / gradient circle accumulators (one vote per point per bin),
  `cv_hough_circles`, drawing helpers.
- `core/contour_ui.py`: scene picker (selectbox + live threshold + invert), pixel-grid figures, `grow` (fixed-size
  polylines for step animations), accumulator heat traces, `two_panel` (image space | parameter space), `image_axes`.

**Widget key prefixes (topic 3).** p01 `ec_`, p02 `px_`, p03 `ct_`, p04 `pa_`, p05 `rd_`, p06 `ms_`, p07 `hm_`,
p08 `hn_`, p09 `hc_`, p10 `hg_`, p11 `rq_` / `sd_` (drill radio `sd_choice` added to `QUIZ_KEYS` in
`tools/apptest_checks.py`).

**Checked against OpenCV / measured (the pages quote these).**
- Moore tracing visits exactly the pixels of `cv2.findContours` (every object of the shapes scene); our tracer walks
  clockwise on screen, OpenCV's object contours anti-clockwise (negative shoelace area in y-down coordinates), holes
  the other way.
- `rdp` / `rdp_closed` give the same vertex counts as `cv2.approxPolyDP` (open and closed, several ε).
- RDP is not optimal: e.g. 36 vertices where the optimum is 25 (random walk, ε = 0.2).
- RDP cost: ≈ n log n distance evaluations on typical curves, n²/2 on a shrinking zigzag (n = 50 … 1600).
- Pick's theorem holds exactly for contours: cv2.contourArea = pixels − B/2 − 1.
- Digital circles: crack length → 4/π ≈ 1.27 × 2πr, 8-connected chain ≈ 1.05 × 2πr, circularity ≈ 0.89 (r = 80).
- Hough peaks match `cv2.HoughLines` and `cv2.HoughCircles` on the same data; a vertical line is found at θ = 0°.

**Problems found while building (and fixes).**

| Symptom | Cause | Fix |
|---|---|---|
| A running server kept using old `core/` code after edits (stale `rdp`, missing `image_axes`) | Streamlit reloads the page scripts but not always edited `core` modules | Restart the server after changing `core/` (the AppTest checks always start fresh) |
| `RecursionError` on the RDP cost chart (browser only) | worst-case zigzag recursed n levels deep | `rdp` uses an explicit stack (same visiting order) |
| Step-player captions ran off narrow figures | `wrap_at` default 105 in half-width columns | `wrap_at` ≈ 48-62 for half / third-width columns |
| Caption overlapped subplot titles in image/parameter-space animations | both are layout annotations at the top | panel names moved into the x-axis titles |
| Outlines invisible in dark mode | fixed dark grey `#3a3a38` | mid grey `#7d7b76` |
| Layout jumps: p06 threshold (object slider vanished), p08 line count (table height) | elements appearing / resizing | disabled slider instead of none; fixed table height |
| segmented_control with `format_func` would not be driven by the fuzzer | AppTest harness | string options ("N₄", "N₈") |

**Tests (2026-10-03, topic 3).**
- AppTest smoke: all 35 pages OK. Fuzz `t03_`: 940 runs, 0 errors (p06/p08 re-fuzzed after the last fixes: 0).
- Validate `t03_` (Plotly validation on, every slider at min/max): 0 invalid values.
- Browser load, light and dark: 11 pages, 0 exceptions, 0 JS errors.
- Browser anim: 29 animated figures (step players and scrubbers) play, step and scrub; 47 live sliders change the
  page while dragged (5-6 states for most; 2-3 for sliders with few values or threshold-like effects; the radius
  range slider checked with `range`: live).
- Quiz/button sweep: pass. Layout-jump check: clean after the two fixes.
- Live section reruns (AppTest `perf`, median per slider): all ≤ 127 ms; heaviest `p09 explore_section`.
