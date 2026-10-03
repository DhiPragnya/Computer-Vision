# Computer Vision - Visual & Interactive Notes

An interactive study site, built only with **Streamlit** and Python libraries (NumPy, OpenCV, Plotly),
for understanding computer-vision concepts through the geometry and intuition behind the maths.
Every lesson ends with **"Questions students ask"** - the *why?* questions that come up in class even when the
notes don't ask them.

## How the pages behave

* **Live sliders.** Every slider updates the picture *while you drag it* - no waiting for the mouse to be
  released. Only the block a slider belongs to is recomputed, and the update rate adapts to how heavy that
  block is, so the page stays responsive on slower laptops too. While you hold a slider, the step-by-step
  figures show their key steps only; the smooth in-between frames are added the moment you let go.
* **Step-by-step derivations.** Each mathematical derivation is an animated figure (not a video): press
  ▶ Play to run all steps, a numbered button to play one step, or drag the step slider to scrub back and
  forth. Between two steps the picture *morphs smoothly*, so you can see what that one algebraic step does to
  the geometry or the signal. The equation and a one-line explanation for the current step sit on top.
* **Scrubbers.** Where a single quantity matters (w̃, h₃₁, σ, the iteration k, ...), an animated scrubber
  sweeps it so you can watch its effect continuously.

## Topic 1 - 2D Geometric Transformations (13 lessons, from `cv_notes.pdf`)

| # | Lesson | What you can play with |
|---|---|---|
| 1 | The big picture | the five families side by side, nested-set diagram, invariants table |
| 2 | Homogeneous coordinates | 2D translation as a 3D shear, points as rays, lines & cross products, points at infinity |
| 3 | Translation | displacement arrows, live matrix-vector product |
| 4 | Scaling | derive the matrix from x′ = sₓx, y′ = s_y y; uniform / non-uniform; negative = mirror |
| 5 | Rotation & rigid motion | polar-coordinate / angle-addition derivation, unit-circle view, pivot rotation, mirror flips |
| 6 | Similarity | measured angles / ratios / areas, the complex-number view |
| 7 | Affine | three ways to control A, det = area, SVD, **solving a, b, tₓ / c, d, t_y from 3 points (det M ≠ 0)** |
| 8 | Projective (homography) | perspective row and vanishing points, **why four points**, step-by-step DLT + SVD, rectification, cross-ratio |
| 9 | Composition & inverses | the p′ = T R S p chain, order matters, undoing a chain |
| 10 | Warping images | forward vs inverse mapping, holes, nearest / bilinear / bicubic |
| 11 | Estimating from matches | least squares vs RANSAC with noise and outliers |
| 12 | Matrix playground | type any 3×3 matrix → class, invariants, decomposition, image warp |
| 13 | Recap & quiz | cheat sheet, lecture-notes summary, "name that transform" drill |

Notation follows the lecture notes: affine entries a, b, c, d, tₓ, t_y; homography entries h₁₁ … h₃₃.

## Topic 2 - Edge Detection (10 lessons, from `CV_lec_edge.pdf`)

| # | Lesson | What you can play with |
|---|---|---|
| 1 | What is an edge? | an image as a 3D landscape, a step and its derivative, gradient arrows ⟂ to the edge |
| 2 | Derivatives on pixels | forward vs central differences, a kernel sliding along a signal, 3×3 products, why noise explodes (frequency gain) |
| 3 | Sobel & Prewitt | Sobel = (1,2,1)ᵀ × (−1,0,1), Pascal → Gaussian, measured Prewitt vs Sobel, limits of first-order operators |
| 4 | Second derivative & Laplacian | peak ↔ zero crossing, deriving [1 −2 1] and the 4/8-neighbour kernels, noise sensitivity |
| 5 | Gaussian & LoG | Gaussian kernel, ∇²(G∗I) = (∇²G)∗I checked numerically, the "Mexican hat", step-by-step LoG derivation, σ as scale, DoG |
| 6 | Canny I: non-maximum suppression | why magnitude is thick, NMS up close (p⁺/p⁻, quantised vs interpolated), direction wheel |
| 7 | Canny II: thresholds & hysteresis | strong/weak classes, hysteresis growing step by step, why two thresholds beat one, graph view |
| 8 | Canny III: why it works & comparison | full pipeline vs cv2.Canny, Canny's three criteria simulated, Sobel vs LoG vs Canny scored on known edges |
| 9 | Learned edge detectors | where classical fails, timeline, a hand-built "network" layer, receptive fields, HED-style fusion with learned weights, attention toy |
| 10 | Recap & quiz | key equations, match-the-hidden-Canny challenge, spot-the-detector drill |

The deep-learning part is explained with interactive imitations of the ideas (no trained network is downloaded).

## Topic 3 - Contours, Polygons & Hough (11 lessons, from `CV_lec_contour.pdf`)

| # | Lesson | What you can play with |
|---|---|---|
| 1 | From edges to contours | threshold → binary image → contours; a pile of edge pixels ordered into a chain p₀ → p₁ → … |
| 2 | Connectivity & boundary pixels | N₄ vs N₈ flood-fill labelling, ∂Ω = Ω minus its erosion, the boundary test pixel by pixel, the connectivity paradox |
| 3 | Contour tracing & hierarchy | one Moore-tracing move in detail, the whole trace and its stopping rule, OpenCV retrieval modes, hierarchy tree, chain codes |
| 4 | Contour length & area | shoelace formula with a movable origin, one term = a signed triangle (cross product), the triangle fan, staircase perimeter (measured 1.05× / 1.27×), Pick's theorem |
| 5 | Polygonal approximation (RDP) | deriving A x + B y + C = 0 and the point-line distance, live ε with ε-bands, the recursion replayed segment by segment + recursion tree, RDP vs optimal, measured cost |
| 6 | ε trade-off & shape measurements | ε sweep (400 → 4 rectangle), circularity derived from same-perimeter shapes, measurement dashboard + rule-based classifier, measured invariance |
| 7 | Hough idea: voting for (m, c) | broken line vs contours, point → line in (m, c), live accumulator with outliers, why vertical lines break it |
| 8 | Hough lines in (ρ, θ) | normal-form derivation, a point's sinusoid, voting replay, live accumulator vs cv2.HoughLines, real images, measured bin-size trade-off |
| 9 | Hough circles | point → circle of centres, slicing the (a, b, r) cone, gradient voting (x ± r cos θ), coins vs cv2.HoughCircles |
| 10 | General Hough, gaps & limits | two fragments → one peak, robustness lab (gap, noise, clutter, bins, RANSAC), cost of more parameters, both pipelines step by step |
| 11 | Recap & quiz | key equations, vertex-budget challenge, "contours or Hough?" drill, concept check |

## Run it

**Windows (easiest):** double-click `run_windows.bat`. The first run creates a private Python
environment in `.venv` and installs the packages; afterwards it starts in a few seconds.

**Any OS, by hand:**

```bash
cd ComputerVision
python -m venv .venv
.venv\Scripts\activate          # macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. Light/dark mode follows your system (change it in ⋮ → Settings).
Needs **Streamlit 1.64 or newer** (the live sliders use its bidirectional custom components) and Plotly 6+.
Tested with Python 3.13, Streamlit 1.64, NumPy 2.5, OpenCV 5.0, Plotly 7.1.

If you already installed an older version, update once with `pip install -U -r requirements.txt` (or delete the `.venv` folder and run `run_windows.bat` again).

## Project layout

```
ComputerVision/
├── app.py                    entry point: builds the sidebar from topics/registry.py
├── home.py                   landing page (lists every topic automatically)
├── requirements.txt · requirements-dev.txt · run_windows.bat · .streamlit/config.toml
├── CLAUDE.md                 handoff notes for Claude Code (VS Code)
├── docs/                     project history + session transcript
├── tools/                    test suite (AppTest fuzzer, browser checks, perf)
├── assets/images/            small public-domain / CC0 sample photos (see LICENSES.md)
├── core/                     shared by ALL topics
│   ├── theme.py              colour language (grey = before, blue = after, orange/green = e1/e2)
│   ├── transforms.py         matrices, homogeneous coords, classify(), DLT, fitting, RANSAC
│   ├── shapes.py · plotting.py · images.py      shapes, Plotly helpers, image warping
│   ├── edges.py              kernels, filtering, LoG, zero crossings, NMS, hysteresis, Canny, F1
│   ├── edge_ui.py            image picker, kernel grids, 1D signal plots for the edge lessons
│   ├── contours.py           connectivity, boundaries, Moore tracing, shoelace, RDP, measurements (topic 3)
│   ├── hough.py              line (m, c) / (ρ, θ) and circle accumulators, peaks, RANSAC line (topic 3)
│   ├── contour_ui.py         pixel grids, accumulator heat-maps, image | parameter space figures, scene picker
│   ├── live.py               live sliders (update while dragging) + @live.section partial reruns
│   ├── anim.py               step_player (animated derivations) and scrubber
│   ├── fastplotly.py         skips Plotly's slow per-trace validation for figures built by our helpers
│   └── ui.py                 lesson header, callouts, why-questions, matrices, quizzes, prev/next
└── topics/
    ├── registry.py           table of contents - add new topics here
    ├── _lesson_template.py   starting point for a new lesson page
    ├── t01_geometric_transforms_2d/   p01 … p13
    ├── t02_edge_detection/            p01 … p10
    └── t03_contours_hough/            p01 … p11
```

## Continuing development (Claude Code / VS Code)

* `CLAUDE.md`: the handoff for Claude Code (read automatically): goals, status, architecture, API cheat-sheet,
  rules learned the hard way, testing, and how to add the next topic.
* `docs/PROJECT_HISTORY.md`: timeline, decisions, bug log and test results. `docs/session-history/` holds the
  original session's transcript, a digest of it, and the summary of the earlier part.
* `tools/`: the test suite (`python tools/run_all.py`). See `tools/README.md`.

## Adding the next topic

1. Create a folder, e.g. `topics/t04_<name>/`, and copy `topics/_lesson_template.py` into it once per lesson.
2. Add an entry to `TOPICS` in `topics/registry.py` (id, number, title, summary, folder, pages).
   The sidebar, the home page and the previous/next links update automatically.
3. Put maths that later topics will reuse in `core/`, and end each lesson with `ui.whys([...])`.
4. Use `live.slider` instead of `st.slider`, wrap each interactive block in a function decorated with
   `@live.section`, and turn every derivation into an `anim.step_player` - the template shows all three.
   A figure builder wrapped in `st.cache_resource` also gets `@live.full_quality()` (see the template).

Each lesson follows the same rhythm:
**header → controls | figure | live maths → step-by-step derivation → intuition / pitfall / takeaway → why-questions → quick check → next.**
