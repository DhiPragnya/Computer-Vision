# Test tools

Every lesson page has to work with **zero exceptions and zero browser errors**. A slider must change the
page *while* it is dragged, and every step animation must play. These scripts check all of that.
They were used to test the app before each delivery.

## One-time setup (Windows, inside the project's `.venv`)

```bat
.venv\Scripts\activate
pip install -r requirements-dev.txt
python -m playwright install chromium
```

`requirements-dev.txt` adds Playwright, which only the browser checks need. The headless checks need just the
app's own packages.

## The whole suite

```bat
python tools\run_all.py            :: about 20-40 min. Starts its own server on port 8599
python tools\run_all.py --quick    :: smoke + validate + page loads (a few minutes)
```

Each step writes a log to `tools/out/run_all_<step>.log`. Screenshots go to `tools/out/shots/` (git-ignored).

## Headless checks: `apptest_checks.py` (no browser)

They use `streamlit.testing.v1.AppTest`. Under AppTest, `live.slider` falls back to `st.slider`, so every
control can be driven from Python.

| command | what it does |
|---|---|
| `smoke` | loads every page once |
| `fuzz [page-substr ...]` | every slider at both ends, every selectbox/radio/toggle option (and every slider again under each option), every quiz answer, every button. Both tab pages (p02, p08) are swept tab by tab |
| `validate [page-substr ...]` | turns Plotly's property validation back on (live sections normally skip it, see `core/fastplotly.py`) and puts every slider at min and max, which catches invalid values such as a font size < 1 |
| `perf [page-substr ...]` | median run time of every `@live.section` per slider (what you feel while dragging) |
| `profile <page> <slider-key> <section> [n] [--sort=tottime]` | cProfile one section while one slider moves |

Lines marked `(harness)` are limits of AppTest, not app errors. For example, `selectbox.select_index` fails
when the selectbox uses `format_func`.

## Browser checks: `browser_checks.py` (needs a running app)

Start the app with `streamlit run app.py` (port 8501) and leave it running, then:

| command | what it does |
|---|---|
| `load --all [--dark] [--tabs]` | every page (and every tab) loads with no exception and no JS error; full-page screenshots |
| `anim --all` | presses the step buttons and ▶ Play of every animated figure, drags its in-figure slider, and drags every live slider, counting how many different page states appear *during* the drag |
| `quiz --all` | answers every quiz/drill radio and presses every button twice |
| `layout --all` | clicks each slider at 3 %, 50 %, 97 % and flags any slider that changes the page height by > 40 px (a jumping page is hard to use while dragging) |
| `drag <url> <index>` | while a slider is held, step players keep only their key frames; after release they rebuild with every frame |
| `range <url>:<index> ...` | drags both thumbs of range sliders (`anim` can't do this: it grabs the middle of the track) |
| `find-exception <url>` | clicks every slider until an exception shows, then prints which slider and value caused it |

`<url>` is the page's URL path: `""` = home, `t01-affine`, `t02-canny_nms`, ... (see `topics/registry.py`).
Live-slider index 0 is usually the invisible page guard in the sidebar.

Expected `anim` results: every plot says `OK`. Nearly every live slider shows 5–6 states during the
drag. Range sliders show 1, which is expected; check them with `range`. Sliders with only a few values
show 2–3.
