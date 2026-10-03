"""Checks in a real browser (Playwright + Chromium) against a RUNNING app.

Start the app first (streamlit run app.py), then e.g.:

    python tools/browser_checks.py load --all [--dark] [--tabs]     every page loads: no exception, no JS error
    python tools/browser_checks.py anim --all                       every step animation plays / scrubs, every live
                                                                    slider changes the page WHILE it is dragged
    python tools/browser_checks.py quiz --all                       answer every quiz / drill, press every button
    python tools/browser_checks.py layout --all                     sliders must not make the page jump in height
    python tools/browser_checks.py drag <url-path> <slider-index>   mid-drag: key frames only; after release: all
    python tools/browser_checks.py range <url-path>:<index> ...     drag both thumbs of range sliders
    python tools/browser_checks.py find-exception <url-path>        click every slider until an exception shows

<url-path> is what the browser shows after the host: "" (home), t01-affine, t02-canny_nms, ...
Options: --port=8501.  Screenshots go to tools/out/shots/.  Slider index 0 is usually the hidden page guard.
"""
from __future__ import annotations

import asyncio
import sys
import time

from _common import OUT, TAB_PAGES, url_paths

IGNORE = ("metrics", "ERR_TUNNEL", "data.streamlit", "Failed to load resource", "Undefined metrics")
LIVE = "[data-testid='stBidiComponentIsolated']"

# a hash of everything visible: plot data, image sources and text - changes whenever the page changes
SIG = """() => {
  let h = 0;
  const add = (v) => { const n = Number(v); if (isFinite(n)) h = (h * 31 + Math.round(n * 1000)) % 1000000007; };
  document.querySelectorAll('.js-plotly-plot').forEach(g => (g._fullData || g.data).forEach(t => {
     ['x', 'y', 'z'].forEach(k => { const a = t[k]; if (a && a.length) { for (let i = 0; i < a.length; i += Math.max(1, Math.floor(a.length / 50))) { const r = a[i]; if (r && r.length !== undefined && typeof r !== 'string') { add(r[0]); add(r[r.length-1]); } else add(r); } } });
  }));
  document.querySelectorAll('img').forEach(im => { const s = im.src || ''; for (let i = 0; i < s.length; i += 97) h = (h * 31 + s.charCodeAt(i)) % 1000000007; h = (h + s.length) % 1000000007; });
  const tex = Array.from(document.querySelectorAll('.katex, [data-testid=stMarkdownContainer] p')).map(k => k.textContent).join('');
  for (let i = 0; i < tex.length; i++) h = (h * 33 + tex.charCodeAt(i)) % 1000000007;
  return h; }"""

CAP = """(i) => { const g = document.querySelectorAll('.js-plotly-plot')[i];
  const a = (g.layout.annotations || []).map(a => a.text).join(' | ');
  const d = (g._fullData || g.data).map(t => (t.x && t.x.length ? Number(t.x[0]) + Number(t.x[t.x.length-1]) : 0) + (t.y && t.y.length ? Number(t.y[0]) : 0)).reduce((p, c) => p + (isNaN(c) ? 0 : c), 0);
  return [a.slice(0, 90), Math.round(d * 1000) / 1000]; }"""

FRAMES = ("() => Array.from(document.querySelectorAll('.js-plotly-plot')).map(g => (g._transitionData && "
          "g._transitionData._frames) ? g._transitionData._frames.length : 0)")
SLIDER_BOX = """(j) => { const r = document.querySelectorAll('[data-testid=stBidiComponentIsolated]')[j].shadowRoot;
  const i = r && r.querySelector('input'); if (!i) return null; const b = i.getBoundingClientRect();
  return {x: b.x, y: b.y, w: b.width, h: b.height, label: r.querySelector('.ls-text').textContent} }"""
RANGE_INFO = """(j) => { const r = document.querySelectorAll('[data-testid=stBidiComponentIsolated]')[j].shadowRoot;
  const ins = Array.from(r.querySelectorAll('input[type=range]')); const b = ins[ins.length - 1].getBoundingClientRect();
  return {x: b.x, y: b.y, w: b.width, h: b.height, label: r.querySelector('.ls-text').textContent,
          vals: ins.map(i => +i.value), min: +ins[0].min, max: +ins[0].max} }"""
RANGE_VALS = ("(j) => Array.from(document.querySelectorAll('[data-testid=stBidiComponentIsolated]')[j].shadowRoot"
              ".querySelectorAll('input[type=range]')).map(i => +i.value)")
HEIGHT = "() => document.querySelector('[data-testid=stMainBlockContainer]').getBoundingClientRect().height"


# --------------------------------------------------------------------------------------------- plumbing
class Session:
    def __init__(self, pw, port, dark=False, height=1100):
        self.pw, self.port, self.dark, self.height = pw, port, dark, height
        self.errs = []

    async def __aenter__(self):
        self.browser = await self.pw.chromium.launch()
        ctx = await self.browser.new_context(viewport={"width": 1440, "height": self.height},
                                             color_scheme="dark" if self.dark else "light")
        self.page = await ctx.new_page()
        self.page.on("console", lambda m: self.errs.append(m.text)
                     if m.type == "error" and not any(s in m.text for s in IGNORE) else None)
        self.page.on("pageerror", lambda e: self.errs.append(f"PAGEERROR {e}"))
        return self

    async def __aexit__(self, *exc):
        await self.browser.close()

    async def goto(self, path):
        self.errs.clear()
        await self.page.goto(f"http://localhost:{self.port}/{path}", wait_until="networkidle")
        await self.idle(first=True)

    async def idle(self, first=False, timeout=30.0):
        """Wait until Streamlit stops running and the number of plots/images settles."""
        page = self.page
        await page.wait_for_timeout(1500 if first else 900)
        t0 = time.time()
        while time.time() - t0 < timeout and await page.locator("[data-testid='stStatusWidget']").count():
            await page.wait_for_timeout(250)
        last = -1
        for _ in range(20):
            n = await page.locator(".js-plotly-plot, [data-testid='stImage'], [data-testid='stException']").count()
            if n == last:
                break
            last = n
            await page.wait_for_timeout(600)
        await page.wait_for_timeout(400)

    async def exceptions(self):
        exc = self.page.locator("[data-testid='stException']")
        return [(await exc.nth(i).inner_text())[:600] for i in range(await exc.count())]

    async def slider_box(self, j):
        h = self.page.locator(LIVE).nth(j)
        await h.scroll_into_view_if_needed()
        await self.page.wait_for_timeout(250)
        return await self.page.evaluate(SLIDER_BOX, j)


def parse(args):
    port = next((int(a.split("=", 1)[1]) for a in args if a.startswith("--port=")), 8501)
    flags = {a for a in args if a.startswith("--") and "=" not in a}
    rest = [a for a in args if not a.startswith("--")]
    return port, flags, rest


def targets(flags, rest):
    return url_paths() if "--all" in flags else rest


def shot(name):
    (OUT / "shots").mkdir(parents=True, exist_ok=True)
    return str(OUT / "shots" / f"{name or 'home'}.png")


# --------------------------------------------------------------------------------------------- load
async def load(pw, args):
    port, flags, rest = parse(args)
    bad = 0
    async with Session(pw, port, dark="--dark" in flags, height=6000) as s:
        for path in targets(flags, rest):
            t0 = time.time()
            await s.goto(path)
            exc = await s.exceptions()
            n_plots = await s.page.locator(".js-plotly-plot").count()
            n_live = await s.page.locator(LIVE).count()
            name = (path or "home") + ("_dark" if "--dark" in flags else "")
            await s.page.screenshot(path=shot(name), full_page=True)
            bad += bool(exc or s.errs)
            print(f"{name}: load {time.time() - t0:.1f}s plots={n_plots} live_sliders={n_live} "
                  f"exceptions={len(exc)} js_errors={len(s.errs)}", flush=True)
            for t in exc:
                print("   EXC:", t.replace("\n", " | ")[:300])
            for e in s.errs[:5]:
                print("   JS:", e[:300])
            if "--tabs" in flags:
                tl = s.page.locator("[role='tab']")
                for i in range(1, await tl.count()):
                    await tl.nth(i).click()
                    await s.idle()
                    exc = await s.exceptions()
                    bad += bool(exc)
                    await s.page.screenshot(path=shot(f"{name}_tab{i + 1}"), full_page=True)
                    print(f"   tab {i + 1}: exceptions={len(exc)} js_errors={len(s.errs)}", flush=True)
    return bad


# --------------------------------------------------------------------------------------------- anim
async def anim_page(s, path, tab):
    page = s.page
    await s.goto(path)
    if tab:
        await page.locator("[role='tab']").nth(tab - 1).click()
        await s.idle()
    print(f"{path or 'home'} tab={tab}", flush=True)
    problems = 0
    for i in range(await page.locator(".js-plotly-plot").count()):
        g = page.locator(".js-plotly-plot").nth(i)
        nfr = await page.evaluate(f"(i) => ({FRAMES})()[i]", i)
        if not nfr or not await g.is_visible():
            continue
        await g.scroll_into_view_if_needed()
        await page.wait_for_timeout(300)
        c0 = await page.evaluate(CAP, i)
        btns = g.locator("g.updatemenu-button")
        nb = await btns.count()
        seen = {tuple(c0)}
        if nb > 2:                                           # step buttons: press the last one, then rewind
            await btns.nth(nb - 1).click(force=True)
            await page.wait_for_timeout(1200)
            seen.add(tuple(await page.evaluate(CAP, i)))
            await btns.nth(1).click(force=True)
            await page.wait_for_timeout(400)
        await btns.nth(0).click(force=True)                  # ▶ Play
        await page.wait_for_timeout(700)
        seen.add(tuple(await page.evaluate(CAP, i)))
        await page.wait_for_timeout(500)
        seen.add(tuple(await page.evaluate(CAP, i)))
        rail = g.locator("rect.slider-rail-rect")            # drag the in-figure slider
        if await rail.count():
            bb = await rail.first.bounding_box()
            y = bb["y"] + bb["height"] / 2
            await page.mouse.move(bb["x"] + bb["width"] * 0.05, y)
            await page.mouse.down()
            for f in [0.2, 0.4, 0.6, 0.8, 0.97]:
                await page.mouse.move(bb["x"] + bb["width"] * f, y, steps=4)
                await page.wait_for_timeout(120)
                seen.add(tuple(await page.evaluate(CAP, i)))
            await page.mouse.up()
            await page.wait_for_timeout(300)
        ok = len(seen) >= 3
        problems += not ok
        print(f"  plot {i}: frames={nfr} buttons={nb} distinct states={len(seen)} {'OK' if ok else 'NO CHANGE?'}  "
              f"{c0[0][:60]!r}", flush=True)
    for j in range(await page.locator(LIVE).count()):       # live sliders: does the page change WHILE dragging?
        h = page.locator(LIVE).nth(j)
        if not await h.is_visible():
            continue
        bb = await s.slider_box(j)
        if bb is None:
            continue
        s0 = await page.evaluate(SIG)
        y = bb["y"] + bb["h"] / 2
        await page.mouse.move(bb["x"] + bb["w"] * 0.5, y)
        await page.mouse.down()
        states = {s0}
        for f in [0.55, 0.62, 0.7, 0.78, 0.85]:
            await page.mouse.move(bb["x"] + bb["w"] * f, y, steps=5)
            await page.wait_for_timeout(260)
            states.add(await page.evaluate(SIG))
        await page.mouse.up()
        await s.idle()
        exc = await s.exceptions()
        problems += bool(exc)
        print(f"  live slider {j} {bb['label']!r}: page states during drag={len(states)} exceptions={len(exc)}"
              + ("   (no change: a range slider - use the 'range' check - or a setting with no visible effect)"
                 if len(states) == 1 else ""), flush=True)
    print("  JS errors:", s.errs[:5], flush=True)
    return problems + bool(s.errs)


async def anim(pw, args):
    port, flags, rest = parse(args)
    tab = next((int(a.split("=", 1)[1]) for a in args if a.startswith("--tab=")), 0)
    bad = 0
    async with Session(pw, port) as s:
        for path in targets(flags, rest):
            bad += await anim_page(s, path, tab)
            if "--all" in flags and path in TAB_PAGES:
                await s.goto(path)
                for t in range(2, await s.page.locator("[role='tab']").count() + 1):
                    bad += await anim_page(s, path, t)
    return bad


# --------------------------------------------------------------------------------------------- quiz
async def quiz(pw, args):
    port, flags, rest = parse(args)
    bad = 0
    async with Session(pw, port, height=1400) as s:
        page = s.page
        for path in targets(flags, rest):
            await s.goto(path)
            groups = page.locator("[data-testid='stRadio']")
            answered = 0
            for i in range(await groups.count()):
                g = groups.nth(i)
                opts = g.locator("[role=radiogroup] label")
                if not await opts.count() or await g.locator("input").first.is_disabled():
                    continue
                await opts.nth(0).scroll_into_view_if_needed()
                await opts.nth(0).click()
                await s.idle()
                answered += 1
            exc = await s.exceptions()
            print(f"{path or 'home'}: radios={await groups.count()} answered={answered} exceptions={len(exc)} "
                  f"js={s.errs[:3]}", flush=True)
            bad += bool(exc or s.errs)
            for rep in range(2):                              # every button (Next round, New target, ...) twice
                btns = page.locator("[data-testid='stButton'] button")
                labels = []
                for j in range(await btns.count()):
                    bt = btns.nth(j)
                    if await bt.is_disabled():
                        continue
                    labels.append((await bt.inner_text()).strip().split("\n")[-1][:30])
                    await bt.scroll_into_view_if_needed()
                    await bt.click()
                    await s.idle()
                    for i in range(await groups.count()):     # answer the new drill round as well
                        g = groups.nth(i)
                        if await g.locator("input").count() and not await g.locator("input").first.is_disabled():
                            lab = g.locator("[role=radiogroup] label")
                            if await lab.count():
                                await lab.nth(min(1, await lab.count() - 1)).click()
                                await s.idle()
                exc = await s.exceptions()
                bad += bool(exc)
                print(f"   buttons pass {rep + 1}: {labels} exceptions={len(exc)} {exc[:1]} js={s.errs[:3]}",
                      flush=True)
    return bad


# --------------------------------------------------------------------------------------------- layout
async def layout(pw, args):
    port, flags, rest = parse(args)
    bad = 0
    async with Session(pw, port, height=1000) as s:
        page = s.page
        for path in targets(flags, rest):
            await s.goto(path)
            n = await page.locator(LIVE).count()
            flagged = []
            for j in range(n):
                h = page.locator(LIVE).nth(j)
                if not await h.is_visible() or await page.evaluate(SLIDER_BOX, j) is None:
                    continue
                hs = []
                for f in (0.03, 0.5, 0.97):
                    bb = await s.slider_box(j)
                    await page.mouse.click(bb["x"] + bb["w"] * f, bb["y"] + bb["h"] / 2)
                    await s.idle()
                    hs.append(round(await page.evaluate(HEIGHT)))
                if max(hs) - min(hs) > 40:
                    flagged.append((j, bb["label"], hs))
            exc = await s.exceptions()
            bad += bool(flagged or exc or s.errs)
            print(f"{path or 'home'}: {n} components, exceptions={len(exc)}, js={s.errs[:2]}", flush=True)
            for f in flagged:
                print(f"   JUMP slider {f[0]} {f[1]!r}: page heights {f[2]}", flush=True)
            for t in exc:
                print("   EXC:", t.replace("\n", " | ")[:300])
    return bad


# --------------------------------------------------------------------------------------------- drag
async def drag(pw, args):
    port, _, rest = parse(args)
    path, j = rest[0], int(rest[1])
    async with Session(pw, port) as s:
        page = s.page
        await s.goto(path)
        f0 = await page.evaluate(FRAMES)
        bb = await s.slider_box(j)
        y = bb["y"] + bb["h"] / 2
        await page.mouse.move(bb["x"] + bb["w"] * 0.3, y)
        await page.mouse.down()
        states, during, t0 = {await page.evaluate(SIG)}, [], time.time()
        for k in range(10):
            await page.mouse.move(bb["x"] + bb["w"] * (0.3 + 0.05 * k), y, steps=4)
            await page.wait_for_timeout(250)
            states.add(await page.evaluate(SIG))
            during.append(await page.evaluate(FRAMES))
        dt = time.time() - t0
        await page.mouse.up()
        await s.idle()
        f1 = await page.evaluate(FRAMES)
        exc = await s.exceptions()
        print(f"{path} slider {j} {bb['label']!r}: {len(states)} distinct page states in {dt:.1f}s of dragging")
        print(f"   animation frames per plot before {f0}\n   mid-drag {during[-1]}\n   after release {f1}\n"
              f"   exceptions={len(exc)} errors={s.errs[:3]}")
        return bool(exc or s.errs)


# --------------------------------------------------------------------------------------------- range
async def range_(pw, args):
    port, _, rest = parse(args)
    bad, cur = 0, None
    async with Session(pw, port) as s:
        page = s.page
        for item in rest:
            path, j = item.rsplit(":", 1)
            j = int(j)
            if path != cur:
                await s.goto(path)
                cur = path
            await page.locator(LIVE).nth(j).scroll_into_view_if_needed()
            info = await page.evaluate(RANGE_INFO, j)
            for thumb in (1, 0):                              # drag the upper thumb right, then the lower one left
                fx = (info["vals"][thumb] - info["min"]) / ((info["max"] - info["min"]) or 1)
                y = info["y"] + info["h"] / 2
                x0 = info["x"] + 8 + (info["w"] - 16) * fx
                states = {await page.evaluate(SIG)}
                await page.mouse.move(x0, y)
                await page.mouse.down()
                for k in range(1, 6):
                    await page.mouse.move(x0 + (12 * k if thumb == 1 else -6 * k), y, steps=4)
                    await page.wait_for_timeout(260)
                    states.add(await page.evaluate(SIG))
                await page.mouse.up()
                await s.idle()
                newv = await page.evaluate(RANGE_VALS, j)
                exc = await s.exceptions()
                bad += bool(exc)
                print(f"{path} #{j} {info['label']!r} thumb {thumb}: {info['vals']} -> {newv}  "
                      f"states during drag={len(states)} exceptions={len(exc)}", flush=True)
                info["vals"] = newv
        print("JS errors", s.errs)
    return bad + bool(s.errs)


# --------------------------------------------------------------------------------------------- find-exception
async def find_exception(pw, args):
    port, _, rest = parse(args)
    path = rest[0]
    async with Session(pw, port, height=1000) as s:
        page = s.page
        await s.goto(path)
        for j in range(await page.locator(LIVE).count()):
            if await page.evaluate(SLIDER_BOX, j) is None:
                continue
            for f in (0.03, 0.5, 0.97):
                bb = await s.slider_box(j)
                await page.mouse.click(bb["x"] + bb["w"] * f, bb["y"] + bb["h"] / 2)
                await s.idle()
                exc = await s.exceptions()
                if exc:
                    print(f"EXCEPTION after slider {j} {bb['label']!r} clicked at {f:.0%}:\n{exc[0][:1500]}")
                    return 1
        print("no exception")
    return 0


COMMANDS = {"load": load, "anim": anim, "quiz": quiz, "layout": layout, "drag": drag, "range": range_,
            "find-exception": find_exception}


async def main(cmd, args):
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        return await COMMANDS[cmd](pw, args)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(2)
    sys.exit(1 if asyncio.run(main(sys.argv[1], sys.argv[2:])) else 0)
