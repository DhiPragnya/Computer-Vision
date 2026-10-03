"""Shared helpers for the test tools (paths, page lists, quiet imports)."""
from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # the ComputerVision folder
OUT = ROOT / "tools" / "out"                         # screenshots and logs (git-ignored)

# pages that use st.tabs (the browser checks also visit tabs 2..N there)
TAB_PAGES = {"t01-homogeneous": "h_tabs", "t01-projective": "pj_tabs"}


def project_on_path() -> None:
    """Make `import core`, `import topics` work and run from the project root (like `streamlit run app.py`)."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)


def quiet() -> None:
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)


def page_files(only: list[str] | None = None, home: bool = True) -> list[str]:
    """Page scripts relative to the project root, in sidebar order (optionally filtered by substrings)."""
    project_on_path()
    from topics.registry import TOPICS, page_path

    pages = (["home.py"] if home else []) + [page_path(t, p) for t in TOPICS for p in t["pages"]]
    return [p for p in pages if not only or any(o in p for o in only)]


def url_paths(only: list[str] | None = None, home: bool = True) -> list[str]:
    """URL paths as the browser sees them ('' = home, 't01-affine', ...)."""
    project_on_path()
    from topics.registry import TOPICS, url_path

    urls = ([""] if home else []) + [url_path(t, p) for t in TOPICS for p in t["pages"]]
    return [u for u in urls if not only or any(o in u for o in only) or (u == "" and "home" in only)]


def new_app():
    """A fresh AppTest of the whole app (live sliders fall back to st.slider under AppTest)."""
    project_on_path()
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.run()
    return at


def open_page(page: str):
    at = new_app()
    if page != "home.py":
        at.switch_page(page)
        at.run()
    return at
