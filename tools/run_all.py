"""The full regression suite - run it before calling a change done.

    python tools/run_all.py            headless checks + browser checks (starts its own server on port 8599)
    python tools/run_all.py --quick    smoke + validate + browser load only (a few minutes)
    python tools/run_all.py --no-browser

Exit code 0 = everything passed. Logs: tools/out/run_all_<step>.log
"""
from __future__ import annotations

import subprocess
import sys
import time
import urllib.request

from _common import OUT, ROOT

PORT = 8599
HERE = ROOT / "tools"


def step(name, cmd):
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / f"run_all_{name}.log"
    t0 = time.time()
    print(f"▶ {name:14s}", end=" ", flush=True)
    with open(log, "w", encoding="utf-8") as f:
        code = subprocess.call([sys.executable, *cmd], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    print(f"{'PASS' if code == 0 else 'FAIL'}  ({time.time() - t0:.0f}s)  log: {log.relative_to(ROOT)}", flush=True)
    return code == 0


def start_server():
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true",
                             "--server.port", str(PORT), "--server.runOnSave", "false"],
                            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/_stcore/health", timeout=2)
            return proc
        except Exception:
            time.sleep(1)
    proc.terminate()
    raise RuntimeError("the Streamlit server did not start")


def main(args):
    quick, browser = "--quick" in args, "--no-browser" not in args
    ok = True
    ok &= step("smoke", [str(HERE / "apptest_checks.py"), "smoke"])
    ok &= step("validate", [str(HERE / "apptest_checks.py"), "validate"])
    if not quick:
        ok &= step("fuzz", [str(HERE / "apptest_checks.py"), "fuzz"])
    if browser:
        server = start_server()
        try:
            port = f"--port={PORT}"
            ok &= step("load_light", [str(HERE / "browser_checks.py"), "load", "--all", "--tabs", port])
            ok &= step("load_dark", [str(HERE / "browser_checks.py"), "load", "--all", "--tabs", "--dark", port])
            if not quick:
                ok &= step("quiz", [str(HERE / "browser_checks.py"), "quiz", "--all", port])
                ok &= step("layout", [str(HERE / "browser_checks.py"), "layout", "--all", port])
                ok &= step("anim", [str(HERE / "browser_checks.py"), "anim", "--all", port])
        finally:
            server.terminate()
    print("ALL PASSED" if ok else "SOMETHING FAILED - read the logs above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
