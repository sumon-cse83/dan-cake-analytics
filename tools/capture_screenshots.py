"""Capture report screenshots of the dashboard, website and GitHub repository with headless Microsoft Edge.

Nothing extra to install: it uses Edge (already on Windows) and Tornado (installed with Streamlit) to drive the
browser through the Chrome DevTools Protocol.

Usage (dashboard must be running:  streamlit run dashboard/app.py --server.port 8510):
    python tools/capture_screenshots.py
    python tools/capture_screenshots.py --only dash_forecast
Output: report/figures/screens/*.png
"""
import argparse
import base64
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from tornado.ioloop import IOLoop
from tornado.websocket import websocket_connect

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures" / "screens"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
PORT = 9333
DASH = "http://localhost:8510"

# name -> (url, width, height, seconds to wait, full page?)
SHOTS = {
    "dash_overview":  (f"{DASH}/?page=overview", 1440, 1500, 14, True),
    "dash_trends":    (f"{DASH}/?page=trends", 1440, 1500, 12, True),
    "dash_products":  (f"{DASH}/?page=products", 1440, 1500, 12, True),
    "dash_markets":   (f"{DASH}/?page=markets", 1440, 1500, 12, True),
    "dash_forecast":  (f"{DASH}/?page=forecast", 1440, 1500, 12, True),
    "dash_models":    (f"{DASH}/?page=models", 1440, 1500, 12, True),
    "dash_anomalies": (f"{DASH}/?page=anomalies", 1440, 1500, 12, True),
    "dash_insights":  (f"{DASH}/?page=insights", 1440, 1500, 12, True),
    "dash_quality":   (f"{DASH}/?page=quality", 1440, 1500, 12, True),
    "website":        ("https://sumon-cse83.github.io/dan-cake-analytics/", 1440, 1400, 8, False),
    "github_repo":    ("https://github.com/sumon-cse83/dan-cake-analytics", 1440, 1300, 8, False),
}
MAX_HEIGHT = 2600          # cap very long pages so they stay readable on an A4 page


class Tab:
    def __init__(self, ws):
        self.ws, self.n = ws, 0

    async def call(self, method, **params):
        self.n += 1
        await self.ws.write_message(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.read_message())
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})


async def capture(names):
    targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json"))
    page = next(t for t in targets if t["type"] == "page")
    tab = Tab(await websocket_connect(page["webSocketDebuggerUrl"], max_message_size=200 * 2**20))
    await tab.call("Page.enable")
    for name in names:
        url, w, h, wait, full = SHOTS[name]
        await tab.call("Emulation.setDeviceMetricsOverride", width=w, height=h, deviceScaleFactor=1, mobile=False)
        await tab.call("Page.navigate", url=url)
        time.sleep(wait)                                    # Streamlit draws its charts after the page load event
        if full:                                            # Streamlit scrolls inside its own container
            js = ("Math.max(...[...document.querySelectorAll('section.main, [data-testid=stMain], "
                  "[data-testid=stAppViewContainer], body')].map(e => e.scrollHeight))")
            r = await tab.call("Runtime.evaluate", expression=js, returnByValue=True)
            hh = min(int(r["result"]["value"]) + 40, MAX_HEIGHT)
            await tab.call("Emulation.setDeviceMetricsOverride", width=w, height=hh, deviceScaleFactor=1, mobile=False)
            time.sleep(3)
        shot = await tab.call("Page.captureScreenshot", format="png")
        (OUT / f"{name}.png").write_bytes(base64.b64decode(shot["data"]))
        print(f"saved {name}.png")


def crop_sidebars():
    """Write <name>_main.png without the 300-px Streamlit sidebar (it is identical on every page), so the page
    content is printed larger in the report. Page 1 keeps its sidebar to show the navigation and filters."""
    from PIL import Image
    for f in OUT.glob("dash_*.png"):
        if f.stem.endswith("_main") or f.stem == "dash_overview":
            continue
        im = Image.open(f)
        im.crop((330, 0, im.width - 50, im.height)).save(OUT / f"{f.stem}_main.png")
        print(f"cropped {f.stem}_main.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="names to capture (default: all)")
    ap.add_argument("--crop", action="store_true", help="only crop the sidebars of existing screenshots")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.crop:
        return crop_sidebars()
    names = args.only or list(SHOTS)
    profile = tempfile.mkdtemp(prefix="edge-shots-")
    edge = subprocess.Popen([EDGE, "--headless=new", f"--remote-debugging-port={PORT}", "--disable-gpu",
                             "--hide-scrollbars", f"--user-data-dir={profile}", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(40):                                  # wait for the debugging endpoint
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/version", timeout=1)
                break
            except OSError:
                time.sleep(0.5)
        IOLoop.current().run_sync(lambda: capture(names), timeout=900)
        crop_sidebars()
    finally:
        edge.terminate()
        time.sleep(1)
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    main()
