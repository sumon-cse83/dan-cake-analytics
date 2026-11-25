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
# The dashboard is captured in a fairly narrow window (about 900 px of page content next to the sidebar) at
# twice the screen resolution, so that its text stays sharp and large enough to read when printed 15 cm wide.
W = 1380
SHOTS = {
    "dash_overview":  (f"{DASH}/?page=overview", W, 1500, 14, True),
    "dash_trends":    (f"{DASH}/?page=trends", W, 1500, 12, True),
    "dash_products":  (f"{DASH}/?page=products", W, 1500, 12, True),
    "dash_markets":   (f"{DASH}/?page=markets", W, 1500, 12, True),
    "dash_forecast":  (f"{DASH}/?page=forecast", W, 1500, 12, True),
    "dash_models":    (f"{DASH}/?page=models", W, 1500, 12, True),
    "dash_anomalies": (f"{DASH}/?page=anomalies", W, 1500, 12, True),
    "dash_insights":  (f"{DASH}/?page=insights", W, 1500, 12, True),
    "dash_quality":   (f"{DASH}/?page=quality", W, 1500, 12, True),
    "website":        ("https://sumon-cse83.github.io/dan-cake-analytics/", 1100, 545, 8, False),
    "github_repo":    ("https://github.com/sumon-cse83/dan-cake-analytics", 1100, 900, 8, False),
}
SCALE = 2                  # device pixels per CSS pixel
MAX_HEIGHT = 3200          # cap very long pages (CSS pixels)
SIDEBAR = 330              # CSS pixels of the Streamlit sidebar, cut off in the *_main images
# part of each page printed in the report: (top, bottom) in CSS pixels of the page content
REPORT_CROP = {
    "dash_overview": (0, 665), "dash_trends": (0, 700), "dash_products": (0, 790), "dash_markets": (0, 615),
    "dash_forecast": (0, 715), "dash_models": (0, 600), "dash_anomalies": (0, 600), "dash_insights": (0, 600),
    "dash_quality": (0, 770),
}


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
        await tab.call("Emulation.setDeviceMetricsOverride", width=w, height=h, deviceScaleFactor=SCALE, mobile=False)
        await tab.call("Page.navigate", url=url)
        time.sleep(wait)                                    # Streamlit draws its charts after the page load event
        if full:                                            # Streamlit scrolls inside its own container
            js = ("Math.max(...[...document.querySelectorAll('section.main, [data-testid=stMain], "
                  "[data-testid=stAppViewContainer], body')].map(e => e.scrollHeight))")
            r = await tab.call("Runtime.evaluate", expression=js, returnByValue=True)
            hh = min(int(r["result"]["value"]) + 40, MAX_HEIGHT)
            await tab.call("Emulation.setDeviceMetricsOverride", width=w, height=hh, deviceScaleFactor=SCALE,
                           mobile=False)
            time.sleep(3)
        shot = await tab.call("Page.captureScreenshot", format="png")
        (OUT / f"{name}.png").write_bytes(base64.b64decode(shot["data"]))
        print(f"saved {name}.png")


def crop_sidebars():
    """Write <name>_main.png without the Streamlit sidebar (it is identical on every page) and <name>_report.png,
    the part of the page that is printed in the report, so that the page content is printed large enough to read."""
    from PIL import Image
    for f in OUT.glob("dash_*.png"):
        if f.stem.endswith(("_main", "_report")):
            continue
        im = Image.open(f)
        s = im.width / SHOTS[f.stem][1]                    # device pixels per CSS pixel of this image
        main = im.crop((int(SIDEBAR * s), 0, im.width - int(20 * s), im.height))
        main.save(OUT / f"{f.stem}_main.png")
        top, bottom = REPORT_CROP.get(f.stem, (0, main.height / s))
        main.crop((0, int(top * s), main.width, min(main.height, int(bottom * s)))).save(OUT / f"{f.stem}_report.png")
        print(f"cropped {f.stem}_main.png and {f.stem}_report.png")


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
