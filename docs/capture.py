"""Screenshot helper for the demo story (uses the locally installed Chrome through Playwright).

  python -m docs.capture grafana <name> [--from now-15m] [--to now] [--rows prod|all]
  python -m docs.capture page <name> <url> [--wait-text TEXT] [--full]

Grafana is expected at http://localhost:3001 (./infra/aws/tunnel.sh 3000 3001), TrueForge at :8791.
Images are written to docs/screenshots/<name>.png.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent / "screenshots"
GRAFANA = "http://localhost:3001"


def shoot(url: str, name: str, width: int, height: int, full: bool, wait_text: str | None, settle_s: float,
          clip_height: int | None = None) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=2,
                                color_scheme="dark")
        page.goto(url, wait_until="networkidle", timeout=90_000)
        if wait_text:
            page.get_by_text(wait_text).first.wait_for(timeout=60_000)
        time.sleep(settle_s)
        if clip_height:
            page.screenshot(path=str(path), clip={"x": 0, "y": 0, "width": width, "height": clip_height})
        else:
            page.screenshot(path=str(path), full_page=full)
        browser.close()
    print(path)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grafana")
    g.add_argument("name")
    g.add_argument("--from", dest="frm", default="now-15m")
    g.add_argument("--to", default="now")
    g.add_argument("--rows", choices=["prod", "all"], default="prod")
    pg = sub.add_parser("page")
    pg.add_argument("name")
    pg.add_argument("url")
    pg.add_argument("--wait-text")
    pg.add_argument("--full", action="store_true")
    pg.add_argument("--width", type=int, default=1600)
    pg.add_argument("--height", type=int, default=1000)
    a = ap.parse_args()
    if a.cmd == "grafana":
        url = f"{GRAFANA}/d/firefighter?orgId=1&from={a.frm}&to={a.to}&kiosk&refresh="
        # prod rows = first ~1,050 px of the dashboard; "all" = prod + shadow
        shoot(url, a.name, 1800, 2300, full=a.rows == "all", wait_text=None, settle_s=8,
              clip_height=1060 if a.rows == "prod" else None)
    else:
        shoot(a.url, a.name, a.width, a.height, full=a.full, wait_text=a.wait_text, settle_s=4)


if __name__ == "__main__":
    main()
