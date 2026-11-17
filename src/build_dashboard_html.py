"""Build reports/dashboard.html: the results dashboard as one file that opens in any browser.

Run from the project folder (after src/export_results.py):
    uv run python src/build_dashboard_html.py

It takes the page from reports/dashboard/dashboard_v2/index.html, puts the data
from reports/dashboard/*.json inside it, and adds a small helper so the page's
charts work outside claude.ai. Charts need internet once to load the d3 library.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_PAGE = ROOT / "reports" / "dashboard" / "dashboard_v2" / "index.html"
DATA_DIR = ROOT / "reports" / "dashboard"
OUT = ROOT / "reports" / "dashboard.html"
DATASETS = ["label_counts", "fires", "model_cv", "test_scores", "calibration", "importance", "new_fires"]

THEME = """
:root {
  --color-fg: #1d2330; --color-fg-muted: #5d6676; --color-bg: #f6f7f9; --color-panel: #ffffff; --color-border-line: #e2e5ea;
  --color-ok: #1f8a4c; --color-warn: #b7791f; --color-bad: #c53030;
  --font-anthropic-sans: "Inter", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --font-anthropic-serif: Georgia, "Times New Roman", serif;
  --cds-font-size-caption: 12px; --cds-font-size-body: 15px; --cds-font-size-heading: 17px; --cds-font-size-title: 30px; --cds-font-weight-medium: 600;
  --cds-gap-xs: 4px; --cds-gap-sm: 8px; --cds-gap-md: 12px; --cds-gap-lg: 16px; --cds-gap-xl: 24px;
  --cds-pad-xs: 4px; --cds-pad-sm: 8px; --cds-pad-md: 12px; --cds-pad-lg: 16px; --cds-pad-xl: 24px; --cds-radius: 10px;
  --cds-chart-muted: #b9c0cc; --cds-chart-status-critical: #d6453d; --cds-chart-grid: #e8ebf0; --cds-chart-axis: #c5cbd5; --cds-chart-reference: #7a8291;
  --cds-text-secondary: #5d6676; --cds-surface-popover: #ffffff; --cds-border: #dfe3e8; --cds-shadow-popover: 0 4px 16px rgba(20, 30, 50, .12);
  --accent-1: #2f6fdb; --accent-2: #e08a1e; --accent-3: #2a9d6f;
}
@media (prefers-color-scheme: dark) {
  :root {
    --color-fg: #e8ebf1; --color-fg-muted: #a1a9b8; --color-bg: #12151b; --color-panel: #1a1e26; --color-border-line: #2c323d;
    --color-ok: #4cc38a; --color-warn: #e3a948; --color-bad: #f07167;
    --cds-chart-muted: #5b6475; --cds-chart-status-critical: #f0645b; --cds-chart-grid: #262c36; --cds-chart-axis: #3b4351; --cds-chart-reference: #9aa3b2;
    --cds-text-secondary: #a1a9b8; --cds-surface-popover: #222833; --cds-border: #353c48; --cds-shadow-popover: 0 4px 16px rgba(0, 0, 0, .4);
    --accent-1: #6aa0ff; --accent-2: #f2a541; --accent-3: #4cc38a;
    color-scheme: dark;
  }
}
body { margin: 0; background: var(--color-bg); color: var(--color-fg); font-family: var(--font-anthropic-sans); }
#dash-root { max-width: 1180px; margin: 0 auto; padding: 28px max(16px, 3vw) 48px; box-sizing: border-box; }
.dash-select { font: inherit; font-size: 13px; color: var(--color-fg); background-color: var(--color-panel); border: 1px solid var(--color-border-line); border-radius: 8px; padding: 5px 10px; }
.footer { margin-top: 32px; font-size: 12px; color: var(--color-fg-muted); }
"""

# A tiny stand-in for the dashboard runtime: data, calcs and drawings.
SHIM = """
window.dash = (() => {
  const raw = window.__DATA__, calcs = {}, draws = {}, order = [];
  const data = (id) => {
    if (raw[id]) return { status: 'ok', data: raw[id], columns: Object.keys(raw[id][0] || {}) };
    const c = calcs[id];
    if (c) return { status: 'ok', data: c.fn(...c.inputs.map((i) => data(i).data)) };
    return { status: 'missing', data: [] };
  };
  const run = (name) => document.querySelectorAll(`[data-draw="${name}"]`).forEach((el) => draws[name](el));
  return {
    colors: ['var(--accent-1)', 'var(--accent-2)', 'var(--accent-3)'],
    data,
    calc: (id, spec) => { calcs[id] = spec; },
    loader: () => {},
    onData: () => {},
    params: () => ({}), setParams: () => {}, resetParams: () => {}, setLink: () => {}, refresh: () => {},
    draw: (name, fn) => {
      if (typeof fn === 'function') { draws[name] = fn; order.push(name); }
      else if (draws[name]) run(name);
    },
    _drawAll: () => order.forEach(run),
  };
})();
"""

START = """
window.addEventListener('DOMContentLoaded', () => {
  dash._drawAll();
  let w = window.innerWidth;
  window.addEventListener('resize', () => { if (Math.abs(window.innerWidth - w) > 20) { w = window.innerWidth; dash._drawAll(); } });
});
"""


def main() -> None:
    """Assemble the standalone page."""
    data = {name: json.loads((DATA_DIR / f"{name}.json").read_text(encoding="utf-8"))["rows"] for name in DATASETS}
    page = SRC_PAGE.read_text(encoding="utf-8").replace('<meta name="dash-one-writer">', "").strip()
    data_js = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wildfire Evacuation Risk Model</title>
<style>{THEME}</style>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script>window.__DATA__ = {data_js};</script>
<script>{SHIM}</script>
</head>
<body>
<div id="dash-root">
{page}
<p class="footer">Built from the fire-risk-model project (src/build_dashboard_html.py). Model results from the WiDS Global Datathon 2026 data.</p>
</div>
<script>{START}</script>
</body>
</html>
"""
    OUT.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
