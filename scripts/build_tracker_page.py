"""
Inject a tracker report into the page template.

    python scripts/build_tracker_page.py \
        --report reports/smart_money_report.json \
        --out reports/web/disclosure_tracker.html

The output is one self-contained HTML file (data embedded), suitable for
publishing as a Claude artifact or any static host.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "reports" / "web" / "tracker_page.template.html"
PLACEHOLDER = "__REPORT_JSON__"


def build(report_path: Path, out_path: Path, template: Path = TEMPLATE) -> int:
    report = json.loads(report_path.read_text())
    data = json.dumps(report, separators=(",", ":"), allow_nan=False)
    # "<" only occurs inside JSON strings, where \u003c is equivalent; this
    # keeps the data from ever closing its own <script> tag.
    data = data.replace("<", "\\u003c")
    html = template.read_text()
    if html.count(PLACEHOLDER) != 1:
        raise SystemExit("Template must contain exactly one " + PLACEHOLDER)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html.replace(PLACEHOLDER, data))
    return out_path.stat().st_size


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--report", default=str(ROOT / "reports" / "smart_money_report.json"))
    p.add_argument("--out", default=str(ROOT / "reports" / "web" / "disclosure_tracker.html"))
    a = p.parse_args()
    size = build(Path(a.report), Path(a.out))
    print(f"wrote {a.out} ({size / 1024 / 1024:.1f} MB)")
