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
# The template is a page body; this makes it a complete document for any host.
SHELL_HEAD = (
    "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
    "<meta charset=\"utf-8\">\n"
    "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1,viewport-fit=cover\">\n"
    "<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}"
    "body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>\n"
    "</head>\n<body>\n"
)


def build(report_path: Path, out_path: Path, template: Path = TEMPLATE) -> int:
    report = json.loads(report_path.read_text())
    data = json.dumps(report, separators=(",", ":"), allow_nan=False)
    # "<" only occurs inside JSON strings, where \u003c is equivalent; this
    # keeps the data from ever closing its own <script> tag.
    data = data.replace("<", "\\u003c")
    html = template.read_text()
    if html.count(PLACEHOLDER) != 1:
        raise SystemExit("Template must contain exactly one " + PLACEHOLDER)
    html = html.replace(PLACEHOLDER, data)
    if not html.lstrip().lower().startswith("<!doctype"):
        html = SHELL_HEAD + html + "\n</body>\n</html>\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path.stat().st_size


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--report", default=str(ROOT / "reports" / "smart_money_report.json"))
    p.add_argument("--out", default=str(ROOT / "reports" / "web" / "disclosure_tracker.html"))
    a = p.parse_args()
    size = build(Path(a.report), Path(a.out))
    print(f"wrote {a.out} ({size / 1024 / 1024:.1f} MB)")
