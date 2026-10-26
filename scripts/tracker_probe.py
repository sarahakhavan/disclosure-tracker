"""
One-off probe: capture raw samples from the official disclosure sources so
parsers can be written against real formats. Writes into ./probe/.
"""

import io
import json
import re
import sys
import zipfile
from pathlib import Path

import httpx

OUT = Path("probe")
OUT.mkdir(exist_ok=True)
UA = {"User-Agent": "DisclosureTracker research (github.com/sarahakhavan/disclosure-tracker)"}
log = {}


def save(name, data):
    p = OUT / name
    p.write_bytes(data if isinstance(data, bytes) else data.encode())


def house(c):
    r = c.get("https://disclosures-clerk.house.gov/public_disc/financial-pdfs/2025FD.zip")
    log["house_zip"] = r.status_code
    z = zipfile.ZipFile(io.BytesIO(r.content))
    log["house_zip_files"] = z.namelist()
    xml = next(n for n in z.namelist() if n.lower().endswith(".xml"))
    text = z.read(xml).decode("utf-8", "replace")
    save("house_2025FD_head.xml", text[:6000])
    members = re.findall(r"<Member>(.*?)</Member>", text, re.S)
    ptrs = [m for m in members if "<FilingType>P</FilingType>" in m]
    log["house_members"] = len(members)
    log["house_ptrs"] = len(ptrs)
    pel = [m for m in ptrs if "Pelosi" in m]
    picks = pel[-2:] + ptrs[100:102] + ptrs[-2:]
    from pypdf import PdfReader

    for i, m in enumerate(picks):
        doc = re.search(r"<DocID>(\d+)</DocID>", m).group(1)
        url = f"https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/2025/{doc}.pdf"
        pr = c.get(url)
        log[f"house_pdf_{i}"] = {"url": url, "status": pr.status_code, "bytes": len(pr.content), "member": re.sub(r"\s+", " ", m)[:300]}
        save(f"house_ptr_{i}_{doc}.pdf", pr.content)
        try:
            rd = PdfReader(io.BytesIO(pr.content))
            save(f"house_ptr_{i}_{doc}.txt", "\n\n=====PAGE=====\n\n".join(p.extract_text() or "" for p in rd.pages))
            save(f"house_ptr_{i}_{doc}.layout.txt", "\n\n=====PAGE=====\n\n".join(p.extract_text(extraction_mode="layout") or "" for p in rd.pages))
        except Exception as e:
            log[f"house_pdf_{i}"]["extract_error"] = str(e)


def senate(c):
    home = c.get("https://efdsearch.senate.gov/search/home/")
    log["senate_home"] = home.status_code
    save("senate_home.html", home.text[:20000])
    tok = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', home.text)
    if not tok:
        return
    r = c.post(
        "https://efdsearch.senate.gov/search/home/",
        data={"prohibition_agreement": "1", "csrfmiddlewaretoken": tok.group(1)},
        headers={"Referer": "https://efdsearch.senate.gov/search/home/"},
    )
    log["senate_agree"] = r.status_code
    csrf = c.cookies.get("csrftoken")
    d = c.post(
        "https://efdsearch.senate.gov/search/report/data/",
        data={
            "start": "0", "length": "25", "report_types": "[11]", "filer_types": "[]",
            "submitted_start_date": "01/01/2025 00:00:00", "submitted_end_date": "",
            "candidate_state": "", "senator_state": "", "office_id": "", "first_name": "", "last_name": "",
            "csrfmiddlewaretoken": csrf,
        },
        headers={"Referer": "https://efdsearch.senate.gov/search/", "X-CSRFToken": csrf or ""},
    )
    log["senate_data"] = d.status_code
    save("senate_data.json", d.text[:20000])
    try:
        rows = d.json().get("data", [])
        log["senate_rows"] = len(rows)
        links = [re.search(r'href="([^"]+)"', str(r[3])) for r in rows]
        links = [l.group(1) for l in links if l]
        for i, link in enumerate([l for l in links if "/ptr/" in l][:2] + [l for l in links if "/paper/" in l][:1]):
            pr = c.get("https://efdsearch.senate.gov" + link)
            log[f"senate_ptr_{i}"] = {"link": link, "status": pr.status_code}
            save(f"senate_ptr_{i}.html", pr.text)
    except Exception as e:
        log["senate_error"] = str(e)


def thirteenf(c):
    for name, cik in {"pershing": 1336528, "tiger": 1167483, "bridgewater": 1350694, "thirdpoint": 1040273}.items():
        sub = c.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
        rec = sub["filings"]["recent"]
        idx = [i for i, f in enumerate(rec["form"]) if f == "13F-HR"]
        info = []
        for i in [idx[0], idx[len(idx) // 2], idx[-1]]:
            acc = rec["accessionNumber"][i].replace("-", "")
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/"
            r = c.get(url)
            files = re.findall(r'href="([^"]+)"', r.text)
            info.append({"period": rec["reportDate"][i], "url": url, "status": r.status_code,
                         "files": [f for f in files if "/Archives/" in f or f.endswith(".xml")][:20]})
        log[f"13f_{name}"] = info


with httpx.Client(headers=UA, timeout=60, follow_redirects=True) as c:
    for fn in (house, senate, thirteenf):
        try:
            fn(c)
        except Exception as e:
            log[fn.__name__ + "_fatal"] = repr(e)[:500]

save("probe.json", json.dumps(log, indent=2))
print(json.dumps(log, indent=2)[:4000])
sys.exit(0)
