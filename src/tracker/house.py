"""
U.S. House Periodic Transaction Reports (PTRs), straight from the Clerk.

* Index: https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{YEAR}FD.zip
  (an XML list of every filing; FilingType "P" = PTR)
* Report: https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{YEAR}/{DocID}.pdf

Electronically filed PTRs are text PDFs whose transaction rows read like::

    SP Broadcom Inc. - Common Stock
    (AVGO) [ST]
    P 06/20/202506/20/2025$1,000,001 -
    $5,000,000

i.e. owner code, asset (ticker in parentheses, asset-type code in brackets),
transaction type, transaction date, notification date, amount range.
Paper filings are scanned images with no text; they are counted and skipped.
Only stocks [ST] and stock options [OP] with a ticker are kept.
"""

from __future__ import annotations

import asyncio
import io
import re
import zipfile
from datetime import date, datetime

import httpx
import structlog

from src.tracker.sources import TradeRecord

logger = structlog.get_logger()

INDEX_URL = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.zip"
PDF_URL = "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/{doc}.pdf"
LEGISLATORS_URLS = [
    "https://unitedstates.github.io/congress-legislators/legislators-current.csv",
    "https://unitedstates.github.io/congress-legislators/legislators-historical.csv",
]
FIRST_YEAR = 2014  # e-filed PTRs (text PDFs) are the norm from here on
KEEP_ASSET_TYPES = {"ST": "stock", "OP": "stock-option"}
OWNERS = {"SP": "spouse", "JT": "joint", "DC": "child"}

_TX_RE = re.compile(
    r"^(?P<pre>.*?)(?P<tx>S \(partial\)|P|S|E)\s?"
    r"(?P<d1>\d{2}/\d{2}/\d{4})\s?(?P<d2>\d{2}/\d{2}/\d{4})\s*(?P<amt>.*)$"
)
_TAG_RE = re.compile(r"\[([A-Z]{2})\]")
_TICKER_RE = re.compile(r"\(([A-Z][A-Z0-9.\-]{0,6})\)")
_AMT_RE = re.compile(r"\$(\d[\d,]*)")
_PAGE_CONT_RE = re.compile(r"^(?P<asset>.*\[[A-Z]{2}\])\s*(?P<amt>\$[\d,]+)\s*$")
# Garbled section labels pypdf produces for "Filing Status:", "Description:" …
_LABEL_RE = re.compile(r"^[A-Z](?:\s{2,}[A-Z]?)+\s*:")
_RESET_PREFIXES = ("$200?", "* For the complete", "Filing ID", "=====PAGE", "ID Owner", "Type", "Date", "Gains >")


def _mdY(s: str) -> date | None:
    try:
        return datetime.strptime(s, "%m/%d/%Y").date()
    except ValueError:
        return None


def parse_amount(text: str) -> tuple[float | None, float | None]:
    nums = [float(n.replace(",", "")) for n in _AMT_RE.findall(text)]
    if not nums:
        return None, None
    if len(nums) == 1:
        return nums[0], nums[0]  # "Over $X" or a single figure
    return nums[0], nums[1]


def parse_ptr_text(text: str) -> list[dict]:
    """Transaction rows from one PTR's extracted text (all asset types)."""
    lines = [ln.rstrip() for ln in text.splitlines()]
    rows: list[dict] = []
    block: list[str] = []  # candidate asset lines since the last reset
    skipping_label = False
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        i += 1
        if not ln:
            continue
        m = _TX_RE.match(ln)
        if m and _mdY(m["d1"]) and _mdY(m["d2"]):
            amt = m["amt"].strip()
            # The upper bound often wraps onto the next line
            if amt.endswith("-") and i < len(lines) and lines[i].strip().startswith("$"):
                amt += " " + lines[i].strip()
                i += 1
            pre = m["pre"].strip()
            tail = ""
            if amt.endswith("-"):
                # Row split by a page break: the rest of the asset name and the
                # upper bound sit right after the next page's table header.
                for j in range(i, min(i + 20, len(lines))):
                    cont = _PAGE_CONT_RE.match(lines[j].strip())
                    if cont and any(lines[k].strip().startswith("$200?") for k in range(i, j)):
                        tail, amt = cont["asset"], amt + " " + cont["amt"]
                        i = j + 1
                        break
            cand = (block + ([pre + (" " + tail if tail else "")] if pre or tail else []))[-4:]
            # Asset = lines up to the last one carrying an [XX] tag, starting at
            # the nearest owner-coded line if there is one
            tag_idx = max((k for k, c in enumerate(cand) if _TAG_RE.search(c)), default=len(cand) - 1)
            cand = cand[: tag_idx + 1]
            start = max((k for k, c in enumerate(cand) if c[:3] in ("SP ", "JT ", "DC ")), default=max(0, len(cand) - 2))
            asset_lines = cand[start:]
            asset = " ".join(asset_lines)
            owner = None
            if asset[:3] in ("SP ", "JT ", "DC "):
                owner, asset = OWNERS[asset[:2]], asset[3:]
            tag = _TAG_RE.findall(asset)
            tickers = _TICKER_RE.findall(asset)
            lo, hi = parse_amount(amt)
            rows.append(
                {
                    "owner": owner or "self",
                    "asset": re.sub(r"\s+", " ", _TAG_RE.sub("", asset)).strip(),
                    "asset_code": tag[-1] if tag else None,
                    "ticker": tickers[-1] if tickers else None,
                    "tx": m["tx"],
                    "trade_date": _mdY(m["d1"]),
                    "notified": _mdY(m["d2"]),
                    "amount_low": lo,
                    "amount_high": hi,
                }
            )
            block, skipping_label = [], True
            continue
        if ln.startswith(_RESET_PREFIXES) or ln.startswith("$"):
            block, skipping_label = [], False
            continue
        if _LABEL_RE.match(ln):
            block, skipping_label = [], True
            continue
        if skipping_label and not (ln[:3] in ("SP ", "JT ", "DC ") or _TAG_RE.search(ln) or _TICKER_RE.search(ln)):
            continue  # wrapped description text
        skipping_label = False
        block.append(ln)
    return rows


def _members(xml_text: str) -> list[dict]:
    out = []
    for m in re.findall(r"<Member>(.*?)</Member>", xml_text, re.S):
        get = lambda tag: (re.search(rf"<{tag}>(.*?)</{tag}>", m, re.S) or [None, ""])[1].strip()  # noqa: E731
        if get("FilingType") != "P":
            continue
        out.append(
            {
                "last": get("Last"),
                "first": get("First"),
                "state_dst": get("StateDst"),
                "filed": _mdY(get("FilingDate")),
                "doc": get("DocID"),
            }
        )
    return out


class HouseClient:
    def __init__(self):
        self._client = httpx.AsyncClient(
            headers={"User-Agent": "DisclosureTracker research (+https://github.com/sarahakhavan/disclosure-tracker)"},
            timeout=60.0,
            follow_redirects=True,
        )
        self._sem = asyncio.Semaphore(4)
        self.diag: dict = {"years": {}, "parties": 0}
        self._party: dict[tuple[str, str], str] = {}

    async def _load_parties(self) -> None:
        """(last name, state) → party initial, from the public legislators dataset."""
        import csv

        for url in LEGISLATORS_URLS:
            try:
                r = await self._client.get(url)
                r.raise_for_status()
            except Exception as e:
                self.diag.setdefault("party_errors", []).append(f"{url}: {str(e)[:120]}")
                continue
            for row in csv.DictReader(io.StringIO(r.text)):
                if row.get("type") != "rep":
                    continue
                k = (row.get("last_name", "").lower(), row.get("state", "").upper())
                self._party.setdefault(k, (row.get("party") or "")[:1].upper())
        self.diag["parties"] = len(self._party)

    def _actor(self, mem: dict) -> tuple[str, str, str]:
        state = mem["state_dst"][:2].upper()
        first = re.sub(r"\s+[A-Z]\.$", "", mem["first"]).strip()  # drop trailing middle initial
        party = self._party.get((mem["last"].lower(), state), "")
        name = f"Rep. {first} {mem['last']} ({party}-{state})" if party else f"Rep. {first} {mem['last']} ({state})"
        actor_id = "house:" + re.sub(r"[^a-z]", "", (mem["last"] + first.split(" ")[0]).lower()) + ":" + state
        group = "House" + (f" · {party}" if party else "")
        return actor_id, name, group

    async def _pdf_text(self, year: int, doc: str) -> str | None:
        from pypdf import PdfReader

        async with self._sem:
            try:
                r = await self._client.get(PDF_URL.format(year=year, doc=doc))
                r.raise_for_status()
            except Exception:
                return None
        try:
            reader = PdfReader(io.BytesIO(r.content))
            return "\n=====PAGE=====\n".join(p.extract_text() or "" for p in reader.pages)
        except Exception:
            return ""  # downloaded but unreadable: don't retry forever

    async def fetch_history(
        self, known_docs: set[str], first_year: int = FIRST_YEAR, last_year: int | None = None
    ) -> list[TradeRecord]:
        """New trades from PTRs not in ``known_docs``. Processed doc ids land in
        ``self.processed`` as {doc_id: n_rows}."""
        await self._load_parties()
        self.processed: dict[str, int] = {}
        last_year = last_year or date.today().year
        out: list[TradeRecord] = []
        for year in range(first_year, last_year + 1):
            yd = self.diag["years"][year] = {"ptrs": 0, "new_docs": 0, "no_text": 0, "rows": 0, "kept": 0}
            try:
                r = await self._client.get(INDEX_URL.format(year=year))
                r.raise_for_status()
                z = zipfile.ZipFile(io.BytesIO(r.content))
                xml = next(n for n in z.namelist() if n.lower().endswith(".xml"))
                members = _members(z.read(xml).decode("utf-8", "replace"))
            except Exception as e:
                yd["error"] = str(e)[:200]
                continue
            yd["ptrs"] = len(members)
            todo = [m for m in members if m["doc"] and m["doc"] not in known_docs]
            yd["new_docs"] = len(todo)
            texts = await asyncio.gather(*(self._pdf_text(year, m["doc"]) for m in todo))
            for mem, text in zip(todo, texts):
                if text is None:
                    yd["download_failed"] = yd.get("download_failed", 0) + 1
                    continue  # try again next run
                self.processed[mem["doc"]] = 0
                if not text.strip() or "Transaction" not in text:
                    yd["no_text"] += 1
                    continue
                try:
                    rows = parse_ptr_text(text)
                except Exception as e:  # one odd filing must not stop the run
                    yd["parse_errors"] = yd.get("parse_errors", 0) + 1
                    if len(self.diag.setdefault("parse_error_samples", [])) < 5:
                        self.diag["parse_error_samples"].append(f"{mem['doc']}: {e!r}"[:200])
                    continue
                yd["rows"] += len(rows)
                for r_ in rows:
                    code = r_["asset_code"] or "none"
                    yd.setdefault("codes", {})[code] = yd.setdefault("codes", {}).get(code, 0) + 1
                yd["with_ticker"] = yd.get("with_ticker", 0) + sum(1 for r_ in rows if r_["ticker"])
                actor_id, name, group = self._actor(mem)
                for k, row in enumerate(rows):
                    kind = KEEP_ASSET_TYPES.get(row["asset_code"] or "")
                    if not kind or not row["ticker"] or row["tx"] == "E" or not row["trade_date"]:
                        continue
                    yd["kept"] += 1
                    self.processed[mem["doc"]] += 1
                    out.append(
                        TradeRecord(
                            source="house",
                            source_id=f"{mem['doc']}|{k}",
                            actor_id=actor_id,
                            actor=name,
                            actor_group=group,
                            ticker=row["ticker"].replace("-", "."),
                            issuer=row["asset"][:200],
                            direction="buy" if row["tx"] == "P" else "sell",
                            action={"P": "purchase", "S": "sale", "S (partial)": "partial sale"}[row["tx"]],
                            asset_type=kind,
                            owner=row["owner"],
                            value_usd=(row["amount_low"] + row["amount_high"]) / 2 if row["amount_low"] else None,
                            trade_date=row["trade_date"],
                            disclosure_date=mem["filed"],
                            raw={"doc": mem["doc"], "year": year, "row": {k2: str(v) for k2, v in row.items()}},
                        )
                    )
            logger.info("tracker.house.year", year=year, **yd)
        self.diag["new"] = len(out)
        return out

    async def close(self):
        await self._client.aclose()
