"""
Historical backfill for the tracker.

* Congress: pages through Capitol Trades newest-disclosure-first. On the first
  run it walks the whole archive; later runs stop once a full page of already
  stored transactions is seen.
* 13F: walks every 13F-HR a tracked fund has filed in the XML era (mid-2013
  onward) and diffs consecutive quarters into new / added / cut / exited events.

Both reuse the parsing in the existing pipeline clients.
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import date, datetime

import structlog

from src.clients.congress import CongressClient
from src.clients.edgar_13f import (
    ADD_THRESHOLD,
    TRACKED_FUNDS,
    Edgar13FClient,
    _normalize_issuer,
)

logger = structlog.get_logger()

# A position cut by at least this fraction counts as a SELL decision.
CUT_THRESHOLD = 0.5
# 13F information tables are machine-readable XML from Q2 2013 onward.
XML_ERA_START = date(2013, 6, 1)
# Before 2023 filers reported 13F values in thousands of dollars.
WHOLE_DOLLAR_SWITCH = date(2023, 1, 1)


@dataclass
class TradeRecord:
    source: str
    source_id: str
    actor_id: str
    actor: str
    actor_group: str | None
    ticker: str
    issuer: str | None
    direction: str
    action: str | None
    asset_type: str | None
    owner: str | None
    value_usd: float | None
    trade_date: date
    disclosure_date: date | None
    raw: dict = field(default_factory=dict)


# ── Congress ────────────────────────────────────────────────────────────────


class CongressHistoryClient(CongressClient):
    PAGE_SIZE = 96

    async def _fetch_page_sorted(self, page: int) -> list[dict]:
        url = (
            f"{self.BASE_URL}?page={page}&pageSize={self.PAGE_SIZE}"
            "&sortBy=-pubDate"
        )
        resp = await self._client.get(url)
        resp.raise_for_status()
        return self._extract_trades_from_rsc(resp.text)

    @staticmethod
    def to_record(trade: dict) -> TradeRecord | None:
        tx_id = trade.get("_txId")
        pol_id = trade.get("_politicianId")
        if tx_id is None or not pol_id:
            return None

        issuer = trade.get("issuer") or {}
        ticker = (issuer.get("issuerTicker") or "").strip().upper()
        if ":" in ticker:
            ticker = ticker.split(":")[0]
        if not ticker or len(ticker) > 12:
            return None

        tx_type = (trade.get("txType") or "").lower()
        if tx_type in ("buy", "purchase"):
            direction = "buy"
        elif tx_type.startswith("sell") or tx_type.startswith("sale"):
            direction = "sell"
        else:
            return None  # exchanges, receipts, etc.

        try:
            trade_date = datetime.strptime(trade["txDate"], "%Y-%m-%d").date()
        except (KeyError, TypeError, ValueError):
            return None

        disclosure_date = None
        pub = trade.get("pubDate")
        if pub:
            try:
                disclosure_date = datetime.fromisoformat(
                    pub.replace("Z", "+00:00")
                ).date()
            except ValueError:
                pass
        # Some records carry a filing date instead of / as well as pubDate
        filed = trade.get("filingDate")
        if filed and not disclosure_date:
            try:
                disclosure_date = datetime.strptime(filed[:10], "%Y-%m-%d").date()
            except ValueError:
                pass

        pol = trade.get("politician") or {}
        chamber = (trade.get("chamber") or "").lower()
        prefix = "Sen." if chamber == "senate" else "Rep."
        party = (pol.get("party") or "")[:1].upper()
        state = (trade.get("_stateId") or pol.get("_stateId") or "").upper()
        name = f"{pol.get('firstName', '')} {pol.get('lastName', '')}".strip()
        suffix = f" ({party}-{state})" if party and state else ""
        actor = f"{prefix} {name}{suffix}".strip()

        asset = trade.get("asset") or {}
        value = trade.get("value")

        return TradeRecord(
            source="congress",
            source_id=str(tx_id),
            actor_id=f"cg:{pol_id}",
            actor=actor,
            actor_group=f"{'Senate' if chamber == 'senate' else 'House'}"
            + (f" · {party}" if party else ""),
            ticker=ticker,
            issuer=issuer.get("issuerName"),
            direction=direction,
            action=tx_type or None,
            asset_type=asset.get("assetType"),
            owner=trade.get("owner"),
            value_usd=float(value) if value else None,
            trade_date=trade_date,
            disclosure_date=disclosure_date,
            raw=trade,
        )

    async def fetch_history(
        self,
        known_ids: set[str],
        max_pages: int = 600,
        delay_s: float = 1.0,
    ) -> list[TradeRecord]:
        """Walk pages newest-first; stop at the end or at a fully-known page."""
        out: list[TradeRecord] = []
        for page in range(1, max_pages + 1):
            try:
                raw = await self._fetch_page_sorted(page)
            except Exception as e:  # network / layout change
                logger.warning("tracker.congress.page_failed", page=page, error=str(e))
                break
            if not raw:
                break
            recs = [r for t in raw if (r := self.to_record(t))]
            new = [r for r in recs if r.source_id not in known_ids]
            out.extend(new)
            if page % 25 == 0:
                logger.info("tracker.congress.progress", page=page, new=len(out))
            if known_ids and not new and recs:
                break  # caught up with what we already have
            await asyncio.sleep(delay_s)
        logger.info("tracker.congress.fetched", new=len(out))
        return out


# ── 13F ─────────────────────────────────────────────────────────────────────


class ThirteenFHistoryClient(Edgar13FClient):
    async def _all_13f_filings(self, cik: int) -> tuple[str, list[tuple]]:
        """Every original 13F-HR → [(accession, period, filed)] oldest first."""
        sub = (await self._get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")).json()
        name = sub.get("name") or str(cik)
        blocks = [sub.get("filings", {}).get("recent", {})]
        for extra in sub.get("filings", {}).get("files", []):
            try:
                blocks.append(
                    (await self._get(f"https://data.sec.gov/submissions/{extra['name']}")).json()
                )
            except Exception as e:
                logger.warning("tracker.13f.extra_index_failed", cik=cik, error=str(e))

        by_period: dict[str, tuple] = {}
        for b in blocks:
            for i, form in enumerate(b.get("form", [])):
                if form != "13F-HR":
                    continue
                period, filed = b["reportDate"][i], b["filingDate"][i]
                if not period or date.fromisoformat(filed) < XML_ERA_START:
                    continue
                # Keep the earliest original filing for each period
                cur = by_period.get(period)
                if cur is None or filed < cur[2]:
                    by_period[period] = (b["accessionNumber"][i], period, filed)
        return name, sorted(by_period.values(), key=lambda t: t[1])

    @staticmethod
    def diff_quarters(
        old: dict[str, dict], new: dict[str, dict]
    ) -> list[tuple[str, str, dict]]:
        """Return [(direction, action, position)] for one quarter-over-quarter diff."""
        events = []
        for cusip, pos in new.items():
            prev = old.get(cusip)
            if prev is None:
                events.append(("buy", "new_position", pos))
            elif prev["shares"] > 0 and pos["shares"] / prev["shares"] >= ADD_THRESHOLD:
                events.append(("buy", "added", pos))
            elif prev["shares"] > 0 and pos["shares"] / prev["shares"] <= 1 - CUT_THRESHOLD:
                events.append(("sell", "cut", pos))
        for cusip, pos in old.items():
            if cusip not in new:
                events.append(("sell", "exited", pos))
        return events

    async def fetch_history(
        self,
        known_ids: set[str],
        funds: dict[str, int] | None = None,
    ) -> list[TradeRecord]:
        name_map = await self._load_name_map()
        out: list[TradeRecord] = []
        for fund_label, cik in (funds or TRACKED_FUNDS).items():
            try:
                _, filings = await self._all_13f_filings(cik)
            except Exception as e:
                logger.warning("tracker.13f.index_failed", fund=fund_label, error=str(e))
                continue

            # Incremental runs: once a fund's history is stored, only the last
            # few quarters can contain anything new, so skip re-downloading
            # a decade of holdings tables.
            if any(k.startswith(f"{cik}|") for k in known_ids):
                filings = filings[-3:]

            prev_holdings: dict[str, dict] | None = None
            for acc, period, filed in filings:
                diff_key = f"{cik}|{period}"
                try:
                    holdings = await self._holdings(cik, acc)
                except Exception as e:
                    logger.warning("tracker.13f.holdings_failed", fund=fund_label, period=period, error=str(e))
                    holdings = {}
                if not holdings:
                    prev_holdings = None  # can't diff across a gap
                    continue
                if prev_holdings is not None:
                    scale = 1000.0 if date.fromisoformat(filed) < WHOLE_DOLLAR_SWITCH else 1.0
                    for direction, action, pos in self.diff_quarters(prev_holdings, holdings):
                        ticker = name_map.get(_normalize_issuer(pos["issuer"]))
                        if not ticker:
                            continue
                        sid = f"{diff_key}|{pos['cusip']}|{action}"
                        if sid in known_ids:
                            continue
                        out.append(
                            TradeRecord(
                                source="13f",
                                source_id=sid,
                                actor_id=f"cik:{cik}",
                                actor=fund_label,
                                actor_group="Fund (13F)",
                                ticker=ticker,
                                issuer=pos["issuer"],
                                direction=direction,
                                action=action,
                                asset_type="stock",
                                owner=None,
                                value_usd=pos["value"] * scale,
                                trade_date=date.fromisoformat(period),
                                disclosure_date=date.fromisoformat(filed),
                                raw={
                                    "cusip": pos["cusip"],
                                    "shares": pos["shares"],
                                    "accession": acc,
                                },
                            )
                        )
                prev_holdings = holdings
            logger.info("tracker.13f.fund_done", fund=fund_label, quarters=len(filings))
        logger.info("tracker.13f.fetched", new=len(out))
        return out

    async def _holdings(self, cik: int, accession: str) -> dict[str, dict]:
        """Like the parent, but picks the information table more carefully."""
        import re

        acc_nodash = accession.replace("-", "")
        index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/"
        resp = await self._get(index_url)
        xml_files = re.findall(r'href="([^"]+\.xml)"', resp.text, flags=re.I)
        candidates = [
            f for f in xml_files
            if "primary_doc" not in f.lower() and "xsl" not in f.lower()
        ]
        if not candidates:
            return {}
        candidates.sort(key=lambda f: ("info" not in f.lower() and "table" not in f.lower()))
        url = candidates[0]
        if url.startswith("/"):
            url = f"https://www.sec.gov{url}"
        elif not url.startswith("http"):
            url = index_url + url
        return self._parse_info_table((await self._get(url)).text)


def record_to_row(rec: TradeRecord) -> dict:
    return {
        "source": rec.source,
        "source_id": rec.source_id,
        "actor_id": rec.actor_id,
        "actor": rec.actor,
        "actor_group": rec.actor_group,
        "ticker": rec.ticker,
        "issuer": rec.issuer,
        "direction": rec.direction,
        "action": rec.action,
        "asset_type": rec.asset_type,
        "owner": rec.owner,
        "value_usd": rec.value_usd,
        "trade_date": rec.trade_date,
        "disclosure_date": rec.disclosure_date,
        "raw_payload": json.dumps(rec.raw, default=str)[:4000],
    }
