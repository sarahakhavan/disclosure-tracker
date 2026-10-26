"""
Tracker CLI.

    python -m src.tracker update            # fetch new House PTR + 13F trades and prices
    python -m src.tracker report            # write reports/smart_money_report.json
    python -m src.tracker alerts            # Slack: new followed trades + new clusters
    python -m src.tracker all               # update, then alerts, then report

Uses TRACKER_DATABASE_URL (default sqlite:///./tracker.db), separate from the
main pipeline database so the two never fight over a cache.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

from datetime import datetime, timezone

import structlog
from sqlalchemy import func
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from src.models.database import get_session_factory
from src.tracker.house import HouseClient
from src.tracker.models import SeenDocument, TrackedTrade, create_tables
from src.tracker.prices import update_prices
from src.tracker.report import build_report_from_db, decision_frame
from src.tracker.sources import (
    CongressHistoryClient,
    ThirteenFHistoryClient,
    record_to_row,
)

logger = structlog.get_logger()

DEFAULT_DB = "sqlite:///./tracker.db"
DEFAULT_OUT = Path("reports/smart_money_report.json")


def _session_factory():
    url = os.environ.get("TRACKER_DATABASE_URL", DEFAULT_DB)
    return get_session_factory(create_tables(url))


def _known_ids(sf, source: str) -> set[str]:
    s = sf()
    try:
        return {r[0] for r in s.query(TrackedTrade.source_id).filter(TrackedTrade.source == source)}
    finally:
        s.close()


def _store(sf, records) -> int:
    rows = [record_to_row(r) for r in records]
    if not rows:
        return 0
    s = sf()
    try:
        for i in range(0, len(rows), 400):
            stmt = sqlite_insert(TrackedTrade).values(rows[i : i + 400])
            s.execute(stmt.on_conflict_do_nothing(index_elements=["source", "source_id"]))
        s.commit()
    finally:
        s.close()
    return len(rows)


DIAG_PATH = Path("reports/tracker_run.json")


def _write_diag(diag: dict) -> None:
    DIAG_PATH.parent.mkdir(parents=True, exist_ok=True)
    DIAG_PATH.write_text(json.dumps(diag, indent=2, default=str))


def _mark_seen(sf, source: str, docs: dict[str, int]) -> None:
    if not docs:
        return
    s = sf()
    try:
        rows = [{"source": source, "doc_id": d, "n_rows": n} for d, n in docs.items()]
        for i in range(0, len(rows), 400):
            stmt = sqlite_insert(SeenDocument).values(rows[i : i + 400])
            s.execute(stmt.on_conflict_do_nothing(index_elements=["source", "doc_id"]))
        s.commit()
    finally:
        s.close()


async def cmd_update(args) -> None:
    """Fetch new data. Writes reports/tracker_run.json with what each source did."""
    sf = _session_factory()
    diag: dict = {"started": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if not args.skip_house:
        client = HouseClient()
        try:
            s = sf()
            try:
                seen = {r[0] for r in s.query(SeenDocument.doc_id).filter(SeenDocument.source == "house")}
            finally:
                s.close()
            recs = await client.fetch_history(seen, first_year=args.house_from)
            stored = _store(sf, recs)
            _mark_seen(sf, "house", client.processed)
            diag["house"] = client.diag | {"stored": stored, "docs_processed": len(client.processed)}
        except Exception as e:
            diag["house"] = client.diag | {"fatal": repr(e)[:300]}
        finally:
            await client.close()
        logger.info("tracker.house", new=diag["house"].get("new"), stored=diag["house"].get("stored"))
        _write_diag(diag)
    if args.with_capitoltrades:
        client = CongressHistoryClient()
        try:
            recs = await client.fetch_history(_known_ids(sf, "congress"), max_pages=args.max_pages)
            diag["congress"] = client.diag | {"stored": _store(sf, recs)}
        except Exception as e:
            diag["congress"] = client.diag | {"fatal": str(e)[:300]}
        finally:
            await client.close()
        logger.info("tracker.congress", **{k: v for k, v in diag["congress"].items() if k != "probes"})
    if not args.skip_13f:
        client = ThirteenFHistoryClient()
        try:
            recs = await client.fetch_history(_known_ids(sf, "13f"))
            diag["13f"] = {"funds": getattr(client, "diag", {}), "stored": _store(sf, recs)}
        except Exception as e:
            diag["13f"] = {"funds": getattr(client, "diag", {}), "fatal": str(e)[:300]}
        finally:
            await client.close()
    _write_diag(diag)
    if not args.skip_prices:
        s = sf()
        try:
            tickers = {r[0] for r in s.query(TrackedTrade.ticker).distinct()}
        finally:
            s.close()
        diag["prices"] = {"tickers": len(tickers), "bars_written": update_prices(sf, tickers)}
    s = sf()
    try:
        diag["db_totals"] = {
            src: n for src, n in s.query(TrackedTrade.source, func.count()).group_by(TrackedTrade.source)
        }
    finally:
        s.close()
    diag["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    DIAG_PATH.parent.mkdir(parents=True, exist_ok=True)
    DIAG_PATH.write_text(json.dumps(diag, indent=2, default=str))


def cmd_report(args) -> dict:
    report = build_report_from_db(_session_factory())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, separators=(",", ":"), allow_nan=False))
    logger.info(
        "tracker.report_written",
        path=str(out),
        kb=round(out.stat().st_size / 1024),
        actors=report["meta"]["n_actors"],
        decisions=report["meta"]["n_decisions"],
    )
    return report


async def cmd_alerts(args, new_report: dict | None = None) -> None:
    """Compare a fresh report with the previous one on disk; post what's new."""
    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    out = Path(args.out)
    if not webhook or not out.exists():
        logger.info("tracker.alerts.skipped", reason="no webhook" if not webhook else "no previous report")
        return
    old = json.loads(out.read_text())
    new = new_report or build_report_from_db(_session_factory())

    followed = set(new.get("follow_default", []))
    old_df, new_df = decision_frame(old), decision_frame(new)
    key = ["actor_id", "ticker", "direction", "td"]
    old_keys = set(map(tuple, old_df[key].values)) if not old_df.empty else set()
    fresh = (
        new_df[new_df["actor_id"].isin(followed) & ~new_df[key].apply(tuple, axis=1).isin(old_keys)]
        if not new_df.empty
        else new_df
    )
    old_clusters = {(c["ticker"], c["first"]) for c in old.get("clusters", [])}
    new_clusters = [c for c in new.get("clusters", []) if (c["ticker"], c["first"]) not in old_clusters]

    lines = []
    for r in fresh.sort_values("dd", ascending=False).head(25).itertuples():
        lines.append(f"• *{r.actor}* {r.direction.upper()} `{r.ticker}` (traded {r.td}, disclosed {r.dd})")
    names = {a["id"]: a["name"] for a in new["actors"]}
    for c in new_clusters[:10]:
        who = ", ".join(names.get(a, a) for a in c["actor_ids"][:5])
        lines.append(f"• :busts_in_silhouette: Cluster: {c['n_actors']} people bought `{c['ticker']}` ({c['first']} → {c['last']}): {who}")
    if not lines:
        logger.info("tracker.alerts.nothing_new")
        return

    from src.alerts.notifier import SlackNotifier

    notifier = SlackNotifier(webhook)
    try:
        await notifier._send(
            {"text": "Smart-money tracker: new activity\n" + "\n".join(lines)},
            "tracker_digest",
            "multi",
        )
    finally:
        await notifier.close()


def main():
    p = argparse.ArgumentParser(prog="python -m src.tracker")
    p.add_argument("command", choices=["update", "report", "alerts", "all"])
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--max-pages", type=int, default=600)
    p.add_argument("--skip-house", action="store_true")
    p.add_argument("--house-from", type=int, default=2014, help="first year of House PTRs to backfill")
    p.add_argument(
        "--with-capitoltrades",
        action="store_true",
        help="also scrape Capitol Trades (blocked from cloud CI; works from a home connection)",
    )
    p.add_argument("--skip-13f", action="store_true")
    p.add_argument("--skip-prices", action="store_true")
    args = p.parse_args()

    if args.command == "update":
        asyncio.run(cmd_update(args))
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "alerts":
        asyncio.run(cmd_alerts(args))
    else:
        asyncio.run(cmd_update(args))
        fresh = build_report_from_db(_session_factory())
        asyncio.run(cmd_alerts(args, fresh))
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(fresh, separators=(",", ":"), allow_nan=False))
        logger.info("tracker.report_written", path=str(out), kb=round(out.stat().st_size / 1024))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        Path("reports").mkdir(exist_ok=True)
        Path("reports/tracker_error.txt").write_text(traceback.format_exc())
        raise
