"""
Tracker CLI.

    python -m src.tracker update            # fetch new congress + 13F trades and prices
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
from src.tracker.models import TrackedTrade, create_tables
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


async def cmd_update(args) -> None:
    """Fetch new data. Writes reports/tracker_run.json with what each source did."""
    sf = _session_factory()
    diag: dict = {"started": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if not args.skip_congress:
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
    p.add_argument("--skip-congress", action="store_true")
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
    main()
