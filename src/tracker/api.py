"""
FastAPI routes for the tracker, served from the latest report JSON so they
stay fast and need no database access.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Body, Depends, HTTPException, Query

from src.tracker.clusters import find_clusters
from src.tracker.follow_list import load_follow_list, resolve_followed, save_follow_list
from src.tracker.report import decision_frame

REPORT_PATH = Path("reports/smart_money_report.json")


def build_router(require_admin) -> APIRouter:
    router = APIRouter(prefix="/api/tracker", tags=["tracker"])
    cache: dict = {"mtime": None, "report": None, "df": None}

    def report() -> dict:
        if not REPORT_PATH.exists():
            raise HTTPException(404, "No tracker report yet. Run `python -m src.tracker all`.")
        m = REPORT_PATH.stat().st_mtime
        if cache["mtime"] != m:
            cache["report"] = json.loads(REPORT_PATH.read_text())
            cache["df"] = decision_frame(cache["report"])
            cache["mtime"] = m
        return cache["report"]

    def followed_ids(r: dict) -> set[str]:
        return resolve_followed(load_follow_list(), {a["id"]: a["name"] for a in r["actors"]})

    @router.get("/actors")
    def actors(
        followed_only: bool = False,
        source: str | None = Query(None, pattern="^(congress|13f)$"),
        min_trades: int = 0,
    ):
        r = report()
        fol = followed_ids(r)
        rows = [
            {k: v for k, v in a.items() if k != "stats"} | {"followed": a["id"] in fol}
            for a in r["actors"]
            if (not followed_only or a["id"] in fol)
            and (source is None or (a["source"] == "13f") == (source == "13f"))
            and a["n"] >= min_trades
        ]
        return {"meta": r["meta"], "actors": rows}

    @router.get("/actors/{actor_id}")
    def actor(actor_id: str, limit: int = Query(200, le=2000)):
        r = report()
        a = next((x for x in r["actors"] if x["id"] == actor_id), None)
        if a is None:
            raise HTTPException(404, "Unknown actor")
        df = cache["df"]
        mine = df[df["actor_id"] == actor_id].sort_values("trade_date", ascending=False).head(limit)
        trades = [
            {
                "ticker": t.ticker,
                "direction": t.direction,
                "trade_date": t.td,
                "disclosure_date": t.dd,
                "value_usd": None if pd.isna(t.value_usd) else t.value_usd,
                "excess_3m_after_disclosure": t.disclosure_3m,
            }
            for t in mine.itertuples()
        ]
        return {"actor": a, "trades": trades}

    @router.get("/clusters")
    def clusters(
        window_days: int = Query(45, ge=1, le=365),
        min_actors: int = Query(2, ge=2, le=20),
        direction: str = Query("buy", pattern="^(buy|sell)$"),
        followed_only: bool = False,
        since: str | None = None,
    ):
        r = report()
        df = cache["df"]
        if df.empty:
            return {"clusters": []}
        ids = followed_ids(r) if followed_only else None
        since_ts = pd.Timestamp(since) if since else pd.Timestamp.today() - pd.DateOffset(years=2)
        out = find_clusters(df, window_days, min_actors, direction, ids, since_ts)[:200]
        for c in out:
            for k in ("first_trade", "last_trade", "complete_disclosure"):
                c[k] = None if c[k] is None or pd.isna(c[k]) else pd.Timestamp(c[k]).strftime("%Y-%m-%d")
        return {"clusters": out}

    @router.get("/follow-list")
    def get_follow_list():
        return {"follow": load_follow_list()}

    @router.put("/follow-list", dependencies=[Depends(require_admin)])
    def put_follow_list(entries: list[dict] = Body(..., embed=True, alias="follow")):
        save_follow_list(entries)
        return {"follow": load_follow_list()}

    return router
