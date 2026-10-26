"""
Builds ``reports/smart_money_report.json``: everything the report page needs,
in a compact columnar shape so a decade of trades fits in a few MB.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.tracker import track_record as tr
from src.tracker.clusters import DEFAULT_MIN_ACTORS, DEFAULT_WINDOW_DAYS, find_clusters
from src.tracker.follow_list import load_follow_list, resolve_followed
from src.tracker.models import TrackedTrade
from src.tracker.prices import BENCHMARK, load_prices

PRICE_CHART_START = "2018-01-01"
MAX_CHART_TICKERS = 900
RECENT_YEARS_FOR_CHARTS = 3

CAVEATS = [
    "Disclosures arrive late: congressional trades up to 45 days after the trade, 13F holdings up to 45 days after quarter-end. Scores use the price after disclosure, which is what a follower could actually get.",
    "Congressional amounts are reported in ranges; values shown are estimates. Spouse and family trades are attributed to the member.",
    "Option trades are measured on the underlying stock, so they understate the leverage of the actual position.",
    "13F filings show long US positions only, at quarter-end. Positions in companies that have since been renamed or delisted can't be matched to a ticker and are left out, which flatters 13F track records.",
    "Delisted tickers without price history are excluded from returns.",
    "Past results of a few dozen trades are weak evidence. Scores are shrunk toward 'market-like' for small samples and hidden below the minimum count.",
]


def _r(x, nd=4):
    if x is None:
        return None
    try:
        if isinstance(x, float) and not math.isfinite(x):
            return None
    except TypeError:
        return None
    return round(float(x), nd)


def _d(x):
    if x is None or (isinstance(x, float) and math.isnan(x)) or pd.isna(x):
        return None
    return pd.Timestamp(x).strftime("%Y-%m-%d")


def load_trades(session_factory) -> pd.DataFrame:
    session = session_factory()
    try:
        rows = session.query(TrackedTrade).all()
        data = [
            {
                "source_id": r.source_id,
                "source": r.source,
                "actor_id": r.actor_id,
                "actor": r.actor,
                "actor_group": r.actor_group,
                "ticker": r.ticker,
                "issuer": r.issuer,
                "direction": r.direction,
                "action": r.action,
                "asset_type": r.asset_type,
                "owner": r.owner,
                "value_usd": r.value_usd,
                "trade_date": r.trade_date,
                "disclosure_date": r.disclosure_date,
            }
            for r in rows
        ]
    finally:
        session.close()
    return pd.DataFrame(data)


def _compact_stats(s: dict) -> list:
    return [s["n"], _r(s["mean"]), _r(s["median"]), _r(s["hit_rate"], 3), _r(s["t_stat"], 2), _r(s["shrunk"])]


def _weekly(series: pd.Series) -> dict | None:
    s = series.dropna()
    s = s[s.index >= PRICE_CHART_START]
    if s.empty:
        return None
    w = s.resample("W-FRI").last().dropna()
    if w.empty:
        return None
    # 4 significant digits is plenty for a chart
    closes = [float(f"{v:.4g}") for v in w.to_numpy()]
    return {"start": w.index[0].strftime("%Y-%m-%d"), "closes": closes}


def build_report(trades: pd.DataFrame, prices: pd.DataFrame, follow_entries: list[dict]) -> dict:
    decisions = tr.collapse_decisions(trades)
    decisions = tr.compute_decision_returns(decisions, prices)
    actors = tr.actor_track_records(decisions)

    actor_ids = actors["actor_id"].tolist() if not actors.empty else []
    actor_idx = {a: i for i, a in enumerate(actor_ids)}
    tickers = sorted(decisions["ticker"].unique().tolist()) if not decisions.empty else []
    ticker_idx = {t: i for i, t in enumerate(tickers)}
    issuers = (
        decisions.dropna(subset=["issuer"]).groupby("ticker")["issuer"].last().to_dict()
        if not decisions.empty
        else {}
    )

    followed = resolve_followed(
        follow_entries, dict(zip(actors["actor_id"], actors["actor"])) if not actors.empty else {}
    )

    # ── actors ──
    actor_rows = []
    for a in actors.to_dict("records"):
        actor_rows.append(
            {
                "id": a["actor_id"],
                "name": a["actor"],
                "group": a["actor_group"],
                "source": a["source"],
                "score": _r(a["score"], 1),
                "score_n": int(a["score_n"]),
                "n": a["n_decisions"],
                "buys": a["n_buys"],
                "sells": a["n_sells"],
                "tickers": a["n_tickers"],
                "first": _d(a["first_trade"]),
                "last": _d(a["last_trade"]),
                "last_disclosed": _d(a["last_disclosure"]),
                "lag": _r(a["median_lag_days"], 0),
                "stats": {
                    k: {side: _compact_stats(v[side]) for side in ("all", "buy", "sell")}
                    for k, v in a["stats"].items()
                },
            }
        )

    # ── decisions (columnar) ──
    cols: dict[str, list] = {
        k: []
        for k in [
            "a", "t", "dir", "td", "dd", "val", "lots", "act", "asset", "owner",
            *[f"{an}_{hz}" for an in tr.ANCHORS for hz in tr.HORIZONS],
        ]
    }
    for row in decisions.to_dict("records"):
        cols["a"].append(actor_idx[row["actor_id"]])
        cols["t"].append(ticker_idx[row["ticker"]])
        cols["dir"].append(1 if row["direction"] == "buy" else -1)
        cols["td"].append(_d(row["trade_date"]))
        cols["dd"].append(_d(row["disclosure_date"]))
        v = row.get("value_usd")
        cols["val"].append(None if v is None or (isinstance(v, float) and math.isnan(v)) else round(v))
        cols["lots"].append(int(row["n_lots"]))
        cols["act"].append(row.get("action") or None)
        cols["asset"].append(row.get("asset_type") or None)
        cols["owner"].append(row.get("owner") or None)
        for an in tr.ANCHORS:
            for hz in tr.HORIZONS:
                cols[f"{an}_{hz}"].append(_r(row[f"{an}_{hz}"]))

    # ── default clusters (all actors, last 3 years) ──
    since = pd.Timestamp.today().normalize() - pd.DateOffset(years=RECENT_YEARS_FOR_CHARTS)
    clusters = []
    for c in find_clusters(decisions, DEFAULT_WINDOW_DAYS, DEFAULT_MIN_ACTORS, "buy", since=since)[:300]:
        clusters.append(
            {
                "ticker": c["ticker"],
                "n_actors": c["n_actors"],
                "actor_ids": c["actor_ids"],
                "first": _d(c["first_trade"]),
                "last": _d(c["last_trade"]),
                "public": _d(c["complete_disclosure"]),
                "value": round(c["total_value_usd"]) if math.isfinite(c["total_value_usd"]) else None,
            }
        )

    # ── weekly prices for charting ──
    recent = decisions[decisions["trade_date"] >= since] if not decisions.empty else decisions
    priority: list[str] = []
    seen: set[str] = set()

    def add(ts):
        for t in ts:
            if t not in seen and t in prices.columns:
                seen.add(t)
                priority.append(t)

    if not decisions.empty:
        add(decisions.loc[decisions["actor_id"].isin(followed), "ticker"].value_counts().index)
        add([c["ticker"] for c in clusters])
        add(recent["ticker"].value_counts().index)
        add(decisions["ticker"].value_counts().index)
    chart_prices = {}
    for t in [BENCHMARK] + priority[:MAX_CHART_TICKERS]:
        if t in prices.columns:
            w = _weekly(prices[t])
            if w:
                chart_prices[t] = w

    n_priced = int(decisions["priced"].sum()) if not decisions.empty else 0
    data_through = prices.index.max().strftime("%Y-%m-%d") if not prices.empty else None

    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "prices_through": data_through,
            "n_trades": int(len(trades)),
            "n_decisions": int(len(decisions)),
            "n_priced": n_priced,
            "n_actors": len(actor_rows),
            "sources": sorted(trades["source"].unique().tolist()) if not trades.empty else [],
            "method": {
                "horizons": tr.HORIZONS,
                "anchors": list(tr.ANCHORS),
                "benchmark": BENCHMARK,
                "score_anchor": tr.SCORE_ANCHOR,
                "score_horizon": tr.SCORE_HORIZON,
                "shrink_k": tr.SHRINK_K,
                "min_decisions": tr.MIN_DECISIONS,
                "score_scale": tr.SCORE_SCALE,
                "cluster_window_days": DEFAULT_WINDOW_DAYS,
                "cluster_min_actors": DEFAULT_MIN_ACTORS,
            },
            "caveats": CAVEATS,
            "sample": False,
        },
        "actors": actor_rows,
        "tickers": tickers,
        "issuers": {t: issuers.get(t) for t in tickers if issuers.get(t)},
        "decisions": cols,
        "clusters": clusters,
        "follow_default": sorted(followed),
        "prices": chart_prices,
    }


def build_report_from_db(session_factory, follow_path=None) -> dict:
    trades = load_trades(session_factory)
    prices = load_prices(session_factory, set(trades["ticker"]) if not trades.empty else set())
    entries = load_follow_list(follow_path) if follow_path else load_follow_list()
    return build_report(trades, prices, entries)


def decision_frame(report: dict) -> pd.DataFrame:
    """Rebuild a decisions frame from a report (used by alerts and tests)."""
    d = report["decisions"]
    if not d["a"]:
        return pd.DataFrame()
    df = pd.DataFrame(d)
    df["actor_id"] = [report["actors"][i]["id"] for i in df["a"]]
    df["actor"] = [report["actors"][i]["name"] for i in df["a"]]
    df["ticker"] = [report["tickers"][i] for i in df["t"]]
    df["direction"] = np.where(df["dir"] > 0, "buy", "sell")
    df["trade_date"] = pd.to_datetime(df["td"])
    df["disclosure_date"] = pd.to_datetime(df["dd"])
    df["value_usd"] = df["val"].astype(float)
    return df
