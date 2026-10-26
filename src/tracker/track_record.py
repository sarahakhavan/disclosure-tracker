"""
Per-person track records.

Method
------
1. Lots are collapsed into *decisions*: one person, one ticker, one direction,
   one trade date. Ten same-day lots of NVDA are one call, not ten.
2. Each decision is priced from two entry points:
   * ``trade``      – the close on/after the trade date (the person's own skill)
   * ``disclosure`` – the first close *after* the disclosure date (what someone
     copying the filing could actually have captured)
3. Returns are measured over 1, 3, 6 and 12 months (21/63/126/252 trading
   days) as excess return over SPY. Sells are scored as SPY minus the stock,
   i.e. a sell is "right" if the stock then lagged the market.
4. Per person: count, mean, median, hit rate and t-stat of excess returns,
   plus a mean shrunk toward zero (``n / (n + SHRINK_K)``) so a person with
   three lucky trades cannot top the table.
5. The headline score (0–100, 50 = matched the market) uses the followable
   entry (disclosure) at 3 months, and is only given with at least
   ``MIN_DECISIONS`` matured decisions.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.tracker.prices import BENCHMARK

HORIZONS = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}
ANCHORS = ("trade", "disclosure")
SHRINK_K = 10
MIN_DECISIONS = 8
SCORE_ANCHOR = "disclosure"
SCORE_HORIZON = "3m"
SCORE_SCALE = 0.08  # 8% shrunk excess → ~88/100


def collapse_decisions(trades: pd.DataFrame) -> pd.DataFrame:
    """Lots → decisions. Expects one row per stored trade."""
    if trades.empty:
        return trades.assign(n_lots=pd.Series(dtype=int))
    t = trades.copy()
    t["trade_date"] = pd.to_datetime(t["trade_date"], errors="coerce")
    t["disclosure_date"] = pd.to_datetime(t["disclosure_date"], errors="coerce")
    # Hand-typed filings contain impossible dates (e.g. year 3031). Drop
    # trades dated before 2008, in the future, or after their own filing.
    today = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    ok = t["trade_date"].between(pd.Timestamp("2008-01-01"), today)
    ok &= t["disclosure_date"].isna() | (t["trade_date"] <= t["disclosure_date"] + pd.Timedelta(days=1))
    t.loc[t["disclosure_date"] > today, "disclosure_date"] = pd.NaT
    t = t[ok]
    if t.empty:
        return t.assign(n_lots=pd.Series(dtype=int))
    keys = ["actor_id", "ticker", "direction", "trade_date"]
    agg = t.groupby(keys, as_index=False).agg(
        actor=("actor", "first"),
        actor_group=("actor_group", "first"),
        source=("source", "first"),
        issuer=("issuer", "first"),
        action=("action", "first"),
        asset_type=("asset_type", lambda s: ",".join(sorted({str(x) for x in s if x}))),
        owner=("owner", lambda s: ",".join(sorted({str(x) for x in s if x}))),
        value_usd=("value_usd", lambda s: float(np.nansum(s.astype(float))) if s.notna().any() else np.nan),
        disclosure_date=("disclosure_date", "min"),
        n_lots=("source_id", "count"),
    )
    return agg.sort_values("trade_date").reset_index(drop=True)


# Entries below this price are skipped: Yahoo's adjusted history for penny and
# OTC names is unreliable (adjusted closes can even go negative).
MIN_ENTRY_PRICE = 1.0


def _forward(px: np.ndarray, start_idx: int, h: int, min_entry: float = 0.0) -> float | None:
    end = start_idx + h
    if start_idx < 0 or end >= len(px):
        return None
    a, b = px[start_idx], px[end]
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0 or b <= 0 or a < min_entry:
        return None
    return float(b / a - 1.0)


def compute_decision_returns(decisions: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Adds ``{anchor}_{horizon}`` signed excess-return columns (None = not matured / no price)."""
    out = decisions.copy()
    for a in ANCHORS:
        out[f"{a}_entry"] = pd.NaT
        for hz in HORIZONS:
            out[f"{a}_{hz}"] = np.nan
    out["priced"] = False
    if out.empty or prices.empty or BENCHMARK not in prices:
        return out

    dates = prices.index.values
    bench = prices[BENCHMARK].to_numpy(dtype=float)
    cols = {c: prices[c].to_numpy(dtype=float) for c in prices.columns}

    results = {k: np.full(len(out), np.nan) for a in ANCHORS for k in [f"{a}_{h}" for h in HORIZONS]}
    entries = {a: np.full(len(out), np.datetime64("NaT", "ns"), dtype="datetime64[ns]") for a in ANCHORS}
    priced = np.zeros(len(out), dtype=bool)

    for i, row in enumerate(out.itertuples(index=False)):
        px = cols.get(row.ticker)
        if px is None:
            continue
        sign = 1.0 if row.direction == "buy" else -1.0
        for a in ANCHORS:
            if a == "trade":
                anchor_dt = row.trade_date
                side = "left"  # close on or after the trade date
            else:
                anchor_dt = row.disclosure_date
                side = "right"  # first close strictly after disclosure
            if pd.isna(anchor_dt):
                continue
            idx = int(np.searchsorted(dates, np.datetime64(anchor_dt), side=side))
            # Skip forward over gaps in this ticker's own history
            while idx < len(px) and not np.isfinite(px[idx]):
                idx += 1
            if idx >= len(px):
                continue
            entries[a][i] = dates[idx]
            for hz, h in HORIZONS.items():
                r = _forward(px, idx, h, MIN_ENTRY_PRICE)
                b = _forward(bench, idx, h)
                if r is None or b is None:
                    continue
                results[f"{a}_{hz}"][i] = sign * (r - b)
                priced[i] = True

    for k, v in results.items():
        out[k] = v
    for a in ANCHORS:
        out[f"{a}_entry"] = entries[a]
    out["priced"] = priced
    return out


def _stats(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    n = int(len(x))
    if n == 0:
        return {"n": 0, "mean": None, "median": None, "hit_rate": None, "t_stat": None, "shrunk": None}
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else float("nan")
    t = mean / (sd / math.sqrt(n)) if n > 1 and sd > 0 else None
    return {
        "n": n,
        "mean": mean,
        "median": float(np.median(x)),
        "hit_rate": float((x > 0).mean()),
        "t_stat": t,
        "shrunk": mean * n / (n + SHRINK_K),
    }


def score_from_shrunk(shrunk: float | None, n: int) -> float | None:
    if shrunk is None or n < MIN_DECISIONS:
        return None
    return round(50 + 50 * math.tanh(shrunk / SCORE_SCALE), 1)


def actor_track_records(dec: pd.DataFrame) -> pd.DataFrame:
    """One row per actor with headline score and the full stats grid."""
    rows = []
    for actor_id, g in dec.groupby("actor_id"):
        rec = {
            "actor_id": actor_id,
            "actor": g["actor"].iloc[-1],
            "actor_group": g["actor_group"].iloc[-1],
            "source": g["source"].iloc[-1],
            "n_decisions": int(len(g)),
            "n_buys": int((g["direction"] == "buy").sum()),
            "n_sells": int((g["direction"] == "sell").sum()),
            "n_tickers": int(g["ticker"].nunique()),
            "first_trade": g["trade_date"].min(),
            "last_trade": g["trade_date"].max(),
            "last_disclosure": g["disclosure_date"].max(),
            "median_lag_days": (
                float((g["disclosure_date"] - g["trade_date"]).dt.days.median())
                if g["disclosure_date"].notna().any()
                else None
            ),
            "stats": {},
        }
        for a in ANCHORS:
            for hz in HORIZONS:
                col = f"{a}_{hz}"
                vals = g[col].to_numpy(dtype=float)
                rec["stats"][col] = {
                    "all": _stats(vals),
                    "buy": _stats(g.loc[g["direction"] == "buy", col].to_numpy(dtype=float)),
                    "sell": _stats(g.loc[g["direction"] == "sell", col].to_numpy(dtype=float)),
                }
        head = rec["stats"][f"{SCORE_ANCHOR}_{SCORE_HORIZON}"]["all"]
        rec["score"] = score_from_shrunk(head["shrunk"], head["n"])
        rec["score_n"] = head["n"]
        rows.append(rec)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["score", "n_decisions"], ascending=[False, False], na_position="last")
    return df.reset_index(drop=True)


def reputation_for(score: float | None, default: float) -> float:
    """Map a 0–100 track-record score onto the signal scorer's 0–1 reputation.

    50 (market-like) keeps the default; the record can move it by up to ±0.25.
    """
    if score is None:
        return default
    return round(min(1.0, max(0.0, default + (score - 50) / 200)), 4)
