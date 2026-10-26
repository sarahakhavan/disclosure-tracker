"""
Cluster detection: flag tickers that several distinct people bought (or sold)
within a rolling window of trade dates.

The same rule is mirrored in the report page's JavaScript so the page can
re-cluster with a viewer's own follow list and window.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_WINDOW_DAYS = 45  # wide enough to catch a 13F quarter-end with congress trades
DEFAULT_MIN_ACTORS = 2


def find_clusters(
    decisions: pd.DataFrame,
    window_days: int = DEFAULT_WINDOW_DAYS,
    min_actors: int = DEFAULT_MIN_ACTORS,
    direction: str = "buy",
    actor_ids: set[str] | None = None,
    since: pd.Timestamp | None = None,
) -> list[dict]:
    """Greedy windows per ticker: start at each decision, extend ``window_days``.

    Returns one cluster per maximal group (overlapping windows for the same
    ticker are merged), newest first.
    """
    d = decisions[decisions["direction"] == direction]
    if actor_ids is not None:
        d = d[d["actor_id"].isin(actor_ids)]
    if since is not None:
        d = d[d["trade_date"] >= since]
    if d.empty:
        return []

    window = pd.Timedelta(days=window_days)
    clusters: list[dict] = []
    for ticker, g in d.sort_values("trade_date").groupby("ticker"):
        rows = g.to_dict("records")
        i = 0
        while i < len(rows):
            start = rows[i]["trade_date"]
            j = i
            members = []
            # Grow the window while each next trade is within `window`
            # of the previous member (chained), capped at 3× window total.
            while j < len(rows) and (
                not members
                or (
                    rows[j]["trade_date"] - members[-1]["trade_date"] <= window
                    and rows[j]["trade_date"] - start <= 3 * window
                )
            ):
                members.append(rows[j])
                j += 1
            actors = {m["actor_id"] for m in members}
            if len(actors) >= min_actors:
                disclosed = [m["disclosure_date"] for m in members if pd.notna(m["disclosure_date"])]
                clusters.append(
                    {
                        "ticker": ticker,
                        "direction": direction,
                        "n_actors": len(actors),
                        "actor_ids": sorted(actors),
                        "actors": sorted({m["actor"] for m in members}),
                        "first_trade": min(m["trade_date"] for m in members),
                        "last_trade": max(m["trade_date"] for m in members),
                        # The cluster is only knowable once the last member is public
                        "complete_disclosure": max(disclosed) if disclosed else None,
                        "n_decisions": len(members),
                        "total_value_usd": float(
                            pd.Series([m.get("value_usd") for m in members], dtype=float).sum()
                        ),
                    }
                )
            i = j
    clusters.sort(key=lambda c: c["last_trade"], reverse=True)
    return clusters
