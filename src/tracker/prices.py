"""
Daily adjusted closes for every traded ticker plus the SPY benchmark,
downloaded in batches with yfinance and cached in ``tracker_prices``.
"""

from datetime import date, datetime, timedelta

import pandas as pd
import structlog
from sqlalchemy import func
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from src.tracker.models import PriceBar, PriceFetchStatus

logger = structlog.get_logger()

BENCHMARK = "SPY"
HISTORY_START = date(2012, 1, 1)
BATCH = 80
# Tickers Yahoo returned nothing for are retried at most this often.
EMPTY_RETRY_DAYS = 30


def yahoo_symbol(ticker: str) -> str:
    return ticker.replace(".", "-").replace("/", "-")


def _download(symbols: list[str], start: date) -> dict[str, pd.Series]:
    import yfinance as yf

    df = yf.download(
        symbols,
        start=start.isoformat(),
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="ticker",
    )
    out: dict[str, pd.Series] = {}
    if df is None or df.empty:
        return out
    for sym in symbols:
        try:
            s = df[sym]["Close"] if isinstance(df.columns, pd.MultiIndex) else df["Close"]
        except KeyError:
            continue
        s = s.dropna()
        if not s.empty:
            out[sym] = s
    return out


def update_prices(session_factory, tickers: set[str]) -> int:
    """Fetch missing / stale price history. Returns bars written."""
    tickers = set(tickers) | {BENCHMARK}
    session = session_factory()
    try:
        last = dict(
            session.query(PriceBar.ticker, func.max(PriceBar.date))
            .group_by(PriceBar.ticker)
            .all()
        )
        status = {s.ticker: s for s in session.query(PriceFetchStatus).all()}
    finally:
        session.close()

    today = date.today()
    now = datetime.utcnow()
    # Group by start date so each batch shares one download window
    plan: dict[date, list[str]] = {}
    for t in sorted(tickers):
        st = status.get(t)
        if t not in last:
            if st and st.n_bars == 0 and now - st.last_attempt < timedelta(days=EMPTY_RETRY_DAYS):
                continue
            plan.setdefault(HISTORY_START, []).append(t)
        elif last[t] < today - timedelta(days=1):
            plan.setdefault(last[t] - timedelta(days=5), []).append(t)

    written = 0
    for start, group in plan.items():
        for i in range(0, len(group), BATCH):
            chunk = group[i : i + BATCH]
            sym_map = {yahoo_symbol(t): t for t in chunk}
            try:
                series = _download(list(sym_map), start)
            except Exception as e:
                logger.warning("tracker.prices.batch_failed", error=str(e), n=len(chunk))
                continue
            session = session_factory()
            try:
                for sym, t in sym_map.items():
                    s = series.get(sym)
                    rows = (
                        [
                            {"ticker": t, "date": d.date(), "close": float(c)}
                            for d, c in s.items()
                        ]
                        if s is not None
                        else []
                    )
                    for j in range(0, len(rows), 500):
                        stmt = sqlite_insert(PriceBar).values(rows[j : j + 500])
                        stmt = stmt.on_conflict_do_update(
                            index_elements=["ticker", "date"],
                            set_={"close": stmt.excluded.close},
                        )
                        session.execute(stmt)
                    written += len(rows)
                    session.merge(
                        PriceFetchStatus(
                            ticker=t,
                            last_attempt=now,
                            n_bars=(status[t].n_bars if t in status else 0) + len(rows),
                        )
                    )
                session.commit()
            finally:
                session.close()
            logger.info("tracker.prices.batch", start=str(start), n=len(chunk), bars=written)
    return written


def load_prices(session_factory, tickers: set[str] | None = None) -> pd.DataFrame:
    """Wide frame: index = trading dates, columns = tickers, values = close."""
    session = session_factory()
    try:
        q = session.query(PriceBar.ticker, PriceBar.date, PriceBar.close)
        if tickers is not None:
            q = q.filter(PriceBar.ticker.in_(list(tickers | {BENCHMARK})))
        rows = q.all()
    finally:
        session.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ticker", "date", "close"])
    wide = df.pivot(index="date", columns="ticker", values="close").sort_index()
    wide.index = pd.to_datetime(wide.index)
    return wide
