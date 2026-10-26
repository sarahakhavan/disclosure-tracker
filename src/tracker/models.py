"""
Tables for the tracker. They live on the project's shared declarative Base,
so ``init_db`` (or ``create_tables`` below) creates them alongside the rest.

``tracked_trades`` is deliberately separate from ``smart_money_events``:
the pipeline table dedups on (ticker, actor, trade_date, source), which
collapses distinct same-day lots, and it only holds recent events. The
tracker needs the full history keyed by each source's own transaction id.
"""

from datetime import datetime

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)

from src.models.database import Base, get_engine


class TrackedTrade(Base):
    __tablename__ = "tracked_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # Stable id from the source (Capitol Trades txId, or fund|period|cusip|action)
    source_id = Column(String(120), nullable=False)
    source = Column(String(20), nullable=False)  # "congress" | "13f"
    actor_id = Column(String(40), nullable=False, index=True)
    actor = Column(String(200), nullable=False)
    actor_group = Column(String(60), nullable=True)  # party/chamber or "Fund"
    ticker = Column(String(12), nullable=False, index=True)
    issuer = Column(String(200), nullable=True)
    direction = Column(String(4), nullable=False)  # "buy" | "sell"
    action = Column(String(30), nullable=True)  # purchase, sale, new_position, added, exited
    asset_type = Column(String(30), nullable=True)
    owner = Column(String(30), nullable=True)  # self / spouse / joint / child
    value_usd = Column(Float, nullable=True)
    trade_date = Column(Date, nullable=False)
    disclosure_date = Column(Date, nullable=True)
    raw_payload = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("source", "source_id", name="uq_tracked_trade_source"),
        Index("ix_tracked_actor_date", "actor_id", "trade_date"),
    )


class PriceBar(Base):
    __tablename__ = "tracker_prices"

    ticker = Column(String(12), primary_key=True)
    date = Column(Date, primary_key=True)
    close = Column(Float, nullable=False)  # split/dividend-adjusted close


class PriceFetchStatus(Base):
    """Remembers tickers Yahoo had nothing for, so we don't re-ask every run."""

    __tablename__ = "tracker_price_status"

    ticker = Column(String(12), primary_key=True)
    last_attempt = Column(DateTime, nullable=False)
    n_bars = Column(Integer, nullable=False, default=0)


def create_tables(database_url: str):
    engine = get_engine(database_url)
    Base.metadata.create_all(
        engine,
        tables=[
            TrackedTrade.__table__,
            PriceBar.__table__,
            PriceFetchStatus.__table__,
        ],
    )
    return engine
