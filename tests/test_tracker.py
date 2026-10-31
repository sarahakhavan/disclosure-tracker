import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.tracker import track_record as tr
from src.tracker.clusters import find_clusters
from src.tracker.follow_list import load_follow_list, resolve_followed, save_follow_list
from src.tracker.report import build_report, decision_frame
from src.tracker.sources import CongressHistoryClient, ThirteenFHistoryClient


def _prices(days=400):
    idx = pd.bdate_range("2023-01-02", periods=days)
    spy = pd.Series(100.0, index=idx)  # flat benchmark keeps the math obvious
    up = pd.Series(np.linspace(100, 200, days), index=idx)  # steadily rising
    down = pd.Series(np.linspace(100, 50, days), index=idx)
    return pd.DataFrame({"SPY": spy, "UP": up, "DOWN": down})


def _trade(i, actor_id, ticker, direction, td, dd, actor=None, source="congress"):
    return {
        "source_id": str(i),
        "source": source,
        "actor_id": actor_id,
        "actor": actor or actor_id,
        "actor_group": "House · D",
        "ticker": ticker,
        "issuer": ticker + " Inc",
        "direction": direction,
        "action": "purchase" if direction == "buy" else "sale",
        "asset_type": "stock",
        "owner": "spouse",
        "value_usd": 15000.0,
        "trade_date": pd.Timestamp(td).date(),
        "disclosure_date": pd.Timestamp(dd).date(),
    }


def test_collapse_merges_same_day_lots():
    t = pd.DataFrame(
        [
            _trade(1, "cg:A", "UP", "buy", "2023-02-01", "2023-03-01"),
            _trade(2, "cg:A", "UP", "buy", "2023-02-01", "2023-03-02"),
            _trade(3, "cg:A", "UP", "sell", "2023-02-01", "2023-03-01"),
        ]
    )
    d = tr.collapse_decisions(t)
    assert len(d) == 2
    buy = d[d.direction == "buy"].iloc[0]
    assert buy.n_lots == 2 and buy.value_usd == 30000
    assert buy.disclosure_date == pd.Timestamp("2023-03-01")


def test_signed_excess_returns_and_anchors():
    t = pd.DataFrame(
        [
            _trade(1, "cg:A", "UP", "buy", "2023-02-01", "2023-03-01"),
            _trade(2, "cg:A", "DOWN", "sell", "2023-02-01", "2023-03-01"),
            _trade(3, "cg:A", "UP", "buy", "2024-07-01", "2024-07-02"),  # not matured
        ]
    )
    d = tr.compute_decision_returns(tr.collapse_decisions(t), _prices())
    buy, sell, late = d.iloc[0], d.iloc[1], d.iloc[2]
    assert buy["trade_3m"] > 0 and buy["disclosure_3m"] > 0
    assert sell["trade_3m"] > 0  # sold before a fall = right call
    # Disclosure entry is strictly after the disclosure date
    assert buy["disclosure_entry"] > pd.Timestamp("2023-03-01")
    assert np.isnan(late["disclosure_12m"])


def test_score_needs_minimum_and_shrinks():
    assert tr.score_from_shrunk(0.2, tr.MIN_DECISIONS - 1) is None
    assert tr.score_from_shrunk(0.0, 50) == 50
    few = tr.score_from_shrunk(0.10 * 8 / (8 + tr.SHRINK_K), 8)
    many = tr.score_from_shrunk(0.10 * 80 / (80 + tr.SHRINK_K), 80)
    assert 50 < few < many < 100


def test_reputation_mapping_bounded():
    assert tr.reputation_for(None, 0.6) == 0.6
    assert tr.reputation_for(50, 0.6) == 0.6
    assert tr.reputation_for(100, 0.6) == 0.85
    assert tr.reputation_for(0, 0.6) == 0.35


def test_clusters_need_distinct_people_in_window():
    d = tr.collapse_decisions(
        pd.DataFrame(
            [
                _trade(1, "cg:A", "UP", "buy", "2023-02-01", "2023-03-01"),
                _trade(2, "cg:A", "UP", "buy", "2023-02-05", "2023-03-01"),
                _trade(3, "cg:B", "UP", "buy", "2023-02-20", "2023-04-01"),
                _trade(4, "cg:C", "UP", "buy", "2023-09-01", "2023-10-01"),  # too late
                _trade(5, "cg:A", "DOWN", "buy", "2023-02-01", "2023-03-01"),
            ]
        )
    )
    cl = find_clusters(d, window_days=30, min_actors=2)
    assert len(cl) == 1
    c = cl[0]
    assert c["ticker"] == "UP" and c["n_actors"] == 2
    assert c["complete_disclosure"] == pd.Timestamp("2023-04-01")
    assert find_clusters(d, window_days=30, min_actors=2, actor_ids={"cg:A"}) == []


def test_follow_list_roundtrip_and_resolution(tmp_path):
    p = tmp_path / "f.json"
    save_follow_list([{"label": "Pelosi", "match": "nancy pelosi"}, {"label": "junk"}], p)
    entries = load_follow_list(p)
    assert entries == [{"label": "Pelosi", "match": "nancy pelosi"}]
    ids = resolve_followed(
        entries + [{"id": "cik:1067983"}],
        {"cg:P000197": "Rep. Nancy Pelosi (D-CA)", "cik:1067983": "Berkshire Hathaway", "cg:X": "Rep. Other"},
    )
    assert ids == {"cg:P000197", "cik:1067983"}


def test_congress_record_parsing():
    raw = {
        "_txId": 123,
        "_politicianId": "P000197",
        "_stateId": "ca",
        "chamber": "house",
        "txType": "buy",
        "txDate": "2024-01-12",
        "pubDate": "2024-01-20T13:00:00Z",
        "value": 750000,
        "owner": "spouse",
        "issuer": {"issuerName": "NVIDIA Corp", "issuerTicker": "NVDA:US"},
        "asset": {"assetType": "stock-option"},
        "politician": {"firstName": "Nancy", "lastName": "Pelosi", "party": "democrat"},
    }
    r = CongressHistoryClient.to_record(raw)
    assert r.ticker == "NVDA" and r.direction == "buy"
    assert r.actor == "Rep. Nancy Pelosi (D-CA)" and r.actor_id == "cg:P000197"
    assert r.disclosure_date == date(2024, 1, 20) and r.asset_type == "stock-option"
    assert CongressHistoryClient.to_record({**raw, "txType": "exchange"}) is None
    assert CongressHistoryClient.to_record({**raw, "txType": "sell_partial"}).direction == "sell"


def test_13f_quarter_diff():
    old = {"A": {"shares": 100, "value": 100, "issuer": "A"}, "B": {"shares": 100, "value": 100, "issuer": "B"},
           "C": {"shares": 100, "value": 100, "issuer": "C"}, "D": {"shares": 100, "value": 100, "issuer": "D"}}
    new = {"A": {"shares": 130, "value": 130, "issuer": "A"}, "B": {"shares": 40, "value": 40, "issuer": "B"},
           "C": {"shares": 100, "value": 100, "issuer": "C"}, "E": {"shares": 10, "value": 10, "issuer": "E"}}
    got = {(d, a, p["issuer"]) for d, a, p in ThirteenFHistoryClient.diff_quarters(old, new)}
    assert got == {("buy", "added", "A"), ("sell", "cut", "B"), ("sell", "exited", "D"), ("buy", "new_position", "E")}


def test_report_is_compact_json_and_roundtrips():
    trades = pd.DataFrame(
        [_trade(i, "cg:P000197", "UP", "buy", f"2023-02-{i+1:02d}", f"2023-03-{i+1:02d}", "Rep. Nancy Pelosi (D-CA)") for i in range(10)]
        + [_trade(100 + i, "cik:1", "DOWN", "buy", "2023-03-31", "2023-05-15", "Berkshire Hathaway", "13f") for i in range(1)]
        + [_trade(200, "cg:Z", "UP", "buy", "2023-02-10", "2023-03-15", "Rep. Z")]
    )
    report = build_report(trades, _prices(), [{"match": "pelosi"}])
    json.dumps(report, allow_nan=False)  # strict JSON, no NaN
    pel = next(a for a in report["actors"] if a["id"] == "cg:P000197")
    assert pel["score"] is not None and pel["score"] > 50
    berk = next(a for a in report["actors"] if a["id"] == "cik:1")
    assert berk["score"] is None  # too few decisions
    assert report["follow_default"] == ["cg:P000197"]
    df = decision_frame(report)
    assert len(df) == report["meta"]["n_decisions"] == 12
    assert "SPY" in report["prices"] and "UP" in report["prices"]


def test_signal_scorer_uses_track_record(tmp_path):
    from src.models.schemas import SourceType
    from src.scoring.signal_scorer import SignalScorer

    p = tmp_path / "r.json"
    p.write_text(json.dumps({"actors": [
        {"name": "Rep. Nancy Pelosi (D-CA)", "score": 90.0},
        {"name": "Berkshire Hathaway", "score": 30.0},
        {"name": "Rep. Few Trades (R-TX)", "score": None},
    ]}))
    s = SignalScorer(p)
    assert s._actor_reputation("Rep. Nancy Pelosi (D-CA)", SourceType.CONGRESSIONAL) == 0.8
    assert s._actor_reputation("BERKSHIRE HATHAWAY INC", SourceType.INSTITUTIONAL) == 0.5
    assert s._actor_reputation("Rep. Few Trades (R-TX)", SourceType.CONGRESSIONAL) == 0.6
    assert SignalScorer(tmp_path / "missing.json")._actor_reputation("x", SourceType.INSIDER) == 0.5


def test_tracker_api_routes(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import src.tracker.api as tapi
    import src.tracker.follow_list as fl

    trades = pd.DataFrame(
        [_trade(i, "cg:P000197", "UP", "buy", f"2023-02-{i+1:02d}", f"2023-03-{i+1:02d}", "Rep. Nancy Pelosi (D-CA)") for i in range(10)]
        + [_trade(50, "cg:Z", "UP", "buy", "2023-02-10", "2023-03-15", "Rep. Z")]
    )
    rp = tmp_path / "report.json"
    rp.write_text(json.dumps(build_report(trades, _prices(), [])))
    fp = tmp_path / "follow.json"
    fp.write_text(json.dumps({"follow": [{"match": "pelosi"}]}))
    monkeypatch.setattr(tapi, "REPORT_PATH", rp)
    monkeypatch.setattr(fl, "DEFAULT_PATH", fp)
    monkeypatch.setattr(tapi, "load_follow_list", lambda: fl.load_follow_list(fp))
    monkeypatch.setattr(tapi, "save_follow_list", lambda e: fl.save_follow_list(e, fp))

    def admin():
        return None

    app = FastAPI()
    app.include_router(tapi.build_router(admin))
    c = TestClient(app)
    body = c.get("/api/tracker/actors", params={"followed_only": True}).json()
    assert [a["id"] for a in body["actors"]] == ["cg:P000197"]
    assert len(c.get("/api/tracker/actors/cg:P000197").json()["trades"]) == 10
    cl = c.get("/api/tracker/clusters", params={"since": "2023-01-01"}).json()["clusters"]
    assert cl and cl[0]["ticker"] == "UP" and cl[0]["n_actors"] == 2
    assert c.put("/api/tracker/follow-list", json={"follow": [{"match": "Rep. Z"}]}).status_code == 200
    assert c.get("/api/tracker/follow-list").json() == {"follow": [{"match": "Rep. Z"}]}


def test_build_tracker_page_embeds_safe_json(tmp_path):
    import importlib.util
    import re

    spec = importlib.util.spec_from_file_location("btp", "scripts/build_tracker_page.py")
    btp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(btp)
    trades = pd.DataFrame([_trade(1, "cg:X", "UP", "buy", "2023-02-01", "2023-03-01", "Rep. </script><b>x</b>")])
    rp = tmp_path / "r.json"
    rp.write_text(json.dumps(build_report(trades, _prices(), [])))
    out = tmp_path / "page.html"
    btp.build(rp, out)
    html = out.read_text()
    m = re.search(r'id="report-data" type="application/json">(.*?)</script>', html, re.S)
    data = json.loads(m.group(1))
    assert data["actors"][0]["name"] == "Rep. </script><b>x</b>"
    assert "__REPORT_JSON__" not in html


def test_13f_weight_filter_drops_small_positions():
    old = {"A": {"shares": 100, "value": 990, "issuer": "A"}, "B": {"shares": 100, "value": 10, "issuer": "B"}}
    new = {"A": {"shares": 100, "value": 990, "issuer": "A"}, "C": {"shares": 5, "value": 2, "issuer": "C"},
           "D": {"shares": 50, "value": 400, "issuer": "D"}}
    got = {(d, a, p["issuer"]) for d, a, p in ThirteenFHistoryClient.diff_quarters(old, new, 0.005)}
    # B was 1% of the old book (kept); C is 0.1% of the new book (dropped)
    assert got == {("sell", "exited", "B"), ("buy", "new_position", "D")}


@pytest.mark.asyncio
async def test_congress_falls_back_to_html_when_json_unparseable(monkeypatch):
    c = CongressHistoryClient()
    good = {"_txId": 1, "_politicianId": "P1", "txType": "buy", "txDate": "2024-01-02",
            "issuer": {"issuerTicker": "AAA"}, "politician": {"firstName": "A", "lastName": "B"}}

    async def bff(page):
        c._probe("bff", page, status=200)
        return [{"unexpected": True}] if page == 1 else []

    async def html(page, page_size=True):
        return [good] if page == 1 else []

    monkeypatch.setattr(c, "_try_bff", bff)
    monkeypatch.setattr(c, "_try_html", html)
    recs = await c.fetch_history(set(), max_pages=3, delay_s=0)
    await c.close()
    assert [r.ticker for r in recs] == ["AAA"]
    assert c.diag["mode"] == "html" and c.diag["bff_unparsed"]


def test_house_ptr_parser_on_real_filings():
    from pathlib import Path

    from src.tracker.house import parse_amount, parse_ptr_text

    fx = Path(__file__).parent / "fixtures" / "house_ptr"
    pel = parse_ptr_text((fx / "house_ptr_0_20030630.txt").read_text())
    assert [(r["owner"], r["ticker"], r["asset_code"], r["tx"]) for r in pel] == [
        ("spouse", "AVGO", "ST", "P"),
        ("spouse", None, "OT", "S"),
    ]
    assert pel[0]["trade_date"] == date(2025, 6, 20)
    assert (pel[0]["amount_low"], pel[0]["amount_high"]) == (1000001, 5000000)
    aapl = parse_ptr_text((fx / "house_ptr_1_20033337.txt").read_text())
    assert [(r["ticker"], r["tx"]) for r in aapl] == [("AAPL", "S (partial)")]
    bonds = parse_ptr_text((fx / "house_ptr_3_20030541.txt").read_text())
    assert len(bonds) >= 15 and all(r["asset_code"] == "GS" and r["owner"] == "joint" for r in bonds)
    tbill = parse_ptr_text((fx / "house_ptr_4_20030933.txt").read_text())
    assert tbill[0]["owner"] == "self" and tbill[0]["asset"].startswith("Treasury Bill")
    assert parse_amount("Over $50,000,000") == (50000000, 50000000)


def test_13f_parser_handles_prefixed_attributes():
    from src.clients.edgar_13f import Edgar13FClient

    x = ('<?xml version="1.0" ?><informationTable xsi:schemaLocation="a b" xmlns="http://x" '
         'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><infoTable><nameOfIssuer>ALPHABET INC'
         '</nameOfIssuer><cusip>02079K305</cusip><value>9</value><shrsOrPrnAmt><sshPrnamt>100</sshPrnamt>'
         '</shrsOrPrnAmt></infoTable></informationTable>')
    assert Edgar13FClient._parse_info_table(x)["02079K305"]["shares"] == 100


def test_house_legacy_one_line_filings():
    from pathlib import Path

    from src.tracker.house import parse_ptr_text

    fx = Path(__file__).parent / "fixtures" / "house_ptr"
    pel = parse_ptr_text((fx / "legacy_20004596.txt").read_text())
    assert [(r["ticker"], r["asset_code"], r["tx"], r["owner"]) for r in pel] == [
        ("AAPL", "OP", "P", "self"), ("DIS", "OP", "P", "self")]
    assert pel[1]["asset"] == "Walt Disney Company" and pel[1]["amount_high"] == 1000000
    sp = parse_ptr_text((fx / "legacy_20005689.txt").read_text())
    assert [(r["ticker"], r["owner"], r["tx"]) for r in sp] == [("MGRC", "spouse", "S"), ("V", "spouse", "S")]
    twc = parse_ptr_text((fx / "legacy_20005128.txt").read_text())
    assert twc[0]["notified"] == date(2016, 6, 7) and twc[0]["asset"] == "Time Warner Cable Inc"


def test_collapse_drops_impossible_dates():
    t = pd.DataFrame(
        [
            _trade(1, "h:A", "UP", "buy", "2023-02-01", "2023-03-01"),
            {**_trade(2, "h:A", "UP", "buy", "2023-02-01", "2023-03-01"), "trade_date": date(3031, 4, 30)},
            _trade(3, "h:A", "UP", "buy", "2023-05-01", "2023-03-01"),  # traded after filing
            _trade(4, "h:A", "UP", "buy", "2001-05-01", "2001-06-01"),
        ]
    )
    d = tr.collapse_decisions(t)
    assert len(d) == 1 and d.iloc[0].trade_date == pd.Timestamp("2023-02-01")


def test_nonpositive_or_penny_prices_are_not_scored():
    idx = pd.bdate_range("2023-01-02", periods=200)
    px = pd.DataFrame({"SPY": 100.0, "OTC": np.r_[np.full(50, 0.5), np.full(150, 0.6)],
                       "NEG": np.r_[np.full(10, 5.0), np.full(190, -1.0)]}, index=idx)
    t = pd.DataFrame([_trade(1, "h:A", "OTC", "buy", "2023-01-03", "2023-01-04"),
                      _trade(2, "h:A", "NEG", "sell", "2023-01-03", "2023-01-04")])
    d = tr.compute_decision_returns(tr.collapse_decisions(t), px)
    assert d["disclosure_1m"].isna().all() and not d["priced"].any()


def test_issuer_names_cleaned_and_longest_kept():
    from src.tracker.report import _issuer_names

    d = pd.DataFrame({"ticker": ["TSM", "TSM", "NVDA"], "issuer": [
        "Manufacturing Company Ltd. (TSM)", "Taiwan Semiconductor Manufacturing Company Ltd. (TSM)",
        "NVIDIA Corporation - Common Stock (NVDA)"]})
    assert _issuer_names(d)["NVDA"] == "NVIDIA Corporation"
    d2 = pd.DataFrame({"ticker": ["TSM"] * 3 + ["NVDA"], "issuer": [
        "Taiwan Semiconductor Manufacturing Co (TSM)", "Taiwan Semiconductor Manufacturing Co (TSM)",
        "Thomas C MacArthur IRAJTTaiwan Semiconductor ManufacturingCo",
        "Morgan Stanley Smith Barney LLC Brokerage AccountNVIDIA Corp"]})
    names = _issuer_names(d2)
    assert names == {"TSM": "Taiwan Semiconductor Manufacturing Co", "NVDA": "NVIDIA Corp"}


def test_clean_issuer_rejects_header_bleed():
    from src.tracker.report import clean_issuer

    assert clean_issuer("P erIODIC TrANSACTION r ePOrTClerk of the House of Represent") is None
    assert clean_issuer("sP M81sPAlcoa Inc.") is None
    assert clean_issuer("Shares g fedcFILINg STATUS: NewSUBHoLDINg oF: Morgan Stanley") is None
    assert clean_issuer("3M Company") == "3M Company"
    assert clean_issuer("iShares Core S&P 500 ETF (IVV)") == "iShares Core S&P 500 ETF"
    assert clean_issuer("Apple Inc. - Common Stock (AAPL)") == "Apple Inc."


def test_scrambled_names_rejected_and_official_names_preferred():
    from src.tracker.report import _title_case, clean_issuer

    for junk in ["sPAmerican Financial Group, Inc.", "MERRILL LYNCH INVESTMENT aCCoUNTSallergan, Inc.",
                 "blDRS Emerging Markets 50 aDRIndex Fund", "JTalpha Pro Tech, ltd."]:
        assert clean_issuer(junk) is None, junk
    for ok in ["YouTube Inc", "PayPal Holdings, Inc.", "AbbVie Inc.", "eBay Inc.", "McDonald's Corp"]:
        assert clean_issuer(ok) == ok, ok
    assert _title_case("NVIDIA CORP") == "Nvidia Corp"
    assert _title_case("BANK OF AMERICA CORP /DE/") == "Bank of America Corp"
    trades = pd.DataFrame([_trade(1, "h:A", "UP", "buy", "2023-02-01", "2023-03-01")])
    r = build_report(trades, _prices(), [], {"UP": "Up Industries Inc", "ZZZ": "Unused"})
    assert r["issuers"] == {"UP": "Up Industries Inc"}
