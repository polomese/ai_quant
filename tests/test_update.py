"""Offline tests for incremental updates: delta snapshots, upserts, rewrite detection, pruning."""

from __future__ import annotations

import shutil
import zipfile
from datetime import date

import polars as pl
import pytest
from test_sharadar_client import FakeResp, client

from pitlab.config import find_project_root, load_settings
from pitlab.data import update as up
from pitlab.data.ingest import latest_snapshot, refresh_processed, upsert, write_manifest


@pytest.fixture
def settings(tmp_path):
    shutil.copytree(find_project_root() / "config", tmp_path / "config")
    return load_settings(tmp_path / "config" / "settings.yaml", root=tmp_path)


def stocks_csv(rows):
    head = "ticker,date,open,high,low,close,volume,closeadj,closeunadj,lastupdated\n"
    return head + "".join(f"{t},{d},{c},{c},{c},{c},100,{c},{c},{lu}\n" for t, d, c, lu in rows)


def put_snapshot(raw, table, stamp, csv_text, kind="full", as_zip=False):
    snap = raw / table / stamp
    snap.mkdir(parents=True)
    if as_zip:
        with zipfile.ZipFile(snap / f"{table}.zip", "w") as z:
            z.writestr(f"SHARADAR_{table}.csv", csv_text)
    else:
        (snap / f"{table}_delta.csv").write_text(csv_text)
    write_manifest(snap, table, csv_text.count("\n") - 1, "paid", {"kind": kind})
    return snap


def fund_csv():
    return (
        "ticker,dimension,calendardate,date,reportperiod,lastupdated,netinc\n"
        "AAA,ARQ,2020-03-31,2020-05-01,2020-03-31,2020-05-01,1\n"
    )


def test_upsert_replaces_matching_keys_and_appends_new():
    base = pl.DataFrame({"ticker": ["A", "A", "B"], "date": [1, 2, 1], "v": [1.0, 2.0, 3.0]})
    delta = pl.DataFrame({"ticker": ["A", "C"], "date": [2, 1], "v": [20.0, 30.0]})
    out = upsert(base, delta, ["ticker", "date"]).sort("ticker", "date")
    assert out.rows() == [("A", 1, 1.0), ("A", 2, 20.0), ("B", 1, 3.0), ("C", 1, 30.0)]


def test_latest_snapshot_ignores_deltas_and_deltas_layer_in_order(settings):
    raw = settings.raw_dir
    put_snapshot(raw, "fundamentals", "20260101T000000Z", fund_csv(), as_zip=True)
    put_snapshot(
        raw,
        "stocks",
        "20260101T000000Z",
        stocks_csv(
            [("AAA", "2026-01-02", 10, "2026-01-02"), ("AAA", "2026-01-05", 11, "2026-01-05")]
        ),
        as_zip=True,
    )
    put_snapshot(
        raw,
        "stocks",
        "20260106T000000Z",
        stocks_csv(
            [("AAA", "2026-01-05", 12, "2026-01-06"), ("AAA", "2026-01-06", 13, "2026-01-06")]
        ),
        kind="delta",
    )
    assert latest_snapshot(raw, "stocks").name == "20260101T000000Z"
    refresh_processed(settings, ["fundamentals", "stocks"])
    s = pl.read_parquet(settings.processed_dir / "stocks.parquet").sort("date")
    assert s["close"].to_list() == [10.0, 12.0, 13.0]
    assert s.schema["date"] == pl.Date and s.schema["close"] == pl.Float64

    # a later delta is merged into the existing Parquet without re-reading the base
    put_snapshot(
        raw,
        "stocks",
        "20260107T000000Z",
        stocks_csv([("AAA", "2026-01-07", 14, "2026-01-07")]),
        kind="delta",
    )
    refresh_processed(settings, ["stocks"])
    s = pl.read_parquet(settings.processed_dir / "stocks.parquet").sort("date")
    assert s["close"].to_list() == [10.0, 12.0, 13.0, 14.0]
    meta = (settings.processed_dir / "_meta.json").read_text()
    assert "20260107T000000Z" in meta and "SHARADAR-FREE" in meta  # 1 ticker -> free mode

    # a new full snapshot becomes the base; older deltas no longer apply
    put_snapshot(
        raw,
        "stocks",
        "20260108T000000Z",
        stocks_csv([("AAA", "2026-01-07", 99, "2026-01-08")]),
        as_zip=True,
    )
    refresh_processed(settings, ["stocks"])
    s = pl.read_parquet(settings.processed_dir / "stocks.parquet")
    assert s["close"].to_list() == [99.0]


def test_rewritten_tickers_from_old_rows_and_actions():
    delta = pl.DataFrame(
        {
            "ticker": ["NEW", "OLD", "OLD"],
            "date": ["2026-10-26", "2026-10-26", "2025-11-03"],
        }
    )
    actions = pl.DataFrame(
        {
            "ticker": ["SPL", "DIV", "IGN", "LATE"],
            "date": ["2026-10-25", "2026-10-26", "2026-10-26", "2026-10-01"],
            "action": ["split", "dividend", "listed", "split"],
        }
    )
    got = up.rewritten_tickers(
        delta, date(2026, 10, 24), 10, actions, ["split", "dividend", "spinoff"]
    )
    assert got == ["DIV", "OLD", "SPL"]


def test_fetch_changed_rows_splits_overflowing_window_by_day(monkeypatch):
    monkeypatch.setattr(up, "PAGE", 3)

    def h(u, p, n):
        days = (p["lastupdated.gte"], p["lastupdated.lte"])
        if days[0] != days[1]:
            body = "ticker,date\n" + "X,2026-01-01\n" * 3  # full page -> must split
        else:
            body = f"ticker,date\nT,{days[0]}\n"
        return FakeResp(200, body.encode())

    c = client(h)
    df = up.fetch_changed_rows(c, "stocks", date(2026, 1, 1), date(2026, 1, 3))
    assert sorted(df["date"].to_list()) == ["2026-01-01", "2026-01-02", "2026-01-03"]
    assert all(p["from"] == "1980-01-01" for _, p in c.session.calls)
    assert all("skip" not in p for _, p in c.session.calls)


def test_prune_keeps_newest_fulls_and_their_deltas(settings):
    raw = settings.raw_dir
    csv = stocks_csv([("AAA", "2026-01-02", 1, "2026-01-02")])
    for stamp, kind in [
        ("20260101T000000Z", "full"),
        ("20260102T000000Z", "delta"),
        ("20260108T000000Z", "full"),
        ("20260109T000000Z", "delta"),
        ("20260115T000000Z", "full"),
    ]:
        put_snapshot(raw, "stocks", stamp, csv, kind=kind, as_zip=kind == "full")
        put_snapshot(raw, "fundamentals", stamp, fund_csv(), kind=kind, as_zip=kind == "full")
    removed = up.prune_full_snapshots(settings)  # default keep: stocks 2
    left = sorted(p.name for p in (raw / "stocks").iterdir())
    assert left == ["20260108T000000Z", "20260109T000000Z", "20260115T000000Z"]
    assert len(removed) == 2
    assert len(list((raw / "fundamentals").iterdir())) == 5  # never pruned


def test_update_incremental_writes_delta_and_refetches_rewritten(settings, monkeypatch):
    raw = settings.raw_dir
    put_snapshot(raw, "fundamentals", "20260101T000000Z", fund_csv(), as_zip=True)
    for t in ("stocks", "daily"):
        body = stocks_csv(
            [("AAA", "2025-06-02", 5, "2026-01-01"), ("BBB", "2026-01-02", 7, "2026-01-02")]
        )
        put_snapshot(raw, t, "20260101T000000Z", body, as_zip=True)
    refresh_processed(settings, ["fundamentals", "stocks", "daily"])
    settings.update.delta_tables = ["stocks"]

    monkeypatch.setattr(up, "download_tables", lambda *a, **k: {})
    monkeypatch.setattr(up, "refresh_processed", refresh_processed)

    def h(u, p, n):
        if "ticker" in p:  # full-history refetch for the rewritten ticker
            assert p["ticker"] == "AAA"
            body = stocks_csv(
                [("AAA", "2025-06-02", 2.5, "2026-01-05"), ("AAA", "2026-01-05", 3, "2026-01-05")]
            )
        else:  # capped delta: AAA shows an old re-stamped row (split) + BBB's new day
            body = stocks_csv(
                [("AAA", "2025-12-01", 2.4, "2026-01-05"), ("BBB", "2026-01-05", 8, "2026-01-05")]
            )
        return FakeResp(200, body.encode())

    summary = up.update(settings, client(h), today=date(2026, 1, 5))
    assert summary["tables"]["stocks"]["rewritten_tickers"] == 1
    s = pl.read_parquet(settings.processed_dir / "stocks.parquet").sort("ticker", "date")
    aaa = s.filter(pl.col("ticker") == "AAA")
    assert aaa["close"].to_list() == [2.5, 3.0]  # refetched history replaced the old value
    assert s.filter(pl.col("ticker") == "BBB")["close"].to_list() == [7.0, 8.0]
    deltas = [p for p in (raw / "stocks").iterdir() if p.name != "20260101T000000Z"]
    assert len(deltas) == 1 and '"kind": "delta"' in (deltas[0] / "manifest.json").read_text()
