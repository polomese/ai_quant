"""Keep the local Sharadar copy current: daily incremental top-off + periodic full refresh.

Behaviour of the direct API that shapes this module (verified 2026-10-27 on a paid key, see
docs/DATA_NOTES.md "Incremental updates"):

* ``lastupdated.gte=YYYY-MM-DD`` filters work on fundamentals / stocks / daily.
* Without a ``ticker`` filter those queries are silently capped (stocks / daily: rows dated in
  the last ~1 year, fundamentals: ~5 years) even with ``from=1980-01-01``. With a ticker filter
  the full history comes back.
* Splits AND dividends rewrite a ticker's whole adjusted price history and stamp every row
  with a new ``lastupdated`` - so a capped delta would leave stale ``closeadj`` behind.

Daily update therefore:
1. re-pulls the small tables in full (bulk zip; REST if bulk is unavailable),
2. pulls ``lastupdated >= watermark - overlap`` for each delta table,
3. finds tickers whose history was rewritten (delta rows dated well before the watermark, or a
   split/dividend/spinoff in ``actions`` since the watermark) and re-fetches their FULL history
   ticker by ticker,
4. stores 1+3 as a ``kind: delta`` raw snapshot and upserts it into data/processed.

Anything the delta path can miss (row deletions, rewrites older than the cap) is repaired by
``update(full=True)`` - a fresh bulk download - which the scheduler runs weekly.
"""

from __future__ import annotations

import io
import shutil
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

from pitlab.config import Settings
from pitlab.data.ingest import (
    complete_snapshots,
    latest_snapshot,
    read_raw_snapshot,
    refresh_processed,
    utc_stamp,
    write_manifest,
)
from pitlab.data.sharadar_client import (
    BATCH,
    EARLIEST,
    SharadarClient,
    SharadarError,
    download_tables,
)

PAGE = 100_000
#: tables whose whole history the vendor rewrites on corporate actions
PRICE_TABLES = {"stocks", "daily"}
#: tickers per request when a single day's delta overflows a page
DELTA_TICKER_BATCH = 200


def watermark(settings: Settings, table: str) -> date | None:
    """max(lastupdated) currently in data/processed/<table>.parquet."""
    p = settings.processed_dir / f"{table}.parquet"
    if not p.exists():
        return None
    v = pl.scan_parquet(p).select(pl.col("lastupdated").max()).collect().item()
    return v


def _get_csv(client: SharadarClient, table: str, params: dict[str, Any]) -> pl.DataFrame:
    q = {"from": EARLIEST, **params, "format": "csv", "limit": PAGE}
    body = client.get(f"data/{table}", q).content
    if not body.strip():
        return pl.DataFrame()
    return pl.read_csv(io.BytesIO(body), infer_schema=False)


def fetch_changed_rows(
    client: SharadarClient,
    table: str,
    since: date,
    until: date,
    universe: list[str] | None = None,
) -> pl.DataFrame:
    """Rows with ``since <= lastupdated <= until``. Never pages with skip: an overflowing
    window is split into single days, and an overflowing day into ticker batches."""
    df = _get_csv(
        client, table, {"lastupdated.gte": since.isoformat(), "lastupdated.lte": until.isoformat()}
    )
    if df.height < PAGE:
        return df
    if since < until:
        logger.info(f"{table}: {since}..{until} filled a page; splitting by day")
        frames = []
        d = since
        while d <= until:
            frames.append(fetch_changed_rows(client, table, d, d, universe))
            d += timedelta(days=1)
        return pl.concat([f for f in frames if f.height], how="diagonal_relaxed")
    if not universe:
        raise SharadarError(f"{table}: {since} delta overflows a page and no ticker universe given")
    logger.info(f"{table}: {since} filled a page; slicing by {DELTA_TICKER_BATCH}-ticker batches")
    frames = []
    for i in range(0, len(universe), DELTA_TICKER_BATCH):
        part = _get_csv(
            client,
            table,
            {
                "ticker": ",".join(universe[i : i + DELTA_TICKER_BATCH]),
                "lastupdated.gte": since.isoformat(),
                "lastupdated.lte": since.isoformat(),
            },
        )
        if part.height >= PAGE:
            raise SharadarError(f"{table}: ticker batch overflow on {since}; lower the batch size")
        frames.append(part)
    return pl.concat([f for f in frames if f.height], how="diagonal_relaxed")


def rewritten_tickers(
    delta: pl.DataFrame,
    since: date,
    lookback_days: int,
    actions: pl.DataFrame | None = None,
    adjustment_actions: list[str] | None = None,
) -> list[str]:
    """Tickers whose history was rewritten since ``since``.

    Two independent signals: (a) the delta contains rows dated more than ``lookback_days``
    before ``since`` (the vendor re-stamped old rows), (b) ``actions`` lists a split / dividend
    / spinoff dated on or after ``since``.
    """
    out: set[str] = set()
    if delta.height and "date" in delta.columns:
        cutoff = (since - timedelta(days=lookback_days)).isoformat()
        out |= set(delta.filter(pl.col("date").str.slice(0, 10) < cutoff)["ticker"].to_list())
    if actions is not None and actions.height and adjustment_actions:
        a = actions.filter(
            pl.col("action").is_in(adjustment_actions)
            & (pl.col("date").str.slice(0, 10) >= since.isoformat())
        )
        out |= set(a["ticker"].to_list())
    return sorted(out)


def fetch_full_history(client: SharadarClient, table: str, tickers: list[str]) -> pl.DataFrame:
    b = BATCH.get(table, 3)
    frames = []
    for i in range(0, len(tickers), b):
        part = _get_csv(client, table, {"ticker": ",".join(tickers[i : i + b])})
        if part.height >= PAGE:
            raise SharadarError(f"{table}: full-history batch overflow; lower BATCH[{table!r}]")
        frames.append(part)
    frames = [f for f in frames if f.height]
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def _universe(settings: Settings) -> list[str]:
    p = settings.processed_dir / "tickers.parquet"
    if not p.exists():
        return []
    t = pl.read_parquet(p)
    if "table" in t.columns:
        t = t.filter(pl.col("table").is_in(["fundamentals", "SF1", "stocks", "SEP"]))
    if "isdelisted" in t.columns:
        t = t.filter(pl.col("isdelisted").cast(pl.String).is_in(["N", "false", "False"]))
    return sorted(t["ticker"].unique().to_list())


def _write_delta(
    raw_dir: Path, table: str, stamp: str, df: pl.DataFrame, extra: dict[str, Any]
) -> Path:
    snap = raw_dir / table / stamp
    snap.mkdir(parents=True, exist_ok=True)
    df.write_csv(snap / f"{table}_delta.csv")
    write_manifest(snap, table, df.height, "paid", {"kind": "delta", "method": "rest", **extra})
    return snap


def prune_full_snapshots(settings: Settings) -> list[Path]:
    """Keep the newest N full snapshots of the configured tables; drop older fulls and the
    deltas that only applied to them. Fundamentals are never pruned (vintage archive)."""
    removed: list[Path] = []
    for table, keep in settings.update.keep_full_snapshots.items():
        if table == "fundamentals" or keep < 1:
            continue
        fulls = complete_snapshots(settings.raw_dir, table, "full")
        if len(fulls) <= keep:
            continue
        oldest_kept = fulls[-keep].name
        for snap in complete_snapshots(settings.raw_dir, table):
            if snap.name < oldest_kept:
                shutil.rmtree(snap)
                removed.append(snap)
    if removed:
        logger.info(f"pruned {len(removed)} old raw snapshots")
    return removed


def update(
    settings: Settings, client: SharadarClient, full: bool = False, today: date | None = None
) -> dict[str, Any]:
    """Run one update cycle and refresh data/processed. Returns a summary dict."""
    cfg = settings.update
    today = today or date.today()
    summary: dict[str, Any] = {"mode": "full" if full else "incremental", "tables": {}}
    no_base = [t for t in cfg.delta_tables if latest_snapshot(settings.raw_dir, t) is None]
    if full or no_base:
        if no_base and not full:
            logger.info(f"no full snapshot yet for {no_base}; doing a full download")
        tables = cfg.small_tables + cfg.delta_tables
        download_tables(client, settings.raw_dir, tables, method="auto", history=cfg.history)
        summary["mode"] = "full"
        summary["tables"] = {t: "full" for t in tables}
        refresh_processed(settings, tables)
        summary["pruned"] = [str(p) for p in prune_full_snapshots(settings)]
        return summary

    # 1. small tables: cheap full re-pull every day
    download_tables(client, settings.raw_dir, cfg.small_tables, method="auto", history=cfg.history)
    refresh_processed(settings, cfg.small_tables)
    snap = latest_snapshot(settings.raw_dir, "actions")
    actions = read_raw_snapshot(snap) if snap is not None else None
    universe = _universe(settings)

    # 2-4. incremental tables
    stamp = utc_stamp()
    for table in cfg.delta_tables:
        wm = watermark(settings, table)
        if wm is None:
            raise SharadarError(
                f"{table}: processed table missing; run `pitlab data update --full`"
            )
        since = wm - timedelta(days=cfg.overlap_days)
        delta = fetch_changed_rows(client, table, since, today, universe)
        # price-history rewrites only; old-dated fundamentals rows are normal (MR* rows carry
        # date == reportperiod), and >5y-old restatements are caught by the weekly full refresh
        rew = (
            rewritten_tickers(
                delta,
                since,
                cfg.rewrite_lookback_days,
                actions if table == "stocks" else None,
                cfg.adjustment_actions,
            )
            if table in PRICE_TABLES
            else []
        )
        if rew:
            hist = fetch_full_history(client, table, rew)
            delta = pl.concat(
                [delta.filter(~pl.col("ticker").is_in(rew)), hist], how="diagonal_relaxed"
            )
        _write_delta(
            settings.raw_dir,
            table,
            stamp,
            delta,
            {"since": since.isoformat(), "until": today.isoformat(), "rewritten_tickers": rew},
        )
        summary["tables"][table] = {
            "since": since.isoformat(),
            "rows": delta.height,
            "rewritten_tickers": len(rew),
        }
        logger.info(
            f"{table}: {delta.height:,} changed rows since {since} "
            f"({len(rew)} tickers re-fetched in full)"
        )
    refresh_processed(settings, cfg.delta_tables)
    summary["pruned"] = [str(p) for p in prune_full_snapshots(settings)]
    return summary
