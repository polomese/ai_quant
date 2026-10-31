"""Universe construction: US common stocks (incl. delisted), eligibility, PIT S&P 500."""

from __future__ import annotations

import numpy as np
import polars as pl

from pitlab.config import Settings


def security_master(tickers: pl.DataFrame, settings: Settings) -> pl.DataFrame:
    """One row per ticker in the allowed categories (delisted names included)."""
    t = tickers
    if "table" in t.columns:
        # prefer the fundamentals-table row, then the stocks-table row
        order = (
            pl.when(pl.col("table") == "fundamentals")
            .then(0)
            .when(pl.col("table") == "stocks")
            .then(1)
            .otherwise(2)
        )
        t = (
            t.with_columns(order.alias("_o"))
            .sort(["ticker", "_o"])
            .unique("ticker", keep="first")
            .drop("_o")
        )
    t = t.filter(pl.col("category").is_in(settings.universe.categories))
    cols = ["ticker"] + [
        c for c in ("sector", "category", "isdelisted", "permaticker") if c in t.columns
    ]
    return t.select(cols).with_columns(
        pl.col("sector").fill_null("Unknown")
        if "sector" in cols
        else pl.lit("Unknown").alias("sector")
    )


def sp500_membership(sp500: pl.DataFrame, dates: np.ndarray) -> pl.DataFrame:
    """(ticker, date) pairs that were S&P 500 members *as of* each date.

    Supports both Sharadar encodings: dated snapshots (``historical`` / ``current``) and
    change events (``added`` / ``removed``). Events win when present. Never uses today's
    constituents for past dates.
    """
    d = pl.DataFrame({"date": pl.Series(dates.astype("datetime64[D]"), dtype=pl.Date)}).sort("date")
    ev = sp500.filter(pl.col("action").is_in(["added", "removed"]))
    if ev.height:
        ev = ev.select("ticker", pl.col("date").alias("ev_date"), "action").sort("ev_date")
        grid = d.join(ev.select("ticker").unique(), how="cross").sort("date")
        out = grid.join_asof(
            ev,
            left_on="date",
            right_on="ev_date",
            by="ticker",
            strategy="backward",
            check_sortedness=False,
        )
        return out.filter(pl.col("action") == "added").select("ticker", "date")
    snaps = sp500.filter(pl.col("action").is_in(["historical", "current"]))
    snap_dates = snaps.select(pl.col("date").alias("snap_date")).unique().sort("snap_date")
    m = d.join_asof(snap_dates, left_on="date", right_on="snap_date", strategy="backward")
    return (
        m.drop_nulls("snap_date")
        .join(snaps.select("ticker", pl.col("date").alias("snap_date")), on="snap_date")
        .select("ticker", "date")
        .unique()
    )


def eligible_panel(
    panel: pl.DataFrame,
    master: pl.DataFrame,
    stocks: pl.DataFrame,
    settings: Settings,
    sp500: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Apply category, price and price-history filters (market cap is checked later)."""
    u = settings.universe
    first = stocks.group_by("ticker").agg(pl.col("date").min().alias("first_date"))
    p = panel.join(master, on="ticker", how="inner").join(first, on="ticker", how="left")
    p = p.filter(
        (pl.col("closeunadj") >= u.min_price)
        & (pl.col("first_date") <= pl.col("date").dt.offset_by(f"-{u.min_history_months}mo"))
    )
    if u.sp500_only:
        if sp500 is None or sp500.height == 0:
            raise ValueError("sp500_only=true but no sp500 table is available")
        mem = sp500_membership(sp500, p.get_column("date").unique().sort().to_numpy())
        p = p.join(mem, on=["ticker", "date"], how="semi")
    return p.drop("first_date")
