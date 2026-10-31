"""Monthly rebalance grid and forward returns (close-to-close from ``closeadj``).

The forward return for rebalance date t runs from the close of t to the close of the next
rebalance date t', i.e. it starts the trading day *after* t (no same-day overlap with the
information set, which is "everything with available_at <= t").

Delistings: if a stock's last price falls in (t, t'], its return is the price return to the
last close compounded with a delisting return: ``delisting_return_mna`` when the actions
table shows an M&A-type exit within 30 days of the last price, otherwise
``delisting_return_unknown`` (default -30%). See docs/METHODOLOGY.md.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from pitlab.config import Settings


def month_end_dates(calendar: np.ndarray) -> np.ndarray:
    """Last trading day of each month in the calendar."""
    months = calendar.astype("datetime64[M]")
    last = np.r_[months[1:] != months[:-1], True]
    return calendar[last]


def monthly_panel(stocks: pl.DataFrame, rebal: np.ndarray) -> pl.DataFrame:
    """Price snapshot of every ticker on each rebalance date it traded.

    Returns ticker, date, closeadj, closeunadj (only rows where the ticker printed a price
    on the rebalance date itself).
    """
    rb = pl.DataFrame({"date": pl.Series(rebal.astype("datetime64[D]"), dtype=pl.Date)})
    px_cols = ["ticker", "date", "closeadj"] + (
        ["closeunadj"] if "closeunadj" in stocks.columns else []
    )
    out = stocks.select(px_cols).join(rb, on="date", how="semi")
    if "closeunadj" not in out.columns:
        out = out.with_columns(pl.col("closeadj").alias("closeunadj"))
    return out


def delisting_info(
    stocks: pl.DataFrame, tickers: pl.DataFrame, actions: pl.DataFrame | None, settings: Settings
) -> pl.DataFrame:
    """Per ticker: last price date, last closeadj, delisting flag and assumed delisting return."""
    last = (
        stocks.sort(["ticker", "date"])
        .group_by("ticker")
        .agg(
            pl.col("date").last().alias("last_date"),
            pl.col("closeadj").last().alias("last_closeadj"),
        )
    )
    if "isdelisted" in tickers.columns:
        flag = tickers.select("ticker", (pl.col("isdelisted") == "Y").alias("is_delisted")).unique(
            "ticker"
        )
    else:
        flag = last.select("ticker", pl.lit(False).alias("is_delisted"))
    last = last.join(flag, on="ticker", how="left").with_columns(
        pl.col("is_delisted").fill_null(False)
    )
    mna = pl.lit(False)
    if actions is not None and actions.height:
        m = (
            actions.filter(pl.col("action").is_in(settings.returns.mna_actions))
            .select("ticker", pl.col("date").alias("act_date"))
            .join(last.select("ticker", "last_date"), on="ticker")
            .filter((pl.col("act_date") - pl.col("last_date")).dt.total_days().abs() <= 30)
            .select("ticker")
            .unique()
            .with_columns(pl.lit(True).alias("is_mna"))
        )
        last = last.join(m, on="ticker", how="left")
        mna = pl.col("is_mna").fill_null(False)
    r = settings.returns
    return last.with_columns(
        pl.when(mna)
        .then(r.delisting_return_mna)
        .otherwise(r.delisting_return_unknown)
        .alias("dl_ret")
    ).select("ticker", "last_date", "last_closeadj", "is_delisted", "dl_ret")


def forward_returns(panel: pl.DataFrame, rebal: np.ndarray, delist: pl.DataFrame) -> pl.DataFrame:
    """Add ``fwd_ret`` (t -> next rebalance) and ``next_date`` to the monthly panel."""
    rb = rebal.astype("datetime64[D]")
    nxt = pl.DataFrame(
        {
            "date": pl.Series(rb[:-1], dtype=pl.Date),
            "next_date": pl.Series(rb[1:], dtype=pl.Date),
        }
    )
    p = panel.join(nxt, on="date", how="inner")  # last rebalance has no forward window
    nxt_px = panel.select(
        "ticker", pl.col("date").alias("next_date"), pl.col("closeadj").alias("next_closeadj")
    )
    p = p.join(nxt_px, on=["ticker", "next_date"], how="left").join(delist, on="ticker", how="left")
    delisted_in_window = (
        pl.col("next_closeadj").is_null()
        & pl.col("is_delisted")
        & (pl.col("last_date") >= pl.col("date"))
        & (pl.col("last_date") < pl.col("next_date"))
    )
    fwd = (
        pl.when(pl.col("next_closeadj").is_not_null())
        .then(pl.col("next_closeadj") / pl.col("closeadj") - 1)
        .when(delisted_in_window)
        .then((pl.col("last_closeadj") / pl.col("closeadj")) * (1 + pl.col("dl_ret")) - 1)
        .otherwise(None)
    )
    return p.with_columns(fwd.alias("fwd_ret"), delisted_in_window.alias("delisted_next")).drop(
        "next_closeadj", "last_closeadj", "last_date", "is_delisted", "dl_ret"
    )
