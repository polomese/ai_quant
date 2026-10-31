"""Price-based inputs for the momentum / reversal / volatility signals.

Prices are not restated, so these features are identical in every arm: price signals carry
no Clock 1 bias and act as a built-in control (their A-B-C differences must be ~0).
Everything is measured at the rebalance close from data up to and including that day.

Columns (per ticker, rebalance date):
  ret_1m      return over the last month                      (t-1 -> t)
  ret_12_1    return from 12 months ago to 1 month ago        (t-12 -> t-1, skips last month)
  ret_60_12   return from 5 years ago to 1 year ago           (t-60 -> t-12)
  vol_1y      std. dev. of daily returns over the last 252 trading days (min 126)
  hi_52w      close / highest close of the last 252 trading days (min 126)
"""

from __future__ import annotations

import numpy as np
import polars as pl

PRICE_FEATURES = ["ret_1m", "ret_12_1", "ret_60_12", "vol_1y", "hi_52w"]
WINDOW = 252
MIN_DAYS = 126


def price_features(stocks: pl.DataFrame, rebal: np.ndarray) -> pl.DataFrame:
    rb = pl.DataFrame(
        {
            "date": pl.Series(rebal.astype("datetime64[D]"), dtype=pl.Date),
            "_i": np.arange(len(rebal)),
        }
    )
    # daily features need ~1 year before the first rebalance; monthly ones reach 60 months back
    first = rb["date"].min()
    daily = (
        stocks.select("ticker", "date", "closeadj")
        .filter(pl.col("date") >= pl.lit(first).dt.offset_by("-2y"))
        .sort("ticker", "date")
        .with_columns(
            (pl.col("closeadj") / pl.col("closeadj").shift(1) - 1).over("ticker").alias("_r")
        )
        .with_columns(
            pl.col("_r").rolling_std(WINDOW, min_samples=MIN_DAYS).over("ticker").alias("vol_1y"),
            (
                pl.col("closeadj")
                / pl.col("closeadj").rolling_max(WINDOW, min_samples=MIN_DAYS).over("ticker")
            ).alias("hi_52w"),
        )
        .join(rb, on="date", how="semi")
        .select("ticker", "date", "vol_1y", "hi_52w")
    )
    # month-end closes indexed by rebalance number; a lagged close must be on that exact
    # month-end (no stale fill), otherwise the return is missing
    hist = stocks.select("ticker", "date", "closeadj").join(
        _month_ends_before(stocks, rebal), on="date", how="inner"
    )
    out = hist.filter(pl.col("_i") >= 0)
    for k in (1, 12, 60):
        lagged = hist.select(
            "ticker", (pl.col("_i") + k).alias("_i"), pl.col("closeadj").alias(f"_p{k}")
        )
        out = out.join(lagged, on=["ticker", "_i"], how="left")
    out = out.select(
        "ticker",
        "date",
        (pl.col("closeadj") / pl.col("_p1") - 1).alias("ret_1m"),
        (pl.col("_p1") / pl.col("_p12") - 1).alias("ret_12_1"),
        (pl.col("_p12") / pl.col("_p60") - 1).alias("ret_60_12"),
    )
    return out.join(daily, on=["ticker", "date"], how="left")


def _month_ends_before(stocks: pl.DataFrame, rebal: np.ndarray) -> pl.DataFrame:
    """Month-end trading days back to 60 months before the first rebalance, indexed so that
    ``_i`` lines up with the rebalance index (earlier month-ends get negative indices)."""
    cal = np.sort(stocks.get_column("date").unique().to_numpy().astype("datetime64[D]"))
    months = cal.astype("datetime64[M]")
    ends = cal[np.r_[months[1:] != months[:-1], True]]
    first = rebal.astype("datetime64[D]")[0]
    start = np.searchsorted(ends, first)
    idx = np.arange(len(ends)) - start
    keep = idx >= -60
    return pl.DataFrame(
        {
            "date": pl.Series(ends[keep], dtype=pl.Date),
            "_i": idx[keep],
        }
    ).filter(pl.col("date") <= pl.lit(rebal.astype("datetime64[D]")[-1]))
