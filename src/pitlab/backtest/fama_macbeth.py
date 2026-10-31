"""Fama-MacBeth: monthly cross-sectional OLS of next-month return on the standardized
signal, controlling for size (z-scored log market cap) and sector dummies; inference on the
time series of slopes with Newey-West standard errors."""

from __future__ import annotations

import numpy as np
import polars as pl

from pitlab.stats import newey_west_mean


def fama_macbeth(
    df: pl.DataFrame,
    by: list[str],
    value: str = "z",
    ret: str = "fwd_ret",
    sector_dummies: bool = True,
    min_names: int = 20,
    nw_lags: int = 6,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (monthly slopes, summary). ``df`` needs value, ret, mcap, sector, date + ``by``."""
    d = df.drop_nulls([value, ret, "mcap"]).filter(pl.col("mcap") > 0)
    d = d.with_columns(pl.col("mcap").log().alias("_lnmc"))
    d = d.with_columns(
        (
            (pl.col("_lnmc") - pl.col("_lnmc").mean().over([*by, "date"]))
            / pl.col("_lnmc").std().over([*by, "date"])
        )
        .fill_nan(0.0)
        .fill_null(0.0)
        .alias("_size")
    )
    monthly = []
    for key, g in d.group_by([*by, "date"], maintain_order=True):
        if g.height < min_names:
            continue
        cols = [np.ones(g.height), g[value].to_numpy(), g["_size"].to_numpy()]
        if sector_dummies and "sector" in g.columns:
            sec = g["sector"].to_numpy()
            levels = np.unique(sec)
            for lv in levels[1:]:  # drop one level (intercept)
                cols.append((sec == lv).astype(float))
        X = np.column_stack(cols)
        y = g[ret].to_numpy()
        if X.shape[0] <= X.shape[1] + 2:
            continue
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        monthly.append(
            {
                **dict(zip([*by, "date"], key, strict=True)),
                "slope": float(beta[1]),
                "size_slope": float(beta[2]),
                "n": g.height,
            }
        )
    mdf = (
        pl.DataFrame(monthly)
        if monthly
        else pl.DataFrame(
            schema={
                **{b: pl.String for b in by},
                "date": pl.Date,
                "slope": pl.Float64,
                "size_slope": pl.Float64,
                "n": pl.Int64,
            }
        )
    )
    rows = []
    if mdf.height:
        for key, g in mdf.group_by(by, maintain_order=True):
            nw = newey_west_mean(g.sort("date")["slope"].to_numpy(), nw_lags)
            rows.append(
                {
                    **dict(zip(by, key, strict=True)),
                    "coef_mean": nw["mean"],
                    "t_nw": nw["t"],
                    "n_months": nw["n"],
                    "avg_names": float(g["n"].mean()),
                }
            )
    return mdf, pl.DataFrame(rows)
