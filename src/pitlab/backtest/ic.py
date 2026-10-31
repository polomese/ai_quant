"""Rank information coefficients."""

from __future__ import annotations

import polars as pl

from pitlab.stats import newey_west_mean


def rank_ic(
    df: pl.DataFrame,
    value: str = "z",
    ret: str = "fwd_ret",
    by: list[str] | None = None,
    min_names: int = 20,
) -> pl.DataFrame:
    """Monthly cross-sectional Spearman correlation of ``value`` with ``ret``.

    ``by`` are the grouping keys besides ``date`` (e.g. arm, signal, slice).
    """
    keys = [*(by or []), "date"]
    d = df.drop_nulls([value, ret])
    return (
        d.group_by(keys)
        .agg(pl.corr(value, ret, method="spearman").alias("ic"), pl.len().alias("n"))
        .filter((pl.col("n") >= min_names) & pl.col("ic").is_not_nan())
        .sort(keys)
    )


def ic_summary(ic: pl.DataFrame, by: list[str], nw_lags: int = 6) -> pl.DataFrame:
    """Mean, std, Newey-West t-stat and hit rate of the IC series per group."""
    rows = []
    for key, g in ic.group_by(by, maintain_order=True):
        s = g.sort("date")["ic"].to_numpy()
        nw = newey_west_mean(s, nw_lags)
        rows.append(
            {
                **dict(zip(by, key, strict=True)),
                "mean_ic": nw["mean"],
                "std_ic": float(s.std(ddof=1)) if len(s) > 1 else float("nan"),
                "t_nw": nw["t"],
                "hit_rate": float((s > 0).mean()) if len(s) else float("nan"),
                "n_months": nw["n"],
                "avg_names": float(g["n"].mean()),
            }
        )
    if not rows:
        return pl.DataFrame(
            schema={
                **{b: pl.String for b in by},
                "mean_ic": pl.Float64,
                "std_ic": pl.Float64,
                "t_nw": pl.Float64,
                "hit_rate": pl.Float64,
                "n_months": pl.Int64,
                "avg_names": pl.Float64,
            }
        )
    return pl.DataFrame(rows)
