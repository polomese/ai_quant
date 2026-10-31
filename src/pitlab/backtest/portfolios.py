"""Quantile long-short portfolios (equal- and value-weighted) with turnover."""

from __future__ import annotations

import polars as pl

from pitlab.stats import perf_stats


def assign_quantiles(df: pl.DataFrame, value: str, n_q: int, by: list[str]) -> pl.DataFrame:
    """Quantile 0 (worst) .. n_q-1 (best) within each (by, date) cross-section."""
    g = [*by, "date"]
    r = pl.col(value).rank("ordinal").over(g)
    n = pl.col(value).count().over(g)
    return df.filter(pl.col(value).is_not_null()).with_columns(
        (((r - 1) * n_q) // n).cast(pl.Int32).alias("q")
    )


def long_short(
    df: pl.DataFrame, value: str, n_q: int, by: list[str], ret: str = "fwd_ret"
) -> pl.DataFrame:
    """Monthly long (top) / short (bottom) returns, EW and VW, plus one-sided turnover."""
    d = assign_quantiles(df.drop_nulls([value, ret]), value, n_q, by)
    if d.height == 0:
        return pl.DataFrame(
            schema={
                **{b: pl.String for b in by},
                "date": pl.Date,
                "ret_long": pl.Float64,
                "ret_short": pl.Float64,
                "turnover": pl.Float64,
                "weighting": pl.String,
                "ret_ls": pl.Float64,
            }
        )
    d = d.filter(pl.col("q").is_in([0, n_q - 1])).with_columns(
        pl.when(pl.col("q") == n_q - 1).then(pl.lit("long")).otherwise(pl.lit("short")).alias("leg")
    )
    g = [*by, "date", "leg"]
    d = d.with_columns(
        (1.0 / pl.len().over(g)).alias("w_ew"),
        (pl.col("mcap") / pl.col("mcap").sum().over(g)).alias("w_vw"),
    )
    out = []
    for wcol, label in (("w_ew", "EW"), ("w_vw", "VW")):
        legs = d.group_by(g).agg((pl.col(wcol) * pl.col(ret)).sum().alias("r"))
        wide = legs.pivot(on="leg", index=[*by, "date"], values="r")
        for leg in ("long", "short"):
            if leg not in wide.columns:
                wide = wide.with_columns(pl.lit(None, dtype=pl.Float64).alias(leg))
        to = _turnover(d, wcol, by)
        wide = wide.join(to, on=[*by, "date"], how="left").with_columns(
            pl.lit(label).alias("weighting"),
            (pl.col("long") - pl.col("short")).alias("ret_ls"),
        )
        out.append(wide.rename({"long": "ret_long", "short": "ret_short"}))
    return pl.concat(out, how="diagonal_relaxed").sort([*by, "weighting", "date"])


def _turnover(d: pl.DataFrame, wcol: str, by: list[str]) -> pl.DataFrame:
    """0.5 * sum |w_t - w_{t-1}| per leg (pre-drift weights), averaged over the two legs."""
    dates = d.select([*by, "date"]).unique().sort([*by, "date"])
    dates = dates.with_columns(pl.col("date").shift(1).over(by).alias("prev_date"))
    cur = d.select([*by, "date", "leg", "ticker", pl.col(wcol).alias("w")])
    prev = cur.rename({"date": "prev_date", "w": "w_prev"})
    x = dates.join(cur, on=[*by, "date"], how="left")
    # names held last month but dropped this month
    dropped = (
        dates.join(prev, on=[*by, "prev_date"], how="inner")
        .join(
            cur.select([*by, "date", "leg", "ticker"]),
            on=[*by, "date", "leg", "ticker"],
            how="anti",
        )
        .with_columns(pl.lit(0.0).alias("w"))
    )
    x = x.join(prev, on=[*by, "prev_date", "leg", "ticker"], how="left")
    x = pl.concat([x, dropped.select(x.columns)], how="vertical_relaxed")
    x = x.filter(pl.col("prev_date").is_not_null()).with_columns(
        (pl.col("w").fill_null(0.0) - pl.col("w_prev").fill_null(0.0)).abs().alias("dw")
    )
    leg_to = x.group_by([*by, "date", "leg"]).agg((0.5 * pl.col("dw").sum()).alias("to"))
    return leg_to.group_by([*by, "date"]).agg(pl.col("to").mean().alias("turnover"))


def portfolio_summary(pr: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    rows = []
    for key, g in pr.group_by([*by, "weighting"], maintain_order=True):
        s = perf_stats(g.sort("date")["ret_ls"].to_numpy())
        rows.append(
            {
                **dict(zip([*by, "weighting"], key, strict=True)),
                **s,
                "turnover": float(g["turnover"].mean())
                if g["turnover"].null_count() < g.height
                else float("nan"),
            }
        )
    if not rows:
        return pl.DataFrame(
            schema={
                **{b: pl.String for b in by},
                "weighting": pl.String,
                "ann_return": pl.Float64,
                "ann_vol": pl.Float64,
                "sharpe": pl.Float64,
                "max_drawdown": pl.Float64,
                "n": pl.Int64,
                "turnover": pl.Float64,
            }
        )
    return pl.DataFrame(rows)
