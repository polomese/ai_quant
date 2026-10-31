"""Arm-vs-arm comparisons: paired IC differences, bias decomposition, contamination."""

from __future__ import annotations

import polars as pl

from pitlab.stats import newey_west_mean

#: effect name -> (minuend arm, subtrahend arm)
EFFECTS = {"timing": ("A", "B"), "revision": ("B", "C"), "total": ("A", "C")}


def paired_ic_differences(ic: pl.DataFrame, by: list[str], nw_lags: int = 6) -> pl.DataFrame:
    """For each effect, the monthly IC difference between two arms (matched dates) with NW t."""
    rows = []
    empty = pl.DataFrame(
        schema={
            **{b: pl.String for b in by},
            "effect": pl.String,
            "arms": pl.String,
            "mean_diff": pl.Float64,
            "t_nw": pl.Float64,
            "n_months": pl.Int64,
        }
    )
    if ic.height == 0:
        return empty
    wide = ic.pivot(on="arm", index=[*by, "date"], values="ic")
    for eff, (a, b) in EFFECTS.items():
        if a not in wide.columns or b not in wide.columns:
            continue
        diff = wide.select([*by, "date", (pl.col(a) - pl.col(b)).alias("d")]).drop_nulls("d")
        for key, g in diff.group_by(by, maintain_order=True):
            nw = newey_west_mean(g.sort("date")["d"].to_numpy(), nw_lags)
            rows.append(
                {
                    **dict(zip(by, key, strict=True)),
                    "effect": eff,
                    "arms": f"{a}-{b}",
                    "mean_diff": nw["mean"],
                    "t_nw": nw["t"],
                    "n_months": nw["n"],
                }
            )
    return pl.DataFrame(rows) if rows else empty


def contamination(
    raw: pl.DataFrame,
    pairs: list[tuple[str, str]] = (("C", "A"), ("C", "B")),  # type: ignore[assignment]
    n_q: int = 10,
    min_names: int = 20,
) -> pl.DataFrame:
    """Return-free contamination metrics per (date, signal, pair).

    ``raw`` holds pre-standardization signal values: date, ticker, signal, arm, value, on the
    common (intersection) universe. For each pair (PIT arm, contaminated arm):
    rank correlation of values, and the share of names whose decile differs.
    """
    out = []
    if raw.height == 0:
        return pl.DataFrame()
    wide = raw.pivot(on="arm", index=["date", "ticker", "signal"], values="value")
    for pit, other in pairs:
        if pit not in wide.columns or other not in wide.columns:
            continue
        w = wide.drop_nulls([pit, other])
        g = ["date", "signal"]
        w = w.with_columns(
            ((pl.col(pit).rank("ordinal").over(g) - 1) * n_q // pl.len().over(g)).alias("q_pit"),
            ((pl.col(other).rank("ordinal").over(g) - 1) * n_q // pl.len().over(g)).alias("q_oth"),
        )
        m = (
            w.group_by(g)
            .agg(
                pl.corr(pit, other, method="spearman").alias("rank_corr"),
                (pl.col("q_pit") != pl.col("q_oth")).mean().alias("share_decile_change"),
                (pl.col(pit) != pl.col(other)).mean().alias("share_value_differs"),
                pl.len().alias("n"),
            )
            .filter(pl.col("n") >= min_names)
            .with_columns(pl.lit(f"{pit}_vs_{other}").alias("pair"))
        )
        out.append(m)
    return pl.concat(out).sort(["pair", "signal", "date"]) if out else pl.DataFrame()


def bias_table(
    ic_sum: pl.DataFrame, diffs: pl.DataFrame, port_sum: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Headline per-signal table: mean IC/t per arm, effects, long-short Sharpe per arm."""
    if ic_sum.height == 0:
        return pl.DataFrame(schema={"signal": pl.String})
    base = ic_sum.pivot(on="arm", index="signal", values=["mean_ic", "t_nw"])
    eff = diffs.pivot(on="effect", index="signal", values=["mean_diff", "t_nw"])
    t = base.join(eff, on="signal", how="left")
    if port_sum is not None and port_sum.height:
        sh = port_sum.filter(pl.col("weighting") == "EW").pivot(
            on="arm", index="signal", values="sharpe"
        )
        sh = sh.rename({c: f"sharpe_EW_{c}" for c in sh.columns if c != "signal"})
        t = t.join(sh, on="signal", how="left")
    return t
