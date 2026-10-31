"""End-to-end experiment: tables -> arms -> signals -> IC / FM / portfolios / comparisons."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from loguru import logger

from pitlab.backtest.compare import bias_table, contamination, paired_ic_differences
from pitlab.backtest.fama_macbeth import fama_macbeth
from pitlab.backtest.ic import ic_summary, rank_ic
from pitlab.backtest.portfolios import long_short, portfolio_summary
from pitlab.config import Settings, load_columns
from pitlab.data.ingest import load_table, read_meta
from pitlab.pit.asof import asof_join
from pitlab.pit.availability import (
    FLOW_ITEMS,
    LAG_ITEMS,
    QUARTER_FLOWS,
    STOCK_ITEMS,
    ArmSpec,
    arm_streams,
    trading_calendar,
)
from pitlab.price_features import PRICE_FEATURES, price_features
from pitlab.returns import delisting_info, forward_returns, month_end_dates, monthly_panel
from pitlab.signals import compute_signals, winsorize_zscore
from pitlab.universe import eligible_panel, security_master

PIT_REFERENCE = ArmSpec("C_ref", "AR", "own_filing", 1)  # used only for the shares fallback


@dataclass
class Inputs:
    settings: Settings
    fund: pl.DataFrame
    stocks: pl.DataFrame
    tickers: pl.DataFrame
    daily: pl.DataFrame | None
    actions: pl.DataFrame | None
    sp500: pl.DataFrame | None
    meta: dict[str, Any] = field(default_factory=dict)


def drop_impossible_filings(fund: pl.DataFrame) -> pl.DataFrame:
    """Drop as-reported rows whose filing date precedes their fiscal period end.

    Such rows cannot be real (a 10-Q is filed after the quarter ends); keeping them would make
    arm C "know" the numbers before they existed. PIT-strict choice: drop, never re-date.
    """
    bad = pl.col("dimension").str.starts_with("AR") & (pl.col("datekey") < pl.col("reportperiod"))
    n = fund.filter(bad).height
    if n:
        logger.warning(f"dropping {n} AR rows filed before their fiscal period ended")
    return fund.filter(~bad.fill_null(False))


def load_inputs(settings: Settings) -> Inputs:
    def opt(name: str, columns: list[str] | None = None) -> pl.DataFrame | None:
        try:
            return load_table(settings, name, columns)
        except FileNotFoundError:
            return None

    return Inputs(
        settings=settings,
        fund=drop_impossible_filings(load_table(settings, "fundamentals")),
        # the full-history price tables are ~40M rows: load only the columns the study uses
        stocks=load_table(settings, "stocks", ["ticker", "date", "closeadj", "closeunadj"]),
        tickers=load_table(settings, "tickers"),
        daily=opt("daily", ["ticker", "date", "marketcap"]),
        actions=opt("actions"),
        sp500=opt("sp500"),
        meta=read_meta(settings),
    )


# ------------------------------------------------------------------------------------------
# base grid: rebalance dates x eligible tickers, with forward returns and market cap
# ------------------------------------------------------------------------------------------


def rebalance_dates(calendar: np.ndarray, settings: Settings) -> np.ndarray:
    rb = month_end_dates(calendar)
    # drop a trailing partial month (data ends before the true month end)
    last = calendar[-1]
    next_weekday = np.busday_offset(last, 1, roll="forward")
    if rb[-1] == last and next_weekday.astype("datetime64[M]") == last.astype("datetime64[M]"):
        rb = rb[:-1]
    start = np.datetime64(settings.sample.start)
    rb = rb[rb >= start]
    if settings.sample.end:
        rb = rb[rb <= np.datetime64(settings.sample.end)]
    return rb


def base_grid(inp: Inputs, calendar: np.ndarray, rebal: np.ndarray) -> pl.DataFrame:
    s = inp.settings
    panel = monthly_panel(inp.stocks, rebal)
    delist = delisting_info(inp.stocks, inp.tickers, inp.actions, s)
    panel = forward_returns(panel, rebal, delist)
    master = security_master(inp.tickers, s)
    grid = eligible_panel(panel, master, inp.stocks, s, inp.sp500)
    # market cap from the price side at t: daily.marketcap, else PIT shares x unadjusted close
    if inp.daily is not None and "marketcap" in inp.daily.columns:
        mult = float(load_columns(s.root)["units"]["daily_marketcap_multiplier"])
        mc = inp.daily.select("ticker", "date", (pl.col("marketcap") * mult).alias("mcap_daily"))
        grid = grid.join(mc, on=["ticker", "date"], how="left")
    else:
        grid = grid.with_columns(pl.lit(None, dtype=pl.Float64).alias("mcap_daily"))
    q = arm_streams(inp.fund, PIT_REFERENCE, calendar)["Q"].select(
        "ticker", "calendardate", "available_at", "sharesbas"
    )
    sh = asof_join(
        grid.select("ticker", "date"),
        q,
        max_staleness_days=s.universe.max_fundamental_staleness_days,
    )
    grid = grid.join(sh.select("ticker", "date", "sharesbas"), on=["ticker", "date"], how="left")
    grid = grid.with_columns(
        pl.coalesce(pl.col("mcap_daily"), pl.col("sharesbas") * pl.col("closeunadj")).alias("mcap")
    ).drop("mcap_daily", "sharesbas")
    grid = grid.filter(pl.col("mcap").is_not_null() & (pl.col("mcap") > 0))
    grid = grid.join(price_features(inp.stocks, rebal), on=["ticker", "date"], how="left")
    terc = ((pl.col("mcap").rank("ordinal").over("date") - 1) * 3 // pl.len().over("date")).cast(
        pl.Int32
    )
    grid = grid.with_columns(
        pl.when(terc == 0)
        .then(pl.lit("small"))
        .when(terc == 1)
        .then(pl.lit("mid"))
        .otherwise(pl.lit("large"))
        .alias("size_tercile")
    )
    return grid


def restater_flags(fund: pl.DataFrame) -> pl.DataFrame:
    """Ticker-level flag: any MR value differs from its AR counterpart (descriptive slice only)."""
    flags = []
    for stream, items in (("Q", ["equity", "assets"]), ("T", ["netinc", "revenue", "gp"])):
        items = [c for c in items if c in fund.columns]
        ar = (
            fund.filter(pl.col("dimension") == f"AR{stream}")
            .group_by(["ticker", "calendardate"])
            .agg([pl.col(c).sort_by("datekey").first() for c in items])
        )
        mr = fund.filter(pl.col("dimension") == f"MR{stream}").select(
            "ticker", "calendardate", *items
        )
        j = ar.join(mr, on=["ticker", "calendardate"], suffix="_mr")
        diff = pl.any_horizontal(
            [
                (
                    (pl.col(c) - pl.col(f"{c}_mr")).abs()
                    > 1e-6 * pl.col(c).abs().clip(lower_bound=1.0)
                )
                for c in items
            ]
        )
        flags.append(j.group_by("ticker").agg(diff.any().alias("r")))
    f = pl.concat(flags).group_by("ticker").agg(pl.col("r").any().alias("restater"))
    return f


# ------------------------------------------------------------------------------------------
# arms -> long signal frame
# ------------------------------------------------------------------------------------------


def arm_signals(
    inp: Inputs, spec: ArmSpec, calendar: np.ndarray, grid: pl.DataFrame
) -> pl.DataFrame:
    s = inp.settings
    streams = arm_streams(inp.fund, spec, calendar)
    keys = grid.select("ticker", "date")
    stale = s.universe.max_fundamental_staleness_days
    q = asof_join(keys, streams["Q"], max_staleness_days=stale, suffix="_q")
    t = asof_join(keys, streams["T"], max_staleness_days=stale, suffix="_t")
    q_items = [
        c
        for c in STOCK_ITEMS + list(QUARTER_FLOWS.values()) + [f"{x}_lag4" for x in LAG_ITEMS["Q"]]
        if f"{c}_q" in q.columns
    ]
    t_items = [
        c for c in FLOW_ITEMS + [f"{x}_lag4" for x in LAG_ITEMS["T"]] if f"{c}_t" in t.columns
    ]
    df = (
        grid.select("ticker", "date", "mcap", *[c for c in PRICE_FEATURES if c in grid.columns])
        .join(
            q.select("ticker", "date", *[pl.col(f"{c}_q").alias(c) for c in q_items]),
            on=["ticker", "date"],
            how="left",
        )
        .join(
            t.select("ticker", "date", *[pl.col(f"{c}_t").alias(c) for c in t_items]),
            on=["ticker", "date"],
            how="left",
        )
    )
    df = compute_signals(df, s.signals)
    sig_cols = [f"sig_{n}" for n in s.signals]
    long = df.select("ticker", "date", *sig_cols).unpivot(
        index=["ticker", "date"], on=sig_cols, variable_name="signal", value_name="value"
    )
    long = long.with_columns(
        pl.col("signal").str.strip_prefix("sig_"), pl.lit(spec.name).alias("arm")
    )
    return long.filter(pl.col("value").is_not_null() & pl.col("value").is_finite())


KEYS = ["date", "ticker", "signal"]


def comparison_keys(
    inp: Inputs, specs: list[ArmSpec], calendar: np.ndarray, grid: pl.DataFrame
) -> pl.DataFrame | None:
    """(date, ticker, signal) keys with a valid signal in every arm, or None for per_arm.

    Built arm by arm from the keys alone, so memory stays at about one arm's panel no matter
    how many arms are compared (the lag sweep has ten).
    """
    if inp.settings.universe.comparison_universe != "intersection":
        return None
    common: pl.DataFrame | None = None
    for spec in specs:
        keys = arm_signals(inp, spec, calendar, grid).select(KEYS)
        common = keys if common is None else common.join(keys, on=KEYS, how="semi")
    return common


def build_signal_panel(
    inp: Inputs,
    specs: list[ArmSpec],
    calendar: np.ndarray,
    grid: pl.DataFrame,
    common: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Raw signals for all arms, restricted to the comparison universe."""
    if common is None:
        common = comparison_keys(inp, specs, calendar, grid)
    frames = []
    for spec in specs:
        t0 = time.time()
        raw = arm_signals(inp, spec, calendar, grid)
        if common is not None:
            raw = raw.join(common, on=KEYS, how="semi")
        frames.append(raw)
        logger.info(
            f"arm {spec.name}: {frames[-1].height:,} signal values ({time.time() - t0:.1f}s)"
        )
    return pl.concat(frames)


def standardize(raw: pl.DataFrame, settings: Settings) -> pl.DataFrame:
    p = settings.preprocess
    z = raw.with_columns(pl.col("value").alias("z"))
    z = winsorize_zscore(z, "z", p.winsor_lower, p.winsor_upper, by=["arm", "signal", "date"])
    return z.filter(pl.col("z").is_not_null() & pl.col("z").is_finite())


# ------------------------------------------------------------------------------------------
# measurements
# ------------------------------------------------------------------------------------------


def era_of(settings: Settings) -> pl.Expr:
    expr = pl.lit(None, dtype=pl.String)
    for name, (lo, hi) in reversed(list(settings.slices.eras.items())):
        cond = pl.lit(True)
        if lo is not None:
            cond = cond & (pl.col("date") >= pl.lit(lo))
        if hi is not None:
            cond = cond & (pl.col("date") <= pl.lit(hi))
        expr = pl.when(cond).then(pl.lit(name)).otherwise(expr)
    return expr


def measure_ic(panel: pl.DataFrame, settings: Settings) -> pl.DataFrame:
    b = settings.backtest
    min_slice = max(10, b.min_names_per_date // 2)
    out = [
        rank_ic(panel, by=["arm", "signal"], min_names=b.min_names_per_date).with_columns(
            pl.lit("all").alias("slice_type"), pl.lit("all").alias("slice")
        )
    ]
    sl = settings.slices
    specs = []
    if sl.size_terciles:
        specs.append(("size", "size_tercile"))
    if sl.sectors:
        specs.append(("sector", "sector"))
    if sl.restaters:
        specs.append(("restater", "restater_lbl"))
    for st, col in specs:
        ic = rank_ic(panel, by=["arm", "signal", col], min_names=min_slice)
        out.append(
            ic.rename({col: "slice"}).with_columns(
                pl.lit(st).alias("slice_type"), pl.col("slice").cast(pl.String)
            )
        )
    allic = out[0]
    if sl.eras:
        out.append(
            allic.with_columns(
                era_of(settings).alias("slice"), pl.lit("era").alias("slice_type")
            ).drop_nulls("slice")
        )
    cols = ["arm", "signal", "slice_type", "slice", "date", "ic", "n"]
    return pl.concat([o.select(cols) for o in out])


@dataclass
class Results:
    tables: dict[str, pl.DataFrame]
    meta: dict[str, Any]


def _concat(frames: list[pl.DataFrame]) -> pl.DataFrame:
    """Concat per-signal results, skipping the schema-less empties some helpers return."""
    frames = [f for f in frames if f.width]
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def run_experiment(inp: Inputs, lag_sweep: bool = True, main: bool = True) -> Results:
    s = inp.settings
    t0 = time.time()
    calendar = trading_calendar(inp.stocks)
    rebal = rebalance_dates(calendar, s)
    grid = base_grid(inp, calendar, rebal)
    flags = restater_flags(inp.fund)
    grid = grid.join(flags, on="ticker", how="left").with_columns(
        pl.when(pl.col("restater").fill_null(False))
        .then(pl.lit("restater"))
        .otherwise(pl.lit("non-restater"))
        .alias("restater_lbl")
    )
    logger.info(
        f"grid: {grid.height:,} ticker-months, {grid['date'].n_unique()} rebalances, {grid['ticker'].n_unique()} tickers"
    )
    attach = grid.select(
        "ticker", "date", "fwd_ret", "mcap", "sector", "size_tercile", "restater_lbl"
    )
    b = s.backtest
    tables: dict[str, pl.DataFrame] = {}
    meta: dict[str, Any] = {
        "data_mode": inp.meta.get("mode", "UNKNOWN"),
        "n_rebalances": int(grid["date"].n_unique()),
        "n_tickers": int(grid["ticker"].n_unique()),
        "first_rebalance": str(grid["date"].min()),
        "last_rebalance": str(grid["date"].max()),
    }

    if main:
        specs = [ArmSpec.from_config(n, a) for n, a in s.arms.items()]
        raw = build_signal_panel(inp, specs, calendar, grid)
        sizes = raw.filter(pl.col("arm") == specs[-1].name).group_by(["signal", "date"]).len()
        med = float(sizes["len"].median()) if sizes.height else 0.0
        n_q = b.n_quantiles if med >= b.small_universe_threshold else b.n_quantiles_small
        meta.update(
            {
                "median_universe": med,
                "n_quantiles": n_q,
                "arms": {sp.name: sp.__dict__ for sp in specs},
            }
        )
        logger.info(f"median comparison universe {med:.0f} names -> {n_q} quantiles")

        # one signal at a time keeps the standardized panel small (all stats are per signal)
        ics, fm_ms, fms, prs, conts = [], [], [], [], []
        for sig in raw["signal"].unique(maintain_order=True).to_list():
            r = raw.filter(pl.col("signal") == sig)
            panel = standardize(r, s).join(attach, on=["ticker", "date"], how="left")
            ics.append(measure_ic(panel, s))
            fm_m1, fm1 = fama_macbeth(
                panel,
                by=["arm", "signal"],
                sector_dummies=b.fama_macbeth_sector_dummies,
                min_names=b.min_names_per_date,
                nw_lags=b.newey_west_lags,
            )
            fm_ms.append(fm_m1)
            fms.append(fm1)
            prs.append(long_short(panel, "z", n_q, by=["arm", "signal"]))
            conts.append(
                contamination(
                    r, pairs=[("C", "A"), ("C", "B")], n_q=10, min_names=b.min_names_per_date
                )
            )
            del panel, r
        del raw
        ic = _concat(ics)
        if ic.filter(pl.col("slice_type") == "all").height == 0:
            logger.warning(
                f"no month had >= {b.min_names_per_date} names with a signal in every arm: the universe "
                "is too small for cross-sectional statistics (outputs will be empty)"
            )
        ic_sum = ic_summary(
            ic, by=["arm", "signal", "slice_type", "slice"], nw_lags=b.newey_west_lags
        )
        diffs = paired_ic_differences(
            ic, by=["signal", "slice_type", "slice"], nw_lags=b.newey_west_lags
        )
        fm_m, fm = _concat(fm_ms), _concat(fms)
        pr = _concat(prs)
        ps = portfolio_summary(pr, by=["arm", "signal"])
        cont = _concat(conts)
        if cont.height:
            cont = cont.sort(["pair", "signal", "date"])
        allsum = ic_sum.filter(pl.col("slice_type") == "all")
        alldiff = diffs.filter(pl.col("slice_type") == "all")
        headline = bias_table(
            allsum.drop("slice_type", "slice"), alldiff.drop("slice_type", "slice"), ps
        )
        tables.update(
            {
                "ic_monthly": ic,
                "ic_summary": ic_sum,
                "effects": diffs,
                "fm_monthly": fm_m,
                "fm_results": fm,
                "portfolio_returns": pr,
                "portfolio_summary": ps,
                "contamination": cont,
                "headline": headline,
            }
        )
    if lag_sweep:
        tables["lag_sweep"] = run_lag_sweep(inp, calendar, grid, attach)
    meta["elapsed_sec"] = round(time.time() - t0, 1)
    return Results(tables=tables, meta=meta)


def run_lag_sweep(
    inp: Inputs, calendar: np.ndarray, grid: pl.DataFrame, attach: pl.DataFrame
) -> pl.DataFrame:
    s = inp.settings
    b = s.backtest
    specs = []
    for arm in s.lag_sweep.arms:
        for k in s.lag_sweep.extra_lags:
            sp = ArmSpec.from_config(arm, s.arms[arm], extra_lag=k)
            specs.append(ArmSpec(f"{arm}+{k}", sp.version, sp.timing, sp.lag))
    # one arm at a time: all ten panels together need >30 GB on the full Sharadar history
    common = comparison_keys(inp, specs, calendar, grid)
    n_q: int | None = None
    ics, pss = [], []
    for spec in specs:
        panel = build_signal_panel(inp, [spec], calendar, grid, common)
        panel = standardize(panel, s).join(attach, on=["ticker", "date"], how="left")
        if n_q is None:  # same names in every arm under "intersection"
            med = float(panel.group_by(["signal", "date"]).len()["len"].median() or 0)
            n_q = b.n_quantiles if med >= b.small_universe_threshold else b.n_quantiles_small
        ic = rank_ic(panel, by=["arm", "signal"], min_names=b.min_names_per_date)
        ics.append(ic_summary(ic, by=["arm", "signal"], nw_lags=b.newey_west_lags))
        pr = long_short(panel, "z", n_q, by=["arm", "signal"])
        pss.append(portfolio_summary(pr, by=["arm", "signal"]))
        del panel, pr
    ps = (
        pl.concat(pss)
        .filter(pl.col("weighting") == "EW")
        .select(
            "arm",
            "signal",
            pl.col("sharpe").alias("sharpe_ew"),
            pl.col("ann_return").alias("ann_return_ew"),
        )
    )
    out = (
        pl.concat(ics)
        .join(ps, on=["arm", "signal"], how="left")
        .with_columns(
            pl.col("arm").str.split("+").list.get(0).alias("base_arm"),
            pl.col("arm").str.split("+").list.get(1).cast(pl.Int32).alias("extra_lag"),
        )
    )
    return out.sort(["signal", "base_arm", "extra_lag"])
