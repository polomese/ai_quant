from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from pitlab.backtest.compare import contamination, paired_ic_differences
from pitlab.backtest.fama_macbeth import fama_macbeth
from pitlab.backtest.ic import ic_summary, rank_ic
from pitlab.backtest.portfolios import assign_quantiles, long_short
from pitlab.stats import block_bootstrap_mean, max_drawdown, newey_west_mean, perf_stats


def test_newey_west_iid_matches_ols_se():
    rng = np.random.default_rng(1)
    x = rng.normal(0.5, 1.0, 400)
    nw = newey_west_mean(x, lags=0)
    assert nw["mean"] == pytest.approx(x.mean())
    assert nw["se"] == pytest.approx(x.std(ddof=1) / np.sqrt(len(x)), rel=1e-6)


def test_newey_west_inflates_se_for_autocorrelated_series():
    rng = np.random.default_rng(2)
    e = rng.normal(0, 1, 3000)
    x = np.empty_like(e)
    x[0] = e[0]
    for i in range(1, len(e)):
        x[i] = 0.7 * x[i - 1] + e[i]
    assert newey_west_mean(x, lags=12)["se"] > 1.5 * newey_west_mean(x, lags=0)["se"]


def test_newey_west_short_series_is_nan():
    assert np.isnan(newey_west_mean([0.1, 0.2], lags=6)["t"])


def test_bootstrap_ci_contains_mean():
    rng = np.random.default_rng(3)
    x = rng.normal(0.2, 1, 300)
    b = block_bootstrap_mean(x)
    assert b["lo"] < x.mean() < b["hi"]


def test_max_drawdown_and_perf():
    assert max_drawdown([0.1, -0.5, 0.2]) == pytest.approx(-0.5)
    s = perf_stats(np.full(24, 0.01) + np.r_[0.001, -0.001].repeat(12))
    assert s["ann_return"] == pytest.approx(0.12, abs=1e-9)


def _panel(n_dates=36, n=100, strength=0.3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_dates):
        z = rng.normal(0, 1, n)
        r = strength * 0.05 * z + rng.normal(0, 0.05, n)
        d = date(2010, 1, 31) + timedelta(days=30 * t)
        for i in range(n):
            rows.append(
                {
                    "date": d,
                    "ticker": f"T{i}",
                    "z": z[i],
                    "fwd_ret": r[i],
                    "mcap": 1e9 * (1 + i),
                    "sector": "S1" if i % 2 else "S2",
                    "arm": "A",
                    "signal": "x",
                }
            )
    return pl.DataFrame(rows)


def test_rank_ic_detects_signal_and_summary():
    df = _panel()
    ic = rank_ic(df, by=["arm", "signal"])
    s = ic_summary(ic, by=["arm", "signal"]).row(0, named=True)
    assert s["mean_ic"] > 0.15 and s["t_nw"] > 5 and s["hit_rate"] > 0.8


def test_rank_ic_respects_min_names():
    df = _panel(n=10)
    assert rank_ic(df, by=["arm"], min_names=20).height == 0


def test_fama_macbeth_positive_slope():
    df = _panel()
    _, summ = fama_macbeth(df, by=["arm", "signal"])
    row = summ.row(0, named=True)
    assert row["coef_mean"] == pytest.approx(0.015, abs=0.005) and row["t_nw"] > 5


def test_quantiles_and_long_short():
    df = _panel()
    q = assign_quantiles(df, "z", 10, ["arm", "signal"])
    counts = q.group_by(["date", "q"]).len()["len"]
    assert counts.min() == counts.max() == 10
    ls = long_short(df, "z", 10, ["arm", "signal"])
    ew = ls.filter(pl.col("weighting") == "EW")
    assert ew["ret_ls"].mean() > 0.01
    assert ew["turnover"].drop_nulls().mean() > 0.5  # random signal -> high turnover


def test_turnover_zero_for_static_portfolio():
    rows = []
    for t in range(5):
        for i in range(20):
            rows.append(
                {
                    "date": date(2020, 1, 31) + timedelta(days=31 * t),
                    "ticker": f"T{i}",
                    "z": float(i),
                    "fwd_ret": 0.0,
                    "mcap": 1.0,
                    "arm": "A",
                    "signal": "x",
                }
            )
    ls = long_short(pl.DataFrame(rows), "z", 5, ["arm", "signal"])
    assert ls.filter(pl.col("weighting") == "EW")["turnover"].drop_nulls().max() == pytest.approx(
        0.0
    )


def test_paired_differences_and_contamination():
    d = [date(2010, 1, 31) + timedelta(days=30 * t) for t in range(40)]
    ic = pl.DataFrame(
        {
            "date": d * 3,
            "signal": ["x"] * 120,
            "arm": ["A"] * 40 + ["B"] * 40 + ["C"] * 40,
            "ic": [0.10] * 40 + [0.05] * 40 + [0.0] * 40,
        }
    ).with_columns(pl.col("ic") + pl.Series(np.random.default_rng(0).normal(0, 0.01, 120)))
    diffs = paired_ic_differences(ic, by=["signal"])
    got = {r["effect"]: r["mean_diff"] for r in diffs.iter_rows(named=True)}
    assert got["timing"] == pytest.approx(0.05, abs=0.01)
    assert got["total"] == pytest.approx(0.10, abs=0.01)

    raw = pl.DataFrame(
        {
            "date": [date(2020, 1, 31)] * 60,
            "ticker": [f"T{i}" for i in range(30)] * 2,
            "signal": ["x"] * 60,
            "arm": ["A"] * 30 + ["C"] * 30,
            "value": list(range(30)) + list(range(30)),
        }
    )
    c = contamination(raw, pairs=[("C", "A")], min_names=10).row(0, named=True)
    assert c["rank_corr"] == pytest.approx(1.0) and c["share_decile_change"] == 0.0
