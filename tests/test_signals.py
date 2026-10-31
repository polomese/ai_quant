from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from pitlab.signals import SIGNALS, compute_signals, winsorize_zscore


def frame():
    return pl.DataFrame(
        {
            "ticker": ["A", "B"],
            "date": [date(2020, 1, 31)] * 2,
            "mcap": [1000.0, 2000.0],
            "netinc": [100.0, -50.0],
            "ncfo": [80.0, 10.0],
            "revenue": [1200.0, 500.0],
            "revenue_lag4": [1000.0, 0.0],
            "gp": [300.0, 100.0],
            "equity": [500.0, 400.0],
            "equity_lag4": [300.0, None],
            "assets": [2000.0, 1000.0],
            "assets_lag4": [1600.0, 1000.0],
            "sharesbas": [110.0, 50.0],
            "sharesbas_lag4": [100.0, 50.0],
            "netinc_qtr": [30.0, -20.0],
            "netinc_qtr_lag4": [20.0, -10.0],
            "ret_1m": [0.05, -0.02],
            "ret_12_1": [0.30, -0.10],
            "ret_60_12": [1.0, -0.5],
            "vol_1y": [0.02, 0.04],
            "hi_52w": [0.95, 0.60],
        }
    )


def test_signal_formulas():
    out = compute_signals(frame(), {k: 1 for k in SIGNALS})
    a, b = out.row(0, named=True), out.row(1, named=True)
    assert a["sig_earnings_yield"] == pytest.approx(0.1)
    assert a["sig_book_to_price"] == pytest.approx(0.5)
    assert a["sig_roe"] == pytest.approx(100 / 400)
    assert b["sig_roe"] == pytest.approx(-50 / 400)  # falls back to current equity
    assert a["sig_accruals"] == pytest.approx(20 / 1800)
    assert a["sig_sales_growth"] == pytest.approx(0.2)
    assert b["sig_sales_growth"] is None  # non-positive base
    assert a["sig_cash_flow_yield"] == pytest.approx(0.08)
    assert a["sig_net_issuance"] == pytest.approx(0.1)
    assert a["sig_earnings_surprise"] == pytest.approx(10 / 1000)
    assert b["sig_earnings_surprise"] == pytest.approx(-10 / 2000)
    assert a["sig_momentum_12_1"] == pytest.approx(0.30)
    assert a["sig_high_52w"] == pytest.approx(0.95)
    assert a["sig_gross_profitability"] == pytest.approx(0.15)
    assert a["sig_asset_growth"] == pytest.approx(0.25)


def test_every_signal_is_defined_and_configured():
    from pitlab.config import load_settings
    from pitlab.signals import SIGNAL_INFO, SIGNAL_INPUTS

    assert set(SIGNALS) == set(SIGNAL_INFO) == set(SIGNAL_INPUTS)
    assert set(load_settings().signals) <= set(SIGNALS)


def test_direction_flip():
    out = compute_signals(frame(), {"accruals": -1, "asset_growth": -1})
    assert out["sig_accruals"][0] == pytest.approx(-20 / 1800)
    assert out["sig_asset_growth"][0] == pytest.approx(-0.25)


def test_unknown_signal_raises():
    with pytest.raises(KeyError):
        compute_signals(frame(), {"nope": 1})


def test_winsorize_zscore_is_per_date_only():
    rng = np.random.default_rng(0)
    x1 = rng.normal(0, 1, 200)
    x1[0] = 1e6  # outlier
    x2 = rng.normal(100, 10, 200)  # different level on another date
    df = pl.DataFrame(
        {
            "date": [date(2020, 1, 31)] * 200 + [date(2020, 2, 28)] * 200,
            "v": np.r_[x1, x2],
        }
    )
    out = winsorize_zscore(df, "v", 0.01, 0.99)
    g = out.group_by("date").agg(
        pl.col("v").mean().alias("m"), pl.col("v").std().alias("s"), pl.col("v").max().alias("mx")
    )
    assert np.allclose(g["m"].to_numpy(), 0, atol=1e-9)
    assert np.allclose(g["s"].to_numpy(), 1, atol=1e-9)
    assert g["mx"].max() < 4  # outlier clipped
    # date 2's z-scores are unaffected by date 1's values
    alone = winsorize_zscore(df.filter(pl.col("date") == date(2020, 2, 28)), "v", 0.01, 0.99)
    assert np.allclose(
        alone["v"].to_numpy(), out.filter(pl.col("date") == date(2020, 2, 28))["v"].to_numpy()
    )
