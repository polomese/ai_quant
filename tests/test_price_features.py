"""Price features: exact month-end lags, skip-month momentum, rolling daily stats."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from pitlab.price_features import price_features
from pitlab.returns import month_end_dates


def test_price_features_use_exact_month_end_lags():
    days = pl.date_range(date(2014, 1, 1), date(2020, 12, 31), "1d", eager=True)
    days = days.filter(days.dt.weekday() <= 5)
    n = len(days)
    px = 100 * 1.01 ** np.arange(n)  # steady 1%/day: every return is a known compound
    stocks = pl.DataFrame({"ticker": ["X"] * n, "date": days, "closeadj": px})
    cal = days.to_numpy().astype("datetime64[D]")
    ends = month_end_dates(cal)
    rebal = ends[ends >= np.datetime64("2020-01-01")]
    f = price_features(stocks, rebal).sort("date")
    assert f.height == len(rebal)
    last = f.row(-1, named=True)
    pos = {d: i for i, d in enumerate(cal)}
    i_t = pos[rebal[-1]]
    i_1, i_12 = pos[ends[-2]], pos[ends[-13]]
    assert last["ret_1m"] == pytest.approx(1.01 ** (i_t - i_1) - 1)
    assert last["ret_12_1"] == pytest.approx(1.01 ** (i_1 - i_12) - 1)
    assert last["ret_60_12"] is not None
    assert last["hi_52w"] == pytest.approx(1.0)  # always at its high
    assert last["vol_1y"] == pytest.approx(0.0, abs=1e-12)  # constant daily return
