from __future__ import annotations

import os
from datetime import date

import numpy as np
import polars as pl
import pytest

# tests always write under their own temp roots, never the machine's data folder
os.environ.pop("PITLAB_DATA_ROOT", None)


def bdays(start: str, end: str) -> np.ndarray:
    d = np.arange(
        np.datetime64(start), np.datetime64(end) + np.timedelta64(1, "D"), dtype="datetime64[D]"
    )
    return d[np.is_busday(d)]


def fund_row(ticker, dim, cd, dk, **vals):
    base = {
        "equity": 1000.0,
        "assets": 2000.0,
        "sharesbas": 10.0,
        "netinc": 0.0,
        "revenue": 500.0,
        "gp": 200.0,
        "ncfo": 0.0,
    }
    base.update(vals)
    return {
        "ticker": ticker,
        "dimension": dim,
        "calendardate": cd,
        "datekey": dk,
        "reportperiod": cd,
        "lastupdated": dk,
        **base,
    }


def make_fund(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(
        pl.col(c).cast(pl.Date) for c in ("calendardate", "datekey", "reportperiod", "lastupdated")
    )


@pytest.fixture
def calendar_2019_2021() -> np.ndarray:
    return bdays("2019-01-01", "2021-12-31")


@pytest.fixture
def tiny_fund() -> pl.DataFrame:
    """One ticker, period 2020Q1: first reported 100 on Fri 2020-05-15, restated to 120 later.

    MR rows follow the Sharadar quirk datekey == reportperiod.
    """
    q1 = date(2020, 3, 31)
    return make_fund(
        [
            fund_row("XYZ", "ARQ", q1, date(2020, 5, 15), equity=1000.0),
            fund_row("XYZ", "ART", q1, date(2020, 5, 15), netinc=100.0),
            fund_row("XYZ", "MRQ", q1, q1, equity=1200.0),
            fund_row("XYZ", "MRT", q1, q1, netinc=120.0),
        ]
    )
