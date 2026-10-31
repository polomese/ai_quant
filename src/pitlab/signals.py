"""Signal registry. Each signal is a pure expression over PIT-filtered inputs.

Inputs available on the (ticker, date) frame after the as-of joins:
  T stream (TTM, ART/MRT): netinc, revenue, gp, ncfo, revenue_lag4
  Q stream (quarter, ARQ/MRQ): equity, assets, sharesbas, netinc_qtr and their _lag4
  price side at t: mcap, plus the price features in ``pitlab.price_features``
Ratios with a non-positive denominator are null (not ranked).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import polars as pl

SignalFn = Callable[[], pl.Expr]


def _safe_div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    return pl.when(den > 0).then(num / den).otherwise(None)


def _avg(a: str, b: str) -> pl.Expr:
    # average of current and year-ago balance; fall back to current if year-ago missing
    return pl.when(pl.col(b).is_not_null()).then((pl.col(a) + pl.col(b)) / 2).otherwise(pl.col(a))


SIGNALS: dict[str, SignalFn] = {
    # value
    "earnings_yield": lambda: _safe_div(pl.col("netinc"), pl.col("mcap")),
    "book_to_price": lambda: _safe_div(pl.col("equity"), pl.col("mcap")),
    "cash_flow_yield": lambda: _safe_div(pl.col("ncfo"), pl.col("mcap")),
    # quality
    "roe": lambda: _safe_div(pl.col("netinc"), _avg("equity", "equity_lag4")),
    "gross_profitability": lambda: _safe_div(pl.col("gp"), pl.col("assets")),
    "accruals": lambda: _safe_div(pl.col("netinc") - pl.col("ncfo"), _avg("assets", "assets_lag4")),
    # growth / investment
    "sales_growth": lambda: _safe_div(pl.col("revenue"), pl.col("revenue_lag4")) - 1,
    "asset_growth": lambda: _safe_div(pl.col("assets"), pl.col("assets_lag4")) - 1,
    "net_issuance": lambda: _safe_div(pl.col("sharesbas"), pl.col("sharesbas_lag4")) - 1,
    # earnings news
    "earnings_surprise": lambda: _safe_div(
        pl.col("netinc_qtr") - pl.col("netinc_qtr_lag4"), pl.col("mcap")
    ),
    # price (identical in every arm: the Clock 1 control group)
    "momentum_12_1": lambda: pl.col("ret_12_1"),
    "reversal_1m": lambda: pl.col("ret_1m"),
    "long_term_reversal": lambda: pl.col("ret_60_12"),
    "low_volatility": lambda: pl.col("vol_1y"),
    "high_52w": lambda: pl.col("hi_52w"),
}

#: inputs each signal needs (used to report coverage)
SIGNAL_INPUTS: dict[str, list[str]] = {
    "earnings_yield": ["netinc", "mcap"],
    "book_to_price": ["equity", "mcap"],
    "cash_flow_yield": ["ncfo", "mcap"],
    "roe": ["netinc", "equity"],
    "gross_profitability": ["gp", "assets"],
    "accruals": ["netinc", "ncfo", "assets"],
    "sales_growth": ["revenue", "revenue_lag4"],
    "asset_growth": ["assets", "assets_lag4"],
    "net_issuance": ["sharesbas", "sharesbas_lag4"],
    "earnings_surprise": ["netinc_qtr", "netinc_qtr_lag4", "mcap"],
    "momentum_12_1": ["ret_12_1"],
    "reversal_1m": ["ret_1m"],
    "long_term_reversal": ["ret_60_12"],
    "low_volatility": ["vol_1y"],
    "high_52w": ["hi_52w"],
}


@dataclass(frozen=True)
class SignalInfo:
    family: str
    formula: str
    idea: str
    clock1: str  # how exposed the signal is to look-ahead in fundamentals
    exposure: str = "none"  # high | medium | low | none (price signals); used by the narrative


SIGNAL_INFO: dict[str, SignalInfo] = {
    "earnings_yield": SignalInfo(
        "Value",
        "TTM net income / market cap",
        "Cheap vs. profits beats expensive.",
        "High: the latest quarter's earnings are known weeks before they're filed in arm A.",
        "high",
    ),
    "book_to_price": SignalInfo(
        "Value",
        "Book equity / market cap",
        "Classic value (Fama-French HML).",
        "Low: book value moves slowly, so knowing it early adds little.",
        "low",
    ),
    "cash_flow_yield": SignalInfo(
        "Value",
        "TTM operating cash flow / market cap",
        "Value measured on cash, harder to dress up than earnings.",
        "High: same timing issue as earnings yield.",
        "high",
    ),
    "roe": SignalInfo(
        "Quality",
        "TTM net income / avg. equity (now and a year ago)",
        "Profitable firms beat unprofitable ones.",
        "High: driven by the latest earnings.",
        "high",
    ),
    "gross_profitability": SignalInfo(
        "Quality",
        "TTM gross profit / total assets",
        "Novy-Marx: gross profit is the cleanest profitability measure.",
        "Medium: gross profit is steadier than net income.",
        "medium",
    ),
    "accruals": SignalInfo(
        "Quality",
        "(TTM net income - operating cash flow) / avg. assets; low is good",
        "Sloan: earnings not backed by cash tend to reverse.",
        "Low/mixed: an earnings beat raises accruals, so look-ahead partly cancels out.",
        "low",
    ),
    "sales_growth": SignalInfo(
        "Growth",
        "TTM revenue / revenue a year earlier - 1",
        "Growing firms beat shrinking ones.",
        "High: the newest quarter's sales are the news.",
        "high",
    ),
    "asset_growth": SignalInfo(
        "Investment",
        "Total assets / assets a year earlier - 1; low is good",
        "Cooper-Gulen-Schill: firms that expand fast underperform.",
        "Low: balance sheets change slowly.",
        "low",
    ),
    "net_issuance": SignalInfo(
        "Investment",
        "Shares outstanding / a year earlier - 1; low is good",
        "Issuers underperform, buyback firms outperform.",
        "Low: share counts are mostly known before filing.",
        "low",
    ),
    "earnings_surprise": SignalInfo(
        "Earnings news",
        "(Quarterly net income - same quarter last year) / market cap",
        "Post-earnings drift: prices underreact to earnings news.",
        "Highest: the surprise itself is the look-ahead in arm A.",
        "high",
    ),
    "momentum_12_1": SignalInfo(
        "Momentum",
        "Return from 12 months ago to 1 month ago",
        "Jegadeesh-Titman: winners keep winning for months.",
        "None: prices aren't restated. Control - arms must match.",
    ),
    "reversal_1m": SignalInfo(
        "Mean reversion",
        "Last month's return; low is good",
        "Short-term losers bounce back (liquidity / overreaction).",
        "None: price-based control.",
    ),
    "long_term_reversal": SignalInfo(
        "Mean reversion",
        "Return from 5 years ago to 1 year ago; low is good",
        "De Bondt-Thaler: long-run losers recover.",
        "None: price-based control.",
    ),
    "low_volatility": SignalInfo(
        "Risk",
        "Std. dev. of daily returns, last 252 days; low is good",
        "Low-risk stocks earn as much or more than high-risk ones.",
        "None: price-based control.",
    ),
    "high_52w": SignalInfo(
        "Momentum",
        "Close / highest close of the last 252 days",
        "George-Hwang: stocks near their 52-week high keep rising.",
        "None: price-based control.",
    ),
}


def compute_signals(df: pl.DataFrame, names: dict[str, int]) -> pl.DataFrame:
    """Add raw signal columns ``sig_<name>`` (direction applied: higher = better)."""
    exprs = []
    for name, direction in names.items():
        if name not in SIGNALS:
            raise KeyError(f"unknown signal {name!r}; known: {sorted(SIGNALS)}")
        exprs.append((SIGNALS[name]() * direction).alias(f"sig_{name}"))
    return df.with_columns(exprs)


def winsorize_zscore(
    df: pl.DataFrame, col: str, lower: float, upper: float, by: str | list[str] = "date"
) -> pl.DataFrame:
    """Cross-sectional winsorize at [lower, upper] quantiles and z-score within each ``by`` group.

    Only same-date statistics are used - never full-sample ones.
    """
    lo = pl.col(col).quantile(lower, "linear").over(by)
    hi = pl.col(col).quantile(upper, "linear").over(by)
    w = pl.col(col).clip(lo, hi)
    tmp = f"_{col}_w"
    df = df.with_columns(w.alias(tmp))
    z = (pl.col(tmp) - pl.col(tmp).mean().over(by)) / pl.col(tmp).std().over(by)
    return df.with_columns(z.alias(col)).drop(tmp)
