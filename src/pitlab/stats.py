"""Time-series statistics: Newey-West means, block bootstrap, performance metrics."""

from __future__ import annotations

import math

import numpy as np
import statsmodels.api as sm


def _clean(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return a[np.isfinite(a)]


def newey_west_mean(x, lags: int = 6) -> dict[str, float]:
    """Mean of ``x`` with a Newey-West (Bartlett kernel) HAC standard error.

    Returns mean, se, t and n. With fewer than ``lags + 3`` observations returns NaN se/t.
    """
    a = _clean(x)
    n = len(a)
    out = {"mean": float(a.mean()) if n else math.nan, "se": math.nan, "t": math.nan, "n": n}
    if n < lags + 3 or np.allclose(a, a[0]):
        return out
    res = sm.OLS(a, np.ones(n)).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True}
    )
    out["se"] = float(res.bse[0])
    out["t"] = float(res.tvalues[0])
    return out


def block_bootstrap_mean(
    x, block: int = 6, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> dict[str, float]:
    """Moving-block bootstrap CI for the mean (keeps short-run autocorrelation)."""
    a = _clean(x)
    n = len(a)
    if n < 2 * block:
        return {"mean": float(a.mean()) if n else math.nan, "lo": math.nan, "hi": math.nan}
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    means = a[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(a.mean()), "lo": float(lo), "hi": float(hi)}


def max_drawdown(returns) -> float:
    a = _clean(returns)
    if not len(a):
        return math.nan
    wealth = np.cumprod(1 + a)
    peak = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    return float((wealth / peak - 1).min())


def perf_stats(returns, periods_per_year: int = 12) -> dict[str, float]:
    a = _clean(returns)
    if len(a) < 2:
        return {
            "ann_return": math.nan,
            "ann_vol": math.nan,
            "sharpe": math.nan,
            "max_drawdown": math.nan,
            "n": len(a),
        }
    mu, sd = a.mean() * periods_per_year, a.std(ddof=1) * math.sqrt(periods_per_year)
    return {
        "ann_return": float(mu),
        "ann_vol": float(sd),
        "sharpe": float(mu / sd) if sd > 0 else math.nan,
        "max_drawdown": max_drawdown(a),
        "n": len(a),
    }
