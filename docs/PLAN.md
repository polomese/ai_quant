# Implementation plan (first pass)

1. **Scaffold**: pyproject (uv), .gitignore, .env.example, pre-commit (5 MB / `data/` guard), CI.
2. **Config**: `config/settings.yaml` + `config/columns.yaml`; `pitlab.config` loads YAML into
   pydantic models; `.env` supplies `SHARADAR_API_KEY`.
3. **Synthetic data**: generate Sharadar-shaped tables (tickers, fundamentals ARQ/ART/MRQ/MRT,
   stocks, daily, sp500, actions, events, descriptions) with injected look-ahead:
   - *timing*: an earnings surprise is in the period's numbers but moves the price only at the
     filing date (20–90 days after period end) → only arm A can trade on it;
   - *revision*: restatement deltas proportional to post-filing returns → arms A and B see it.
4. **PIT core**: `pit/availability.py` builds one wide row per (ticker, calendardate) per arm with
   an `available_at`; `pit/asof.py` does the backward as-of join onto rebalance dates.
   Hand-built tests first.
5. **Signals**: registry of 7 pure functions + cross-sectional winsorize/z-score.
6. **Returns/universe**: month-end grid, forward returns from `closeadj`, delisting returns,
   eligibility (price, mcap, history, category, optional S&P 500 PIT membership).
7. **Backtests**: rank IC (+NW), Fama–MacBeth, decile/quintile EW/VW long–short, turnover.
8. **Compare**: paired IC differences, contamination (C vs A rank corr, decile switches), slices.
9. **Pipeline + report**: `pitlab run` → `outputs/<run_id>/…`; `pitlab report` → md + html.
10. **Dashboard**: Streamlit multipage-in-one with the 8 required views.
11. **Sharadar client**: bulk (`years=`) first, per-ticker REST fallback, backoff, resume, raw
    snapshots + manifest, ingest to Parquet + DuckDB, validation.
12. **Docs**: README, METHODOLOGY, DATA_NOTES; end-to-end test asserting A > B > C.
