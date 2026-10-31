# Methodology

## 1. Question

Backtests built on today's database quietly use information a real investor did not have.
This study measures the size of that gap for classic fundamental factors along **two clocks**:

- **Clock 1, the evolution of the data.** A reported number is only knowable from its filing
  date, not from its fiscal period end. It can also be revised (restated) later. *Implemented.*
- **Clock 2, the evolution of the model.** A model trained after an event can "remember" the
  outcome. *Scaffolded only (see §8).*

## 2. Arms

All three arms share the universe, signals, rebalance dates, and portfolio rules. They differ
only in **which version** of each fundamentals row is used and **when** it becomes usable
(`available_at`).

| Arm | Values | `available_at` |
|---|---|---|
| **A, naive** | MRQ / MRT (most recent, restated) | `shift(calendardate, 0)` |
| **B, timing-fixed** | MRQ / MRT (restated) | `shift(original filing date, 1)` |
| **C, true PIT** | ARQ / ART (as first reported) | `shift(own datekey, 1)` |

`shift(d, k)` is the k-th trading day after the first trading day on or after `d`. A filing on
trading day T with k = 1 becomes usable on T+1. A weekend filing becomes usable on the second
trading day after it. Rebalances happen at the close of the last trading day of each month, so
a row is visible at rebalance t only if `available_at <= t`.

- **Timing effect** = A − B. This is the value of using data before it was filed.
- **Revision effect** = B − C. This is the value of using restated rather than first-reported
  numbers.
- **Total bias** = A − C.

**Original filing date (arm B).** This is the *earliest* `datekey` among ARQ (or ART) rows with
the same ticker and `calendardate`. MR rows without a matching AR row are dropped
(conservative). Arm B uses the same +1 trading-day base lag as arm C, so B − C isolates
revisions only.

**Lag sweep.** Arms A and C are rerun with +0, +1, +5, +20, and +60 *extra* trading days on top
of their base rule. Because rebalancing is monthly, extra lags of 1 to 20 days usually move
visibility by exactly one month. That is why A+1, A+5, and A+20 often coincide.

## 3. Point-in-time machinery

1. **Streams.** Stock items (equity, assets, sharesbas) come from the quarterly dimension (Q =
   ARQ/MRQ). Flow items (netinc, revenue, gp, ncfo) come from trailing twelve months (T =
   ART/MRT). Each stream gets its own `available_at` and its own as-of lookup.
2. **As-of join.** `join_asof(strategy="backward", by="ticker")` on `available_at`. Fundamentals
   are never merged on period. A post-join assertion fails if any matched row has
   `available_at > date`.
3. **Several versions.** Among visible rows, the one with the latest `available_at` wins. That
   is how an amendment of the current period replaces the original from its own date onward.
4. **Cumulative-max rule.** A row for an *older* period that becomes available after a newer
   period was already known (for example, a late amendment of last year) never replaces the
   newer period.
5. **Year-ago values** (sales growth, asset growth, average equity, average assets) come from a
   second as-of join keyed on `(ticker, calendardate − 1y)`. It returns the version of the
   year-ago period that was available **when the current row became available**, never a later
   revision.
6. **Staleness.** Fundamentals whose period ended more than 456 days (about 15 months) before
   the rebalance are ignored. This is the same rule for all arms.

These rules are covered by hand-built tests in `tests/test_asof.py` and
`tests/test_availability.py`. The tests prove that a filing dated after the rebalance is
invisible to C, that A sees it, and that B uses restated values but only after the original
filing date.

## 4. Universe

- The Sharadar `tickers` table, filtered to `Domestic Common Stock` (plus the Primary and
  Secondary Class variants). **Delisted names are included.**
- Monthly rebalance on the last trading day of each month. A trailing partial month is dropped.
- Eligibility at t: the stock printed a price on t, unadjusted close is at least $5, there are at
  least 12 months of price history, and market cap is valid.
- **Market cap** comes from the price side at t. The first choice is `daily.marketcap`
  (millions → USD). The fallback is PIT `sharesbas` (as-reported, available by t) × unadjusted
  close. It is never taken from a fundamentals row that could be dated later. The same market
  cap is used for all arms.
- **Comparison universe (conservative).** At each date and for each signal, IC and portfolios
  use only names with a valid signal in *every* compared arm (the intersection). Differences
  between arms are therefore not driven by which names are covered. Set
  `universe.comparison_universe: per_arm` to disable this.
- An optional S&P 500 filter uses the `sp500` table as of each date. Sharadar encodes this
  table as quarterly `historical` snapshots plus a `current` snapshot. Paid keys may also carry
  `added`/`removed` events, which take precedence. Today's constituent list is never used for
  past dates.

## 5. Signals

Each signal is a pure expression over the as-of-joined inputs. Ratios with non-positive
denominators are null.

The single source of truth is `pitlab.signals` (`SIGNALS` for the formulas, `SIGNAL_INFO`
for the plain-English definitions shown on the dashboard's Signals page).

**Fundamental signals.** These test for look-ahead.

| Signal | Family | Formula | Direction | Idea |
|---|---|---|---|---|
| earnings_yield | Value | TTM netinc / mcap | + | Cheap vs. profits beats expensive |
| book_to_price | Value | equity / mcap | + | Classic value (HML) |
| cash_flow_yield | Value | TTM ncfo / mcap | + | Value on cash, harder to dress up |
| roe | Quality | TTM netinc / avg(equity, equity 4Q ago) | + | Profitable beats unprofitable |
| gross_profitability | Quality | TTM gp / assets | + | Novy-Marx |
| accruals | Quality | (TTM netinc − TTM ncfo) / avg(assets, assets 4Q ago) | − | Sloan: non-cash earnings reverse |
| sales_growth | Growth | TTM revenue / TTM revenue 4Q ago − 1 | + | Growers beat shrinkers |
| asset_growth | Investment | assets / assets 4Q ago − 1 | − | Fast expanders underperform |
| net_issuance | Investment | sharesbas / sharesbas 4Q ago − 1 | − | Issuers lag, buybacks lead |
| earnings_surprise | Earnings news | (quarterly netinc − same quarter a year ago) / mcap | + | Post-earnings drift |

**Price signals (control group).** Prices are not restated and do not depend on a filing
date, so these are identical in arms A, B and C. Their bias must be about zero; a non-zero
bias means a pipeline bug. They also give the fundamental results a benchmark of a
"normal" factor.

| Signal | Family | Formula | Direction | Idea |
|---|---|---|---|---|
| momentum_12_1 | Momentum | closeadj(t−1m) / closeadj(t−12m) − 1 | + | Winners keep winning |
| reversal_1m | Mean reversion | closeadj(t) / closeadj(t−1m) − 1 | − | Last month's losers bounce |
| long_term_reversal | Mean reversion | closeadj(t−12m) / closeadj(t−60m) − 1 | − | Long-run losers recover |
| low_volatility | Risk | std of daily returns, last 252 days (min 126) | − | Low risk earns as much or more |
| high_52w | Momentum | close / max close of last 252 days (min 126) | + | Near the high keeps rising |

Monthly lags use the exact earlier month-end close. If the stock didn't trade that day, the
value is missing; it is not stale-filled.

Notes:
- If the year-ago balance is missing, the average falls back to the current balance.
- earnings_surprise is the seasonal random-walk surprise scaled by price, a simple proxy for
  SUE that needs no analyst estimates. Quarterly net income comes from the ARQ/MRQ rows
  (`netinc_qtr`). Its year-ago value is taken as it was known when the current quarter
  became available.

**Standardization.** Values are winsorized at the 1st/99th percentile and z-scored within each
(arm, signal, date) cross-section. Full-sample statistics are never used.

## 6. Returns

- The forward return for rebalance t runs from close(t) to close(t'), where t' is the next
  rebalance, using `closeadj` (split-, dividend-, and spinoff-adjusted). It therefore starts
  the trading day after the information cut-off.
- **Delistings.** If the last price falls in (t, t'], the return is the price return to the last
  close compounded with a delisting return: **−30%** when the reason is unknown or
  performance-related (default, configurable), and **0%** when the `actions` table shows
  `acquisitionby`, `mergerfrom`, or `spinoffdividend` within 30 days of the last price.
  Sharadar does not provide CRSP-style delisting returns, so −30% is a conventional assumption
  in the Shumway (1997) range. Results on delisting-heavy slices are sensitive to it.
- A missing next price without a delisting flag gives a null return, and the name is dropped
  from that month.

## 7. Statistics

- **Rank IC.** This is the monthly Spearman correlation of the z-scored signal with the forward
  return. Months with fewer than `min_names_per_date` names (default 20) are skipped. The run
  reports the mean, standard deviation, **Newey-West t (6 lags)**, and hit rate.
- **Paired arm tests.** These compute the monthly IC difference between two arms on matched
  dates, with a Newey-West t on the mean difference.
- **Fama-MacBeth.** Each month, the forward return is regressed by OLS on the signal z-score, a
  z-scored log market cap, and sector dummies. The run reports the time-series mean slope with
  a Newey-West t (6 lags).
- **Portfolios.** Stocks are sorted into deciles, or quintiles when the median comparison
  universe has fewer than 300 names. Long-short returns are computed both equal-weighted and
  value-weighted (by mcap at t). The run reports annualized return, volatility, Sharpe, maximum
  drawdown, and turnover (0.5 × Σ|Δw| per leg, before price drift, averaged over the two legs).
- **Contamination** (independent of returns). For each month and signal, the run reports the
  Spearman correlation of C values with A (and with B) values, the share of names whose decile
  differs, and the share whose value differs.
- **Slices.** Size terciles (by mcap at t), sectors, restater versus non-restater, and eras
  (pre-2002, 2002–2008, 2009–2015, 2016+). The restater flag is a *descriptive, full-sample*
  ticker attribute: the firm has any MR value that differs from its AR counterpart. It is used
  only to slice results, never to trade.
- `stats.block_bootstrap_mean` provides moving-block bootstrap confidence intervals for further
  work.

## 8. Clock 2 (planned, not implemented)

`src/pitlab/clock2/interfaces.py` defines two interfaces:

- `NewsItem(id, ticker, published_at, headline, body, source)` is the news record, with its own
  PIT timestamp.
- `DatedScorer` is a Protocol with `model_id: str`, `knowledge_cutoff: date`, and
  `score(text, as_of) -> float`. `is_pit_usable(scorer, as_of)` is true only if
  `knowledge_cutoff <= as_of`.

Planned tests:

1. **IC discontinuity at the model cutoff.** Score news before and after a model's
   `knowledge_cutoff`. A drop in sentiment IC right after the cutoff is evidence that pre-cutoff
   IC is partly memorization.
2. **Model-vintage ladder.** Score the same history with models of successive cutoffs. A
   PIT-honest backtest uses, at each date, only the latest model whose cutoff precedes it. The
   gap to the "newest model everywhere" backtest is the Clock 2 bias, analogous to A − C.
3. **Entity anonymization.** Replace company names, tickers, people, and dates with placeholders
   and rescore. The IC lost to anonymization estimates how much of the "sentiment" is
   recognition of a known company and period.
4. **Memorization probes.** Prompt the model to complete post-event facts (next-quarter
   results, price moves, headlines). Accuracy above chance flags contamination for that
   period.

To plug in later: add a news source that yields `NewsItem`s, a scorer that implements
`DatedScorer`, and a signal that aggregates scores per (ticker, rebalance date) using only items
with `published_at <= t`. It flows through the same IC, portfolio, and compare code with an arm
dimension of "scorer vintage".

## 9. Synthetic data (validation of the method)

`pitlab data synthetic` builds a Sharadar-shaped universe (default 800 stocks, 25 years, seed
42). It mimics Sharadar quirks such as MR `datekey == reportperiod`, and plants two leaks with
known structure:

- **Timing leak.** Each firm-quarter has an earnings surprise. It is included in that quarter's
  netinc, ncfo, revenue, and gp, but the price reacts only on the filing date, 20–90 calendar
  days after period end. Arm A can front-run the reaction. B and C cannot.
- **Revision leak.** Each firm-quarter has a news shock that drives a post-filing price drift
  over 63 trading days. For "restater" firms (30%), half of their quarters are later restated by
  an amount proportional to that shock, 120–400 days after filing. MR values therefore predict
  returns after the filing date, while AR values do not. A and B can trade on this. C cannot.
- A small genuine premium on persistent profitability gives C a little true IC.
- **Controls.** book_to_price, asset_growth, and accruals carry no planted leak. The surprise
  is real cash (it is also in ncfo), and equity and assets are never restated. They show
  revision effect ≈ 0 by construction. A small *timing* effect can still appear, because
  knowing balance-sheet data earlier has some value in any world.

`tests/test_end_to_end_synthetic.py` asserts that A > B > C in mean IC for the four leaked
signals, that the total bias is significant, that revision ≈ 0 for controls, that the revision
effect concentrates in restaters, that contamination metrics behave as expected, and that
arm A's IC decays in the lag sweep.

## 10. Conservative choices made (where the spec was ambiguous)

| Choice | Alternative not taken | Why |
|---|---|---|
| Intersection comparison universe | per-arm universes | Arm differences should not reflect coverage |
| B's lag equals C's (+1 trading day) | B at the raw filing date | B − C then isolates revisions |
| Earliest AR datekey for B | latest AR datekey | "Original filing" |
| Weekend or holiday filing +1 means the second trading day after | next trading day | Errs late |
| Drop MR rows without an AR match (arm B) | fall back to calendardate | Never guess timing |
| Year-ago value as known at the current row's availability | latest revision of year-ago | No back-filled revisions |
| Cumulative-max calendardate rule | pure latest `available_at` | Late old-period amendments must not overwrite newer periods |
| Market cap from price side, same for all arms | fundamentals `marketcap` | Spec, and avoids arm-specific size |
| 15-month staleness cutoff | none | Dead fundamentals should not be traded |
| Unknown delisting −30% | 0% | Avoids survivorship-flattering returns |

## 11. Known limitations

- **Daily granularity.** Sharadar's filing date is a date, not a timestamp. An after-close
  filing on day T is treated as usable from T+1's close. The +1 day absorbs this, but intraday
  effects are out of scope.
- **Vintages.** Sharadar exposes first-reported (AR) and *current* restated (MR) values, not the
  full revision history. Arm B uses today's restated value, not the value as of some
  intermediate date. `pitlab data download` keeps every snapshot (`data/raw/<table>/<timestamp>/`),
  so you build your own vintage archive over time.
- **Restater flag** is a full-sample attribute. It is fine for slicing, not for trading.
- **Delisting returns** are assumed (see §6).
- **Free tier** covers about 30 surviving Dow mega-caps with about 5 years of history (measured
  with the public `test-api-key`). It validates the pipeline only. Statistics on it are
  illustrative, and the dashboard says so.
- **Corporate actions** rely on Sharadar's `closeadj`. Ticker recycling is handled by
  Sharadar's ticker renaming. `permaticker` is carried but not used as the join key.

### Addendum (2026-10-28): impossible filing dates
As-reported rows whose filing date (`datekey`) precedes their fiscal period end
(`reportperiod`) are dropped before any arm is built. On the full dataset this is 20 of 1.56M
AR rows. Re-dating them, for example to the period end, would still guess at an unknown real
filing date. Dropping is the PIT-strict option.
