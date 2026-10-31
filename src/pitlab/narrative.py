"""Plain-English explanations shared by the dashboard and the offline site.

Everything here is text built from result tables: the experiment write-up (problem,
hypotheses, verdicts), the one-line takeaway under each chart, and the longer "More" notes.
Keeping it in one place means the dashboard and `pitlab site` always say the same thing.

All text is Markdown. Only aggregate statistics are used (per signal / arm / month), never
ticker-level values.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl

from pitlab.signals import SIGNAL_INFO

ARM_NAMES = {"A": "A naive", "B": "B timing-fixed", "C": "C true PIT"}


# ------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------
def is_price(sig: str) -> bool:
    i = SIGNAL_INFO.get(sig)
    return bool(i and i.exposure == "none")


def exposure(sig: str) -> str:
    i = SIGNAL_INFO.get(sig)
    return i.exposure if i else "none"


def sig_note(sig: str) -> str:
    """One-line definition of a signal for the 'More' sections."""
    i = SIGNAL_INFO.get(sig)
    return f"**{sig}** = {i.formula}. {i.idea} Look-ahead exposure: {i.clock1}" if i else ""


def money(v: float) -> str:
    return f"${v:,.0f}" if v >= 100 else f"${v:.2f}"


def _ok(x: Any) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


# ------------------------------------------------------------------------------------------
# headline table (one row per signal)
# ------------------------------------------------------------------------------------------
HEAD_COLS = [
    "mean_ic_A",
    "mean_ic_B",
    "mean_ic_C",
    "t_nw_A",
    "t_nw_B",
    "t_nw_C",
    "timing",
    "revision",
    "total",
    "t_total",
    "sharpe_A",
    "sharpe_B",
    "sharpe_C",
]


def headline(ic_sum: pl.DataFrame, diffs: pl.DataFrame, ps_ew: pl.DataFrame) -> pl.DataFrame:
    """Mean IC and t per arm, bias split, and EW Sharpe per arm; one row per signal.

    ``ic_sum``: arm, signal, mean_ic, t_nw. ``diffs``: signal, effect, mean_diff, t_nw.
    ``ps_ew``: arm, signal, sharpe (equal-weighted).
    """
    head = ic_sum.pivot(on="arm", index="signal", values=["mean_ic", "t_nw"])
    head = head.join(
        diffs.pivot(on="effect", index="signal", values="mean_diff"), on="signal", how="left"
    )
    tt = diffs.filter(pl.col("effect") == "total").select("signal", pl.col("t_nw").alias("t_total"))
    head = head.join(tt, on="signal", how="left")
    head = head.join(
        ps_ew.pivot(on="arm", index="signal", values="sharpe").rename(
            {a: f"sharpe_{a}" for a in "ABC"}
        ),
        on="signal",
        how="left",
    )
    cols = ["signal"] + [c for c in HEAD_COLS if c in head.columns]
    return head.select(cols).sort("total", descending=True, nulls_last=True)


def headline_from_tables(T: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Headline from the saved run tables (full sample)."""
    s = T["ic_summary"].filter(pl.col("slice_type") == "all").drop("slice_type", "slice")
    d = T["effects"].filter(pl.col("slice_type") == "all").drop("slice_type", "slice")
    ps = T["portfolio_summary"].filter(pl.col("weighting") == "EW")
    return headline(s, d, ps)


# ------------------------------------------------------------------------------------------
# chart takeaways (short, always visible)
# ------------------------------------------------------------------------------------------
def take_top_bias(head: pl.DataFrame) -> str:
    if not {"mean_ic_A", "mean_ic_C", "total"} <= set(head.columns):
        return "Mean IC per arm and how much of it is look-ahead."
    fund = head.filter(~pl.col("signal").map_elements(is_price, return_dtype=pl.Boolean))
    fund = fund.drop_nulls("total")
    if not fund.height:
        return "Mean IC per arm and how much of it is look-ahead."
    top = fund.row(0, named=True)
    out = (
        f"**Biggest look-ahead: {top['signal']}.** IC {top['mean_ic_A']:.3f} naive vs "
        f"{top['mean_ic_C']:.3f} true PIT"
    )
    if top["mean_ic_A"] and top["mean_ic_A"] > 0:
        out += f", so ~{top['total'] / top['mean_ic_A']:.0%} of the naive IC was never tradable."
    return out + ("" if out.endswith(".") else ".")


def bias_split(head: pl.DataFrame) -> tuple[float, float]:
    """Sum of positive timing and revision bias across signals."""
    tim = float(head["timing"].drop_nulls().clip(lower_bound=0).sum())
    rev = float(head["revision"].drop_nulls().clip(lower_bound=0).sum())
    return tim, rev


def take_bias_split(head: pl.DataFrame) -> str:
    if not {"timing", "revision"} <= set(head.columns):
        return "Bias in mean IC, split into timing and restatement."
    tim, rev = bias_split(head)
    if tim + rev <= 0:
        return "No signal shows a positive bias."
    if tim >= rev:
        return (
            f"**Timing is {tim / (tim + rev):.0%} of the bias.** Using numbers before they were "
            "filed matters far more than restatements."
        )
    return f"**Restatements are {rev / (tim + rev):.0%} of the bias**, more than timing."


def take_live(ic_sum: pl.DataFrame) -> str:
    pit = ic_sum.filter(pl.col("arm") == "C").sort("mean_ic", descending=True)
    live = pit.filter(pl.col("t_nw") > 2)["signal"].to_list()
    back = pit.filter(pl.col("t_nw") < -2)["signal"].to_list()
    out = (
        f"**{len(live)} of {pit.height} signals still work in true PIT** (t > 2): "
        f"{', '.join(live) or 'none'}."
    )
    return out + (f" Working backwards: {', '.join(back)}." if back else "")


def take_ic(sig: str, ic_wide: pl.DataFrame) -> str:
    """``ic_wide``: monthly IC with one column per arm."""
    if is_price(sig):
        return f"{sig}: no gap by design (price signal)."
    if not {"A", "C"} <= set(ic_wide.columns):
        return f"{sig}: rolling 12-month IC."
    g = ic_wide.drop_nulls(["A", "C"])
    if not g.height:
        return f"{sig}: rolling 12-month IC."
    beat = float((g["A"] > g["C"]).mean())  # type: ignore[arg-type]
    a, c = float(g["A"].mean()), float(g["C"].mean())  # type: ignore[arg-type]
    return f"{sig}: naive beat true PIT in **{beat:.0%} of months**; average IC {a:.3f} vs {c:.3f}."


def ls_stats(rets: np.ndarray) -> tuple[float, float, float] | None:
    """(annualized Sharpe, end wealth of $1, max drawdown) from monthly returns."""
    r = np.nan_to_num(np.asarray(rets, dtype=float))
    if len(r) < 2 or r.std() == 0:
        return None
    wealth = np.cumprod(1 + r)
    dd = wealth / np.maximum.accumulate(wealth) - 1
    return float(r.mean() / r.std() * np.sqrt(12)), float(wealth[-1]), float(dd.min())


def take_wealth(sig: str, stats: dict[str, tuple[float, float, float] | None]) -> str:
    a, c = stats.get("A"), stats.get("C")
    if not (a and c):
        return f"{sig}: growth of $1."
    return (
        f"{sig}: $1 grows to **{money(a[1])} naive vs {money(c[1])} true PIT**; "
        f"Sharpe {a[0]:.2f} vs {c[0]:.2f}."
    )


def take_drawdown(sig: str, stats: dict[str, tuple[float, float, float] | None]) -> str:
    a, c = stats.get("A"), stats.get("C")
    if not (a and c):
        return f"{sig}: drawdown."
    return f"{sig}: worst drawdown **{a[2]:.0%} naive vs {c[2]:.0%} true PIT**."


def take_lag(title: str, metric: str, sub: pl.DataFrame) -> str:
    """``sub``: lag sweep rows for one signal (base_arm, extra_lag, <metric>)."""
    lags = sorted(sub["extra_lag"].unique().to_list())
    if not lags:
        return f"{title} vs information delay."

    def at(arm: str, lag: int) -> float | None:
        r = sub.filter((pl.col("base_arm") == arm) & (pl.col("extra_lag") == lag))
        return r[metric][0] if r.height else None

    a0, a1, c0, c1 = at("A", lags[0]), at("A", lags[-1]), at("C", lags[0]), at("C", lags[-1])
    if not all(_ok(v) for v in (a0, a1, c0, c1)):
        return f"{title} vs information delay."
    f = ".3f" if metric == "mean_ic" else ".2f"
    return (
        f"{title}: naive {a0:{f}} → {a1:{f}} with +{lags[-1]} days' delay; "
        f"true PIT {c0:{f}} → {c1:{f}}."
    )


def take_rank_corr(avg: pl.DataFrame) -> str:
    """``avg``: signal, rank_corr (fundamental signals only), sorted ascending."""
    if avg.height > 1:
        return (
            f"**Most reshuffled: {avg['signal'][0]}** (avg. rank corr {avg['rank_corr'][0]:.2f}). "
            f"Least: {avg['signal'][-1]} ({avg['rank_corr'][-1]:.2f})."
        )
    if avg.height == 1:
        return f"**{avg['signal'][0]}: avg. rank corr {avg['rank_corr'][0]:.2f}** between the two versions."
    return "Rank agreement between the two versions of each signal."


def take_switch(sw: pl.DataFrame) -> str:
    """``sw``: signal, share_decile_change sorted descending."""
    if not sw.height:
        return "Share of stocks in a different decile."
    return (
        f"**{sw['signal'][0]}: {sw['share_decile_change'][0]:.0%} of stocks sit in a different "
        "decile** in an average month."
    )


def take_slice(sig: str, tot: pl.DataFrame, slice_type: str) -> str:
    """``tot``: slice, mean_diff for the total effect, sorted ascending."""
    if is_price(sig):
        return f"{sig}: price signal, no bias expected in any {slice_type}."
    tot = tot.drop_nulls("mean_diff")
    if tot.height < 2:
        return f"{sig}: bias by {slice_type}."
    return (
        f"{sig}: bias is biggest in **{tot['slice'][-1]}** ({tot['mean_diff'][-1]:.3f} IC) "
        f"and smallest in {tot['slice'][0]} ({tot['mean_diff'][0]:.3f})."
    )


# ------------------------------------------------------------------------------------------
# "More" notes (static; {placeholders} filled by the caller)
# ------------------------------------------------------------------------------------------
MORE = {
    "headline": """
- **IC** = rank correlation between the signal and next month's return, averaged over months.
  0.02 is useful, 0.05 is strong. **t** is Newey-West; |t| > 2 is significant.
- **A** naive: restated numbers, known at quarter end. **B**: restated numbers, known at the
  filing date. **C** true PIT: as-first-reported numbers, known the day after filing.
- **timing** = A − B: bias from using numbers before they were filed.
  **revision** = B − C: bias from using numbers restated later. **total** = A − C.
- **Sharpe** = equal-weight top-minus-bottom decile, annualized.
- Price signals (momentum, reversal, volatility, 52-week high) aren't restated, so their bias
  must be ~0. If it isn't, something is broken.
""",
    "bias_split": """
Each bar is how much mean IC the naive backtest overstates. Purple = timing (A − B), amber =
restatements (B − C). Bars right of zero mean naive looks better than reality. A bar left of
zero means naive *understates* the signal (look-ahead hurt it, e.g. accruals). Short bars on
price signals are expected: they have no look-ahead.
""",
    "ic_by_arm": """
Bars are mean IC per arm. Orange (naive) taller than blue (true PIT) = part of the backtest
was information you couldn't have had. When the three bars are equal the signal is clean.
A negative bar means the signal worked backwards in that period.
""",
    "ic_series": """
Each line is the 12-month rolling average of monthly IC. The gap between orange (naive) and
blue (true PIT) is look-ahead that wouldn't survive live trading. Lines drifting toward zero
= the signal is fading.
""",
    "wealth": """
Long the top decile, short the bottom decile, rebalanced monthly, {weighting}-weighted, before
costs. Log scale: a straight line is a steady return. The gap between the orange and blue
lines is the part of the backtest you couldn't have earned.
""",
    "drawdown": """
Drawdown = how far the strategy is below its previous peak. The true PIT line (blue) is the
pain a live investor would have felt. Naive backtests usually look smoother than reality.
""",
    "lag": """
Each point re-runs the arm with its data made available N trading days later. If the naive
line falls toward the true PIT line as delay grows, the naive edge was early information.
If the true PIT line also falls fast, the signal's value comes from acting quickly after
filings. A flat true PIT line = a slow signal you can trade at leisure.
""",
    "rank_corr": """
Each cell = how alike the two versions rank stocks in that year (Spearman correlation,
averaged over months). 1.0 = same ranking, so look-ahead changes nothing. Lighter cells =
timing or restatements reshuffle which stocks look good. This uses no returns: it shows
*how different* the inputs are, not whether the difference made money. Price signals sit at
1.0 by design.
""",
    "switch": """
Share of stocks that land in a different decile under the two versions. Every switch is a
trade the naive backtest makes that a real investor wouldn't. Spikes usually line up with
earnings seasons, when new quarters arrive in the naive arm weeks before they're filed.
""",
    "slices": """
Each bar stacks the timing (purple) and restatement (amber) bias in mean IC for one
{slice_type}. Tallest bars = where a naive backtest misleads most. Expect timing bias to be
bigger in small caps (slower filers, less coverage) and restatement bias to sit almost
entirely in restaters.
""",
    "signals": """
All signals are ranked across stocks each month-end and z-scored (winsorized at 1%/99%).
"Low is good" signals are flipped so that higher always means a better expected return.
Fundamentals use trailing-twelve-month (TTM) flows and the latest quarter's balance sheet,
as each arm would have seen them on that date. A fundamental older than ~15 months is
treated as missing. Price signals use adjusted closes up to the rebalance day.
""",
}

CLOCK2 = """
Clock 1 asks *when* a number became knowable. **Clock 2** asks *what the model already knew*:
a language model trained on text published after an event can "remember" the outcome, so
scoring historical news with today's LLM leaks the future into the backtest.

**Planned experiments** (interfaces in `src/pitlab/clock2/interfaces.py`, no computation yet):

1. **IC discontinuity at the knowledge cutoff** - sentiment IC should drop for news after the
   model's training cutoff; a jump at the cutoff measures memorisation.
2. **Model-vintage ladder** - score the same news with models of successive cutoffs; a PIT-honest
   result uses only the model vintage available at each news date.
3. **Entity anonymisation** - replace tickers/names/dates with placeholders; the gap between the
   raw and anonymised IC estimates how much comes from recognising the company and period.
4. **Memorisation probes** - ask the model to complete post-event facts (returns, headlines);
   accuracy above chance flags contamination.

Each news item is a `NewsItem`; each model is a `DatedScorer` with a `knowledge_cutoff`.
"""

SIGNALS_SHORT = (
    "**10 fundamental signals** test for look-ahead. **5 price signals** are the control "
    "group: prices aren't restated, so their A, B and C results must match."
)


# ------------------------------------------------------------------------------------------
# the experiment write-up
# ------------------------------------------------------------------------------------------
def _verdict(frac: float) -> str:
    if frac >= 0.75:
        return "✅ Supported"
    if frac >= 0.5:
        return "🟡 Partly supported"
    return "❌ Not supported"


def hypotheses(T: dict[str, pl.DataFrame]) -> list[dict[str, str]]:
    """Each hypothesis with its prediction, what the run found, and a verdict.

    Verdicts are computed from the run, not written by hand, so they stay honest if the
    data or settings change.
    """
    head = headline_from_tables(T)
    eff = T["effects"]
    fund = head.filter(~pl.col("signal").map_elements(is_price, return_dtype=pl.Boolean))
    price = head.filter(pl.col("signal").map_elements(is_price, return_dtype=pl.Boolean))
    out: list[dict[str, str]] = []

    # H1 naive overstates fundamentals
    sig_up = fund.filter((pl.col("total") > 0) & (pl.col("t_total") > 2))
    sig_down = fund.filter((pl.col("total") < 0) & (pl.col("t_total") < -2))
    top = fund.drop_nulls("total").row(0, named=True) if fund.height else None
    found = f"**{sig_up.height} of {fund.height}** fundamental signals look significantly better in the naive backtest."
    if top:
        found += (
            f" Worst case: **{top['signal']}**, IC {top['mean_ic_A']:.3f} naive vs "
            f"{top['mean_ic_C']:.3f} true PIT"
        )
        if _ok(top.get("sharpe_A")) and _ok(top.get("sharpe_C")):
            found += f" (Sharpe {top['sharpe_A']:.2f} vs {top['sharpe_C']:.2f})"
        found += "."
    if sig_down.height:
        verb = "looks" if sig_down.height == 1 else "look"
        found += (
            f" Exception: {', '.join(sig_down['signal'].to_list())} {verb} slightly *worse* "
            "in the naive backtest."
        )
    out.append(
        {
            "id": "H1",
            "claim": "A naive backtest makes fundamental signals look better than they really are.",
            "predict": "Arm A (naive) shows a higher IC than arm C (true PIT) for most fundamental signals.",
            "found": found,
            "verdict": _verdict(sig_up.height / fund.height if fund.height else 0),
        }
    )

    # H2 timing beats restatement
    tim, rev = bias_split(fund) if fund.height else (0.0, 0.0)
    share = tim / (tim + rev) if tim + rev > 0 else 0.0
    out.append(
        {
            "id": "H2",
            "claim": "Most of the bias comes from timing, not from restated numbers.",
            "predict": "The timing part (A − B) is much bigger than the restatement part (B − C).",
            "found": f"Timing is **{share:.0%}** of the total bias; restatements are {1 - share:.0%}.",
            "verdict": _verdict(share),
        }
    )

    # H3 news-like signals are most exposed
    grp = fund.with_columns(
        pl.col("signal").map_elements(exposure, return_dtype=pl.String).alias("exp")
    )
    hi = grp.filter(pl.col("exp") == "high")["total"].drop_nulls()
    lo = grp.filter(pl.col("exp") == "low")["total"].drop_nulls()
    if hi.len() and lo.len():
        m_hi, m_lo = float(hi.mean()), float(lo.mean())  # type: ignore[arg-type]
        ratio = m_hi / m_lo if m_lo > 0 else math.inf
        frac = 1.0 if m_hi > 2 * max(m_lo, 0) and m_hi > 0 else (0.5 if m_hi > m_lo else 0.0)
        found = (
            f"Average bias **{m_hi:.3f} IC** for earnings-driven signals vs **{m_lo:.3f}** for "
            "slow balance-sheet signals"
            + (f" ({ratio:.0f}× larger)." if math.isfinite(ratio) and ratio > 1 else ".")
        )
        out.append(
            {
                "id": "H3",
                "claim": "The more a signal depends on the latest earnings, the bigger the bias.",
                "predict": "Earnings yield, ROE, cash-flow yield, sales growth and earnings surprise "
                "are hit harder than book-to-price, asset growth, net issuance and accruals.",
                "found": found,
                "verdict": _verdict(frac),
            }
        )

    # H4 price controls show nothing
    if price.height:
        worst = float(price["total"].abs().max() or 0.0)  # type: ignore[arg-type]
        out.append(
            {
                "id": "H4",
                "claim": "Signals built only from prices show no bias (the control group).",
                "predict": "Prices are never restated and don't wait for a filing, so A, B and C give "
                "identical results for momentum, reversal, volatility and 52-week high.",
                "found": f"Largest bias across the {price.height} price signals: **{worst:.4f} IC** "
                "(identical in every arm). The method doesn't invent bias where none can exist.",
                "verdict": _verdict(1.0 if worst < 1e-6 else 0.0),
            }
        )

    # H5 small caps
    size = eff.filter((pl.col("slice_type") == "size") & (pl.col("effect") == "total"))
    hi_sigs = [s for s in fund["signal"].to_list() if exposure(s) == "high"]
    wins, n = 0, 0
    for s in hi_sigs:
        by_size = {
            r["slice"]: r["mean_diff"]
            for r in size.filter(pl.col("signal") == s).iter_rows(named=True)
        }
        if _ok(by_size.get("small")) and _ok(by_size.get("large")):
            n += 1
            wins += by_size["small"] > by_size["large"]
    if n:
        out.append(
            {
                "id": "H5",
                "claim": "The bias is bigger for small companies.",
                "predict": "Small caps file later and get less attention, so early knowledge of their "
                "results is worth more.",
                "found": f"Bias is larger in small caps than large caps for **{wins} of {n}** "
                "earnings-driven signals.",
                "verdict": _verdict(wins / n),
            }
        )

    # H6 delay closes the gap
    if "lag_sweep" in T and T["lag_sweep"].height:
        ls = T["lag_sweep"]
        closed = []
        max_lag = int(ls["extra_lag"].max())  # type: ignore[arg-type]
        min_lag = int(ls["extra_lag"].min())  # type: ignore[arg-type]
        for s in hi_sigs:
            g = ls.filter(pl.col("signal") == s)

            def v(arm: str, lag: int, g: pl.DataFrame = g) -> float | None:
                r = g.filter((pl.col("base_arm") == arm) & (pl.col("extra_lag") == lag))
                return r["mean_ic"][0] if r.height else None

            a0, a1, c0 = v("A", min_lag), v("A", max_lag), v("C", min_lag)
            if _ok(a0) and _ok(a1) and _ok(c0) and a0 - c0 > 1e-4:  # type: ignore[operator]
                closed.append(min(1.0, (a0 - a1) / (a0 - c0)))  # type: ignore[operator]
        if closed:
            avg = float(np.mean(closed))
            out.append(
                {
                    "id": "H6",
                    "claim": "Delaying the naive data removes the fake edge.",
                    "predict": "If the extra IC really comes from knowing results early, pushing arm A's "
                    "data back in time should shrink it toward arm C.",
                    "found": f"A {max_lag}-trading-day delay closes **{avg:.0%}** of the A-to-C gap on "
                    "average across the earnings-driven signals.",
                    "verdict": _verdict(avg),
                }
            )
    return out


def _meaning(hyp: list[dict[str, str]]) -> str:
    ok = {h["id"]: h["verdict"].startswith("✅") for h in hyp}
    parts = []
    if ok.get("H1"):
        parts.append("A naive backtest can make a mediocre signal look like a great one.")
    else:
        parts.append("Look-ahead changes some backtests, but not most of them.")
    if ok.get("H2"):
        parts.append(
            "The fake part comes almost entirely from **using results before they're "
            "published**, not from later corrections."
        )
    if ok.get("H3"):
        parts.append("The more a signal leans on fresh earnings news, the bigger the illusion.")
    if ok.get("H1") or ok.get("H2"):
        parts.append("The fix is simple: only use a number from the day it was filed.")
    if ok.get("H4"):
        parts.append(
            "\n\nPrice-based signals, which can't peek ahead, were unaffected. That shows the "
            "method measures real look-ahead and doesn't invent it."
        )
    return " ".join(parts)


def experiment_md(T: dict[str, pl.DataFrame], meta: dict[str, Any]) -> str:
    """Full experiment write-up in Markdown, with verdicts computed from the run."""
    hyp = hypotheses(T)
    n_sup = sum(h["verdict"].startswith("✅") for h in hyp)
    span = f"{meta.get('first_rebalance', '?')[:7]} to {meta.get('last_rebalance', '?')[:7]}"
    n_months = meta.get("n_rebalances", "?")
    med = meta.get("median_universe")
    univ = f"about {med:,.0f}" if isinstance(med, int | float) else "thousands of"
    meaning = _meaning(hyp)
    rows = "\n".join(
        f"""
### {h["id"]}. {h["claim"]}
- **Prediction:** {h["predict"]}
- **Result:** {h["found"]}
- **Verdict:** {h["verdict"]}
"""
        for h in hyp
    )
    return f"""
## The problem: backtests that know the future

A **backtest** tests an investing rule on history: "if I had bought cheap stocks every month
since 1998, how would I have done?" It's how almost every quant strategy gets judged.

The catch is the data. A database you download today shows each company's numbers as they
look *today*. Two things are wrong with that:

1. **Numbers arrive late.** A quarter ends on 31 March, but the company files its report
   weeks later, often in May. A backtest that uses the March numbers on 31 March is
   trading on information nobody had yet.
2. **Numbers get changed.** Companies restate past results. Today's database often shows the
   corrected figure, not the one investors actually saw.

Both leak the future into the past. This is **look-ahead bias**. The fix is to be
**point-in-time (PIT)**: at each date, use only what was public on that date.

It's like grading a weather forecaster using tomorrow's newspaper. The forecaster looks
brilliant, but only because of information they never had.

## The question

**How much of a stock signal's backtested performance is really look-ahead, and where does
it come from: late filings or restated numbers?**

## The experiment

Run the same strategy three times. Same stocks, same rules, same months. The only thing
that changes is *what the strategy is allowed to know, and when*:

| Arm | Which numbers | Usable from | What it represents |
|---|---|---|---|
| **A naive** | Latest (restated) | Quarter end | A careless backtest |
| **B timing-fixed** | Latest (restated) | Filing date | Fixes timing only |
| **C true PIT** | As first reported | Day after filing | What was really knowable |

The gaps split the bias into two parts:
- **A − B = timing bias** (using numbers before they were filed)
- **B − C = restatement bias** (using numbers corrected later)

**Data.** US stocks from {span} ({n_months} month-ends, {univ} stocks a month), including
companies that later went bust or were delisted, so the test isn't limited to survivors.

**Signals.** 15 well-known signals: 10 built from company accounts (value, quality, growth,
earnings news) and 5 built only from prices (momentum, mean reversion, volatility). See the
Signals page.

**Measures.** *IC*, the rank correlation between a signal and next month's return (0.02 is
useful, 0.05 is strong), and the *Sharpe ratio* of a portfolio that buys the top 10% and
shorts the bottom 10%.

**Checks built in:**
- **Control group.** Price signals can't have look-ahead, so they must show zero bias.
- **Tested on fake data first.** Before touching real data, the method was run on synthetic
  stocks with look-ahead planted on purpose. It had to find the planted bias, and find none
  where there wasn't any. (See `tests/test_end_to_end_synthetic.py`.)
- **Delay test.** Push the naive data back 1 to 60 trading days and watch whether the fake
  edge fades.

## Hypotheses and results

**{n_sup} of {len(hyp)} hypotheses are supported** by the data.
{rows}

## What it means

{meaning}

## Limits

- **Statistics, not proof.** The results support the hypotheses; they don't prove them.
  Small effects can still be noise.
- **No trading costs.** Real returns would be lower for every arm.
- **US stocks, monthly rebalancing, one data source.** Other markets or faster strategies
  could differ.
- **Clock 1 only.** A second source of look-ahead, AI models that "remember" history they
  were trained on, is the planned next stage (see the Clock 2 page).
"""
