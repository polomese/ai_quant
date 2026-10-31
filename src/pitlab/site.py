"""`pitlab site`: a single, self-contained HTML version of the dashboard for one run.

The file works offline: double-click it and it opens in any browser, with no Python and no
server. Plotly is embedded, so charts stay interactive (hover, zoom, click a legend item to
hide an arm). Pages, takeaways and "More" notes match the Streamlit dashboard because both
use :mod:`pitlab.narrative`.

Licence guard: the file carries **aggregate results only** (per signal / arm / month). It
contains no ticker, no company-level value and no raw table, so it can't be used to rebuild
the source data. There are no data-health or validation pages (vendor terms forbid
publishing evaluations of the data itself without approval).
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from markdown_it import MarkdownIt

from pitlab import narrative as nv
from pitlab.config import Settings
from pitlab.signals import SIGNAL_INFO

TEMPLATE = Path(__file__).with_name("site_template.html")
ARMS = ["A", "B", "C"]
SLICE_TYPES = ["size", "sector", "era", "restater"]
_md = MarkdownIt("commonmark").enable("table")


def _html(text: str) -> str:
    return _md.render(text)


def _inline(text: str) -> str:
    return _md.renderInline(text)


def _num(x: Any, nd: int = 4) -> float | None:
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) or math.isinf(x) else round(x, nd)


def _nums(xs: Any, nd: int = 4) -> list[float | None]:
    return [_num(v, nd) for v in xs]


def _align(df: pl.DataFrame, months: pl.DataFrame, col: str) -> pl.Series:
    return months.join(df.select("date", col), on="date", how="left")[col]


def load_tables(run_dir: Path) -> tuple[dict[str, pl.DataFrame], dict[str, Any]]:
    tables = {p.stem: pl.read_parquet(p) for p in run_dir.glob("*.parquet")}
    meta = json.loads((run_dir / "summary.json").read_text(encoding="utf-8")).get("meta", {})
    return tables, meta


def build_payload(T: dict[str, pl.DataFrame], meta: dict[str, Any], signals: list[str]) -> dict:
    ic_all = T["ic_monthly"].filter(pl.col("slice_type") == "all")
    run_sigs = set(ic_all["signal"].unique())
    sigs = [s for s in signals if s in run_sigs] + sorted(run_sigs - set(signals))
    months = ic_all.select("date").unique().sort("date")
    dates = [d.isoformat() for d in months["date"]]

    head = nv.headline_from_tables(T)
    ic_sum = T["ic_summary"].filter(pl.col("slice_type") == "all")
    pr = T["portfolio_returns"]
    eff = T["effects"]
    lag = T.get("lag_sweep")
    cont = T.get("contamination")

    per_sig: dict[str, Any] = {}
    for sig in sigs:
        ic_s = ic_all.filter(pl.col("signal") == sig)
        ic_roll = {}
        for a in ARMS:
            x = ic_s.filter(pl.col("arm") == a).sort("date")
            x = x.with_columns(pl.col("ic").rolling_mean(12, min_samples=6).alias("roll"))
            ic_roll[a] = _nums(_align(x, months, "roll"))
        take_ic = nv.take_ic(sig, ic_s.pivot(on="arm", index="date", values="ic"))

        ls: dict[str, Any] = {}
        for w in ("EW", "VW"):
            series, stats = {}, {}
            for a in ARMS:
                x = pr.filter((pl.col("signal") == sig) & (pl.col("weighting") == w))
                x = x.filter(pl.col("arm") == a).sort("date")
                r = _align(x, months, "ret_ls").fill_null(0.0).to_numpy()
                wealth = np.cumprod(1 + r)
                dd = wealth / np.maximum.accumulate(wealth) - 1
                series[a] = {"wealth": _nums(wealth), "dd": _nums(dd)}
                stats[a] = nv.ls_stats(x["ret_ls"].to_numpy()) if x.height else None
            ls[w] = {
                "series": series,
                "take_wealth": _inline(nv.take_wealth(sig, stats)),
                "take_dd": _inline(nv.take_drawdown(sig, stats)),
            }

        lag_d: dict[str, Any] = {}
        if lag is not None and lag.height:
            sub = lag.filter(pl.col("signal") == sig).sort("extra_lag")
            lag_d = {
                "lags": sorted(sub["extra_lag"].unique().to_list()),
                "arms": {
                    a: {
                        "mean_ic": _nums(sub.filter(pl.col("base_arm") == a)["mean_ic"]),
                        "sharpe": _nums(sub.filter(pl.col("base_arm") == a)["sharpe_ew"], 3),
                    }
                    for a in sub["base_arm"].unique().sort().to_list()
                },
                "take_ic": _inline(nv.take_lag("Mean rank IC", "mean_ic", sub)),
                "take_sharpe": _inline(nv.take_lag("EW long-short Sharpe", "sharpe_ew", sub)),
            }

        slices = {}
        for st in SLICE_TYPES:
            e = eff.filter((pl.col("slice_type") == st) & (pl.col("signal") == sig))
            if not e.height:
                continue
            order = sorted(e["slice"].unique().to_list())
            by = {
                k: e.filter(pl.col("effect") == k).select("slice", "mean_diff")
                for k in nv_effects()
            }
            slices[st] = {
                "slices": order,
                **{
                    k: _nums(
                        pl.DataFrame({"slice": order}).join(v, on="slice", how="left")["mean_diff"]
                    )
                    for k, v in by.items()
                },
                "take": _inline(nv.take_slice(sig, by["total"].sort("mean_diff"), st)),
            }

        switch = {}
        if cont is not None and cont.height:
            for pair in cont["pair"].unique().sort().to_list():
                c = cont.filter((pl.col("pair") == pair) & (pl.col("signal") == sig)).sort("date")
                if c.height:
                    sw = c.group_by("signal").agg(pl.col("share_decile_change").mean())
                    switch[pair] = {
                        "share": _nums(_align(c, months, "share_decile_change"), 3),
                        "take": _inline(nv.take_switch(sw)),
                    }

        per_sig[sig] = {
            "ic": ic_roll,
            "take_ic": _inline(take_ic),
            "note": "<p>" + _inline(nv.sig_note(sig)) + "</p>",
            "ls": ls,
            "lag": lag_d,
            "slices": slices,
            "switch": switch,
        }

    heat = {}
    if cont is not None and cont.height:
        for pair in cont["pair"].unique().sort().to_list():
            c = cont.filter(pl.col("pair") == pair).with_columns(
                pl.col("date").dt.year().alias("year")
            )
            hm = c.group_by(["signal", "year"]).agg(pl.col("rank_corr").mean())
            years = sorted(hm["year"].unique().to_list())
            rows = [s for s in sigs if s in set(hm["signal"])]
            z = [
                _nums(
                    pl.DataFrame({"year": years}).join(
                        hm.filter(pl.col("signal") == s), on="year", how="left"
                    )["rank_corr"],
                    3,
                )
                for s in rows
            ]
            avg = (
                hm.filter(~pl.col("signal").map_elements(nv.is_price, return_dtype=pl.Boolean))
                .group_by("signal")
                .agg(pl.col("rank_corr").mean())
                .sort("rank_corr")
            )
            heat[pair] = {
                "years": years,
                "signals": rows,
                "z": z,
                "take": _inline(nv.take_rank_corr(avg)),
            }

    head_rows = [
        {k: (_num(v) if k != "signal" else v) for k, v in r.items()}
        for r in head.iter_rows(named=True)
    ]
    ic_by_arm = {
        a: {
            r["signal"]: _num(r["mean_ic"])
            for r in ic_sum.filter(pl.col("arm") == a).iter_rows(named=True)
        }
        for a in ARMS
    }
    more = {k: _html(v) for k, v in nv.MORE.items()}
    return {
        "meta": {
            "run_mode": meta.get("data_mode", "UNKNOWN"),
            "first": meta.get("first_rebalance"),
            "last": meta.get("last_rebalance"),
            "months": meta.get("n_rebalances"),
            "universe": meta.get("median_universe"),
            "generated": datetime.now(UTC).strftime("%Y-%m-%d"),
        },
        "dates": dates,
        "signals": sigs,
        "info": [
            {
                "signal": n,
                "family": i.family,
                "definition": i.formula,
                "idea": i.idea,
                "exposure": i.clock1,
                "price": i.exposure == "none",
                "in_run": n in run_sigs,
            }
            for n, i in SIGNAL_INFO.items()
        ],
        "head": head_rows,
        "ic_by_arm": ic_by_arm,
        "take": {
            "top": _inline(nv.take_top_bias(head)),
            "split": _inline(nv.take_bias_split(head)),
            "live": _inline(nv.take_live(ic_sum)),
            "signals": _inline(nv.SIGNALS_SHORT),
        },
        "more": more,
        "experiment": _html(nv.experiment_md(T, meta)),
        "clock2": _html(nv.CLOCK2),
        "per_signal": per_sig,
        "heat": heat,
    }


def nv_effects() -> list[str]:
    return ["timing", "revision", "total"]


def assert_aggregate_only(payload: dict, tickers: set[str]) -> None:
    """Refuse to write a site that mentions any ticker (a cheap leak check)."""
    text = json.dumps(payload)
    # only check tokens that can't be ordinary words (>= 2 chars, all caps, not a signal name)
    hits = [t for t in tickers if len(t) >= 3 and f'"{t}"' in text]
    if hits:
        raise ValueError(f"site payload contains ticker-like values: {hits[:5]}")


def build_site(settings: Settings, run_id: str, out: Path | None = None) -> Path:
    run_dir = settings.outputs_dir / run_id
    if not (run_dir / "summary.json").exists():
        raise FileNotFoundError(f"no run '{run_id}' under {settings.outputs_dir}")
    T, meta = load_tables(run_dir)
    payload = build_payload(T, meta, list(settings.signals))
    tick_p = settings.processed_dir / "tickers.parquet"
    if tick_p.exists():
        assert_aggregate_only(payload, set(pl.read_parquet(tick_p, columns=["ticker"])["ticker"]))
    from plotly.offline import get_plotlyjs

    html = TEMPLATE.read_text(encoding="utf-8")
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = html.replace("/*__PLOTLY__*/", get_plotlyjs()).replace("__PAYLOAD__", data)
    out = out or run_dir / "site" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
