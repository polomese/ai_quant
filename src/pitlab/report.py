"""Persist run results to outputs/<run_id>/ and render a static markdown / HTML summary."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

from pitlab.config import Settings
from pitlab.pipeline import Results

ILLUSTRATIVE = (
    "Illustrative only: the Sharadar free tier covers ~30 surviving mega-caps, far too few and too "
    "survivorship-biased for valid cross-sectional conclusions."
)


def run_dir(settings: Settings, run_id: str) -> Path:
    return settings.outputs_dir / run_id


def _clean(v: Any) -> Any:
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def write_results(settings: Settings, run_id: str, res: Results) -> Path:
    out = run_dir(settings, run_id)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in res.tables.items():
        df.write_parquet(out / f"{name}.parquet")
    # merge meta with any previous summary (e.g. a lag-sweep-only rerun into the same run)
    prev = {}
    if (out / "summary.json").exists():
        prev = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    summary = build_summary(settings, run_id, res, out, prev)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    if "headline" in res.tables or not (out / "config_used.yaml").exists():
        (out / "config_used.yaml").write_text(settings.to_yaml(), encoding="utf-8")
    (out / "summary.md").write_text(render_markdown(out), encoding="utf-8")
    logger.info(f"wrote results to {out}")
    return out


def _rows(df: pl.DataFrame) -> list[dict[str, Any]]:
    return [{k: _clean(v) for k, v in r.items()} for r in df.iter_rows(named=True)]


def build_summary(
    settings: Settings, run_id: str, res: Results, out: Path, prev: dict
) -> dict[str, Any]:
    s = dict(prev)
    s.update({"run_id": run_id, "updated_at": datetime.now(UTC).isoformat(timespec="seconds")})
    s.setdefault("meta", {}).update(res.meta)
    if "headline" in res.tables:
        s["headline"] = _rows(res.tables["headline"])
        s["ic_summary_all"] = _rows(
            res.tables["ic_summary"]
            .filter(pl.col("slice_type") == "all")
            .drop("slice_type", "slice")
        )
        s["effects_all"] = _rows(
            res.tables["effects"].filter(pl.col("slice_type") == "all").drop("slice_type", "slice")
        )
        s["portfolio_summary"] = _rows(res.tables["portfolio_summary"])
        s["fama_macbeth"] = _rows(res.tables["fm_results"])
    if "lag_sweep" in res.tables:
        s["lag_sweep"] = _rows(
            res.tables["lag_sweep"].select(
                "arm", "base_arm", "extra_lag", "signal", "mean_ic", "t_nw", "sharpe_ew"
            )
        )
    s["files"] = sorted(p.name for p in out.glob("*.parquet"))
    return s


def _fmt(v: Any, nd: int = 3) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return "" if math.isnan(v) else f"{v:.{nd}f}"
    return str(v)


def md_table(rows: list[dict[str, Any]], cols: list[str], headers: list[str] | None = None) -> str:
    headers = headers or cols
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |")
    return "\n".join(lines)


def render_markdown(out: Path) -> str:
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    m = s.get("meta", {})
    lines = [f"# pit-two-clocks run `{s['run_id']}`", ""]
    lines.append(f"- data mode: **{m.get('data_mode')}**")
    lines.append(
        f"- rebalances: {m.get('n_rebalances')} ({m.get('first_rebalance')} to {m.get('last_rebalance')}), "
        f"tickers: {m.get('n_tickers')}, median comparison universe: {m.get('median_universe')}, "
        f"quantiles: {m.get('n_quantiles')}"
    )
    if m.get("data_mode") == "SHARADAR-FREE":
        lines += ["", f"> **{ILLUSTRATIVE}**"]
    if m.get("data_mode") == "SYNTHETIC":
        lines += [
            "",
            "> Synthetic data with *planted* look-ahead (see docs/METHODOLOGY.md). Numbers validate the "
            "method, they say nothing about real markets.",
        ]
    if "headline" in s:
        lines += ["", "## Mean rank IC by arm and bias decomposition", ""]
        cols = [
            "signal",
            "mean_ic_A",
            "mean_ic_B",
            "mean_ic_C",
            "t_nw_A",
            "t_nw_C",
            "mean_diff_timing",
            "mean_diff_revision",
            "mean_diff_total",
            "t_nw_total",
            "sharpe_EW_A",
            "sharpe_EW_B",
            "sharpe_EW_C",
        ]
        hdr = [
            "signal",
            "IC A",
            "IC B",
            "IC C",
            "t A",
            "t C",
            "timing A-B",
            "revision B-C",
            "total A-C",
            "t(total)",
            "Sharpe A",
            "Sharpe B",
            "Sharpe C",
        ]
        lines.append(md_table(s["headline"], cols, hdr))
    if "portfolio_summary" in s:
        lines += ["", "## Long-short portfolios", ""]
        lines.append(
            md_table(
                s["portfolio_summary"],
                [
                    "signal",
                    "arm",
                    "weighting",
                    "ann_return",
                    "ann_vol",
                    "sharpe",
                    "max_drawdown",
                    "turnover",
                ],
            )
        )
    if "fama_macbeth" in s:
        lines += ["", "## Fama-MacBeth (signal slope, controls: size + sector)", ""]
        lines.append(
            md_table(s["fama_macbeth"], ["signal", "arm", "coef_mean", "t_nw", "n_months"])
        )
    if "lag_sweep" in s:
        lines += ["", "## Lag sweep (extra trading days of delay)", ""]
        lines.append(
            md_table(
                s["lag_sweep"], ["signal", "base_arm", "extra_lag", "mean_ic", "t_nw", "sharpe_ew"]
            )
        )
    lines += ["", f"_generated {s.get('updated_at')}_", ""]
    return "\n".join(lines)


def render_html(out: Path) -> Path:
    """Very small static HTML wrapper around the markdown tables (no external assets)."""
    md = render_markdown(out)
    body = []
    in_table = False
    for line in md.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set("".join(cells)) <= set("-"):
                continue
            tag = "th" if not in_table else "td"
            if not in_table:
                body.append("<table>")
                in_table = True
            body.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
            continue
        if in_table:
            body.append("</table>")
            in_table = False
        if line.startswith("# "):
            body.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("> "):
            body.append(f"<blockquote>{line[2:]}</blockquote>")
        elif line.startswith("- "):
            body.append(f"<p>{line[2:]}</p>")
        elif line.strip():
            body.append(f"<p>{line}</p>")
    if in_table:
        body.append("</table>")
    html = (
        "<!doctype html><html><head><meta charset='utf-8'><title>pit-two-clocks report</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:2em auto;padding:0 16px}"
        "table{border-collapse:collapse;font-size:13px;margin:1em 0}td,th{border:1px solid #ccc;padding:3px 7px;"
        "text-align:right}th{background:#f3f3f3}blockquote{background:#fff4e5;padding:8px 12px;margin:0}</style>"
        "</head><body>" + "\n".join(body) + "</body></html>"
    )
    p = out / "report.html"
    p.write_text(html, encoding="utf-8")
    return p
