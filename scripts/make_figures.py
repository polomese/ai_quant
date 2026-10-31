"""Render README figures (aggregate charts) from a SYNTHETIC run into docs/img/.

    uv run python scripts/make_figures.py --run-id demo

Refuses to run on Sharadar-derived runs: the public repo only carries synthetic figures.
"""

from __future__ import annotations

import argparse
import json
import sys

import plotly.graph_objects as go
import polars as pl

from pitlab.config import find_project_root, load_settings

ARM_COLORS = {"C": "#2a78d6", "A": "#eb6834", "B": "#1baf7a"}
ARM_NAMES = {"A": "A naive", "B": "B timing-fixed", "C": "C true PIT"}
EFFECT_COLORS = {"timing": "#6250d6", "revision": "#eda100"}
LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="#fcfcfb",
    plot_bgcolor="#fcfcfb",
    font=dict(family="Arial", size=13, color="#0b0b0b"),
    margin=dict(l=20, r=20, t=60, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="demo")
    args = ap.parse_args()
    s = load_settings()
    out = s.outputs_dir / args.run_id
    summ = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    if summ.get("meta", {}).get("data_mode") != "SYNTHETIC":
        print("refusing: README figures must come from SYNTHETIC runs (Sharadar licence)")
        return 1
    img = find_project_root() / "docs" / "img"
    img.mkdir(parents=True, exist_ok=True)

    # 1. bias decomposition
    eff = pl.read_parquet(out / "effects.parquet").filter(pl.col("slice_type") == "all")
    order = eff.filter(pl.col("effect") == "total").sort("mean_diff")["signal"].to_list()
    fig = go.Figure()
    for e in ("timing", "revision"):
        x = eff.filter(pl.col("effect") == e)
        m = dict(zip(x["signal"], x["mean_diff"], strict=True))
        fig.add_bar(
            y=order,
            x=[m[o] for o in order],
            name=f"{e} effect",
            orientation="h",
            marker_color=EFFECT_COLORS[e],
        )
    fig.update_layout(
        barmode="relative",
        title="Synthetic demo: look-ahead bias in mean rank IC (A - C), split",
        xaxis_title="IC overstatement",
        **LAYOUT,
    )
    fig.write_image(img / "bias_decomposition.png", width=900, height=420, scale=2)

    # 2. rolling IC for roe
    ic = pl.read_parquet(out / "ic_monthly.parquet").filter(
        (pl.col("slice_type") == "all") & (pl.col("signal") == "roe")
    )
    fig = go.Figure()
    for a in ("A", "B", "C"):
        x = (
            ic.filter(pl.col("arm") == a)
            .sort("date")
            .with_columns(pl.col("ic").rolling_mean(12).alias("r"))
        )
        fig.add_scatter(
            x=x["date"].to_list(),
            y=x["r"].to_list(),
            name=ARM_NAMES[a],
            line=dict(color=ARM_COLORS[a], width=2),
        )
    fig.add_hline(y=0, line_color="#999", line_width=1)
    fig.update_layout(title="Synthetic demo: ROE rolling 12-month rank IC by arm", **LAYOUT)
    fig.write_image(img / "rolling_ic_roe.png", width=900, height=380, scale=2)

    # 3. lag sweep
    ls = pl.read_parquet(out / "lag_sweep.parquet").filter(pl.col("signal") == "roe")
    fig = go.Figure()
    for a in ("A", "C"):
        x = ls.filter(pl.col("base_arm") == a).sort("extra_lag")
        fig.add_scatter(
            x=[str(v) for v in x["extra_lag"]],
            y=x["mean_ic"].to_list(),
            name=ARM_NAMES[a],
            mode="lines+markers",
            marker=dict(size=9),
            line=dict(color=ARM_COLORS[a], width=2),
        )
    fig.update_layout(
        title="Synthetic demo: ROE mean IC vs extra information delay",
        xaxis_title="extra lag (trading days)",
        yaxis_title="mean rank IC",
        **LAYOUT,
    )
    fig.write_image(img / "lag_sweep_roe.png", width=900, height=380, scale=2)
    print(f"wrote figures to {img}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
