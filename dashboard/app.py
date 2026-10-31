"""pit-two-clocks dashboard: `pitlab dashboard` (or `streamlit run dashboard/app.py`)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from pitlab import narrative as nv
from pitlab.config import find_project_root, load_settings
from pitlab.narrative import is_price, sig_note
from pitlab.signals import SIGNAL_INFO

# Palette: validated categorical slots (CVD-safe as a set of three). Colour follows the arm.
ARM_COLORS = {"C": "#2a78d6", "A": "#eb6834", "B": "#1baf7a"}
ARM_NAMES = {"A": "A naive", "B": "B timing-fixed", "C": "C true PIT"}
EFFECT_COLORS = {"timing": "#6250d6", "revision": "#eda100"}
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ERA_SHADE = "rgba(128,128,128,0.08)"
ILLUSTRATIVE = (
    "Free-tier data (~30 surviving Dow mega-caps): results are ILLUSTRATIVE ONLY - far too few, "
    "survivorship-biased names for valid cross-sectional statistics."
)

st.set_page_config(page_title="pit-two-clocks", layout="wide")
ROOT = (
    Path(os.environ["PITLAB_ROOT"])
    if os.environ.get("PITLAB_ROOT")
    else find_project_root(Path(__file__).resolve().parent)
)
SETTINGS = load_settings(root=ROOT)
OUT = SETTINGS.outputs_dir


# ------------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_run(run: str, mtime: float) -> dict[str, pl.DataFrame]:
    d = OUT / run
    return {p.stem: pl.read_parquet(p) for p in d.glob("*.parquet")}


def run_summary(run: str) -> dict:
    p = OUT / run / "summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def runs() -> list[str]:
    if not OUT.exists():
        return []
    rs = [p for p in OUT.iterdir() if p.is_dir() and (p / "summary.json").exists()]
    return [p.name for p in sorted(rs, key=lambda p: p.stat().st_mtime, reverse=True)]


def explain(short: str, more: str = "") -> None:
    """Takeaway under a chart: one or two lines always visible, the rest on demand."""
    st.markdown(short)
    if more:
        with st.expander("More"):
            st.markdown(more)


def table_view(df: pl.DataFrame, label: str = "Table view") -> None:
    with st.expander(label):
        st.dataframe(df.to_pandas(), width="stretch", hide_index=True)


def base_layout(fig: go.Figure, height: int = 380, hovermode: str = "x unified", **kw) -> go.Figure:
    fig.update_layout(
        height=height,
        # title pinned to the top of the figure, legend on its own row just above the plot,
        # with enough top margin that the two never overlap
        margin=dict(l=10, r=10, t=88, b=10),
        title=dict(
            x=0, xanchor="left", yref="container", y=0.98, yanchor="top", font=dict(size=16)
        ),
        hovermode=hovermode,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
        **kw,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)", zerolinecolor="rgba(128,128,128,0.5)")
    return fig


def era_shading(fig: go.Figure) -> None:
    for i, (name, (lo, hi)) in enumerate(SETTINGS.slices.eras.items()):
        if i % 2 == 0:
            fig.add_vrect(
                x0=lo or "1990-01-01",
                x1=hi or "2030-12-31",
                fillcolor=ERA_SHADE,
                line_width=0,
                layer="below",
            )
        fig.add_annotation(
            x=lo or "1990-01-01",
            y=1,
            yref="paper",
            text=name,
            showarrow=False,
            xanchor="left",
            font=dict(size=10),
            opacity=0.7,
        )


# ------------------------------------------------------------------------------------------
# sidebar
# ------------------------------------------------------------------------------------------
available = runs()
st.sidebar.title("pit-two-clocks")
if not available:
    st.title("No runs yet")
    st.code("uv run pitlab data synthetic\nuv run pitlab run --run-id demo", language="bash")
    st.stop()

run = st.sidebar.selectbox("Run", available)
summ = run_summary(run)
T = load_run(run, (OUT / run / "summary.json").stat().st_mtime)
mode = summ.get("meta", {}).get("data_mode", "UNKNOWN")
badge = {
    "SYNTHETIC": ":blue-background[SYNTHETIC]",
    "SHARADAR-FREE": ":orange-background[SHARADAR-FREE]",
    "SHARADAR-FULL": ":green-background[SHARADAR-FULL]",
}.get(mode, mode)
st.sidebar.markdown(f"Data mode: **{badge}**")

page = st.sidebar.radio(
    "Page",
    [
        "Experiment",
        "Overview",
        "Signals",
        "IC time series",
        "Long-short returns",
        "Lag sweep",
        "Contamination",
        "Slices",
        "Data health",
        "Clock 2",
    ],
)
# signals present in this run, in config order (grouped by family)
_run_sigs = set(T["ic_monthly"]["signal"].unique()) if "ic_monthly" in T else set(SETTINGS.signals)
all_signals = [s_ for s_ in SETTINGS.signals if s_ in _run_sigs] + sorted(
    _run_sigs - set(SETTINGS.signals)
)
signals = st.sidebar.multiselect("Signals", all_signals, default=all_signals)
arms = [a for a in ("A", "B", "C") if st.sidebar.checkbox(ARM_NAMES[a], value=True)]
slice_type = st.sidebar.selectbox("Slice (Slices page)", ["size", "sector", "era", "restater"])
has_stats = "ic_monthly" in T and T["ic_monthly"].height > 0
if has_stats and T["ic_monthly"]["date"].min() < T["ic_monthly"]["date"].max():
    dmin, dmax = T["ic_monthly"]["date"].min(), T["ic_monthly"]["date"].max()
    dr = st.sidebar.slider(
        "Date range", min_value=dmin, max_value=dmax, value=(dmin, dmax), format="YYYY-MM"
    )
else:
    dr = None


def in_range(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.filter(pl.col("date").is_between(dr[0], dr[1]))
        if dr is not None and "date" in df.columns
        else df
    )


if mode == "SHARADAR-FREE":
    st.warning(ILLUSTRATIVE, icon="⚠️")
elif mode == "SYNTHETIC":
    st.info(
        "Synthetic data with planted look-ahead: these numbers test the method, not the market.",
        icon="🧪",
    )

if not has_stats and page not in ("Data health", "Clock 2", "Signals", "Experiment"):
    st.warning(
        f"No month in run '{run}' had at least {SETTINGS.backtest.min_names_per_date} names with a signal in "
        "every arm, so the cross-sectional statistics are empty. See the Data health page."
    )
    st.stop()
if not signals and page not in ("Data health", "Clock 2", "Signals", "Experiment"):
    st.warning("Select at least one signal in the sidebar.")
    st.stop()


def need(*names: str) -> bool:
    missing = [n for n in names if n not in T]
    if missing:
        st.info(
            f"This run has no {', '.join(missing)}. Run `pitlab run --run-id {run}` (full run) to create it."
        )
        return False
    return True


# ------------------------------------------------------------------------------------------
# pages
# ------------------------------------------------------------------------------------------
def page_overview() -> None:
    st.header("Overview - how much of each signal's backtest is look-ahead?")
    if not need("ic_monthly", "portfolio_returns"):
        return
    ic = in_range(T["ic_monthly"]).filter(
        (pl.col("slice_type") == "all") & pl.col("signal").is_in(signals)
    )
    from pitlab.backtest.compare import paired_ic_differences
    from pitlab.backtest.ic import ic_summary
    from pitlab.backtest.portfolios import portfolio_summary

    nw = SETTINGS.backtest.newey_west_lags
    s = ic_summary(ic, by=["arm", "signal"], nw_lags=nw)
    d = paired_ic_differences(ic, by=["signal"], nw_lags=nw)
    pr = in_range(T["portfolio_returns"]).filter(
        pl.col("signal").is_in(signals) & (pl.col("weighting") == "EW")
    )
    ps = portfolio_summary(pr, by=["arm", "signal"])
    head = nv.headline(s, d, ps)
    st.subheader("Headline: mean rank IC (NW t) per arm, bias decomposition, EW long-short Sharpe")
    st.dataframe(head.to_pandas().round(3), width="stretch", hide_index=True)
    explain(nv.take_top_bias(head), nv.MORE["headline"])

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for eff in ("timing", "revision"):
            if eff in head.columns:
                fig.add_bar(
                    y=head["signal"].to_list(),
                    x=head[eff].to_list(),
                    name=f"{eff} effect",
                    orientation="h",
                    marker_color=EFFECT_COLORS[eff],
                    hovertemplate="%{y}: %{x:.4f}<extra>" + eff + "</extra>",
                )
        fig.update_layout(
            barmode="relative", title="Total bias in mean IC, split into timing and revision"
        )
        st.plotly_chart(base_layout(fig, hovermode="closest"), width="stretch")
        explain(nv.take_bias_split(head), nv.MORE["bias_split"])
    with c2:
        fig = go.Figure()
        for a in arms:
            sub = s.filter(pl.col("arm") == a).sort("signal")
            fig.add_bar(
                x=sub["signal"].to_list(),
                y=sub["mean_ic"].to_list(),
                name=ARM_NAMES[a],
                marker_color=ARM_COLORS[a],
                hovertemplate="%{x}: %{y:.4f}<extra>" + ARM_NAMES[a] + "</extra>",
            )
        fig.update_layout(
            barmode="group", title="Mean rank IC by arm", bargap=0.25, bargroupgap=0.08
        )
        st.plotly_chart(base_layout(fig, hovermode="closest"), width="stretch")
        explain(nv.take_live(s), nv.MORE["ic_by_arm"])
    table_view(d, "Paired IC difference tests (Newey-West)")


def page_ic() -> None:
    st.header("IC time series")
    if not need("ic_monthly"):
        return
    ic = in_range(T["ic_monthly"]).filter(
        (pl.col("slice_type") == "all") & pl.col("arm").is_in(arms)
    )
    for sig in signals:
        sub = ic.filter(pl.col("signal") == sig).sort("date")
        fig = go.Figure()
        for a in arms:
            x = sub.filter(pl.col("arm") == a).with_columns(
                pl.col("ic").rolling_mean(12, min_samples=6).alias("roll")
            )
            fig.add_scatter(
                x=x["date"].to_list(),
                y=x["roll"].to_list(),
                name=ARM_NAMES[a],
                mode="lines",
                line=dict(color=ARM_COLORS[a], width=2),
                hovertemplate="%{y:.3f}",
            )
        era_shading(fig)
        fig.add_hline(y=0, line_width=1, line_color="rgba(128,128,128,0.6)")
        fig.update_layout(title=f"{sig}: rolling 12-month rank IC")
        st.plotly_chart(base_layout(fig, height=320), width="stretch")
        explain(
            nv.take_ic(sig, sub.pivot(on="arm", index="date", values="ic")),
            sig_note(sig)
            + "\n\n"
            + nv.MORE["ic_series"]
            + "\nShaded bands are the eras used on the Slices page.",
        )
    table_view(
        ic.filter(pl.col("signal").is_in(signals)).select("date", "signal", "arm", "ic", "n")
    )


def page_ls() -> None:
    st.header("Long-short returns")
    if not need("portfolio_returns"):
        return
    w = st.radio("Weighting", ["EW", "VW"], horizontal=True)
    pr = in_range(T["portfolio_returns"]).filter(
        (pl.col("weighting") == w) & pl.col("arm").is_in(arms)
    )
    for sig in signals:
        sub = pr.filter(pl.col("signal") == sig).sort("date")
        c1, c2 = st.columns(2)
        f1, f2 = go.Figure(), go.Figure()
        stats: dict[str, tuple[float, float, float] | None] = {}  # arm -> sharpe, $1, max dd
        for a in arms:
            x = sub.filter(pl.col("arm") == a).with_columns(pl.col("ret_ls").fill_null(0.0))
            wealth = np.cumprod(1 + x["ret_ls"].to_numpy())
            dd = wealth / np.maximum.accumulate(wealth) - 1
            stats[a] = nv.ls_stats(x["ret_ls"].to_numpy())
            f1.add_scatter(
                x=x["date"].to_list(),
                y=wealth,
                name=ARM_NAMES[a],
                line=dict(color=ARM_COLORS[a], width=2),
                hovertemplate="%{y:.2f}",
            )
            f2.add_scatter(
                x=x["date"].to_list(),
                y=dd,
                name=ARM_NAMES[a],
                line=dict(color=ARM_COLORS[a], width=2),
                hovertemplate="%{y:.1%}",
            )
        f1.update_yaxes(type="log")
        f1.update_layout(title=f"{sig}: growth of $1, {w} top-minus-bottom quantile (log scale)")
        f2.update_layout(title=f"{sig}: drawdown")
        f2.update_yaxes(tickformat=".0%")
        weighting = "equal" if w == "EW" else "value"
        with c1:
            st.plotly_chart(base_layout(f1, height=320), width="stretch")
            explain(
                nv.take_wealth(sig, stats),
                sig_note(sig) + "\n\n" + nv.MORE["wealth"].format(weighting=weighting),
            )
        with c2:
            st.plotly_chart(base_layout(f2, height=320), width="stretch")
            explain(nv.take_drawdown(sig, stats), nv.MORE["drawdown"])
    table_view(
        pr.filter(pl.col("signal").is_in(signals)).select(
            "date", "signal", "arm", "ret_long", "ret_short", "ret_ls", "turnover"
        )
    )


def page_lag() -> None:
    st.header("Lag sweep - performance vs information delay")
    if not need("lag_sweep"):
        return
    ls = T["lag_sweep"].filter(pl.col("signal").is_in(signals) & pl.col("base_arm").is_in(arms))
    sig = st.selectbox("Signal", signals)
    sub = ls.filter(pl.col("signal") == sig).sort("extra_lag")
    c1, c2 = st.columns(2)
    for col, metric, title in (
        (c1, "mean_ic", "Mean rank IC"),
        (c2, "sharpe_ew", "EW long-short Sharpe"),
    ):
        fig = go.Figure()
        for a in sub["base_arm"].unique().sort().to_list():
            x = sub.filter(pl.col("base_arm") == a)
            fig.add_scatter(
                x=x["extra_lag"].to_list(),
                y=x[metric].to_list(),
                name=ARM_NAMES.get(a, a),
                mode="lines+markers",
                line=dict(color=ARM_COLORS.get(a), width=2),
                marker=dict(size=8),
                hovertemplate="+%{x} days: %{y:.3f}",
            )
        fig.update_xaxes(title="extra availability lag (trading days)", type="category")
        fig.update_layout(title=f"{sig}: {title} vs information delay")
        with col:
            st.plotly_chart(base_layout(fig, height=360), width="stretch")
            explain(nv.take_lag(title, metric, sub), sig_note(sig) + "\n\n" + nv.MORE["lag"])
    table_view(
        sub.select("signal", "base_arm", "extra_lag", "mean_ic", "t_nw", "sharpe_ew", "n_months")
    )


def page_contam() -> None:
    st.header("Contamination - how different are PIT and restated signals?")
    if not need("contamination"):
        return
    c = in_range(T["contamination"]).filter(pl.col("signal").is_in(signals))
    pair = st.radio(
        "Compare",
        c["pair"].unique().sort().to_list(),
        horizontal=True,
        format_func=lambda p: {
            "C_vs_A": "C (PIT) vs A (naive)",
            "C_vs_B": "C (PIT) vs B (restated)",
        }.get(p, p),
    )
    c = c.filter(pl.col("pair") == pair).with_columns(pl.col("date").dt.year().alias("year"))
    hm = c.group_by(["signal", "year"]).agg(pl.col("rank_corr").mean()).sort(["signal", "year"])
    piv = hm.pivot(on="year", index="signal", values="rank_corr").sort("signal")
    years = [c_ for c_ in piv.columns if c_ != "signal"]
    fig = go.Figure(
        go.Heatmap(
            z=piv.select(years).to_numpy(),
            x=years,
            y=piv["signal"].to_list(),
            colorscale=[[i / (len(SEQ_BLUE) - 1), col] for i, col in enumerate(SEQ_BLUE)],
            xgap=2,
            ygap=2,
            colorbar=dict(title="rank corr"),
            hovertemplate="%{y}, %{x}: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(title="Mean cross-sectional rank correlation, by signal and year")
    st.plotly_chart(
        base_layout(fig, height=110 + 40 * len(piv), hovermode="closest"), width="stretch"
    )
    avg = (
        hm.filter(~pl.col("signal").map_elements(is_price, return_dtype=pl.Boolean))
        .group_by("signal")
        .agg(pl.col("rank_corr").mean())
        .sort("rank_corr")
    )
    explain(nv.take_rank_corr(avg), nv.MORE["rank_corr"])
    fig = go.Figure()
    for _i, sig in enumerate(signals):
        x = c.filter(pl.col("signal") == sig).sort("date")
        fig.add_scatter(
            x=x["date"].to_list(),
            y=x["share_decile_change"].to_list(),
            name=sig,
            mode="lines",
            line=dict(width=2),
            hovertemplate="%{y:.1%}",
        )
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(title="Share of names that land in a different decile")
    st.plotly_chart(base_layout(fig, height=340), width="stretch")
    sw = (
        c.filter(pl.col("signal").is_in(signals))
        .group_by("signal")
        .agg(pl.col("share_decile_change").mean())
        .sort("share_decile_change", descending=True)
    )
    explain(nv.take_switch(sw), nv.MORE["switch"])
    table_view(
        c.select("date", "signal", "rank_corr", "share_decile_change", "share_value_differs", "n")
    )


def page_slices() -> None:
    st.header(f"Slices - bias decomposition by {slice_type}")
    if not need("ic_monthly"):
        return
    from pitlab.backtest.compare import paired_ic_differences

    ic = in_range(T["ic_monthly"]).filter(
        (pl.col("slice_type") == slice_type) & pl.col("signal").is_in(signals)
    )
    if ic.height == 0:
        st.info("No data for this slice (universe too small per slice?).")
        return
    d = paired_ic_differences(ic, by=["signal", "slice"], nw_lags=SETTINGS.backtest.newey_west_lags)
    for sig in signals:
        sub = d.filter(pl.col("signal") == sig)
        if sub.height == 0:
            continue
        fig = go.Figure()
        for eff in ("timing", "revision"):
            x = sub.filter(pl.col("effect") == eff).sort("slice")
            fig.add_bar(
                x=x["slice"].to_list(),
                y=x["mean_diff"].to_list(),
                name=f"{eff} effect",
                marker_color=EFFECT_COLORS[eff],
                hovertemplate="%{x}: %{y:.4f}<extra>" + eff + "</extra>",
            )
        fig.update_layout(
            barmode="relative", title=f"{sig}: timing + revision effect on mean IC, by {slice_type}"
        )
        st.plotly_chart(base_layout(fig, height=320, hovermode="closest"), width="stretch")
        tot = sub.filter(pl.col("effect") == "total").sort("mean_diff")
        explain(
            nv.take_slice(sig, tot, slice_type),
            sig_note(sig) + "\n\n" + nv.MORE["slices"].format(slice_type=slice_type),
        )
    table_view(d)


def page_experiment() -> None:
    st.header("The experiment - do backtests know the future?")
    if not need("ic_summary", "effects", "portfolio_summary"):
        return
    st.markdown(nv.experiment_md(T, summ.get("meta", {})))


def page_signals() -> None:
    st.header("Signals - what each one measures")
    explain(nv.SIGNALS_SHORT, nv.MORE["signals"])
    rows = [
        {
            "signal": n,
            "family": i.family,
            "definition": i.formula,
            "why it should work": i.idea,
            "look-ahead exposure": i.clock1,
            "in this run": n in all_signals,
        }
        for n, i in SIGNAL_INFO.items()
    ]
    st.dataframe(pl.DataFrame(rows).to_pandas(), width="stretch", hide_index=True)


def page_health() -> None:
    st.header("Data health")
    meta_p = SETTINGS.processed_dir / "_meta.json"
    val_p = SETTINGS.processed_dir / "_validation.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        c = st.columns(3)
        c[0].metric("Data mode", meta.get("mode"))
        c[1].metric("Fundamentals tickers", f"{meta.get('n_fundamentals_tickers', 0):,}")
        c[2].metric("Built", meta.get("created_at"))
        rc = pl.DataFrame(
            {
                "table": list(meta.get("row_counts", {})),
                "rows": list(meta.get("row_counts", {}).values()),
            }
        )
        st.subheader("Row counts")
        st.dataframe(rc.to_pandas(), hide_index=True)
    else:
        st.info("No processed data found.")
    if val_p.exists():
        v = json.loads(val_p.read_text(encoding="utf-8"))
        st.subheader(f"Validation ({v.get('validated_at')})")
        checks = pl.DataFrame(v["checks"])
        n_fail = checks.filter(~pl.col("passed") & (pl.col("severity") == "error")).height
        (st.error if n_fail else st.success)(
            f"{len(checks) - n_fail}/{len(checks)} error-level checks passed"
        )
        st.dataframe(checks.to_pandas(), hide_index=True, width="stretch")
        st.subheader("Coverage")
        st.json(v.get("coverage", {}))
    else:
        st.info("Run `pitlab data validate` to populate validation results.")
    st.subheader("Download manifests (vintage archive)")
    rows = []
    for m in (
        sorted(SETTINGS.raw_dir.glob("*/*/manifest.json")) if SETTINGS.raw_dir.exists() else []
    ):
        j = json.loads(m.read_text(encoding="utf-8"))
        rows.append(
            {
                "table": j.get("table"),
                "snapshot": j.get("timestamp"),
                "rows": j.get("row_count"),
                "tier": j.get("license_tier"),
                "method": j.get("method", ""),
                "sha256": j.get("sha256", "")[:16],
            }
        )
    if rows:
        st.dataframe(pl.DataFrame(rows).to_pandas(), hide_index=True, width="stretch")
    else:
        st.caption("No manifests yet.")
    st.subheader("Run")
    st.json(summ.get("meta", {}))


def page_clock2() -> None:
    st.header("Clock 2 - the evolution of the model (planned)")
    st.markdown(nv.CLOCK2)


{
    "Experiment": page_experiment,
    "Overview": page_overview,
    "Signals": page_signals,
    "IC time series": page_ic,
    "Long-short returns": page_ls,
    "Lag sweep": page_lag,
    "Contamination": page_contam,
    "Slices": page_slices,
    "Data health": page_health,
    "Clock 2": page_clock2,
}[page]()
