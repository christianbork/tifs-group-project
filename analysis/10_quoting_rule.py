# %% [markdown]
# # 10 Quoting-rule simulation: what are the spread forecasts worth to a market maker?
# Scripts 07-09 compare forecasts by RMSE. A market maker does not care about RMSE; it cares
# about the P&L of the quotes it posts. This script turns every forecast (and every
# benchmark) into a quoting rule and simulates it on the signed trade feed:
#
# - at the start of each 5-min bar the market maker posts a half-spread h_t (bps) around
#   the prevailing mid, floored at half a tick;
# - each aggressive lit trade is a potential fill: if the aggressor paid at least h_t
#   (its effective half-spread e >= h_t) the market maker's quote was at least as good as
#   the one that was hit, so it takes the trade up to its quote size;
# - revenue per filled share = h_t; adverse-selection cost = the 5-min mid move after the
#   trade (decomposition from script 05); realised = revenue - adverse selection.
#
# Rules: in-sample average, yesterday's average, last bar's spread, the 07/08 models
# (predictions saved by 08 for the OOS window and by 09 for the forward week) and the
# realised quoted half-spread of the bar as an oracle. Tighter quotes fill more often but
# earn less per fill and face the same adverse selection, so the comparison is a real
# trade-off, not a forecasting score. Outputs: 10_quoting_<sample>.csv, 10_quoting_frontier.png,
# 10_quoting_daily_pnl.png.

# %% Setup
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tifs_group_project import config
from tifs_group_project.bars import tick_size
from tifs_group_project.impact import decomposition
from tifs_group_project.quoting import fills, summarise

GROUPS = {"constrained": config.CONSTRAINED, "unconstrained": config.UNCONSTRAINED}
GROUP_COLORS = {"constrained": "#2a78d6", "unconstrained": "#eb6834"}
SAMPLES = ["OOS", "FWD"]
plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e5e4e0",
        "grid.linewidth": 0.6,
    }
)

# %% Trades: signed continuous lit trades with the 5-min decomposition, keyed to their bar
frames = []
for stock in config.STOCKS:
    d = (
        pd.read_parquet(config.PROCESSED / f"signed_{stock}.parquet")
        .sort_values("ts", kind="stable")
        .reset_index(drop=True)
    )
    dec = decomposition(d)
    d["eff_bps"], d["impact_5m_bps"] = dec["eff_bps"], dec["impact_5m_bps"]
    d["half_tick_bps"] = tick_size(d["bid"], d["ask"]) / 2 / d["mid"] * 1e4
    d["bar_ts"] = d["ts"].dt.floor("5min")
    d["stock"] = stock
    frames.append(d)
trades = pd.concat(frames, ignore_index=True)
trades = trades[
    (trades["side"] != 0) & trades["impact_5m_bps"].notna() & (trades["eff_bps"] >= 0)
].copy()
trades["group"] = np.where(
    trades["stock"].isin(config.CONSTRAINED), "constrained", "unconstrained"
)
is_trades = trades[trades["sample"] == "IS"]
quote_size = (is_trades if len(is_trades) else trades).groupby("stock")["vol"].median()
print(
    "Quote size per stock (median lit trade, shares):\n",
    quote_size.round(0).to_string(),
    "\n",
)

# %% Rules per stock-bar: benchmarks from the bars, model forecasts from 08 / 09
bars = (
    pd.read_parquet(config.PROCESSED / "bars_5min.parquet")
    .sort_values(["stock", "ts"])
    .reset_index(drop=True)
)
g = bars.groupby("stock")
rules = bars[["stock", "date", "ts", "sample", "quoted_hs_bps"]].copy()
rules["rule_oracle (realised bar spread)"] = bars["quoted_hs_bps"]
rules["rule_last bar"] = g["quoted_hs_bps"].shift(1)
day_mean = bars.groupby(["stock", "date"]).apply(
    lambda x: np.average(x["quoted_hs_bps"].fillna(0), weights=x["q_sec"]),
    include_groups=False,
)
rules = rules.merge(
    day_mean.groupby(level=0).shift(1).rename("rule_yesterday").reset_index(),
    on=["stock", "date"],
    how="left",
)
is_bars = bars[bars["sample"] == "IS"]
if len(is_bars):
    rules["rule_IS average"] = rules["stock"].map(
        is_bars.groupby("stock")["quoted_hs_bps"].mean()
    )
for name in ["predictions_oos.parquet", "predictions_fwd.parquet"]:
    f = config.PROCESSED / name
    if f.exists():
        pred = pd.read_parquet(f)
        model_cols = [
            c
            for c in pred.columns
            if c.startswith("pred_")
            and c not in ("pred_IS average", "pred_yesterday", "pred_last bar")
        ]
        pred = pred[["stock", "ts", *model_cols]].rename(
            columns={c: "rule_" + c[5:] for c in model_cols}
        )
        rules = (
            rules.merge(pred, on=["stock", "ts"], how="left")
            if not any(c in rules for c in pred.columns if c.startswith("rule_"))
            else rules.set_index(["stock", "ts"])
            .combine_first(pred.set_index(["stock", "ts"]))
            .reset_index()
        )
        print(f"loaded {name}: {len(model_cols)} model rules")
    else:
        print(
            f"{name} not found (run 08 / 09 first); model rules skipped for that sample"
        )
RULES = [c for c in rules.columns if c.startswith("rule_")]
print("Rules:", [r[5:] for r in RULES], "\n")

tr = trades.merge(
    rules[["stock", "ts", *RULES]].rename(columns={"ts": "bar_ts"}),
    on=["stock", "bar_ts"],
    how="inner",
)

# %% Simulate every rule, per sample, per group and stock
results, daily = [], []
for sample in SAMPLES:
    t_s = tr[tr["sample"] == sample]
    if t_s.empty:
        print(f"{sample}: no trades with this sample label, skipped")
        continue
    for rule in RULES:
        if t_s[rule].isna().all():
            continue
        t_r = t_s[t_s[rule].notna()].copy()
        f = fills(
            t_r, t_r[rule], np.inf, t_r["half_tick_bps"]
        )  # placeholder size, replaced per stock below
        size = np.minimum(
            t_r["vol"].to_numpy(float), t_r["stock"].map(quote_size).to_numpy(float)
        )
        f["size"] = np.where(f["filled"], size, 0.0)
        f["pnl_dkk"] = np.where(
            f["filled"],
            f["size"] * t_r["mid"].to_numpy(float) * f["realised_bps"] / 1e4,
            0.0,
        )
        f["signed_size"] = -t_r["side"].to_numpy(float) * f["size"]
        for unit in [*GROUPS, *config.STOCKS]:
            sel = (t_r["group"] == unit) if unit in GROUPS else (t_r["stock"] == unit)
            if not sel.any():
                continue
            results.append(
                {
                    "sample": sample,
                    "rule": rule[5:],
                    "unit": unit,
                    "trades": int(sel.sum()),
                    **summarise(t_r[sel], f[sel]),
                }
            )
        pd_day = (
            f["pnl_dkk"]
            .groupby([t_r["group"], t_r["date"]])
            .sum()
            .rename("pnl_dkk")
            .reset_index()
        )
        daily.append(pd_day.assign(sample=sample, rule=rule[5:]))

res = pd.DataFrame(results)
for sample in res["sample"].unique():
    out = (
        res[res["sample"] == sample].drop(columns="sample").set_index(["unit", "rule"])
    )
    out.round(3).to_csv(config.TABLES / f"10_quoting_{sample.lower()}.csv")
    show = [
        "fills",
        "fill_rate_pct",
        "avg_half_spread_bps",
        "adverse_selection_bps",
        "realised_bps",
        "pnl_dkk_per_day",
        "pnl_sharpe_daily",
        "max_abs_inventory_dkk",
    ]
    print(
        f"Quoting-rule simulation, {sample} sample (per group):\n",
        out.loc[list(GROUPS), show].round(2).to_string(),
        "\n",
    )
daily = pd.concat(daily, ignore_index=True) if daily else pd.DataFrame()

# %% Figures: fill rate vs realised spread (the trade-off), and daily P&L by rule
if not res.empty:
    samples = [s for s in SAMPLES if s in res["sample"].unique()]
    fig, axes = plt.subplots(
        len(samples), 2, figsize=(10, 3.8 * len(samples)), squeeze=False
    )
    for i, sample in enumerate(samples):
        for j, grp in enumerate(GROUPS):
            ax = axes[i, j]
            r = res[(res["sample"] == sample) & (res["unit"] == grp)]
            ax.scatter(
                r["fill_rate_pct"],
                r["realised_bps"],
                color=GROUP_COLORS[grp],
                s=40,
                zorder=3,
            )
            for _, row in r.iterrows():
                ax.annotate(
                    row["rule"],
                    (row["fill_rate_pct"], row["realised_bps"]),
                    fontsize=7,
                    xytext=(4, 3),
                    textcoords="offset points",
                )
            ax.axhline(0, color="#52514e", linewidth=0.8)
            ax.set_xlabel("Fill rate (% of lit trades)")
            ax.set_ylabel("Realised half-spread per fill (bps)")
            ax.set_title(f"{grp.capitalize()} group, {sample}", loc="left", fontsize=10)
    fig.suptitle(
        "Quoting rules: more fills vs more spread captured net of adverse selection",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(config.FIGURES / "10_quoting_frontier.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(GROUPS), figsize=(10, 3.8), sharey=False)
    for ax, grp in zip(axes, GROUPS):
        d = daily[daily["group"] == grp]
        piv = d.pivot_table(
            index="date", columns="rule", values="pnl_dkk", aggfunc="sum"
        ).cumsum()
        for col in piv:
            ax.plot(piv.index, piv[col] / 1e3, label=col, linewidth=1.4)
        ax.axhline(0, color="#52514e", linewidth=0.8)
        ax.set_title(
            f"{grp.capitalize()} group: cumulative P&L (DKK thousand)",
            loc="left",
            fontsize=10,
        )
        ax.tick_params(axis="x", labelrotation=30)
    axes[-1].legend(
        frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1)
    )
    fig.tight_layout()
    fig.savefig(config.FIGURES / "10_quoting_daily_pnl.png", dpi=150)
    plt.close(fig)
    print("Saved 10_quoting_frontier.png and 10_quoting_daily_pnl.png")
