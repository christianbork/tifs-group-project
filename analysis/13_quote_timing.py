# %% [markdown]
# # 13 Quote timing: do the drivers tell a market maker WHEN to widen?
# Script 10 simulates every forecast as a quoting rule. Its rules quote different average
# widths, and a wider quote mechanically fills fewer, more selective orders and loses less per
# fill, so part of any ranking there is a width effect. This script holds the width fixed:
# 1. Rules = the refined linear forecasts with controls only / + volatility / + impact /
#    + both / + lagged spread, LightGBM, and the benchmarks (in-sample average, yesterday,
#    last bar), all saved by script 08 for every bar (fitted on in-sample days only).
# 2. Each rule is scaled per stock so its IN-SAMPLE average quote equals the in-sample
#    average quoted half-spread. Equal average width -> P&L differences come from timing.
# 3. Fills, spread capture and adverse selection use script 10's engine
#    (src/tifs_group_project/quoting.py): one potential fill per aggressive order, quote on the
#    half-tick grid, filled if the order's first fill paid at least h_t, quote size = the
#    stock's in-sample median order, P&L = h_t - 5-min price impact.
# 4. Out-of-sample (22-30 Oct) and forward week (2-6 Nov); paired stock-day tests of daily P&L
#    vs the constant rule and vs the controls-only rule (the controls-only rule isolates the
#    value of the volatility and impact drivers).
# 5. Stress case: ISS 29-Oct (15:20-15:35 sell-off), how the rules quoted through it.
# History: written in v2 as 10_quote_rule.py with its own fill model; renumbered and moved
# onto script 10's engine when v2 was merged with the forward test (8-Nov), so the project
# has one fill model.

# %% Setup
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tifs_group_project import config
from tifs_group_project.bars import tick_size
from tifs_group_project.impact import decomposition
from tifs_group_project.quoting import day_level_test, fills, orders, paired_daily_test, scale_to_is_width, summarise

RULES = {
    "constant": "bench_is_avg",
    "yesterday": "bench_yday",
    "last bar": "lag_quoted_hs",
    "controls only": "refined: controls only",
    "vol rule": "refined: vol only",
    "impact rule": "refined: impact only",
    "vol + impact rule": "refined: combined",
    "lagged spread + drivers": "refined + lagged spread",
    "LightGBM": "LightGBM",
}
BENCH = ["constant", "yesterday", "last bar"]
GROUPS = {"constrained": config.CONSTRAINED, "unconstrained": config.UNCONSTRAINED}
SAMPLES = ["OOS", "FWD"]
plt.rcParams.update(
    {"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e5e4e0", "grid.linewidth": 0.6}
)

# %% Quotes: scaled forecasts per stock-bar (scale set on in-sample rows only)
fc = pd.read_parquet(config.PROCESSED / "spread_forecasts_5min.parquet")
quotes = fc[["stock", "group", "date", "ts", "sample", "quoted_hs_bps"]].copy()
scale_rows = []
for rule, col in RULES.items():
    quotes[rule] = np.nan
    for s in config.STOCKS:
        m = fc["stock"] == s
        is_m = m & (fc["sample"] == "IS")
        k = 1.0 if rule == "constant" else scale_to_is_width(fc.loc[is_m, col], fc.loc[is_m, "quoted_hs_bps"])
        quotes.loc[m, rule] = (k * fc.loc[m, col]).clip(lower=0.0)
        scale_rows.append({"rule": rule, "stock": s, "scale": k})
scales = pd.DataFrame(scale_rows).pivot(index="rule", columns="stock", values="scale")[list(config.STOCKS)]
scales.round(4).to_csv(config.TABLES / "13_scale_factors.csv")
print("Scale factors (in-sample average quote = in-sample average spread):\n", scales.round(3).to_string(), "\n")

# %% Aggressive orders with the 5-min decomposition (same construction as script 10)
frames = []
for stock in config.STOCKS:
    d = pd.read_parquet(config.PROCESSED / f"signed_{stock}.parquet").sort_values("ts", kind="stable").reset_index(drop=True)
    dec = decomposition(d)
    d["eff_bps"], d["impact_5m_bps"] = dec["eff_bps"], dec["impact_5m_bps"]
    tick_day = d.groupby("date", sort=False).apply(lambda x: tick_size(x["bid"], x["ask"]), include_groups=False)
    d["half_tick_bps"] = d["date"].map(tick_day) / 2 / d["mid"] * 1e4
    d["bar_ts"] = d["ts"].dt.floor("5min")
    d["stock"] = stock
    frames.append(d)
trades = orders(pd.concat(frames, ignore_index=True))
trades = trades[(trades["side"] != 0) & trades["impact_5m_bps"].notna() & (trades["eff_bps"] >= 0)].copy()
trades["group"] = np.where(trades["stock"].isin(config.CONSTRAINED), "constrained", "unconstrained")
quote_size = trades[trades["sample"] == "IS"].groupby("stock")["vol"].median()

tr = (
    trades.merge(quotes[["stock", "ts", *RULES]].rename(columns={"ts": "bar_ts"}), on=["stock", "bar_ts"], how="inner")
    .sort_values("ts", kind="stable")
    .reset_index(drop=True)
)

# %% Simulate: every rule on the same orders (bars where every rule has a quote)
results, tests, daily_pnl, fills_by = [], [], {}, {}
for sample in SAMPLES:
    t_s = tr[(tr["sample"] == sample)].dropna(subset=list(RULES)).reset_index(drop=True)
    if t_s.empty:
        print(f"{sample}: no orders, skipped")
        continue
    size_cap = t_s["stock"].map(quote_size).to_numpy(float)
    for rule in RULES:
        f = fills(t_s, t_s[rule], np.inf, t_s["half_tick_bps"])
        f["size"] = np.where(f["filled"], np.minimum(t_s["vol"].to_numpy(float), size_cap), 0.0)
        f["pnl_dkk"] = np.where(f["filled"], f["size"] * t_s["mid"].to_numpy(float) * f["realised_bps"] / 1e4, 0.0)
        f["signed_size"] = -t_s["side"].to_numpy(float) * f["size"]
        fills_by[(sample, rule)] = f
        daily_pnl[(sample, rule)] = f["pnl_dkk"].groupby([t_s["stock"], t_s["date"]]).sum()
        for unit in ["all", *GROUPS, *config.STOCKS]:
            sel = np.ones(len(t_s), bool) if unit == "all" else ((t_s["group"] == unit) if unit in GROUPS else (t_s["stock"] == unit)).to_numpy()
            results.append(
                {"sample": sample, "unit": unit, "rule": rule, "avg_quote_bps": float(t_s.loc[sel, rule].mean()),
                 **summarise(t_s[sel], f[sel])}
            )
    for unit in ["all", *GROUPS]:
        stocks = list(config.STOCKS) if unit == "all" else GROUPS[unit]
        for rule in RULES:
            for base in ["constant", "controls only"]:
                if rule == base:
                    continue
                a = daily_pnl[(sample, rule)]
                b = daily_pnl[(sample, base)]
                a = a[a.index.get_level_values("stock").isin(stocks)]
                b = b[b.index.get_level_values("stock").isin(stocks)]
                tests.append({"sample": sample, "unit": unit, "rule": rule, "vs": base, **paired_daily_test(a, b)})

res = pd.DataFrame(results)
tst = pd.DataFrame(tests)

# %% Robustness of the timing result (added 18-Nov after review)
# "Equal width" above is equal IN-SAMPLE width. In the test weeks a rule that follows volatility
# quotes wider on average (volatility was higher), fills less, and every rule loses per fill, so
# filling less alone looks like "earning more". Two checks, each rule against controls only:
#   (a) day-level test: sum the P&L difference over stocks per day, t with n_days - 1 df
#       (stock-days on one date are not independent);
#   (b) width-matched: rescale the rule, per stock, so that its average quote over the SAME test
#       sample equals the controls-only average. This uses test-sample information, so it is a
#       diagnostic of what drives the result, not a tradable rule.
def _daily(t_s: pd.DataFrame, h: pd.Series) -> pd.Series:
    cap = t_s["stock"].map(quote_size).to_numpy(float)
    f = fills(t_s, h, np.inf, t_s["half_tick_bps"])
    size = np.where(f["filled"], np.minimum(t_s["vol"].to_numpy(float), cap), 0.0)
    pnl = pd.Series(np.where(f["filled"], size * t_s["mid"].to_numpy(float) * f["realised_bps"] / 1e4, 0.0), index=t_s.index)
    return pnl.groupby([t_s["stock"], t_s["date"]]).sum(), float(100 * f["filled"].mean())


rob = []
for sample in SAMPLES:
    t_s = tr[(tr["sample"] == sample)].dropna(subset=list(RULES)).reset_index(drop=True)
    if t_s.empty:
        continue
    bars_s = quotes[quotes["sample"] == sample]
    base_pnl, base_fill = _daily(t_s, t_s["controls only"])
    for rule in ["vol rule", "impact rule", "vol + impact rule", "lagged spread + drivers", "LightGBM"]:
        a, fill_a = _daily(t_s, t_s[rule])
        k = (bars_s.groupby("stock")["controls only"].mean() / bars_s.groupby("stock")[rule].mean())  # per stock, test sample
        a_m, fill_m = _daily(t_s, t_s[rule] * t_s["stock"].map(k))
        width_ratio = float((bars_s.groupby("stock")[rule].mean() / bars_s.groupby("stock")["controls only"].mean()).mean())
        rob.append({
            "sample": sample, "rule": rule, "vs": "controls only",
            "avg_width_ratio_test_sample": width_ratio, "fill_rate_pct": fill_a, "fill_rate_controls_pct": base_fill,
            "t_stock_day": paired_daily_test(a, base_pnl)["t"], **day_level_test(a, base_pnl),
            "fill_rate_matched_pct": fill_m, "t_stock_day_matched": paired_daily_test(a_m, base_pnl)["t"],
            **{f"{k_}_matched": v for k_, v in day_level_test(a_m, base_pnl).items() if k_ in ("t_day", "p_day", "days_better")},
        })
rob = pd.DataFrame(rob)
rob.round(4).to_csv(config.TABLES / "13_quote_timing_robustness.csv", index=False)
print("Timing result, robustness (vs controls only, all stocks):\n", rob.round(2).to_string(index=False), "\n")
res.round(4).to_csv(config.TABLES / "13_quote_timing.csv", index=False)
tst.round(4).to_csv(config.TABLES / "13_quote_timing_tests.csv", index=False)
show = ["avg_quote_bps", "fill_rate_pct", "avg_half_spread_bps", "adverse_selection_bps", "realised_bps", "pnl_dkk_per_day"]
for sample in res["sample"].unique():
    for unit in ["all", *GROUPS]:
        print(f"{sample}, {unit}, equal in-sample width:\n",
              res[(res["sample"] == sample) & (res["unit"] == unit)].set_index("rule")[show].round(2).to_string(), "\n")
print("Paired stock-day tests of daily DKK P&L (t > 0: rule earns more than 'vs'):\n",
      tst.pivot_table(index=["vs", "rule"], columns=["sample", "unit"], values="t", sort=False).round(2).to_string(), "\n")

# %% Figure: daily P&L per day by rule, by sample and group (equal width)
samples = [s for s in SAMPLES if s in res["sample"].unique()]
fig, axes = plt.subplots(len(samples), 2, figsize=(11, 3.6 * len(samples)), squeeze=False)
for i, sample in enumerate(samples):
    for j, grp in enumerate(GROUPS):
        ax = axes[i, j]
        m = res[(res["sample"] == sample) & (res["unit"] == grp)].set_index("rule").loc[list(RULES)]
        yy = np.arange(len(m))[::-1]
        ax.barh(yy, m["pnl_dkk_per_day"] / 1e3, color=["#c3c2b7" if r in BENCH else "#2a78d6" for r in m.index])
        ax.set_yticks(yy, m.index)
        ax.axvline(0, color="#52514e", linewidth=0.8)
        ax.set_xlabel("P&L per day (DKK thousand)")
        ax.set_title(f"{grp.capitalize()} group, {sample}", loc="left", fontsize=10)
fig.suptitle("Quoting rules at equal in-sample average width (grey = benchmarks)", x=0.01, ha="left", fontsize=11)
fig.tight_layout()
fig.savefig(config.FIGURES / "13_quote_timing.png", dpi=150)
plt.close(fig)

# %% Stress case: ISS 29-Oct
EV_STOCK, EV_DAY = "ISS", pd.Timestamp("2026-10-29")
b1 = pd.read_parquet(config.PROCESSED / "bars_1min.parquet")
b1 = b1[(b1["stock"] == EV_STOCK) & (b1["date"] == EV_DAY)]
b5 = pd.read_parquet(config.PROCESSED / "bars_5min.parquet")
b5 = b5[(b5["stock"] == EV_STOCK) & (b5["date"] == EV_DAY)].set_index("ts")
q_ev = quotes[(quotes["stock"] == EV_STOCK) & (quotes["date"] == EV_DAY)].set_index("ts")
shock = b5["rv_bps"].idxmax()
event_tab = pd.DataFrame(
    {
        "bar": b5.index.strftime("%H:%M"),
        "mid_end": b5["mid_end"],
        "trades": b5["trades"],
        "rv_bps": b5["rv_bps"],
        "market_half_spread_bps": b5["quoted_hs_bps"],
        **{f"quote: {r}": q_ev[r].reindex(b5.index) for r in ["constant", "vol + impact rule", "LightGBM"]},
    }
).loc[shock - pd.Timedelta(minutes=20) : shock + pd.Timedelta(minutes=40)]
event_tab.round(2).to_csv(config.TABLES / "13_iss_event_bars.csv", index=False)
print(f"ISS 29-Oct around the largest 5-min move (bar {shock:%H:%M}):\n", event_tab.round(2).to_string(index=False), "\n")
t_oos = tr[tr["sample"] == "OOS"].dropna(subset=list(RULES)).reset_index(drop=True)
ev_rows = (t_oos["stock"] == EV_STOCK) & (t_oos["date"] == EV_DAY)
ev_pnl = pd.DataFrame(
    [{"rule": r, **{k: v for k, v in summarise(t_oos[ev_rows], fills_by[("OOS", r)][ev_rows]).items()
                    if k in ("fills", "fill_rate_pct", "avg_half_spread_bps", "adverse_selection_bps", "realised_bps", "pnl_dkk_per_day")}}
     for r in RULES]
)
ev_pnl.round(2).to_csv(config.TABLES / "13_iss_event_pnl.csv", index=False)
print("ISS 29-Oct, rule outcomes for the day:\n", ev_pnl.round(2).to_string(index=False), "\n")

win = (EV_DAY + pd.Timedelta("13:30:00"), EV_DAY + pd.Timedelta("16:55:00"))
fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True, gridspec_kw={"height_ratios": [1.2, 1.2, 0.8]})
x1 = b1[(b1["ts"] >= win[0]) & (b1["ts"] <= win[1])]
axes[0].plot(x1["ts"], x1["mid_end"], color="#008300")
axes[0].set_ylabel("Mid (DKK)")
axes[0].set_title("ISS, 29 Oct 2026: intraday sell-off and how quoting rules respond", loc="left", fontsize=11)
x5 = b5.loc[win[0] : win[1]]
axes[1].step(x5.index, x5["quoted_hs_bps"], where="post", color="#52514e", linewidth=2, label="Market half-spread")
for r, c, ls in [("constant", "#8a8984", ":"), ("vol + impact rule", "#2a78d6", "-"), ("LightGBM", "#eb6834", "--")]:
    q = q_ev[r].loc[win[0] : win[1]]
    axes[1].step(q.index, q.values, where="post", color=c, linestyle=ls, label=f"Quote: {r}")
axes[1].set_ylabel("Half-spread (bps)")
axes[1].legend(frameon=False, fontsize=8, ncol=2)
axes[2].bar(x5.index, x5["trades"], width=pd.Timedelta(minutes=4), color="#c3c2b7", align="edge")
axes[2].set_ylabel("Trades / 5 min")
for ax in axes:
    ax.axvline(shock, color="#eb6834", linewidth=0.8, linestyle="--")
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(config.FIGURES / "13_iss_event.png", dpi=150)
plt.close(fig)
