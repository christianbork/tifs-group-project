# %% [markdown]
# # 10 Quote-widening rule: does the risk adjustment pay a market maker?
# Closes the loop with the Week 6 market-making diagram: the spread forecasts of script 08
# become quoting rules, scored by a stylised market maker's markout P&L out of sample.
# 1. Rules (quoted half-spread h_t per stock-bar, set before the bar):
#    constant (in-sample average), yesterday's spread, last bar's spread, and the refined
#    linear forecasts with controls only / + vol / + impact / + both, refined + lagged
#    spread, and LightGBM.
# 2. Every rule is scaled per stock so its IN-SAMPLE average quote equals the in-sample
#    average spread. Equal average width -> P&L differences come from WHEN a rule widens.
# 3. Fill: one fill per aggressive ORDER (fills sharing a timestamp, as in 02/05): if the
#    order reached at least h_t from its pre-trade mid (its deepest fill), the market maker's
#    single quote is hit once, at its own price, for PARTICIPATION x the order's size. Markout
#    at the mid 5 min after the order (1 min as robustness), from the pre-trade mid:
#    P&L = h_t - price impact (bps). See src/tifs_group_project/quote_rule.py for assumptions.
# 4. Out-of-sample days only (22-30 Oct); paired stock-day tests vs the constant rule and vs
#    the controls-only rule (isolates the value of the volatility / impact drivers).
# 5. Stress case: ISS 29-Oct (15:20-15:35 sell-off) - how each rule quoted through it.

# %% Setup
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tifs_group_project import config
from tifs_group_project.impact import decomposition
from tifs_group_project.quote_rule import fills, paired_daily_test, scale_to_is_width, summarise

PARTICIPATION = 0.10  # share of each fill the market maker takes (scales DKK P&L only)
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
GROUPS = {"constrained": config.CONSTRAINED, "unconstrained": config.UNCONSTRAINED}
plt.rcParams.update(
    {"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e5e4e0", "grid.linewidth": 0.6}
)

# %% Quotes: scaled forecasts per stock-bar
fc = pd.read_parquet(config.PROCESSED / "spread_forecasts_5min.parquet").rename(columns={"ts": "bar_ts"})
quotes = fc[["stock", "group", "date", "bar_ts", "sample", "quoted_hs_bps"]].copy()
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
print("Scale factors (in-sample average quote = in-sample average spread):\n", scales.round(3).to_string(), "\n")
oosq = quotes[quotes["sample"] == "OOS"].dropna(subset=list(RULES))  # common OOS bars for all rules
print(f"OOS stock-bars with a quote from every rule: {len(oosq):,} of {(quotes['sample'] == 'OOS').sum():,}\n")

# %% Trades with their distance from the mid and markouts
tr = []
for s in config.STOCKS:
    d = pd.read_parquet(config.PROCESSED / f"signed_{s}.parquet")
    d = d[d["date"] > config.IS_END].sort_values("ts", kind="stable").reset_index(drop=True)
    dec = decomposition(d, horizons_min=(1, 5))
    t = pd.DataFrame(
        {
            "order_id": d["order_id"],
            "fill_pos": d["fill_pos"],
            "date": d["date"],
            "ts": d["ts"],
            "dkk": d["price"] * d["vol"],
            "vol": d["vol"],
            "eff_bps": dec["eff_bps"],
            "impact_5m": dec["impact_5m_bps"],
            "impact_1m": dec["impact_1m_bps"],
        }
    )
    # one row per aggressive order: deepest level reached, total size, markout from the
    # first fill (its attached quote is the pre-trade quote)
    first = t[t["fill_pos"] == 0].set_index("order_id")
    o = t.groupby("order_id").agg(eff_bps=("eff_bps", "max"), vol=("vol", "sum"), dkk=("dkk", "sum"))
    o = o.join(first[["date", "ts", "impact_5m", "impact_1m"]]).reset_index()
    o["price"] = o["dkk"] / o["vol"]
    o["stock"] = s
    o["bar_ts"] = o["ts"].dt.floor("5min")
    tr.append(o)
tr = pd.concat(tr, ignore_index=True)
print(f"OOS aggressive orders: {len(tr):,}\n")

# %% Simulate every rule (5-min markout main, 1-min robustness)
keys = pd.MultiIndex.from_frame(oosq[["stock", "date"]].drop_duplicates().sort_values(["stock", "date"]))
results, tests, fills_main = [], [], {}
for horizon in ["impact_5m", "impact_1m"]:
    t = tr.rename(columns={horizon: "impact_bps"})
    f_all = {rule: fills(t, oosq, rule, PARTICIPATION) for rule in RULES}
    if horizon == "impact_5m":
        fills_main = f_all
    for unit in ["all", *GROUPS, *config.STOCKS]:
        stocks = list(config.STOCKS) if unit == "all" else GROUPS.get(unit, [unit])
        q_u = oosq[oosq["stock"].isin(stocks)]
        k_u = keys[keys.get_level_values("stock").isin(stocks)]
        for rule in RULES:
            f_u = f_all[rule][f_all[rule]["stock"].isin(stocks)]
            results.append({"markout": horizon[-2:], "unit": unit, **summarise(f_u, q_u, rule)})
            for base in ["constant", "controls only"]:
                if rule != base:
                    b_u = f_all[base][f_all[base]["stock"].isin(stocks)]
                    tests.append({"markout": horizon[-2:], "unit": unit, "rule": rule, "vs": base, **paired_daily_test(f_u, b_u, k_u)})
res = pd.DataFrame(results)
tst = pd.DataFrame(tests)
res.round(4).to_csv(config.TABLES / "10_quote_rule_pnl.csv", index=False)
tst.round(4).to_csv(config.TABLES / "10_quote_rule_tests.csv", index=False)

main = res[res["markout"] == "5m"]
show = ["avg_quote_bps", "fills", "notional_dkk_m", "spread_capture_bps", "adverse_sel_bps", "markout_pnl_bps", "pnl_dkk_per_stock_day"]
for unit in ["all", *GROUPS]:
    print(f"OOS quoting rules, 5-min markout, {unit}:\n", main[main["unit"] == unit].set_index("rule")[show].round(2).to_string(), "\n")
print("Markout P&L (bps per DKK filled) by stock, 5-min markout:\n",
      main[main["unit"].isin(config.STOCKS)].pivot(index="rule", columns="unit", values="markout_pnl_bps")[list(config.STOCKS)].round(2).to_string(), "\n")
tv = tst[(tst["markout"] == "5m") & tst["unit"].isin(["all", *GROUPS])]
print("Paired stock-day tests of daily DKK P&L (t > 0: rule earns more than 'vs'):\n",
      tv.pivot_table(index=["vs", "rule"], columns="unit", values="t", sort=False).round(2).to_string(), "\n")
print("1-min markout robustness, markout P&L (bps):\n",
      res[(res["markout"] == "1m") & res["unit"].isin(["all", *GROUPS])].pivot(index="rule", columns="unit", values="markout_pnl_bps").round(2).to_string(), "\n")

# %% Figure: average quote vs markout P&L per rule, by group
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, grp in zip(axes, GROUPS):
    m = main[main["unit"] == grp].set_index("rule")
    ax.barh(np.arange(len(m))[::-1], m["markout_pnl_bps"], color=["#c3c2b7" if r in ("constant", "yesterday", "last bar") else "#2a78d6" for r in m.index])
    for k, (r, v) in enumerate(m["markout_pnl_bps"].items()):
        ax.text(v, len(m) - 1 - k, f" {v:.2f}", va="center", fontsize=8)
    ax.axvline(0, color="#52514e", linewidth=0.8)
    ax.set_yticks(np.arange(len(m))[::-1], m.index)
    ax.set_xlabel("Markout P&L per DKK filled (bps, 5-min markout)")
    ax.set_title(f"{grp.capitalize()} group", loc="left", fontsize=10)
fig.suptitle("Quoting rules out of sample at equal average width (grey = benchmarks)", x=0.01, ha="left", fontsize=11)
fig.tight_layout()
fig.savefig(config.FIGURES / "10_quote_rule_pnl.png", dpi=150)
plt.close(fig)

# %% Stress case: ISS 29-Oct
EV_STOCK, EV_DAY = "ISS", pd.Timestamp("2026-10-29")
b1 = pd.read_parquet(config.PROCESSED / "bars_1min.parquet")
b1 = b1[(b1["stock"] == EV_STOCK) & (b1["date"] == EV_DAY)]
q_ev = quotes[(quotes["stock"] == EV_STOCK) & (quotes["date"] == EV_DAY)].set_index("bar_ts")
b5 = pd.read_parquet(config.PROCESSED / "bars_5min.parquet")
b5 = b5[(b5["stock"] == EV_STOCK) & (b5["date"] == EV_DAY)].set_index("ts")
win = (EV_DAY + pd.Timedelta("13:30:00"), EV_DAY + pd.Timedelta("16:55:00"))
shock = b5["rv_bps"].idxmax()
event_tab = pd.DataFrame(
    {
        "bar": b5.index.strftime("%H:%M"),
        "mid_end": b5["mid_end"],
        "trades": b5["trades"],
        "rv_bps": b5["rv_bps"],
        "market_half_spread_bps": b5["quoted_hs_bps"],
        **{f"quote: {r}": q_ev[r].reindex(b5.index) for r in ["constant", "vol + impact rule", "lagged spread + drivers", "LightGBM"]},
    }
).loc[shock - pd.Timedelta(minutes=20) : shock + pd.Timedelta(minutes=40)]
event_tab.round(2).to_csv(config.TABLES / "10_iss_event_bars.csv", index=False)
print(f"ISS 29-Oct around the largest 5-min move (bar {shock:%H:%M}):\n", event_tab.round(2).to_string(index=False), "\n")
ev_pnl = pd.DataFrame(
    [{"rule": r, **{k: v for k, v in summarise(f[(f["stock"] == EV_STOCK) & (f["date"] == EV_DAY)], q_ev.reset_index(), r).items()
                    if k in ("fills", "spread_capture_bps", "adverse_sel_bps", "markout_pnl_bps", "pnl_dkk")}}
     for r, f in fills_main.items()]
)
ev_pnl.round(2).to_csv(config.TABLES / "10_iss_event_pnl.csv", index=False)
print("ISS 29-Oct, rule P&L for the day (5-min markout):\n", ev_pnl.round(2).to_string(index=False), "\n")

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
fig.savefig(config.FIGURES / "10_iss_event.png", dpi=150)
plt.close(fig)
