# %% [markdown]
# # 15 Quoting refinement: markout horizon, selective quoting and inventory control
# Script 10 shows that no quoting rule makes money at the 5-minute markout, and script 13 that the
# drivers help mainly through *when* a rule widens. This script refines the strategy itself, using
# the same fill engine as 10 (one fill per aggressive order, first fill's quote, half-tick grid,
# quote size = median in-sample order, revenue = the posted half-spread):
#
# 1. Markout curve. Realised half-spread per fill at 5 s, 30 s, 1, 2 and 5 minutes, for the
#    market (every aggressive order) and for the market maker quoting the last bar's spread.
#    Whether quoting loses depends on how fast inventory is unwound.
# 2. Selective quoting. Pull both quotes for a bar when the HAR volatility forecast (relative to
#    its seasonal level) or the rolling price impact is above its in-sample percentile p. Both
#    inputs are known before the bar starts.
# 3. Inventory control (Ho and Stoll, 1981; Avellaneda and Stoikov, 2008). Skew the quotes against
#    inventory q (in units of the quote size): bid half-spread h + kappa*q half-ticks, ask
#    h - kappa*q, snapped to the grid, floored at half a tick; and stop quoting the side that would
#    take |q| beyond a limit L. Inventory is marked to the mid and flattened at the last mid of the
#    day, so P&L here is mark-to-market (spread earned plus inventory gains), not the 5-min markout.
#
# p, kappa and L are chosen on the in-sample days (1-21 Oct) by a mean-risk criterion on the group's
# daily mark-to-market P&L: mean minus one standard deviation. (The Sharpe ratio misranks rules whose
# mean is negative: cutting risk then makes it MORE negative.) Inventory control (kappa, L) is chosen
# first; p is then chosen with that inventory control in place. The 22-30 Oct and 2-6 Nov weeks are
# scored once with those values.
# Base rule: the last bar's quoted half-spread (the market-matching rule of script 10).
# Outputs: 15_markout_curve.csv, 15_calibration.csv, 15_refined_quoting.csv, 15_refined_tests.csv,
# 15_markout_curve.png, 15_refined_pnl.png.

# %% Setup
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tifs_group_project import config
from tifs_group_project.bars import tick_size
from tifs_group_project.impact import decomposition
from tifs_group_project.quoting import day_level_test, orders

GROUPS = {"constrained": config.CONSTRAINED, "unconstrained": config.UNCONSTRAINED}
GROUP_COLORS = {"constrained": "#2a78d6", "unconstrained": "#eb6834"}
HORIZONS_MIN = (5 / 60, 0.5, 1, 2, 5)
HORIZON_LABEL = {5 / 60: "5s", 0.5: "30s", 1: "1m", 2: "2m", 5: "5m"}
P_GRID = [None, 0.95, 0.90, 0.80]
KAPPA_GRID = [0.0, 0.1, 0.25, 0.5, 1.0]
LIMIT_GRID = [np.inf, 20.0, 10.0, 5.0, 3.0, 2.0, 1.0]
plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e5e4e0",
        "grid.linewidth": 0.6,
    }
)


def hcol(h):
    return f"impact_{HORIZON_LABEL[h]}_bps"


# %% Orders with the markout at every horizon, keyed to their bar
frames = []
for stock in config.STOCKS:
    d = (
        pd.read_parquet(config.PROCESSED / f"signed_{stock}.parquet")
        .sort_values("ts", kind="stable")
        .reset_index(drop=True)
    )
    dec = decomposition(d, horizons_min=HORIZONS_MIN)
    d["eff_bps"] = dec["eff_bps"]
    for h in HORIZONS_MIN:
        d[hcol(h)] = dec[f"impact_{h}m_bps"]
    tick_day = d.groupby("date", sort=False).apply(
        lambda x: tick_size(x["bid"], x["ask"]), include_groups=False
    )
    d["half_tick_bps"] = d["date"].map(tick_day) / 2 / d["mid"] * 1e4
    d["bar_ts"] = d["ts"].dt.floor("5min")
    frames.append(d)
tr = orders(pd.concat(frames, ignore_index=True))
tr = tr[(tr["side"] != 0) & tr[hcol(5)].notna() & (tr["eff_bps"] >= 0)].copy()
tr["group"] = np.where(
    tr["stock"].isin(config.CONSTRAINED), "constrained", "unconstrained"
)
quote_size = tr[tr["sample"] == "IS"].groupby("stock")["vol"].median()

# bar-level inputs, all known before the bar starts
bars = pd.read_parquet(config.PROCESSED / "bars_5min.parquet").sort_values(
    ["stock", "ts"]
)
bars["h_last"] = bars.groupby("stock")["quoted_hs_bps"].shift(1)
vol = pd.read_parquet(config.PROCESSED / "vol_forecast_5min.parquet")[
    ["stock", "ts", "vol_fc_har_ratio"]
]
imp = pd.read_parquet(config.PROCESSED / "impact_intraday_5min.parquet")[
    ["stock", "ts", "impact_roll_ewm_bps"]
]
feat = (
    bars[["stock", "ts", "h_last"]]
    .merge(vol, on=["stock", "ts"], how="left")
    .merge(imp, on=["stock", "ts"], how="left")
)
tr = tr.merge(
    feat.rename(columns={"ts": "bar_ts"}), on=["stock", "bar_ts"], how="inner"
)
tr = (
    tr[tr["h_last"].notna()]
    .sort_values(["stock", "date", "ts"], kind="stable")
    .reset_index(drop=True)
)

# in-sample percentiles of the two toxicity signals, per stock (bar level, IS bars only)
is_feat = feat.merge(bars[["stock", "ts", "sample"]], on=["stock", "ts"])
is_feat = is_feat[is_feat["sample"] == "IS"]
print("orders:", tr.groupby("sample").size().to_dict())

# %% 1. Markout curve: the market vs the market maker quoting the last bar's spread
rows = []
for sample in ["IS", "OOS", "FWD"]:
    for grp, stocks in GROUPS.items():
        t = tr[(tr["sample"] == sample) & tr["stock"].isin(stocks)]
        ht = t["half_tick_bps"].to_numpy()
        h = np.maximum(1, np.round(t["h_last"].to_numpy() / ht)) * ht
        filled = h <= t["eff_bps"].to_numpy()
        w_all = np.minimum(
            t["vol"].to_numpy(float), t["stock"].map(quote_size).to_numpy(float)
        )
        for hz in HORIZONS_MIN:
            imp_h = t[hcol(hz)].to_numpy()
            ok = np.isfinite(imp_h)
            mkt = np.average(t["eff_bps"].to_numpy()[ok] - imp_h[ok], weights=w_all[ok])
            sel = ok & filled
            mm = np.average(h[sel] - imp_h[sel], weights=w_all[sel])
            rows.append(
                {
                    "sample": sample,
                    "group": grp,
                    "horizon": HORIZON_LABEL[hz],
                    "market_realised_bps": mkt,
                    "mm_last_bar_realised_bps": mm,
                    "mm_fill_rate_pct": 100 * filled.mean(),
                    "market_eff_bps": np.average(
                        t["eff_bps"].to_numpy()[ok], weights=w_all[ok]
                    ),
                    "mm_eff_paid_by_filled_bps": np.average(
                        t["eff_bps"].to_numpy()[sel], weights=w_all[sel]
                    ),
                    "mm_quoted_bps": np.average(h[sel], weights=w_all[sel]),
                }
            )
markout = pd.DataFrame(rows)
markout.round(3).to_csv(config.TABLES / "15_markout_curve.csv", index=False)
print(markout[markout["sample"] != "IS"].round(2).to_string(index=False), "\n")


# %% 2-3. Sequential simulator with selective quoting and inventory control
def simulate(
    t: pd.DataFrame, pull: np.ndarray, kappa: float, limit: float
) -> pd.DataFrame:
    """Mark-to-market daily P&L for one stock's orders (sorted by date, ts). Returns per-day rows."""
    side = t["side"].to_numpy(int)  # +1 = aggressive buy, hits the market maker's ask
    eff = t["eff_bps"].to_numpy(float)
    mid = t["mid"].to_numpy(float)
    ht = t["half_tick_bps"].to_numpy(float)
    h0 = t["h_last"].to_numpy(float)
    vol_ = t["vol"].to_numpy(float)
    imp5 = t[hcol(5)].to_numpy(float)
    date = t["date"].to_numpy()
    qs = float(quote_size[t["stock"].iloc[0]])
    out = []
    n = len(t)
    i = 0
    while i < n:
        j = i
        while j < n and date[j] == date[i]:
            j += 1
        inv = cash = 0.0
        fills = rev_dkk = mk5_dkk = notional = 0.0
        max_inv = 0.0
        for k in range(i, j):
            if pull[k]:
                continue
            q = inv / qs
            s = side[k]
            if s > 0:  # market maker sells at the ask
                if q <= -limit:
                    continue
                x = h0[k] - kappa * q * ht[k]
            else:  # market maker buys at the bid
                if q >= limit:
                    continue
                x = h0[k] + kappa * q * ht[k]
            hk = max(1.0, round(x / ht[k])) * ht[k]
            if hk > eff[k]:
                continue
            size = min(vol_[k], qs)
            px = mid[k] * (1 + s * hk / 1e4)
            inv -= s * size
            cash += s * size * px
            fills += 1
            notional += size * mid[k]
            rev_dkk += size * mid[k] * hk / 1e4
            mk5_dkk += size * mid[k] * (hk - imp5[k]) / 1e4
            max_inv = max(max_inv, abs(inv * mid[k]))
        pnl = cash + inv * mid[j - 1]  # flatten at the last mid of the day
        out.append(
            {
                "date": date[i],
                "orders": j - i,
                "fills": fills,
                "notional_dkk": notional,
                "spread_dkk": rev_dkk,
                "markout5_dkk": mk5_dkk,
                "mtm_pnl_dkk": pnl,
                "max_abs_inv_dkk": max_inv,
                "end_inv_dkk": abs(inv * mid[j - 1]),
            }
        )
        i = j
    return pd.DataFrame(out)


def pull_mask(t: pd.DataFrame, p) -> np.ndarray:
    if p is None:
        return np.zeros(len(t), bool)
    thr_v = is_feat.groupby("stock")["vol_fc_har_ratio"].quantile(p)
    thr_i = is_feat.groupby("stock")["impact_roll_ewm_bps"].quantile(p)
    v = t["vol_fc_har_ratio"] > t["stock"].map(thr_v)
    m = t["impact_roll_ewm_bps"] > t["stock"].map(thr_i)
    return (v | m).to_numpy()


def run(sample: str, p, kappa: float, limit: float) -> pd.DataFrame:
    res = []
    t_s = tr[tr["sample"] == sample]
    for stock, t in t_s.groupby("stock", sort=False):
        d = simulate(t, pull_mask(t, p), kappa, limit)
        res.append(d.assign(stock=stock))
    out = pd.concat(res, ignore_index=True)
    out["group"] = np.where(
        out["stock"].isin(config.CONSTRAINED), "constrained", "unconstrained"
    )
    return out


def score(daily: pd.DataFrame) -> pd.DataFrame:
    """Per group: daily P&L summed over the group's stocks, then summary statistics."""
    rows = []
    for grp, d in daily.groupby("group"):
        by_day = d.groupby("date")[
            [
                "mtm_pnl_dkk",
                "markout5_dkk",
                "spread_dkk",
                "fills",
                "orders",
                "notional_dkk",
            ]
        ].sum()
        sd = by_day["mtm_pnl_dkk"].std(ddof=1)
        rows.append(
            {
                "group": grp,
                "fill_rate_pct": 100 * by_day["fills"].sum() / by_day["orders"].sum(),
                "notional_mdkk_per_day": by_day["notional_dkk"].mean() / 1e6,
                "markout5_bps": 1e4
                * by_day["markout5_dkk"].sum()
                / by_day["notional_dkk"].sum(),
                "mtm_bps": 1e4
                * by_day["mtm_pnl_dkk"].sum()
                / by_day["notional_dkk"].sum(),
                "mtm_pnl_dkk_per_day": by_day["mtm_pnl_dkk"].mean(),
                "mtm_pnl_sd_day": sd,
                "sharpe_daily": by_day["mtm_pnl_dkk"].mean() / sd if sd > 0 else np.nan,
                "pct_days_positive": 100 * (by_day["mtm_pnl_dkk"] > 0).mean(),
                "max_abs_inv_mdkk": d["max_abs_inv_dkk"].max() / 1e6,
                "avg_end_inv_mdkk": d["end_inv_dkk"].mean() / 1e6,
                "days": len(by_day),
            }
        )
    return pd.DataFrame(rows)


# %% Calibration on the in-sample days only
def objective(row) -> float:
    return row["mtm_pnl_dkk_per_day"] - row["mtm_pnl_sd_day"]


cal = []
for kappa, limit in itertools.product(KAPPA_GRID, LIMIT_GRID):
    cal.append(
        score(run("IS", None, kappa, limit)).assign(
            p=None, kappa=kappa, limit=limit, stage="inventory"
        )
    )
cal = pd.concat(cal, ignore_index=True)
cal["objective"] = cal.apply(objective, axis=1)
best = {}
for grp in GROUPS:
    c = cal[(cal["group"] == grp) & (cal["stage"] == "inventory")]
    inv = c.sort_values("objective", ascending=False).iloc[0]
    best[grp] = {"kappa": float(inv["kappa"]), "limit": float(inv["limit"])}
stage2 = []
for p in P_GRID:
    for grp, b in best.items():
        sc = score(run("IS", p, b["kappa"], b["limit"]))
        stage2.append(
            sc[sc["group"] == grp].assign(
                p=p, kappa=b["kappa"], limit=b["limit"], stage="selective + inventory"
            )
        )
    sc = score(run("IS", p, 0.0, np.inf))
    stage2.append(sc.assign(p=p, kappa=0.0, limit=np.inf, stage="selective"))
stage2 = pd.concat(stage2, ignore_index=True)
stage2["objective"] = stage2.apply(objective, axis=1)
for grp in GROUPS:
    for stage, key in [("selective + inventory", "p"), ("selective", "p_alone")]:
        c = stage2[(stage2["group"] == grp) & (stage2["stage"] == stage)].copy()
        c["p_key"] = c["p"].fillna(1.0)
        best[grp][key] = c.sort_values(
            ["objective", "p_key"], ascending=[False, False]
        ).iloc[0]["p"]
        best[grp][key] = None if pd.isna(best[grp][key]) else float(best[grp][key])
print("Chosen in-sample:", best)
cal = pd.concat([cal, stage2], ignore_index=True)
cal.round(4).to_csv(config.TABLES / "15_calibration.csv", index=False)

# %% Score the test weeks once with the in-sample choices
RULES = {
    "last bar (base)": lambda b: (None, 0.0, np.inf),
    "selective": lambda b: (b["p_alone"], 0.0, np.inf),
    "inventory control": lambda b: (None, b["kappa"], b["limit"]),
    "selective + inventory": lambda b: (b["p"], b["kappa"], b["limit"]),
}
summary, tests, daily_all = [], [], []
for sample in ["IS", "OOS", "FWD"]:
    daily_by_rule = {}
    for rule, params in RULES.items():
        parts = []
        for grp, b in best.items():
            p, kappa, limit = params(b)
            d = run(sample, p, kappa, limit)
            parts.append(d[d["group"] == grp])
        d = pd.concat(parts, ignore_index=True)
        daily_by_rule[rule] = d
        daily_all.append(d.assign(sample=sample, rule=rule))
        summary.append(score(d).assign(sample=sample, rule=rule))
    base = daily_by_rule["last bar (base)"]
    for rule in list(RULES)[1:]:
        d = daily_by_rule[rule]
        for grp in [*GROUPS, "all"]:
            a = d if grp == "all" else d[d["group"] == grp]
            b0 = base if grp == "all" else base[base["group"] == grp]
            for metric in ["mtm_pnl_dkk", "markout5_dkk"]:
                ta = a.set_index(["stock", "date"])[metric]
                tb = b0.set_index(["stock", "date"])[metric]
                tests.append(
                    {
                        "sample": sample,
                        "rule": rule,
                        "vs": "last bar (base)",
                        "group": grp,
                        "metric": metric,
                        **day_level_test(ta, tb),
                    }
                )
summary = pd.concat(summary, ignore_index=True)
cols = ["sample", "group", "rule"] + [
    c for c in summary.columns if c not in ("sample", "group", "rule")
]
summary = summary[cols].sort_values(["sample", "group"], kind="stable")
summary.round(3).to_csv(config.TABLES / "15_refined_quoting.csv", index=False)
tests = pd.DataFrame(tests)
tests.round(4).to_csv(config.TABLES / "15_refined_tests.csv", index=False)
print(summary[summary["sample"] != "IS"].round(2).to_string(index=False))
print(tests[(tests["sample"] != "IS")].round(3).to_string(index=False))
daily_all = pd.concat(daily_all, ignore_index=True)

# %% Figures
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=False)
for ax, grp in zip(axes, GROUPS):
    for sample, ls in [("OOS", "-"), ("FWD", "--")]:
        m = markout[(markout["sample"] == sample) & (markout["group"] == grp)]
        ax.plot(
            m["horizon"],
            m["market_realised_bps"],
            ls,
            color="#52514e",
            marker="o",
            ms=3,
            label=f"market, {'22-30 Oct' if sample == 'OOS' else '2-6 Nov'}",
        )
        ax.plot(
            m["horizon"],
            m["mm_last_bar_realised_bps"],
            ls,
            color=GROUP_COLORS[grp],
            marker="o",
            ms=3,
            label=f"market maker (last-bar quote), {'22-30 Oct' if sample == 'OOS' else '2-6 Nov'}",
        )
    ax.axhline(0, color="#52514e", linewidth=0.8)
    ax.set_title(f"{grp.capitalize()} group", loc="left", fontsize=10)
    ax.set_xlabel("Markout horizon")
    ax.set_ylabel("Realised half-spread per fill (bps)")
axes[1].legend(frameon=False, fontsize=7, loc="lower left")
fig.tight_layout()
fig.savefig(config.FIGURES / "15_markout_curve.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(10, 6.2))
for i, sample in enumerate(["OOS", "FWD"]):
    for j, grp in enumerate(GROUPS):
        ax = axes[i, j]
        d = daily_all[(daily_all["sample"] == sample) & (daily_all["group"] == grp)]
        piv = d.pivot_table(
            index="date", columns="rule", values="mtm_pnl_dkk", aggfunc="sum"
        ).cumsum()
        for col in RULES:
            ax.plot(range(len(piv)), piv[col] / 1e3, marker="o", ms=3, label=col)
        ax.set_xticks(
            range(len(piv)),
            [pd.Timestamp(x).strftime("%d-%b") for x in piv.index],
            fontsize=7,
        )
        ax.axhline(0, color="#52514e", linewidth=0.8)
        ax.set_title(
            f"{grp.capitalize()}, {'22-30 Oct' if sample == 'OOS' else '2-6 Nov'}: cumulative P&L (DKK k)",
            loc="left",
            fontsize=9,
        )
axes[0, 1].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(config.FIGURES / "15_refined_pnl.png", dpi=150)
plt.close(fig)
print("Saved 15_markout_curve.png and 15_refined_pnl.png")
