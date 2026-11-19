"""Build the Word version of the report from report/REPORT_DRAFT.md.

Figures come from outputs/figures, tables are rebuilt from outputs/tables so the
document never carries a number that is not in a table. Run from the repo root:

    .venv/bin/python report/build_docx.py          -> report/Group_Report.docx
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
T, F = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"
import sys

_APPX = len(sys.argv) > 1 and sys.argv[1] == "appendices"
SRC = ROOT / "report" / ("APPENDICES.md" if _APPX else "REPORT_DRAFT.md")
OUT = ROOT / "report" / ("Group_Report_Appendices.docx" if _APPX else "Group_Report.docx")

BODY_FONT, BODY_PT = "Calibri", 10.5
GREY = RGBColor(0x55, 0x55, 0x55)


# ----------------------------------------------------------------------------- tables
def table1() -> tuple[pd.DataFrame, str]:
    d = pd.read_csv(T / "report_table1_data_summary.csv")
    d["Group"] = d["Group"].str.replace("constrained", "C").str.replace("unC", "U")
    d = d.rename(columns={"Group": "Grp", "Lit trades per day": "Lit trades /day", "Turnover (DKK m/day)": "Turnover DKK m/day",
                          "Signed by quote rule (%)": "Quote rule (%)"})
    return d, ("Table 1. Continuous-session lit trades, 1 to 30 October 2026 (22 days). C = tick-constrained group, "
               "U = unconstrained. Source: report_table1_data_summary.csv.")


def table2() -> tuple[pd.DataFrame, str]:
    c = pd.read_csv(T / "08_r1_coefficients.csv")
    c = c[c["spec"].isin(["original", "refined"])]
    names = {"impact_roll_ewm_bps": "Impact, rolling (last 300 orders)", "impact_prevday_bps": "Impact, previous day",
             "log_vol_har": "Volatility forecast, log(HAR / seasonal)"}
    rows = []
    for grp in ["constrained", "unconstrained"]:
        for v, lab in names.items():
            r = {"Group": grp, "Driver": lab}
            for spec in ["original", "refined"]:
                x = c[(c.group == grp) & (c.spec == spec) & (c["var"] == v)].iloc[0]
                star = "***" if x.p < 0.01 else "**" if x.p < 0.05 else "*" if x.p < 0.1 else ""
                r[f"{spec}: coef [t]"] = f"{x.coef:.3f}{star} [{x.t:.1f}]"
                r[f"{spec}: +1 SD (bps)"] = f"{x.effect_1sd:.3f}"
            rows.append(r)
        r = {"Group": grp, "Driver": "N / R²"}
        for spec in ["original", "refined"]:
            x = c[(c.group == grp) & (c.spec == spec)].iloc[0]
            r[f"{spec}: coef [t]"] = f"{int(x.n)} / {x.r2:.2f}"
            r[f"{spec}: +1 SD (bps)"] = ""
        rows.append(r)
    d = pd.DataFrame(rows)
    # robustness of the original spec: lowest |t| over six SE choices (07_se_robustness) and the dynamic spec
    se = pd.read_csv(T / "07_se_robustness.csv")
    min_t = se.assign(a=se["t"].abs()).groupby(["group", "var"])["a"].min()
    dyn = pd.read_csv(T / "07_dynamic.csv").query("spec == 'dynamic, Driscoll-Kraay L=24'").set_index(["group", "var"])["t"]
    inv = {lab: v for v, lab in names.items()}
    grp_col = d["Group"].copy()
    d["min |t|, 6 SE choices"] = [f"{min_t[(g, inv[x])]:.1f}" if x in inv else "" for g, x in zip(grp_col, d["Driver"])]
    d["dynamic spec t (DK)"] = [f"{dyn[(g, inv[x])]:.1f}" if x in inv else "" for g, x in zip(grp_col, d["Driver"])]
    d.loc[d["Group"].duplicated(), "Group"] = ""
    return d, ("Table 3. In-sample (1 to 21 Oct) regressions of the 5-minute quoted half-spread (bps) by group. "
               "Cells: coefficient [Newey-West t, 7 lags] and the effect of a one within-stock SD increase in bps. "
               "Original = shared time-of-day dummies and stock fixed effects; refined = stock-by-time-bucket dummies (Section 6.1). "
               "All specifications include lagged trades, turnover and |order imbalance| (lagged within the day). "
               "min |t| = lowest |t| of the original specification across Newey-West with 7, 24 and 95 lags, Driscoll-Kraay, and "
               "stock-day and day clusters; dynamic spec = original plus the last observed spread, Driscoll-Kraay t. "
               "*** p<0.01, ** p<0.05, * p<0.1. Source: 08_r1_coefficients.csv, 07_se_robustness.csv, 07_dynamic.csv.")


RACE = ["IS average", "yesterday", "last bar", "original: combined", "refined: combined", "refined + lagged spread", "LightGBM"]
RACE_LABELS = {"original: combined": "linear (original)", "refined: combined": "linear (refined)",
               "refined + lagged spread": "linear + lagged spread"}


def race_table(rmse_file: str, dm_file: str, subset: str, title: str) -> tuple[pd.DataFrame, str]:
    r = pd.read_csv(T / rmse_file).query("subset == @subset").set_index("unit")
    dm = pd.read_csv(T / dm_file).query("subset == @subset")
    units = ["constrained", "unconstrained", "Novo", "DSV", "ISS", "Vestas", "Pandora", "GN"]
    d = r.loc[units, RACE].round(2).rename(columns=RACE_LABELS)
    for model, lab in [("refined: combined", "DM linear (refined) vs last bar"), ("LightGBM", "DM LightGBM vs last bar")]:
        x = dm[(dm.model == model) & (dm.vs == "last bar")].set_index("unit")["dm_stat"]
        d[lab] = x.reindex(units).round(1)
    d.index.name = "Unit"
    return d.reset_index(), title


def race_combined() -> tuple[pd.DataFrame, str]:
    """One table for the OOS window and the forward week, group rows only (per-stock rows are in the CSVs)."""
    oos = pd.read_csv(T / "08_oos_rmse.csv").query("subset == 'all OOS days'").set_index("unit")
    fwd = pd.read_csv(T / "09_fwd_rmse.csv").query("subset == 'all FWD days'").set_index("unit")
    dmo = pd.read_csv(T / "08_oos_diebold_mariano.csv").query("subset == 'all OOS days' and vs == 'last bar'")
    dmf = pd.read_csv(T / "09_fwd_diebold_mariano.csv").query("subset == 'all FWD days' and vs == 'last bar'")
    rows = []
    for rule in RACE:
        r = {"Rule": RACE_LABELS.get(rule, rule)}
        for grp, g in [("constrained", "C"), ("unconstrained", "U")]:
            r[f"OOS RMSE {g}"] = f"{oos.loc[grp, rule]:.2f}"
            r[f"FWD RMSE {g}"] = f"{fwd.loc[grp, rule]:.2f}"
            do = dmo[(dmo.unit == grp) & (dmo.model == rule)]["dm_stat"]
            df_ = dmf[(dmf.unit == grp) & (dmf.model == rule)]["dm_stat"]
            r[f"DM vs last bar {g} (OOS / FWD)"] = (f"{do.iloc[0]:.1f}" if len(do) else "") + " / " + (f"{df_.iloc[0]:.1f}" if len(df_) else "") if (len(do) or len(df_)) else ""
        rows.append(r)
    d = pd.DataFrame(rows)[["Rule", "OOS RMSE C", "FWD RMSE C", "DM vs last bar C (OOS / FWD)", "OOS RMSE U", "FWD RMSE U", "DM vs last bar U (OOS / FWD)"]]
    return d, ("Table 4. Forecast race for the 5-minute quoted half-spread (RMSE, bps): out-of-sample window 22 to 30 October "
               "(used three times) and forward week 2 to 6 November (never used for fitting or tuning; all parameters frozen on 1 to 21 October). "
               "C = constrained group, U = unconstrained. DM = Diebold-Mariano statistic against the last bar's spread, "
               "negative favours the model. Per-stock rows: 08_oos_rmse.csv, 09_fwd_rmse.csv.")


def table3() -> tuple[pd.DataFrame, str]:
    return race_table("08_oos_rmse.csv", "08_oos_diebold_mariano.csv", "all OOS days",
                      "Table 3. Out-of-sample RMSE of the quoted half-spread (bps per 5-minute bar), 22 to 30 October. "
                      "Benchmarks: in-sample average, yesterday's average, last bar's spread. DM = Diebold-Mariano statistic, "
                      "negative favours the model. Source: 08_oos_rmse.csv, 08_oos_diebold_mariano.csv.")


def table4() -> tuple[pd.DataFrame, str]:
    return race_table("09_fwd_rmse.csv", "09_fwd_diebold_mariano.csv", "all FWD days",
                      "Table 4. Forward test, 2 to 6 November: RMSE of the quoted half-spread (bps per 5-minute bar) "
                      "with every parameter frozen on 1 to 21 October. Same layout as Table 3. "
                      "Source: 09_fwd_rmse.csv, 09_fwd_diebold_mariano.csv.")


def table5() -> tuple[pd.DataFrame, str]:
    rules = ["yesterday", "last bar", "oracle (realised bar spread)", "original: combined", "refined: combined",
             "refined + lagged spread", "LightGBM"]
    lab = {**RACE_LABELS, "oracle (realised bar spread)": "realised spread (oracle)"}
    oos = pd.read_csv(T / "10_quoting_oos.csv").set_index(["unit", "rule"])
    fwd = pd.read_csv(T / "10_quoting_fwd.csv").set_index(["unit", "rule"])
    rows = []
    for grp in ["constrained", "unconstrained"]:
        for rule in rules:
            o, f = oos.loc[(grp, rule)], fwd.loc[(grp, rule)]
            rows.append({"Group": grp, "Quoting rule": lab.get(rule, rule),
                         "OOS fill rate (%)": f"{o.fill_rate_pct:.0f}", "OOS realised (bps/fill)": f"{o.realised_bps:.2f}",
                         "FWD fill rate (%)": f"{f.fill_rate_pct:.0f}", "FWD half-spread quoted (bps)": f"{f.avg_half_spread_bps:.2f}",
                         "FWD adverse selection (bps)": f"{f.adverse_selection_bps:.2f}", "FWD realised (bps/fill)": f"{f.realised_bps:.2f}"})
    d = pd.DataFrame(rows)
    d.loc[d["Group"].duplicated(), "Group"] = ""
    return d, ("Table 5. Quoting-rule simulation. The market maker posts the rule's half-spread at the start of each 5-minute bar "
               "(snapped to the tick grid, never below half a tick); an aggressive lit order fills it if the aggressor paid at least that "
               "much on its first print. Fill rate = share of aggressive orders filled. Realised = half-spread earned minus the "
               "5-minute mid move after the fill, per filled order. OOS = 22 to 30 Oct, FWD = 2 to 6 Nov. "
               "Source: 10_quoting_oos.csv, 10_quoting_fwd.csv.")


def table7() -> tuple[pd.DataFrame, str]:
    """Do the drivers add anything once the last spread is known? Nested tests (script 14) and quote timing (script 13)."""
    rm = pd.read_csv(T / "14_nested_driver_rmse.csv").set_index(["sample", "unit"])
    dm = pd.read_csv(T / "14_nested_driver_diebold_mariano.csv")
    cw = pd.read_csv(T / "14_nested_driver_clark_west.csv")
    rob = pd.read_csv(T / "13_quote_timing_robustness.csv").query("rule == 'vol + impact rule'").set_index("sample")
    rows = []
    for smp, lab in [("OOS", "22-30 Oct"), ("FWD", "2-6 Nov")]:
        for grp in ["constrained", "unconstrained"]:
            d = dm[(dm["sample"] == smp) & (dm.unit == grp)]
            c = cw[(cw["sample"] == smp) & (cw.unit == grp)]
            g = lambda fam, m: d[(d.model == f"{fam}: {m}") & (d.vs == f"{fam}: controls only")]["dm_stat"].iloc[0]
            rows.append({
                "Week": lab, "Group": grp,
                "RMSE, lagged spread + controls": f"{rm.loc[(smp, grp), 'linear + lag: controls only']:.3f}",
                "RMSE, + vol + impact": f"{rm.loc[(smp, grp), 'linear + lag: combined']:.3f}",
                "DM (linear)": f"{g('linear + lag', 'combined'):.1f}",
                "Clark-West z (linear)": f"{c[(c.model == 'linear + lag: combined') & (c.vs == 'linear + lag: controls only')]['cw_stat'].iloc[0]:.1f}",
                "DM (LightGBM)": f"{g('LightGBM', 'combined'):.1f}",
                "Quote timing t, daily totals": f"{rob.loc[smp, 't_day']:.1f}" if grp == "constrained" else "",
                "Same, widths matched": f"{rob.loc[smp, 't_day_matched']:.1f}" if grp == "constrained" else "",
            })
    return pd.DataFrame(rows), (
        "Table 6. Do volatility and impact add anything once the last spread is known? Nested comparison with the lagged spread, "
        "time-of-day and activity controls held fixed. DM = Diebold-Mariano of the larger model against the smaller (negative "
        "favours adding the drivers; biased against the larger nested model). Clark-West z above 1.645 means the drivers carry "
        "information (one-sided 5%). Last two columns, all six stocks: paired test on daily P&L totals of the quoting rule with "
        "the volatility and impact forecasts against the same rule without them, at equal in-sample width and with widths also "
        "matched inside the test week. Source: 14_nested_driver_*.csv, 13_quote_timing_robustness.csv.")


def table8() -> tuple[pd.DataFrame, str]:
    a = pd.read_csv(T / "12_ae_appendix.csv")
    a.columns = ["Applied Econometrics check", "Script", "Result", "Implication"]
    return a, ("Table C1. Applied Econometrics checks (Newey-West lag rule, SE robustness, ADF, Granger, QLR, VAR lag choice, "
               "IRF bootstrap, ARCH-LM and standardised GARCH residuals). Source: 12_ae_appendix.csv.")


def table6() -> tuple[pd.DataFrame, str]:
    d = pd.read_csv(T / "11_vol_signature.csv").set_index("stock")
    out = pd.DataFrame({
        "Stock": d.index,
        "σ 5s": d["5s"].round(1).values, "σ 1min": d["1min"].round(1).values, "σ 5min": d["5min"].round(1).values,
        "σ 15min": d["15min"].round(1).values, "σ 30min": d["30min"].round(1).values,
        "Noise share at 1min (%)": d["noise_share_1min_pct"].round(0).astype(int).values,
        "Flattens at": d["flattens_at"].values,
        "Excess per 5s sample (bps)": d["excess_rms_5s_bps"].round(2).values,
        "Half-spread (bps)": d["half_spread_bps"].round(2).values,
    })
    return out, ("Table 2. Volatility signature, in-sample days: realised volatility of mid returns in bps per √minute by sampling "
                 "interval. Noise share = 1 − σ²(15min)/σ²(1min). 'Flattens at' = first interval within 5% of the 15-minute level. "
                 "Excess per 5-second sample = RMS of the variance above the 15-minute level, to compare with the median half-spread. "
                 "Source: 11_vol_signature.csv.")


TABLES = {1: table1, 2: table6, 3: table2, 4: race_combined, 5: table5, 6: table7, 7: table8}
def c_table1():
    d = pd.read_csv(T / "report_table1_data_summary.csv")
    d["Group"] = d["Group"].str.replace("unconstrained", "U").str.replace("constrained", "C")
    d = d[["Stock", "Group", "Lit trades per day", "Median spread (bps)", "Time at 1 tick (%)"]]
    d.columns = ["Stock", "Group", "Trades/day", "Spread (bps)", "At 1 tick (%)"]
    return d, "Table 1. Lit continuous trades, October 2026. C = tick-constrained, U = unconstrained."


def c_table2():
    c = pd.read_csv(T / "08_r1_coefficients.csv").query("spec == 'original'")
    se = pd.read_csv(T / "07_se_robustness.csv")
    names = {"impact_roll_ewm_bps": "Rolling impact", "impact_prevday_bps": "Yesterday's impact", "log_vol_har": "Volatility forecast"}
    rows = []
    for grp in ["constrained", "unconstrained"]:
        for v, lab in names.items():
            x = c[(c.group == grp) & (c["var"] == v)].iloc[0]
            mint = se[(se.group == grp) & (se["var"] == v)]["t"].abs().min()
            rows.append({"Group": grp[0].upper(), "Driver": lab, "Coef": f"{x.coef:.2f}", "t": f"{x.t:.1f}",
                         "+1 SD (bps)": f"{x.effect_1sd:.2f}", "Lowest |t|": f"{mint:.1f}"})
    return pd.DataFrame(rows), ("Table 2. In-sample regressions of the 5-minute quoted half-spread (bps). "
                                "Lowest |t| is over six standard-error choices.")


def c_table3():
    d, _ = race_combined()
    d.columns = ["Rule", "Oct C", "Nov C", "DM C (Oct/Nov)", "Oct U", "Nov U", "DM U (Oct/Nov)"]
    return d, ("Table 3. Forecast RMSE (bps) in the two test weeks. DM = Diebold-Mariano against the last bar; "
               "negative favours the rule.")


def c_table4():
    d, _ = table7()
    d = d.iloc[:, :7]
    d.columns = ["Week", "Group", "RMSE without", "RMSE with drivers", "DM", "Clark-West z", "DM (LightGBM)"]
    d["Group"] = d["Group"].str[0].str.upper()
    return d, ("Table 4. Adding volatility and impact to a model that already has the lagged spread. "
               "Negative DM or z above 1.645 favours the drivers.")


def c_table5():
    oos = pd.read_csv(T / "10_quoting_oos.csv").set_index(["unit", "rule"])
    fwd = pd.read_csv(T / "10_quoting_fwd.csv").set_index(["unit", "rule"])
    lab = {"last bar": "last bar", "refined: combined": "linear", "refined + lagged spread": "linear + lag", "LightGBM": "LightGBM"}
    rows = []
    for grp in ["constrained", "unconstrained"]:
        for rule, l in lab.items():
            o, f = oos.loc[(grp, rule)], fwd.loc[(grp, rule)]
            rows.append({"Group": grp[0].upper(), "Rule": l, "Oct fills (%)": f"{o.fill_rate_pct:.0f}", "Oct bps/fill": f"{o.realised_bps:.2f}",
                         "Nov fills (%)": f"{f.fill_rate_pct:.0f}", "Nov bps/fill": f"{f.realised_bps:.2f}"})
    return pd.DataFrame(rows), "Table 5. Quoting simulation: share of orders filled and profit per fill after five minutes."


def c_table6():
    """Refined quoting (script 15): base rule vs inventory control vs inventory control + selective quoting."""
    q = pd.read_csv(T / "15_refined_quoting.csv")
    lab = {"last bar (base)": "last bar (base)", "inventory control": "inventory control",
           "selective + inventory": "selective + inventory"}
    cols = [("Fills (%)", "fill_rate_pct", "{:.0f}", 1), ("bps/fill", "markout5_bps", "{:.2f}", 1),
            ("P&L/day (DKK k)", "mtm_pnl_dkk_per_day", "{:.1f}", 1e3), ("SD/day (DKK k)", "mtm_pnl_sd_day", "{:.1f}", 1e3),
            ("Max position (DKK m)", "max_abs_inv_mdkk", "{:.2f}", 1)]
    rows = []
    for grp in ["constrained", "unconstrained"]:
        for rule, l in lab.items():
            r = {"Group": grp[0].upper(), "Rule": l}
            o = q[(q["sample"] == "OOS") & (q.group == grp) & (q.rule == rule)].iloc[0]
            f = q[(q["sample"] == "FWD") & (q.group == grp) & (q.rule == rule)].iloc[0]
            for name, c, fmt, div in cols:
                r[f"{name}, Oct / Nov"] = fmt.format(o[c] / div) + " / " + fmt.format(f[c] / div)
            rows.append(r)
    return pd.DataFrame(rows), ("Table 6. Refined quoting, parameters chosen on 1 to 21 October. bps/fill = half-spread earned minus the "
                                "5-minute mid move; P&L = daily mark-to-market P&L with inventory flattened at the day's last mid "
                                "(mean and SD across days); max position = largest absolute inventory.")


CTABLES = {1: c_table1, 2: c_table2, 3: c_table3, 4: c_table4, 5: c_table5, 6: c_table6, 9: lambda: table8()}
CFIGURES = {
    1: ("report_fig1_intraday_tick.png", "Figure 1. Spread, turnover and volatility through the day, and time spent at one tick.", 14),
    2: ("11_vol_surface.png", "Figure 2. Volatility by sampling interval and hour, relative to each stock's 15-minute level.", 14),
    3: ("report_fig2_impact_by_stock.png", "Figure 3. Price impact against the effective half-spread.", 11),
    4: ("09_fwd_rmse.png", "Figure 4. Forecast error by rule: October test week (grey) and November (colour).", 13),
    5: ("15_markout_curve.png", "Figure 5. Realised half-spread per fill by markout horizon: the market's liquidity providers "
                                "against our market maker quoting the last bar's spread.", 15),
    6: ("report_fig4_shap.png", "Figure 6. What LightGBM relies on: SHAP values and the cost of dropping each feature group.", 13),
    7: ("15_refined_pnl.png", "Figure 7. Cumulative mark-to-market P&L of the base rule and the refined rules in the two test weeks.", 14),
}


FIGURES = {  # number: (file, caption, width in cm)
    1: ("report_fig1_intraday_tick.png", "Figure 1. Intraday pattern of spread, turnover and volatility (in-sample days, indexed to each "
                                         "stock's daily mean) and the share of time the spread sits at one tick.", 15),
    2: ("11_vol_surface.png", "Figure 2. Intraday volatility surface: realised volatility per √minute by sampling interval (5 seconds to "
                              "30 minutes) and hour of day, in-sample days, relative to each stock's whole-day 15-minute level. "
                              "Short intervals and the opening hour carry the transitory part (quote flicker and tick discreteness in the mid).", 16),
    3: ("report_fig2_impact_by_stock.png", "Figure 3. Permanent price impact (Hasbrouck VAR) against the five-minute impact from the spread "
                                           "decomposition and the effective half-spread, medians across days.", 13),
    4: ("09_fwd_rmse.png", "Figure 4. Forecast error by rule: out-of-sample window (grey, used three times) against the forward week "
                           "(colour, never used for fitting or tuning).", 14),
    5: ("10_quoting_frontier.png", "Figure 5. Quoting rules: fill rate against realised half-spread per fill, net of five-minute adverse "
                                   "selection. No rule is above zero.", 13.5),
    7: ("report_fig4_shap.png", "Figure 7. What the LightGBM forecast relies on: mean |SHAP| on in-sample bars by group (left) and the "
                                "increase in cross-validated error when a feature group is dropped (right).", 15),
    8: ("report_fig5_hmm_regimes.png", "Figure 8. Driver coefficients in the low- and high-volatility states (two-state HMM, "
                                       "forward-filtered), with 95% Newey-West intervals.", 15),
    6: ("13_iss_event.png", "Figure 6. ISS, 29 October: the 15:20 to 15:35 sell-off, the market half-spread and the quotes of three "
                            "rules set at the start of each 5-minute bar.", 12),
}


# ----------------------------------------------------------------------------- docx helpers
def set_cell_shading(cell, hex_fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tc_pr.append(shd)


def add_caption(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size, run.font.color.rgb, run.italic = Pt(8.5), GREY, False
    p.paragraph_format.space_after = Pt(10)


def keep_together(table) -> None:
    """Keep a table on one page: no row may split, every row but the last keeps with the next."""
    rows = table.rows
    for k, row in enumerate(rows):
        tr_pr = row._tr.get_or_add_trPr()
        cant = OxmlElement("w:cantSplit")
        tr_pr.append(cant)
        if k < len(rows) - 1:
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.keep_with_next = True


def add_math(doc: Document, text: str) -> None:
    """Formula line: x_k, x_{k-i}, Σ_{i=0..p} and ^{...} become real sub/superscripts."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for tok in re.split(r"(_\{[^}]*\}|\^\{[^}]*\}|_[A-Za-z0-9₁₂]|\^[A-Za-z0-9])", text):
        if not tok:
            continue
        if tok.startswith("_"):
            r = p.add_run(tok[2:-1] if tok.startswith("_{") else tok[1:])
            r.font.subscript = True
        elif tok.startswith("^"):
            r = p.add_run(tok[2:-1] if tok.startswith("^{") else tok[1:])
            r.font.superscript = True
        else:
            r = p.add_run(tok)
        r.font.name, r.font.size = "Cambria Math", Pt(10.5)


def add_table(doc: Document, df: pd.DataFrame, caption: str) -> None:
    add_caption(doc, caption)
    doc.paragraphs[-1].paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, col in enumerate(df.columns):
        cell = t.rows[0].cells[j]
        cell.text = ""
        run = cell.paragraphs[0].add_run(str(col))
        run.bold, run.font.size = True, Pt(8)
        cell.paragraphs[0].paragraph_format.space_after = Pt(0)
        set_cell_shading(cell, "EDEDED")
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for j, val in enumerate(row):
            cells[j].text = ""
            txt = "" if (isinstance(val, float) and pd.isna(val)) else str(val)
            run = cells[j].paragraphs[0].add_run(txt)
            run.font.size = Pt(8)
            cells[j].paragraphs[0].paragraph_format.space_after = Pt(0)
            if j > 0 and re.fullmatch(r"[-+−]?[\d.,]+%?( / [-+−]?[\d.,]+)?", txt.strip()):
                cells[j].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
    keep_together(t)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_figure(doc: Document, file: str, caption: str, width_cm: float = 15) -> None:
    doc.add_picture(str(F / file), width=Cm(width_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_caption(doc, caption)


def add_markdown_runs(p, text: str) -> None:
    """Bold (**x**), italic (*x*) and code (`x`) inline markup."""
    for tok in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)", text):
        if not tok:
            continue
        if tok.startswith("**"):
            p.add_run(tok[2:-2]).bold = True
        elif tok.startswith("`"):
            r = p.add_run(tok[1:-1])
            r.font.name, r.font.size = "Consolas", Pt(9.5)
        elif tok.startswith("*"):
            p.add_run(tok[1:-1]).italic = True
        else:
            p.add_run(tok)


# ----------------------------------------------------------------------------- build
def build() -> Path:
    global TABLES, FIGURES
    TABLES, FIGURES = CTABLES, CFIGURES  # compact exhibits (whole document near 2,500 words)
    doc = Document()
    sec = doc.sections[0]
    sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
    sec.left_margin = sec.right_margin = Cm(2.0)
    sec.top_margin = sec.bottom_margin = Cm(2.0)
    st = doc.styles["Normal"]
    st.font.name, st.font.size = BODY_FONT, Pt(BODY_PT)
    st.paragraph_format.space_after = Pt(6)
    st.paragraph_format.line_spacing = 1.08
    for name, size in [("Title", 20), ("Heading 1", 14), ("Heading 2", 12), ("Heading 3", 11)]:
        s = doc.styles[name]
        s.font.name, s.font.size, s.font.bold = BODY_FONT, Pt(size), True
        s.font.color.rgb = RGBColor(0, 0, 0)

    lines = SRC.read_text().splitlines()
    in_refs = False
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        i += 1
        if not line.strip() or line.startswith("---") or line.startswith(">"):
            continue
        if line.startswith("# "):
            doc.add_paragraph(line[2:], style="Title")
            continue
        if line.startswith("## "):
            h = line[3:]
            in_refs = h.strip().lower() == "references"
            doc.add_paragraph(h, style="Heading 1")
            continue
        if line.startswith("### "):
            doc.add_paragraph(line[4:], style="Heading 2")
            continue
        m = re.match(r"^\*\[(Table|Figure|Figures) (\d+)(?: / (Table|Figure) (\d+))?[^\]]*\]\*$", line)
        if m:
            kinds = [(m.group(1).rstrip("s"), int(m.group(2)))]
            if m.group(3):
                kinds.append((m.group(3), int(m.group(4))))
            if m.group(1) == "Figures":  # "Figures 4 and 5"
                kinds = [("Figure", n) for n in map(int, re.findall(r"\d+", line.split(":")[0]))]
            for kind, n in kinds:
                if kind == "Table":
                    df, cap = TABLES[n]()
                    add_table(doc, df, cap)
                else:
                    add_figure(doc, *FIGURES[n])
            continue
        if line.startswith("|"):  # markdown table: header, separator, rows
            block = [line]
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i].rstrip())
                i += 1
            cells = [[c.strip() for c in row.strip("|").split("|")] for row in block if not re.match(r"^\|[\s:|-]+\|$", row)]
            df = pd.DataFrame(cells[1:], columns=cells[0])
            t = doc.add_table(rows=1, cols=len(df.columns))
            t.style = "Table Grid"
            for j, col in enumerate(df.columns):
                t.rows[0].cells[j].text = ""
                r = t.rows[0].cells[j].paragraphs[0].add_run(col)
                r.bold, r.font.size = True, Pt(8.5)
                set_cell_shading(t.rows[0].cells[j], "EDEDED")
            for _, row in df.iterrows():
                cs = t.add_row().cells
                for j, val in enumerate(row):
                    cs[j].text = ""
                    add_markdown_runs(cs[j].paragraphs[0], str(val))
                    for r in cs[j].paragraphs[0].runs:
                        r.font.size = Pt(8.5)
            doc.add_paragraph()
            continue
        if line.startswith("- "):
            p = doc.add_paragraph(style="List Bullet")
            add_markdown_runs(p, line[2:])
            p.paragraph_format.space_after = Pt(2)
            continue
        if re.match(r"^\d+\. ", line):
            p = doc.add_paragraph(style="List Number")
            add_markdown_runs(p, re.sub(r"^\d+\. ", "", line))
            continue
        if in_refs:
            p = doc.add_paragraph()
            p.paragraph_format.left_indent, p.paragraph_format.first_line_indent = Cm(1), Cm(-1)
            add_markdown_runs(p, line)
            continue
        if line.startswith("[") and line.endswith("]"):
            p = doc.add_paragraph()
            r = p.add_run(line)
            r.italic, r.font.color.rgb = True, GREY
            continue
        if "Σ" in line or line.startswith("HalfSpread_"):
            add_math(doc, line)
            continue
        p = doc.add_paragraph()
        add_markdown_runs(p, line)
    doc.save(OUT)
    return OUT


if __name__ == "__main__":
    print(build())
