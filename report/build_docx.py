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
SRC, OUT = ROOT / "report" / "REPORT_DRAFT.md", ROOT / "report" / "Group_Report.docx"

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
    d.loc[d["Group"].duplicated(), "Group"] = ""
    return d, ("Table 3. In-sample (1 to 21 Oct) regressions of the 5-minute quoted half-spread (bps) by group. "
               "Cells: coefficient [Newey-West t] and the effect of a one within-stock SD increase in bps. "
               "Original = shared time-of-day dummies and stock fixed effects; refined = stock-by-time-bucket dummies (Section 6.1). "
               "All specifications include lagged trades, turnover and |order imbalance|. *** p<0.01, ** p<0.05, * p<0.1. "
               "Source: 08_r1_coefficients.csv.")


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
               "(used three times) and forward week 2 to 6 November (used once, all parameters frozen on 1 to 21 October). "
               "C = constrained group, U = unconstrained. DM = Diebold-Mariano statistic against the last bar's spread, "
               "negative favours the model. Per-stock rows: 08_oos_rmse.csv, 09_fwd_rmse.csv.")


def table3() -> tuple[pd.DataFrame, str]:
    return race_table("08_oos_rmse.csv", "08_oos_diebold_mariano.csv", "all OOS days",
                      "Table 3. Out-of-sample RMSE of the quoted half-spread (bps per 5-minute bar), 22 to 30 October. "
                      "Benchmarks: in-sample average, yesterday's average, last bar's spread. DM = Diebold-Mariano statistic, "
                      "negative favours the model. Source: 08_oos_rmse.csv, 08_oos_diebold_mariano.csv.")


def table4() -> tuple[pd.DataFrame, str]:
    return race_table("09_fwd_rmse.csv", "09_fwd_diebold_mariano.csv", "all FWD days",
                      "Table 4. Forward test, 2 to 6 November (used once): RMSE of the quoted half-spread (bps per 5-minute bar) "
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


TABLES = {1: table1, 2: table6, 3: table2, 4: race_combined, 5: table5}
FIGURES = {  # number: (file, caption, width in cm)
    1: ("report_fig1_intraday_tick.png", "Figure 1. Intraday pattern of spread, turnover and volatility (in-sample days, indexed to each "
                                         "stock's daily mean) and the share of time the spread sits at one tick.", 15),
    2: ("11_vol_surface.png", "Figure 2. Intraday volatility surface: realised volatility per √minute by sampling interval (5 seconds to "
                              "30 minutes) and hour of day, in-sample days, relative to each stock's whole-day 15-minute level. "
                              "Short intervals and the opening hour carry the transitory (bid-ask bounce) part.", 16),
    3: ("report_fig2_impact_by_stock.png", "Figure 3. Permanent price impact (Hasbrouck VAR) against the five-minute impact from the spread "
                                           "decomposition and the effective half-spread, medians across days.", 13),
    4: ("09_fwd_rmse.png", "Figure 4. Forecast error by rule: out-of-sample window (grey, used three times) against the forward week "
                           "(colour, used once).", 15),
    5: ("10_quoting_frontier.png", "Figure 5. Quoting rules: fill rate against realised half-spread per fill, net of five-minute adverse "
                                   "selection. No rule is above zero.", 13.5),
    6: ("report_fig4_shap.png", "Figure 6. Mean |SHAP| of the LightGBM forecast on in-sample bars, by group.", 12),
    7: ("report_fig5_hmm_regimes.png", "Figure 7. Driver coefficients by volatility regime (two-state HMM, forward-filtered), with 95% "
                                       "Newey-West intervals.", 15),
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
            if j > 0:
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
