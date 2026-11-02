# tifs-group-project
FINS5566 group project: what drives bid-ask spreads on Nasdaq Copenhagen stocks, informed trading (price impact) vs volatility.

- Run tests: `pytest`
- Format: `ruff format .`
- Data source: FactSet trade exports (trades with the bid/ask at each trade), 6 CSE stocks (NOVO.B, DSV, VWS, PNDORA, GN, ISS), 22 days 1–30 Oct 2026, 08:45–17:20 local. One file per stock-day in data/raw/<stock>/; busy days split as -part1/-part2 (export cap ~25k rows, truncation drops the morning silently). data/raw/quotes_sample/ = ISS 30-Oct full quotes file (validation only).
- Full process, methods and assumptions: docs/PLAN.md (read first)
- Notes and decisions:
  - Use local Copenhagen time from `Time (ms)`; files are newest-first.
  - Drop IND rows (auction indicative prices). AUC = auction prints, CLS = close, MPM = dark midpoint (venue DCSE).
  - In-sample 1–21 Oct, out-of-sample 22–30 Oct. DSV 21-Oct = earnings day.
  - Remove duplicates only in the overlap between -partN files; some files contain genuinely repeated rows.
  - Pipeline = numbered scripts in analysis/; shared functions in src/tifs_group_project/; report figures from notebooks/report_figures.ipynb.
  - Signing (02): use the quote attached to each trade (fits better than lagged quotes; matches the prevailing quote 95% of the time on ISS 30-Oct; 99.8% sign agreement). Quote rule, then tick test; ~0% left unsigned.
  - Fills sharing a timestamp = one aggressive order walking the book (order_id); ~2.3 fills per order and 98% of multi-fill orders are one-sided. Trades outside the quote are later fills of these orders, not bad data.
