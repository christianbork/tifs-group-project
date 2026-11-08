# Forward-test data: Nasdaq Copenhagen, 2-6 November 2026

Exported by Emanuel on 8 Nov 2026 from the FactSet web workstation, Time & Sales app
(`my.apps.factset.com/workstation/time-sales/`), using the Download button exactly as
Christian did for the October sample. Nothing in this folder was seen by the models.
Use it once, with every parameter frozen, as the final out-of-sample / forward test.

## Settings used
- Type: Trades (quotes files: Trades + Quotes ticked)
- Date & Time: one trading day per file, 08:45:00 to 17:20:00
- Time zone: Date & Time gear -> "Exchange" (Copenhagen local time; the request body
  then carries `streamingDataTimeZone = NONE`, `exchangeOffset = 60`)
- Settings: Off Hour ticked; At Only / Primary Only / Block Only / Exclude Odd Trades off
- Columns (Settings gear -> Columns): the default 13 columns plus Tick, Venue, Sell Name,
  Sell ID, Exch Time, Curr Yld, Buy Name, Buy ID. `Venue` carries the MIC codes the
  loader needs (XCSE lit, DCSE dark, MCSE other). Buy ID / Sell ID are always "001"
  (Nasdaq Copenhagen post-trade is anonymous), Buy/Sell Name are empty, Curr Yld is 0.
  Extra columns are ignored by `tifs_group_project.io`.
- Files are newest row first, same as the October exports.

## Export cap
FactSet caps a download at 25,000 rows and silently drops the oldest rows (the morning).
Every file was checked: full-day files start at 08:45:00 and end with the 17:20:00 CLS print.
Busy days were exported in time windows and saved as `-partN` files for `io._stitch`:

| Stock-day | Parts (Copenhagen time) | Rows |
|---|---|---|
| Novo 4 Nov (Q3 report day, 12.1m shares) | 08:45-09:30, 09:30-11:00, 11:00-13:30, 13:30-15:30, 15:30-17:20 | 12,206 / 7,990 / 11,982 / 8,913 / 15,724 |
| Novo 5 Nov | 08:45-12:00, 12:00-14:30, 14:30-17:20 | 13,186 / 4,764 / 24,310 |
| Novo 6 Nov | 08:45-12:00, 12:00-17:20 | 8,569 / 18,913 |

Adjacent parts share the boundary second; `_stitch` removes the duplicates in the overlap.
All other stock-days fit in one file (largest: GN 5 Nov 13,880 rows, Vestas 4 Nov 11,660).

## Event days in the window (for the stress test)
- Novo Nordisk 4 Nov: ~12.1m shares vs ~3.6m on 2 Nov; price 270 -> 285 by the close.
- Pandora 4 Nov: price 829 -> 919 (+11%), turnover ~2.6x normal.
- GN Store Nord 5 Nov: price 111 -> 124 (+12%), ~3.9m shares vs ~0.6m.
Check the company announcements to label these (Q3 reports are the obvious candidates).

## quotes_sample/
- `ISS-CSE-20261105-20261105-quotes.csv`: full L1 quote stream for ISS on 5 Nov (one side
  per update row; trade rows carry `Exch Time`), same format as Christian's ISS 30 Oct file.
- `NOVO.B-CSE-20261106-20261106-quotes-1000-1100.csv`: Novo quote stream 10:00-11:00 on
  6 Nov (9,562 rows for one hour), to quantify the trade-time sampling bias on the most
  liquid stock.
- `GN-CSE-20261105-20261105-quotes-part1.csv` / `-part2.csv`: GN quote stream on its event day (08:45-12:00 and 12:00-17:20; the full day exceeded the 25,000-row cap).

## Layout
Mirrors the repo: copy `raw/<Stock>/` into `data/raw/<Stock>/` of `tifs-group-project`
(folder names Novo, DSV, Vestas, Pandora, GN, ISS) and `raw/quotes_sample/` into
`data/raw/quotes_sample/`. Extend `config.IS_END` / sample labels so that 2-6 Nov is a
third sample ("FWD"), never merged into IS or OOS.
