# Appendices to the group report: What drives bid-ask spreads on Nasdaq Copenhagen?

**FINS5566 Trading in Financial Securities, T3 2026.** Christian Bork, Emanuel Annvik, Mathilde Sørensen. Submitted alongside the report and the code.

## Appendix A. AI transparency statement 

We used generative AI in this project and this is how.

Emanuel used Claude (Anthropic, Claude Code) for: drafting and debugging the forward-test script (`analysis/09_forward_holdout.py`), the quoting-rule simulation (`analysis/10_quoting_rule.py`, `src/tifs_group_project/quoting.py`) and their unit tests; driving the FactSet Time and Sales exports for 1 to 30 October and 2 to 6 November and checking each file for the 25,000-row cap; re-running the full pipeline and comparing the regenerated tables with the committed ones; an independent code review of all scripts, which found and fixed three errors in the first version of the quoting simulation (fills counted per print instead of per order, quotes not snapped to the tick grid, a wrong inventory figure); drafting the report text from the output tables, and editing it for length and language. Every number in the report was checked against the table it comes from, and the tables were produced by the scripts in the repository, not by the model.

[Draft, Christian to confirm:] Christian used Claude (Anthropic, Claude Code) for: writing and debugging the data loader, signing, bars, price-impact, volatility, spread-model and refinement scripts (01 to 08), their unit tests and the figures notebook; a review of the first draft against the Student Guide and the Applied Econometrics material, which led to the second version (standard-error robustness, a dynamic specification, bootstrap intervals for the impact, GARCH diagnostics, ADF, Granger and QLR tests, the equal-width quote-timing test, scripts 12 and 13); and merging that version with the forward test. Outputs were checked with unit tests, the quote-stream validation, the cross-check of the VAR impact against the spread decomposition, an exact re-run of the previous version before any change, and a reconciliation of every number in the report with the output tables.

[Draft, Mathilde to confirm:] Mathilde used OpenAI Codex to help draft and review the nested driver tests (`analysis/14_nested_driver_tests.py`), which ask whether volatility and impact add forecasting information once the last spread is known. She checked the design, ran the script, audited that every model is scored on identical rows (`14_nested_driver_sample_audit.csv`) and interpreted the results. Emanuel added the Clark-West version of the test with Claude after a review.

The research question, the tick-constraint design, the model specifications, the decision to add a forward week and a quoting simulation, and the interpretation of the results are our own. AI output was treated as a draft to be verified, and we are responsible for all content.

## Appendix B. Contribution statement 

Roles follow the Student Guide. The work was coordinated through a shared private GitHub repository: numbered pipeline scripts with unit tests, pull requests reviewed before merging, and a plan file (`docs/PLAN.md`) that records every design decision and who made it.

| Member | Role (Student Guide) | Deliverables | Evidence |
|---|---|---|---|
| Christian Bork | Quantitative Analysis and Technology | Repository and pipeline design, data loader and stitching, Lee-Ready signing, bars, Hasbrouck VAR and spread decomposition, volatility models, spread regressions, out-of-sample race, refinements (stock-by-time controls, LightGBM and SHAP, HMM), second version (standard-error robustness, dynamic model, bootstrap, econometrics checks, equal-width quote test, ISS event), merge of all branches, figures notebook | `analysis/01` to `08`, `12`, `13`, `src/tifs_group_project/`, `notebooks/`, `CHANGELOG.md`, 27 commits |
| Emanuel Annvik | Risk and Sales (evaluation, refinement, report) | FactSet data exports (October re-export and November forward week), forward-test holdout (09), market-maker quoting simulation (10), volatility surface (11), quote-stream validation, two code reviews and fixes, Clark-West and timing robustness checks, report drafts 1 to 5 and 7, Word and PDF build | `analysis/09` to `11`, `src/tifs_group_project/quoting.py`, `data/README.md`, `report/`, 20+ commits |
| Mathilde Sørensen | Trading (market interpretation and validation) [to confirm] | Nested driver tests (14) answering the second research question, sample audit, review of the merged version before submission [add: report review, peer coordination] | `analysis/14`, `outputs/tables/14_*`, pull request 2, 3 commits |

## Appendix C. Applied Econometrics checks 

The checks below follow the Applied Econometrics course. The structural-break (QLR) dates rest on 21 daily observations and are descriptive. Full tables for every result in the report, including the volatility signature, the standard-error robustness, the per-stock forecast errors, the quoting simulation for all rules and the ISS event study, are in `outputs/tables/` of the code submission (index in `CHANGELOG.md`).

*[Table 9: 12_ae_appendix.csv.]*

