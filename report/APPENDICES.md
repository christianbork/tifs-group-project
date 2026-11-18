# Appendices to the group report: What drives bid-ask spreads on Nasdaq Copenhagen?

**FINS5566 Trading in Financial Securities, T3 2026.** Christian Bork, Emanuel Annvik, Mathilde Sørensen. Submitted alongside the report and the code.

## Appendix A. AI transparency statement

We used AI coding assistants (Claude Code by Anthropic and OpenAI Codex) to help write, debug and review the Python code in this project, to run and check the data pipeline, and to help draft and edit the report text. All output was checked by us: the code with unit tests, validation against full quote streams and exact re-runs, and every number in the report against the table it comes from. The research question, the design, the choice of models and the interpretation of the results are our own, and we are responsible for all content.

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

