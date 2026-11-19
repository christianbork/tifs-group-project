# Appendices to the group report: What drives bid-ask spreads on Nasdaq Copenhagen?

**FINS5566 Trading in Financial Securities, T3 2026.** Christian Bork, Emanuel Annvik, Mathilde Sørensen. Submitted alongside the report and the code.

## Appendix A. AI transparency statement

**Tools.** Claude Code (Anthropic) and OpenAI Codex.

**How we used them.**

- Code: writing, debugging and refactoring the Python pipeline (scripts 01 to 15 and the shared package), writing unit tests, and running the pipeline end to end.
- Review: two code reviews and two report reviews against the Student Guide, which led to the standard-error robustness checks, the Clark-West tests, the timing robustness check and the quoting refinement (script 15).
- Writing: drafting and editing report text and captions. Commits made directly by an assistant appear under the author "Claude" in the repository history.

**Why.** The project combines a large tick dataset (about 720,000 trades), several econometric models and a simulation. AI assistance let a three-person team test more specifications and robustness checks within the deadline, and an independent review caught errors we had missed (for example, overnight lags in the controls and a misleading Sharpe ranking).

**How we checked it.** Every result rests on code we ran: unit tests (28), validation of trade signs against full quote streams, exact re-runs from a clean checkout, and a check of every number in the report against the table it comes from. The research question, design, model choices and interpretation are our own, and we are responsible for all content.

## Appendix B. Contribution statement 

Roles follow the Student Guide. The work was coordinated through a shared private GitHub repository (`christianbork/tifs-group-project`): numbered pipeline scripts with unit tests, branches merged through pull requests (#1 forward test, #2 nested tests), a plan file (`docs/PLAN.md`) that records every design decision, and a changelog (`CHANGELOG.md`) for each review round.

| Member | Role (Student Guide) | Deliverables | Evidence |
|---|---|---|---|
| Christian Bork | Quantitative Analysis and Technology | Repository and pipeline design, data loader and stitching, Lee-Ready signing, bars, Hasbrouck VAR and spread decomposition, volatility models, spread regressions, out-of-sample race, refinements (stock-by-time controls, LightGBM and SHAP, HMM), second version (standard-error robustness, dynamic model, bootstrap, econometrics checks, equal-width quote test, ISS event), quoting refinement (15), merge of all branches, figures notebook | `analysis/01` to `08`, `12`, `13`, `15`, `src/tifs_group_project/`, `notebooks/`, `CHANGELOG.md`, 29 commits |
| Emanuel Annvik | Risk and Sales (evaluation, refinement, report) | FactSet data exports (October re-export and November forward week), forward-test holdout (09), market-maker quoting simulation (10), volatility surface (11), quote-stream validation, two code reviews and fixes, Clark-West and timing robustness checks, report drafts 1 to 5 and 7, Word and PDF build | `analysis/09` to `11`, `src/tifs_group_project/quoting.py`, `data/README.md`, `report/`, 20+ commits |
| Mathilde Sørensen | Trading (market interpretation and validation) | Nested driver tests (14) answering the second research question (does each driver add to a model that already knows the last spread?), sample audit of the nested comparison, dependency update, review and merge of the merged version (pull request 2) | `analysis/14`, `outputs/tables/14_*`, pull request 2, 3 commits |

## Appendix C. Applied Econometrics checks 

The checks below follow the Applied Econometrics course. The structural-break (QLR) dates rest on 21 daily observations and are descriptive. Full tables for every result in the report, including the volatility signature, the standard-error robustness, the per-stock forecast errors, the quoting simulation for all rules and the ISS event study, are in `outputs/tables/` of the code submission (index in `CHANGELOG.md`).

*[Table 9: 12_ae_appendix.csv.]*

