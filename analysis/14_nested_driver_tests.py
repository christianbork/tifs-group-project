# %% [markdown]
# # 14 Nested driver tests: do impact and volatility add forecasting information?
#
# This is a post-review diagnostic requested to answer the project's second research
# question directly. It is deliberately additive: scripts 01-13 and their output files are
# not changed. Both the October OOS period and the November forward period have already
# been inspected elsewhere, so these results are not a new pristine holdout test.
#
# Within each model family, the non-driver information is held fixed and four nested
# specifications are compared:
#   1. controls only;
#   2. controls + volatility;
#   3. controls + impact;
#   4. controls + volatility + impact.
#
# The linear family always includes the lagged spread and stock-by-time fixed effects.
# The LightGBM family always includes the lagged spread, stock, time and activity controls.
# LightGBM hyperparameters are frozen at the values selected on in-sample days in script 08.
# Every comparison uses a common training sample and a common evaluation sample.
#
# Outputs:
#   14_nested_driver_rmse.csv
#   14_nested_driver_diebold_mariano.csv
#   14_nested_driver_predictions.parquet
#   14_nested_driver_sample_audit.csv

# %% Setup
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from tifs_group_project import config
from tifs_group_project.spread_model import (
    build_panel,
    coarse_bucket,
    dm_test,
    fit_ols,
    nw_lags,
    predict,
)

SEED = 42
warnings.filterwarnings("ignore", message=".*eval_set.*")

IMPACT = ["impact_roll_ewm_bps", "impact_prevday_bps"]
VOLATILITY = ["log_vol_har"]
LINEAR_CONTROLS = ["lag_log_trades", "lag_log_turnover", "lag_abs_ofi"]
LAGGED_SPREAD = ["lag_quoted_hs"]
GROUPS = {
    "constrained": config.CONSTRAINED,
    "unconstrained": config.UNCONSTRAINED,
}

DRIVER_BLOCKS = {
    "controls only": [],
    "vol only": VOLATILITY,
    "impact only": IMPACT,
    "combined": VOLATILITY + IMPACT,
}

# LightGBM's non-driver information matches the full model in scripts 08 and 09.
ML_BASE = [
    "minute_of_day",
    "min_since_us_open",
    "bucket",
    "stock",
    "lag_quoted_hs",
    "lag_log_trades",
    "lag_log_turnover",
    "lag_ofi",
    "lag_abs_ofi",
]
ML_CATEGORICAL = ["bucket", "stock"]

LINEAR_NAMES = {name: f"linear + lag: {name}" for name in DRIVER_BLOCKS}
ML_NAMES = {name: f"LightGBM: {name}" for name in DRIVER_BLOCKS}
MODEL_NAMES = [*LINEAR_NAMES.values(), *ML_NAMES.values()]
BENCHMARKS = ["last bar", "yesterday"]


def linear_features(driver_block: str) -> list[str]:
    """Features for one member of the nested dynamic-linear family."""
    return LINEAR_CONTROLS + LAGGED_SPREAD + DRIVER_BLOCKS[driver_block]


def ml_features(driver_block: str) -> list[str]:
    """Features for one member of the nested LightGBM family."""
    return ML_BASE + DRIVER_BLOCKS[driver_block]


panel = build_panel("5min")
panel["bucket"] = coarse_bucket(panel["minute_of_day"], panel["min_since_us_open"])

# Requiring the union of all variables makes the four variants comparable. In particular,
# a model cannot win merely because it was evaluated on an easier subset of bars.
COMMON_FEATURES = list(
    dict.fromkeys(LINEAR_CONTROLS + LAGGED_SPREAD + ML_BASE + VOLATILITY + IMPACT)
)
TRAIN_REQUIRED = ["quoted_hs_bps", *COMMON_FEATURES]
EVAL_REQUIRED = ["quoted_hs_bps", "bench_yday", *COMMON_FEATURES]

train = panel[panel["sample"] == "IS"].dropna(subset=TRAIN_REQUIRED).copy()
evaluation = panel[panel["sample"].isin(["OOS", "FWD"])].dropna(
    subset=EVAL_REQUIRED
).copy()

if train.empty:
    raise ValueError("No complete in-sample rows. Run the upstream data pipeline first.")
if evaluation.empty:
    raise ValueError("No complete OOS/FWD rows. Run the upstream data pipeline first.")
for required_sample in ["OOS", "FWD"]:
    if not (evaluation["sample"] == required_sample).any():
        raise ValueError(
            f"No complete {required_sample} rows. Run the upstream data pipeline first."
        )

evaluation["pred_last bar"] = evaluation["lag_quoted_hs"]
evaluation["pred_yesterday"] = evaluation["bench_yday"]

L = nw_lags(int(train.groupby("stock").size().median()))

# %% Nested dynamic-linear models
for group, stocks in GROUPS.items():
    group_train = train[train["stock"].isin(stocks)]
    eval_rows = evaluation["stock"].isin(stocks)
    for block in DRIVER_BLOCKS:
        features = linear_features(block)
        result, fitted_rows = fit_ols(
            group_train,
            "quoted_hs_bps",
            features,
            L,
            stock_tod=True,
        )
        if len(fitted_rows) != len(group_train):
            raise AssertionError(
                f"Linear {group}/{block} did not use the common training sample."
            )
        model = LINEAR_NAMES[block]
        evaluation.loc[eval_rows, f"pred_{model}"] = predict(
            result,
            evaluation.loc[eval_rows],
            features,
            stock_tod=True,
        )

# %% Nested LightGBM models with script 08's frozen settings
cv_path = config.TABLES / "08_r2_lgbm_cv.csv"
if not cv_path.exists():
    raise FileNotFoundError(
        "08_r2_lgbm_cv.csv is required to freeze the LightGBM settings from script 08."
    )

cv = pd.read_csv(cv_path).sort_values("cv_rmse")
best = cv.iloc[0]
LGBM_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.03,
    "n_estimators": int(round(best["best_iter"])),
    "num_leaves": int(best["num_leaves"]),
    "min_child_samples": int(best["min_child_samples"]),
    "colsample_bytree": float(best["colsample_bytree"]),
    "subsample": 1.0,
    "random_state": SEED,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}


def ml_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """LightGBM frame with stable category levels across train and test periods."""
    frame = df[features].copy()
    if "stock" in features:
        frame["stock"] = pd.Categorical(
            frame["stock"], categories=list(config.STOCKS)
        )
    if "bucket" in features:
        frame["bucket"] = pd.Categorical(
            frame["bucket"], categories=sorted(panel["bucket"].unique())
        )
    return frame


for block in DRIVER_BLOCKS:
    features = ml_features(block)
    categorical = [col for col in ML_CATEGORICAL if col in features]
    model = lgb.LGBMRegressor(**LGBM_PARAMS).fit(
        ml_frame(train, features),
        train["quoted_hs_bps"],
        categorical_feature=categorical,
    )
    name = ML_NAMES[block]
    evaluation[f"pred_{name}"] = model.predict(ml_frame(evaluation, features))

# %% Common-sample audit and squared errors
prediction_columns = [f"pred_{name}" for name in [*BENCHMARKS, *MODEL_NAMES]]
if evaluation[prediction_columns].isna().any().any():
    missing = evaluation[prediction_columns].isna().sum()
    raise AssertionError(
        "Nested models produced missing predictions:\n"
        + missing[missing > 0].to_string()
    )

for name in [*BENCHMARKS, *MODEL_NAMES]:
    evaluation[f"se_{name}"] = (
        evaluation["quoted_hs_bps"] - evaluation[f"pred_{name}"]
    ) ** 2

audit_rows = []
for sample in ["OOS", "FWD"]:
    sample_rows = evaluation[evaluation["sample"] == sample]
    for unit in [*GROUPS, *config.STOCKS]:
        rows = (
            sample_rows[sample_rows["group"] == unit]
            if unit in GROUPS
            else sample_rows[sample_rows["stock"] == unit]
        )
        audit_rows.append(
            {
                "sample": sample,
                "unit": unit,
                "rows": len(rows),
                "days": rows["date"].nunique(),
                "first_date": rows["date"].min(),
                "last_date": rows["date"].max(),
                "unique_stock_timestamps": rows[["stock", "ts"]]
                .drop_duplicates()
                .shape[0],
                "missing_predictions": int(rows[prediction_columns].isna().sum().sum()),
            }
        )

audit = pd.DataFrame(audit_rows)
audit.to_csv(config.TABLES / "14_nested_driver_sample_audit.csv", index=False)

# %% RMSE on the identical evaluation rows
rmse_rows = []
for sample in ["OOS", "FWD"]:
    sample_rows = evaluation[evaluation["sample"] == sample]
    for unit in [*GROUPS, *config.STOCKS]:
        rows = (
            sample_rows[sample_rows["group"] == unit]
            if unit in GROUPS
            else sample_rows[sample_rows["stock"] == unit]
        )
        rmse_rows.append(
            {
                "sample": sample,
                "unit": unit,
                "n": len(rows),
                **{
                    name: np.sqrt(rows[f"se_{name}"].mean())
                    for name in [*BENCHMARKS, *MODEL_NAMES]
                },
            }
        )

rmse = pd.DataFrame(rmse_rows)
rmse.to_csv(
    config.TABLES / "14_nested_driver_rmse.csv", index=False, float_format="%.4f"
)

# %% Diebold-Mariano tests within each model family
# These are the comparisons that identify whether volatility and/or impact add information.
NESTED_PAIRS = [
    ("vol only", "controls only"),
    ("impact only", "controls only"),
    ("combined", "controls only"),
    ("combined", "vol only"),
    ("combined", "impact only"),
]

dm_rows = []
for sample in ["OOS", "FWD"]:
    sample_rows = evaluation[evaluation["sample"] == sample]
    dm_lags = nw_lags(sample_rows["ts"].nunique())
    for unit in [*GROUPS, *config.STOCKS]:
        rows = (
            sample_rows[sample_rows["group"] == unit]
            if unit in GROUPS
            else sample_rows[sample_rows["stock"] == unit]
        )
        for family, names in [("linear + lag", LINEAR_NAMES), ("LightGBM", ML_NAMES)]:
            loss_columns = [f"se_{names[block]}" for block in DRIVER_BLOCKS]
            # For group results, give each stock equal weight within a timestamp, matching
            # scripts 07-09. A stock result has one observation per timestamp already.
            loss = rows.groupby("ts")[loss_columns].mean()
            for model_block, baseline_block in NESTED_PAIRS:
                model_name = names[model_block]
                baseline_name = names[baseline_block]
                dm_rows.append(
                    {
                        "sample": sample,
                        "unit": unit,
                        "family": family,
                        "model": model_name,
                        "vs": baseline_name,
                        **dm_test(
                            loss[f"se_{model_name}"],
                            loss[f"se_{baseline_name}"],
                            dm_lags,
                        ),
                    }
                )

dm = pd.DataFrame(dm_rows)
dm.to_csv(
    config.TABLES / "14_nested_driver_diebold_mariano.csv",
    index=False,
    float_format="%.4f",
)

# %% Clark-West tests for the nested linear models (added 18-Nov after review)
# Diebold-Mariano compares the two FORECASTS. When the models are nested it is biased against the
# larger one: under the null that the extra regressors carry no information, the larger model still
# estimates their coefficients, and that noise raises its squared error. Clark and West (2007)
# remove that term: f_t = e_small^2 - [e_big^2 - (yhat_small - yhat_big)^2], and test mean(f) > 0
# one-sided. DM answers "is the bigger model's forecast better as it stands?"; Clark-West answers
# "do the drivers carry information?". Reported for the linear family only: the adjustment is
# derived for linear models estimated by least squares, not for boosted trees.
from scipy import stats as _stats

cw_rows = []
for sample in ["OOS", "FWD"]:
    sample_rows = evaluation[evaluation["sample"] == sample]
    cw_lags = nw_lags(sample_rows["ts"].nunique())
    for unit in [*GROUPS, *config.STOCKS]:
        rows = (
            sample_rows[sample_rows["group"] == unit]
            if unit in GROUPS
            else sample_rows[sample_rows["stock"] == unit]
        )
        for model_block, baseline_block in NESTED_PAIRS:
            big, small = LINEAR_NAMES[model_block], LINEAR_NAMES[baseline_block]
            y = rows["quoted_hs_bps"]
            f = (y - rows[f"pred_{small}"]) ** 2 - (
                (y - rows[f"pred_{big}"]) ** 2 - (rows[f"pred_{small}"] - rows[f"pred_{big}"]) ** 2
            )
            f = f.groupby(rows["ts"]).mean().dropna().to_numpy()  # equal weight per stock within a bar
            t_n = len(f)
            u = f - f.mean()
            lrv = u @ u / t_n + 2 * sum(
                (1 - k / (cw_lags + 1)) * (u[k:] @ u[:-k]) / t_n for k in range(1, cw_lags + 1)
            )
            z = f.mean() / np.sqrt(lrv / t_n)
            cw_rows.append(
                {
                    "sample": sample,
                    "unit": unit,
                    "model": big,
                    "vs": small,
                    "cw_mean": f.mean(),
                    "cw_stat": z,
                    "p_one_sided": float(_stats.norm.sf(z)),
                    "n": t_n,
                }
            )
cw = pd.DataFrame(cw_rows)
cw.to_csv(
    config.TABLES / "14_nested_driver_clark_west.csv", index=False, float_format="%.4f"
)
print(
    "Clark-West, linear family, group results (stat > 1.645: the added driver carries information):\n",
    cw[cw["unit"].isin(GROUPS)].round(3).to_string(index=False),
    "\n",
)

# Save only this script's predictions. Existing files used by scripts 10 and 13 are untouched.
keep = [
    "stock",
    "group",
    "date",
    "ts",
    "sample",
    "quoted_hs_bps",
    *prediction_columns,
]
evaluation[keep].to_parquet(
    config.PROCESSED / "14_nested_driver_predictions.parquet", index=False
)

print(
    "Nested driver RMSE (bps):\n",
    rmse.set_index(["sample", "unit"]).round(3).to_string(),
    "\n",
)
print(
    "Nested driver DM tests for group results "
    "(negative = model beats the comparison):\n",
    dm[dm["unit"].isin(GROUPS)]
    .set_index(["sample", "unit", "family", "model", "vs"])[
        ["dm_stat", "p_value", "n"]
    ]
    .round(4)
    .to_string(),
    "\n",
)
