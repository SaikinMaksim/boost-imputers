"""Compare the three boosting imputers with mean imputation on the diabetes dataset.

Run:  python examples/benchmark.py
"""

import time

import pandas as pd
from sklearn.datasets import load_diabetes

from boost_imputers import CatBoostImputer, LightGBMImputer, XGBoostImputer, imputation_error, make_missing

X = load_diabetes(as_frame=True, scaled=False).frame
X["sex"] = X["sex"].map({1.0: "F", 2.0: "M"})  # a categorical column
X_missing, mask = make_missing(X, frac=0.2, random_state=0)

params = {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4}
imputers = {
    "mean/mode": LightGBMImputer(max_iter=0),  # initial fill only = baseline
    "catboost": CatBoostImputer(max_iter=5, params=params, random_state=0, verbose=1),
    "xgboost": XGBoostImputer(max_iter=5, params=params, random_state=0, verbose=1),
    "lightgbm": LightGBMImputer(max_iter=5, params=params, random_state=0, verbose=1),
}

scores = {}
for name, imputer in imputers.items():
    start = time.perf_counter()
    X_filled = imputer.fit_transform(X_missing)
    print(f"{name}: {time.perf_counter() - start:.1f}s")
    scores[name] = imputation_error(X, X_filled, mask).reindex(columns=["nrmse", "error_rate"]).bfill(axis=1).iloc[:, 0]

print("\nNRMSE (numeric) / error rate (sex) on hidden values, lower is better:")
print(pd.DataFrame(scores).round(3).to_string())
print("mean", pd.DataFrame(scores).mean().round(3).to_dict())
