# boost-imputers

Iterative (MICE-style) missing-value imputers powered by **CatBoost**, **XGBoost** and **LightGBM**,
with a scikit-learn API. Inspired by the
[Iterative CatBoost Imputer](https://www.kaggle.com/code/arunklenin/ps3e15-iterative-catboost-imputer-ensemble)
Kaggle notebook.

## How it works

1. Missing cells are pre-filled with the mean (numeric) / mode (categorical), or left as NaN.
2. For each round, every column with gaps is modelled from all other columns: a regressor for
   numeric columns, a classifier for categorical ones. The model is trained on rows where the
   column is observed and its predictions overwrite the missing cells.
3. Rounds repeat until the imputed values stop changing (`tol`) or `max_iter` is reached.

For a detailed walkthrough of the algorithm and the design decisions, see
[docs/how-it-works.md](docs/how-it-works.md).

## Installation

```bash
pip install "boost-imputers[all]"        # all three backends
pip install "boost-imputers[catboost]"   # or just the one you need
```

From source:

```bash
pip install -e ".[dev]"
```

## Quick start

```python
import pandas as pd
from boost_imputers import CatBoostImputer, LightGBMImputer, XGBoostImputer

df = pd.concat([train, test])  # typical Kaggle usage: impute train + test together

imputer = CatBoostImputer(
    max_iter=10,
    tol=1e-3,
    params={"iterations": 300, "depth": 6, "learning_rate": 0.05},
    random_state=42,
    verbose=1,
)
df_filled = imputer.fit_transform(df)

imputer.history_          # change of imputed values per round and per column
imputer.transform(new_df) # replays the fitted models on new data
```

All three classes share the same parameters; only `params` are backend-specific.
`n_estimators`, `max_depth` and `learning_rate` are accepted by every backend.

| Parameter | Default | Meaning |
|---|---|---|
| `max_iter` | `10` | Maximum number of rounds over all columns |
| `tol` | `1e-3` | Early stop when the mean change of imputed values falls below it (`None` disables) |
| `initial_strategy` | `"mean"` | `"mean"`, `"median"`, `"most_frequent"`, `"constant"` or `"none"` (keep NaN, use native NaN handling) |
| `imputation_order` | `"ascending"` | `"ascending"` / `"descending"` (by number of NaNs), `"roman"`, `"arabic"`, `"random"` |
| `columns` | all with NaN | Columns to model |
| `categorical_features` | auto | Labels, positions or boolean mask; auto = object, string, category and bool columns |
| `params` | `{}` | Parameters for every model |
| `regressor_params` / `classifier_params` | `{}` | Overrides for numeric / categorical targets |
| `max_samples` | `None` | Subsample training rows per model (int or fraction): faster on big data |
| `clip` | `True` | Clip numeric imputations to the observed range |
| `random_state` | `None` | Seed |
| `verbose` | `0` | `1` = per round, `2` = per model |

Fitted attributes: `n_iter_`, `history_`, `imputation_order_`, `imputation_sequence_`,
`categorical_mask_`, `n_features_in_`, `feature_names_in_`.

### Notes

- DataFrames come back as DataFrames with the original dtypes (including `category`,
  `boolean` and `Int64`); NumPy arrays come back as arrays.
- Observed values are never changed. Only cells that were NaN are filled.
- Categorical columns use each library's native categorical support (CatBoost `cat_features`,
  pandas `category` dtype for LightGBM and XGBoost).
- `transform` replays every fitted model, so all of them are kept in memory.
  For large models, lower `max_iter` or use early stopping via `tol`.
- Works inside `sklearn.pipeline.Pipeline`, supports `clone`, `get_params` and pickling.

## Evaluating imputation quality

```python
from boost_imputers import make_missing, imputation_error

X_missing, mask = make_missing(X_complete, frac=0.2, random_state=0)  # hide 20% of values
X_filled = LightGBMImputer().fit_transform(X_missing)
imputation_error(X_complete, X_filled, mask)  # RMSE / NRMSE per numeric column, error rate per categorical
```

On the diabetes dataset with 20% of values hidden (`examples/benchmark.py`), mean NRMSE:

| mean/mode | CatBoost | XGBoost | LightGBM |
|---|---|---|---|
| 0.969 | 0.658 | 0.690 | 0.689 |

## Writing your own backend

Subclass `BaseBoostingImputer` and implement `_make_estimator`:

```python
from boost_imputers import BaseBoostingImputer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

class HGBImputer(BaseBoostingImputer):
    def _make_estimator(self, task, params):
        cls = HistGradientBoostingClassifier if task == "classification" else HistGradientBoostingRegressor
        return cls(categorical_features="from_dtype", **params)
```

Override `_prepare_features` / `_fit_estimator` if the library needs a special feature format
or fit arguments.

## Development

```bash
pip install -e ".[dev]"
pytest
```
