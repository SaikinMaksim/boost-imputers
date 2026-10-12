"""Helpers for benchmarking imputers: hide known values, then score how well they come back."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.utils.validation import check_random_state


def make_missing(X: pd.DataFrame, frac: float = 0.1, columns=None, random_state=None):
    """Hide a random share of the observed values (missing completely at random).

    Returns
    -------
    X_missing : DataFrame
        Copy of ``X`` with the selected cells set to NaN.
    mask : DataFrame of bool
        True where a value was hidden by this function.
    """
    if not 0 <= frac < 1:
        raise ValueError(f"frac must be in [0, 1), got {frac}.")
    rng = check_random_state(random_state)
    X = pd.DataFrame(X)
    columns = list(X.columns) if columns is None else list(columns)
    mask = pd.DataFrame(False, index=X.index, columns=X.columns)
    # Each cell is hidden with probability frac, independently of the others (MCAR).
    # Cells that are already empty are left out of the mask: there is no true value
    # to compare their imputation against.
    mask[columns] = (rng.random_sample((len(X), len(columns))) < frac) & X[columns].notna().to_numpy()
    X_missing = X.copy()
    for col in columns:
        X_missing.loc[mask[col], col] = np.nan
    return X_missing, mask


def imputation_error(X_true: pd.DataFrame, X_imputed: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Score imputed cells against the true values, column by column.

    Numeric columns get RMSE and NRMSE (RMSE / std of the true column);
    categorical columns get the error rate (share of wrong labels).
    """
    X_true, X_imputed, mask = pd.DataFrame(X_true), pd.DataFrame(X_imputed), pd.DataFrame(mask)
    rows = []
    for col in mask.columns:
        # Compare only the hidden cells: the imputer didn't change any others.
        cells = mask[col].to_numpy(dtype=bool)
        if not cells.any():
            continue
        true = X_true[col].to_numpy()[cells]
        pred = X_imputed[col].to_numpy()[cells]
        dtype = X_true[col].dtype
        if is_numeric_dtype(dtype) and not is_bool_dtype(dtype):
            rmse = float(np.sqrt(np.mean((pred.astype(float) - true.astype(float)) ** 2)))
            # NRMSE = RMSE / std. About 1.0 is what mean imputation gives,
            # 0 is perfect recovery. Makes columns on different scales comparable.
            std = float(X_true[col].std()) or 1.0
            rows.append({"column": col, "n_missing": int(cells.sum()), "rmse": rmse, "nrmse": rmse / std})
        else:
            error = float(np.mean(pred != true))
            rows.append({"column": col, "n_missing": int(cells.sum()), "error_rate": error})
    return pd.DataFrame(rows).set_index("column")
