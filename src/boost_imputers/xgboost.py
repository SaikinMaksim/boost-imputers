"""Iterative imputer backed by XGBoost."""

from __future__ import annotations

import pandas as pd

from .base import BaseBoostingImputer


class XGBoostImputer(BaseBoostingImputer):
    """Iterative imputer using ``XGBRegressor`` / ``XGBClassifier``.

    Categorical columns use XGBoost's native categorical support
    (``enable_categorical=True`` with the ``hist`` tree method).
    See :class:`~boost_imputers.base.BaseBoostingImputer` for all parameters.

    Examples
    --------
    >>> from boost_imputers import XGBoostImputer
    >>> imputer = XGBoostImputer(params={"n_estimators": 300, "max_depth": 6})
    >>> X_filled = imputer.fit_transform(X)  # doctest: +SKIP
    """

    # enable_categorical=True: XGBoost handles pandas "category" columns natively
    # (only with tree_method="hist" or "approx").
    _default_params = {"tree_method": "hist", "enable_categorical": True}

    def _make_estimator(self, task, params):
        # Imported inside the method so the package works without xgboost installed.
        from xgboost import XGBClassifier, XGBRegressor

        cls = XGBClassifier if task == "classification" else XGBRegressor
        return cls(**params)

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        # XGBoost accepts category labels only as str or int and fails on bool and float
        # (e.g. categories [False, True]). Rename the categories to their positions
        # 0..k-1: the codes themselves don't change, NaN stays NaN and XGBoost treats it as missing.
        categorical = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
        if not categorical:
            return X
        X = X.copy()  # don't modify the shared work frame
        for c in categorical:
            X[c] = X[c].cat.rename_categories(range(len(X[c].cat.categories)))
        return X
