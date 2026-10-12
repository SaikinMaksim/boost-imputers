"""Iterative imputer backed by CatBoost."""

from __future__ import annotations

import pandas as pd

from .base import BaseBoostingImputer


class CatBoostImputer(BaseBoostingImputer):
    """Iterative imputer using ``CatBoostRegressor`` / ``CatBoostClassifier``.

    Categorical columns are passed to CatBoost as native ``cat_features``.
    See :class:`~boost_imputers.base.BaseBoostingImputer` for all parameters.

    Examples
    --------
    >>> from boost_imputers import CatBoostImputer
    >>> imputer = CatBoostImputer(max_iter=5, params={"iterations": 300, "depth": 6})
    >>> X_filled = imputer.fit_transform(X)  # doctest: +SKIP
    """

    # verbose=0: otherwise CatBoost prints a training log for each of the hundreds of models.
    # allow_writing_files=False: otherwise every fit creates a catboost_info/ folder.
    _default_params = {"verbose": 0, "allow_writing_files": False}
    _seed_param = "random_seed"

    def _make_estimator(self, task, params):
        # Imported inside the method so the package can be installed without catboost
        # when only XGBoost or LightGBM is used.
        from catboost import CatBoostClassifier, CatBoostRegressor

        cls = CatBoostClassifier if task == "classification" else CatBoostRegressor
        return cls(**params)

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        # CatBoost accepts categorical features only as int or str, without NaN.
        # Use the category codes: a missing value gets code -1, which CatBoost
        # treats as a separate "no value" category.
        categorical = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
        if not categorical:
            return X
        X = X.copy()  # don't modify the shared work frame
        for c in categorical:
            X[c] = X[c].cat.codes.astype("int64")
        return X

    def _fit_estimator(self, estimator, X, y, cat_columns):
        # Tell CatBoost explicitly which columns are categorical: it can't guess that from
        # int codes. The list is stored in the model, so predict doesn't need it again.
        estimator.fit(X, y, cat_features=cat_columns or None)
