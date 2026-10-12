"""Iterative imputer backed by LightGBM."""

from __future__ import annotations

from .base import BaseBoostingImputer


class LightGBMImputer(BaseBoostingImputer):
    """Iterative imputer using ``LGBMRegressor`` / ``LGBMClassifier``.

    Categorical columns use LightGBM's native handling of pandas ``category`` dtype.
    See :class:`~boost_imputers.base.BaseBoostingImputer` for all parameters.

    Examples
    --------
    >>> from boost_imputers import LightGBMImputer
    >>> imputer = LightGBMImputer(params={"n_estimators": 300, "num_leaves": 31})
    >>> X_filled = imputer.fit_transform(X)  # doctest: +SKIP
    """

    # verbose=-1 silences LightGBM's info messages ("No further splits..." etc.).
    # Categories need no special preparation: LightGBM recognises the "category" dtype
    # and remembers the category list, which we keep fixed between fit and transform.
    _default_params = {"verbose": -1}

    def _make_estimator(self, task, params):
        # Imported inside the method so the package works without lightgbm installed.
        from lightgbm import LGBMClassifier, LGBMRegressor

        cls = LGBMClassifier if task == "classification" else LGBMRegressor
        return cls(**params)
