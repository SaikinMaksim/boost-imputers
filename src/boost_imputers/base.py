"""Core iterative (MICE-style) imputation algorithm shared by all boosting backends.

Algorithm
---------
1. Missing cells are pre-filled with a simple statistic (mean / median / mode),
   or left as NaN (``initial_strategy="none"``) since boosting models handle NaN natively.
2. For ``max_iter`` rounds, every column with missing values is visited in turn:
   a regressor (numeric column) or classifier (categorical column) is trained on the
   rows where that column was observed, using all other columns as features, and its
   predictions overwrite the missing cells.
3. After each round the change of the imputed values is measured; the loop stops
   early once it drops below ``tol``.

All fitted models are kept in ``imputation_sequence_`` so ``transform`` can replay
exactly the same procedure on new data.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_timedelta64_dtype,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_is_fitted, check_random_state

_INITIAL_STRATEGIES = ("mean", "median", "most_frequent", "constant", "none")
_IMPUTATION_ORDERS = ("ascending", "descending", "roman", "arabic", "random")
# Libraries name the seed differently. If the user put any of these keys into params,
# we don't inject our own seed, to avoid a conflict
# (CatBoost fails when both random_seed and random_state are set).
_SEED_ALIASES = ("random_state", "random_seed", "seed")


@dataclass
class _ColumnModel:
    """A model fitted for one column during one imputation round."""

    index: int  # position of the column this model imputes
    estimator: Any = None  # fitted boosting model (None when the column is constant)
    # For a classifier: the model predicts a class number 0..k-1,
    # and classes[number] gives back the original category code.
    classes: np.ndarray | None = None
    # If the column has a single value in the training rows, no model is needed:
    # we simply fill with that value (CatBoost fails on a constant target).
    constant: Any = None


class BaseBoostingImputer(TransformerMixin, BaseEstimator):
    """Iterative imputer driven by gradient-boosting models.

    Subclasses only implement :meth:`_make_estimator` and, if needed,
    :meth:`_prepare_features` / :meth:`_fit_estimator`.

    Parameters
    ----------
    max_iter : int, default=10
        Maximum number of imputation rounds over all columns.
    tol : float, default=1e-3
        Early-stopping threshold on the mean change of imputed values between rounds
        (RMSE divided by the column's std for numeric columns, share of changed labels
        for categorical columns). ``None`` disables early stopping.
    initial_strategy : {"mean", "median", "most_frequent", "constant", "none"}, default="mean"
        How missing cells are pre-filled before the first round. Categorical columns
        always use the most frequent value. ``"none"`` keeps NaN and relies on the
        boosting library's native missing-value handling.
    fill_value : float, default=0.0
        Value for numeric columns when ``initial_strategy="constant"``.
    imputation_order : {"ascending", "descending", "roman", "arabic", "random"}, default="ascending"
        Order in which columns are imputed: by number of missing values
        (ascending = fewest first), by position (roman = left to right) or random.
    columns : list, optional
        Columns (labels or positions) to model. By default every column that has
        missing values during ``fit``. Other columns that are missing in ``transform``
        only receive the initial fill.
    categorical_features : list or array of bool, optional
        Columns (labels, positions or a boolean mask) to treat as categorical.
        By default object, string, category and bool columns are categorical.
    params : dict, optional
        Parameters passed to every model (regressors and classifiers).
    regressor_params, classifier_params : dict, optional
        Extra parameters for regressors / classifiers only (override ``params``).
    max_samples : int or float, optional
        Subsample the observed rows used to train each model: an absolute number
        or a fraction. Speeds up imputation of large datasets.
    clip : bool, default=True
        Clip numeric imputations to the range observed during ``fit``.
    random_state : int, RandomState instance or None, default=None
        Seeds the models and the random parts of the algorithm.
    verbose : int, default=0
        0 = silent, 1 = one line per round, 2 = one line per fitted model.
    """

    # Backend subclasses override these two attributes:
    # the parameters every model receives (e.g. silent mode)
    # and the name of the seed parameter in that library.
    _default_params: dict = {}
    _seed_param: str = "random_state"

    def __init__(
        self,
        *,
        max_iter: int = 10,
        tol: float | None = 1e-3,
        initial_strategy: str = "mean",
        fill_value: float = 0.0,
        imputation_order: str = "ascending",
        columns: list | None = None,
        categorical_features: list | None = None,
        params: dict | None = None,
        regressor_params: dict | None = None,
        classifier_params: dict | None = None,
        max_samples: int | float | None = None,
        clip: bool = True,
        random_state=None,
        verbose: int = 0,
    ):
        # scikit-learn rule: __init__ only stores the parameters as given, with no checks
        # or computation. Otherwise clone(), get_params() and GridSearchCV break.
        # Parameters are validated in fit (_validate_params).
        self.max_iter = max_iter
        self.tol = tol
        self.initial_strategy = initial_strategy
        self.fill_value = fill_value
        self.imputation_order = imputation_order
        self.columns = columns
        self.categorical_features = categorical_features
        self.params = params
        self.regressor_params = regressor_params
        self.classifier_params = classifier_params
        self.max_samples = max_samples
        self.clip = clip
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------ backend hooks
    # The three hooks through which the shared algorithm talks to a specific library.
    # CatBoostImputer / XGBoostImputer / LightGBMImputer override only these.

    def _make_estimator(self, task: str, params: dict):
        """Return an unfitted estimator for ``task`` ("regression" or "classification")."""
        raise NotImplementedError

    def _prepare_features(self, X: pd.DataFrame):
        """Convert the feature frame into what the backend expects.

        Categorical columns arrive as pandas ``category`` dtype with fixed categories.
        """
        return X

    def _fit_estimator(self, estimator, X, y: np.ndarray, cat_columns: list[str]) -> None:
        estimator.fit(X, y)

    # ------------------------------------------------------------------ public API

    def fit(self, X, y=None):
        """Fit the imputer on ``X``."""
        # Fitting and imputing are the same process: round-2 models train on data
        # imputed in round 1. So fit simply discards the result.
        # y is unused; it is only there for Pipeline compatibility.
        self._fit_transform(X)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit the imputer on ``X`` and return ``X`` with missing values imputed."""
        # Overridden to avoid doing the work twice: sklearn's default
        # fit_transform would call fit() and then transform().
        return self._fit_transform(X)

    def transform(self, X):
        """Impute missing values in ``X`` by replaying the fitted models."""
        check_is_fitted(self, "imputation_sequence_")
        df, is_frame = self._validate_input(X, reset=False)
        # Encode new data with the same categories that were seen in fit.
        work = self._encode(df)
        mask = df.isna().to_numpy()
        # Initial fill uses the statistics from fit (training means, not the new data's).
        self._initial_fill(work, mask)
        # Nothing is trained here: we replay ALL stored models in the same order as in fit
        # (round 1: a, b, c; round 2: a, b, c; ...). The last round alone is not enough:
        # its models were trained on features that were already well imputed, while
        # here the features are still filled with means.
        for model in self.imputation_sequence_:
            rows = mask[:, model.index]
            if rows.any():  # the column may have no missing values in the new data
                self._impute_column(work, model, rows)
        return self._build_output(df, work, mask, is_frame)

    def get_feature_names_out(self, input_features=None):
        # The imputer doesn't change the set of columns, so output names equal input names.
        # Needed by Pipeline/ColumnTransformer and set_output(transform="pandas").
        check_is_fitted(self, "n_features_in_")
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if hasattr(self, "feature_names_in_"):
            return np.asarray(self.feature_names_in_, dtype=object)
        return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)

    def __sklearn_tags__(self):
        # Tell sklearn (>= 1.6) that NaN, strings and categories are valid input;
        # otherwise its checks would reject such data before our code runs.
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.categorical = True
        tags.input_tags.string = True
        return tags

    def _more_tags(self):  # scikit-learn < 1.6
        return {"allow_nan": True, "X_types": ["2darray", "categorical", "string"]}

    # ------------------------------------------------------------------ fitting

    def _fit_transform(self, X):
        # The main method: the whole algorithm from start to finish.
        self._validate_params()
        # One random generator for the whole fit: model seeds, max_samples subsamples
        # and the "random" order all come from it. With a fixed random_state
        # the result is reproducible.
        rng = check_random_state(self.random_state)
        df, is_frame = self._validate_input(X, reset=True)
        if df.shape[1] < 2:
            # A column is predicted from the other columns; without them there is nothing to predict from.
            raise ValueError("At least two columns are required for iterative imputation.")

        # --- Step 1. Prepare the data -------------------------------------------------
        self.categorical_mask_ = self._resolve_categorical(df)
        # Internal names f0, f1, ...: LightGBM and XGBoost fail on column names with
        # special characters ("[", "<", quotes), and user column names can be anything.
        self._names = [f"f{i}" for i in range(df.shape[1])]
        # The category list is fixed once during fit, so "A" always gets the same code
        # in both fit and transform.
        self._categories = {
            j: _observed_categories(df.iloc[:, j])
            for j in range(df.shape[1])
            if self.categorical_mask_[j]
        }
        # work is the working copy in which the models overwrite missing cells.
        work = self._encode(df)
        # mask is the map of the ORIGINAL missing cells (True = was NaN). It never changes
        # and defines which cells may be overwritten. Observed values are never touched.
        mask = df.isna().to_numpy()
        self._compute_statistics(work)

        # --- Step 2. Which columns to impute, and in what order ------------------------
        targets = self._resolve_targets(df, mask)
        self.imputation_order_ = self._order_targets(targets, mask, rng)

        # --- Step 3. Initial fill (mean / mode) so the models have something to start from
        self._initial_fill(work, mask)

        # --- Step 4. Iterations -------------------------------------------------------
        self.imputation_sequence_ = []  # every fitted model, in order, for transform
        self.history_ = []  # change of the imputed values per round
        self.n_iter_ = 0
        converged = False
        for it in range(self.max_iter):
            start = time.perf_counter()
            # Snapshot the imputed values before the round, to measure how far they move.
            previous = {j: self._values_at(work, j, mask[:, j]) for j in self.imputation_order_}
            for j in self.imputation_order_:
                # Train a "column j from all the others" model on rows where j is known...
                model = self._fit_column(work, j, mask[:, j], rng)
                self.imputation_sequence_.append(model)
                # ...and write its predictions into work right away. The next column in
                # this same round already sees the fresh values, not last round's.
                if mask[:, j].any():
                    self._impute_column(work, model, mask[:, j])

            # --- Step 5. Convergence check ---------------------------------------------
            per_column = self._changes(previous, work, mask)
            change = float(np.mean(list(per_column.values()))) if per_column else 0.0
            self.n_iter_ = it + 1
            self.history_.append(
                {
                    "iteration": it + 1,
                    "change": change,
                    "columns": {self._label(j): c for j, c in per_column.items()},
                }
            )
            if self.verbose:
                print(
                    f"[{type(self).__name__}] iteration {it + 1}/{self.max_iter}: "
                    f"change={change:.6f} ({time.perf_counter() - start:.1f}s)"
                )
            # The imputed values have almost stopped changing: more rounds won't help, stop.
            if self.tol is not None and change < self.tol:
                converged = True
                break

        # Boosting rarely converges to zero (each retrained model gives slightly different
        # predictions), so this is a warning, not an error: the result is usable.
        if not converged and self.tol is not None and self.max_iter > 0 and self.imputation_order_:
            warnings.warn(
                f"{type(self).__name__} did not converge after {self.max_iter} iterations "
                f"(last change {self.history_[-1]['change']:.6f} > tol={self.tol}).",
                ConvergenceWarning,
                stacklevel=3,
            )
        # --- Step 6. Copy the imputations into a copy of the input, with the original dtypes
        return self._build_output(df, work, mask, is_frame)

    def _validate_params(self) -> None:
        # Catch bad parameters before a long fit, not ten minutes into it.
        if not isinstance(self.max_iter, (int, np.integer)) or self.max_iter < 0:
            raise ValueError(f"max_iter must be a non-negative integer, got {self.max_iter!r}.")
        if self.initial_strategy not in _INITIAL_STRATEGIES:
            raise ValueError(
                f"initial_strategy must be one of {_INITIAL_STRATEGIES}, got {self.initial_strategy!r}."
            )
        if self.imputation_order not in _IMPUTATION_ORDERS:
            raise ValueError(
                f"imputation_order must be one of {_IMPUTATION_ORDERS}, got {self.imputation_order!r}."
            )
        if self.max_samples is not None and not (
            (isinstance(self.max_samples, float) and 0 < self.max_samples <= 1)
            or (isinstance(self.max_samples, (int, np.integer)) and self.max_samples > 0)
        ):
            raise ValueError(
                "max_samples must be a positive int or a float in (0, 1], "
                f"got {self.max_samples!r}."
            )

    def _validate_input(self, X, reset: bool) -> tuple[pd.DataFrame, bool]:
        # Internally everything works on a DataFrame. is_frame records what came in,
        # so the result can be returned as the same type.
        is_frame = isinstance(X, pd.DataFrame)
        if is_frame:
            df = X
        else:
            arr = np.asarray(X)
            if arr.ndim != 2:
                raise ValueError(f"Expected a 2D array, got an array with shape {arr.shape}.")
            # A mixed-type array arrives as object; infer_objects gives numeric columns
            # a numeric dtype again, otherwise they would be taken for categories.
            df = pd.DataFrame(arr).infer_objects()

        # reset=True means fit: remember the shape of the data.
        if reset:
            self.n_features_in_ = df.shape[1]
            # sklearn convention: feature_names_in_ is set only when all column names are strings.
            if is_frame and all(isinstance(c, str) for c in df.columns):
                self.feature_names_in_ = np.asarray(df.columns, dtype=object)
            elif hasattr(self, "feature_names_in_"):
                del self.feature_names_in_  # left over from an earlier fit on another DataFrame
            return df, is_frame

        # reset=False means transform: the data must match what was seen in fit.
        if df.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {df.shape[1]} features, but {type(self).__name__} "
                f"was fitted with {self.n_features_in_} features."
            )
        if is_frame and hasattr(self, "feature_names_in_"):
            missing = [c for c in self.feature_names_in_ if c not in df.columns]
            if missing:
                raise ValueError(f"Columns seen during fit are missing from X: {missing}.")
            # Models refer to columns by position, so restore the fit-time order:
            # columns passed in a different order still work correctly.
            df = df[list(self.feature_names_in_)]
        return df, is_frame

    def _column_positions(self, df: pd.DataFrame, spec, name: str) -> list[int]:
        """Translate labels / positions / boolean mask into column positions."""
        # Lets columns / categorical_features be given in any convenient form:
        # ["city", "age"], [3, 0] or a mask [False, True, ...].
        spec = list(spec)
        if spec and all(isinstance(s, (bool, np.bool_)) for s in spec):
            if len(spec) != df.shape[1]:
                raise ValueError(f"Boolean mask {name} must have one entry per column.")
            return [j for j, flag in enumerate(spec) if flag]
        labels = list(df.columns)
        positions = []
        for s in spec:
            # Look it up as a column name first, then as a position: for numpy input
            # the column names are the numbers 0, 1, 2... themselves.
            if s in labels:
                positions.append(labels.index(s))
            elif isinstance(s, (int, np.integer)) and 0 <= s < df.shape[1]:
                positions.append(int(s))
            else:
                raise ValueError(f"Unknown column {s!r} in {name}.")
        return positions

    def _resolve_categorical(self, df: pd.DataFrame) -> np.ndarray:
        # Returns a mask: True = categorical column (imputed by a classifier),
        # False = numeric (imputed by a regressor).
        # Dates are not supported: they can't be averaged, nor treated as categories.
        for col in df.columns:
            dtype = df[col].dtype
            if is_datetime64_any_dtype(dtype) or is_timedelta64_dtype(dtype):
                raise TypeError(
                    f"Column {col!r} has dtype {dtype}; convert datetimes to numeric features first."
                )
        if self.categorical_features is None:
            # Auto-detection: everything non-numeric (strings, object, category) plus bool.
            # Numeric codes such as Gender = 0/1 are not detected this way; list them
            # explicitly in categorical_features.
            return np.array(
                [
                    is_bool_dtype(df[c].dtype) or not is_numeric_dtype(df[c].dtype)
                    for c in df.columns
                ]
            )
        flags = np.zeros(df.shape[1], dtype=bool)
        flags[self._column_positions(df, self.categorical_features, "categorical_features")] = True
        return flags

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Numeric columns -> float64, categorical -> category dtype with fixed categories."""
        # Brings the data into one internal format that every backend can work with.
        data = {}
        for j, name in enumerate(self._names):
            series = df.iloc[:, j]
            if self.categorical_mask_[j]:
                # Going through object makes any input type (str, bool, category, numbers)
                # match the category list the same way. Values not seen during fit
                # become NaN: the model knows nothing about them.
                values = np.asarray(series.astype(object))
                data[name] = pd.Categorical(values, categories=self._categories[j])
            else:
                try:
                    # Plain float64 with np.nan. pandas nullable dtypes (Int64, Float64)
                    # store a missing value as pd.NA, which the boosters don't understand; NaN they do.
                    values = pd.to_numeric(series).to_numpy(dtype="float64", na_value=np.nan)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Column {self._label(j)!r} is not numeric; "
                        "list it in categorical_features."
                    ) from exc
                data[name] = values
        return pd.DataFrame(data)

    def _compute_statistics(self, work: pd.DataFrame) -> None:
        # Everything is computed from observed values only and stored, so that transform
        # uses the training data's statistics rather than the new data's.
        self._initial_values = {}  # what to fill missing cells with before round 1
        self._bounds = {}  # (min, max) for clip
        self._scales = {}  # std, to normalise the convergence metric
        for j, name in enumerate(self._names):
            col = work[name]
            if self.categorical_mask_[j]:
                # Categories always use the mode, whatever initial_strategy says:
                # categories have no mean. Code -1 means NaN, so it is dropped.
                codes = col.cat.codes.to_numpy()
                codes = codes[codes >= 0]
                self._initial_values[j] = int(np.bincount(codes).argmax()) if codes.size else None
                continue
            values = col.to_numpy()
            values = values[~np.isnan(values)]
            if values.size == 0:
                # The column is entirely empty: no statistics, nothing to fill it with.
                self._initial_values[j] = None
                continue
            if self.initial_strategy == "median":
                self._initial_values[j] = float(np.median(values))
            elif self.initial_strategy == "constant":
                self._initial_values[j] = float(self.fill_value)
            elif self.initial_strategy == "most_frequent":
                uniques, counts = np.unique(values, return_counts=True)
                self._initial_values[j] = float(uniques[counts.argmax()])
            else:
                self._initial_values[j] = float(values.mean())
            self._bounds[j] = (float(values.min()), float(values.max()))
            std = float(values.std())
            self._scales[j] = std if std > 0 else 1.0  # guard against division by zero

    def _resolve_targets(self, df: pd.DataFrame, mask: np.ndarray) -> list[int]:
        # Target columns are the ones we train models for.
        if self.columns is None:
            # By default: every column with at least one missing value during fit.
            targets = [j for j in range(df.shape[1]) if mask[:, j].any()]
        else:
            # dict.fromkeys removes duplicates while keeping the order.
            targets = list(dict.fromkeys(self._column_positions(df, self.columns, "columns")))
        # A fully empty column has nothing to train on: exclude it with a warning.
        # It stays as a feature (all NaN), which the boosters tolerate.
        empty = [j for j in targets if mask[:, j].all()]
        if empty:
            warnings.warn(
                f"Columns {[self._label(j) for j in empty]} have no observed values and "
                "will not be imputed.",
                UserWarning,
                stacklevel=4,
            )
        return [j for j in targets if j not in empty]

    def _order_targets(self, targets: list[int], mask: np.ndarray, rng) -> list[int]:
        counts = mask.sum(axis=0)  # number of missing values in each column
        # ascending (default): columns with the fewest missing values first.
        # Their imputations are the most reliable and give better features to columns
        # with more gaps. Ties are broken by position (j) so the result is stable.
        if self.imputation_order == "ascending":
            return sorted(targets, key=lambda j: (counts[j], j))
        if self.imputation_order == "descending":
            return sorted(targets, key=lambda j: (-counts[j], j))
        if self.imputation_order == "roman":
            return sorted(targets)
        if self.imputation_order == "arabic":
            return sorted(targets, reverse=True)
        return [targets[i] for i in rng.permutation(len(targets))]

    def _initial_fill(self, work: pd.DataFrame, mask: np.ndarray) -> None:
        # A rough first guess. Without it the first model would see NaN in every feature
        # that has gaps; the iterations later replace these values with predictions.
        if self.initial_strategy == "none":
            # Keep NaN: CatBoost, XGBoost and LightGBM handle missing values in features
            # natively, and sometimes that works better than plugging in the mean.
            return
        for j, name in enumerate(self._names):
            rows = mask[:, j]
            value = self._initial_values[j]
            if not rows.any() or value is None:
                continue
            if self.categorical_mask_[j]:
                # value is the code of the mode; the Categorical stores the label itself.
                work.loc[rows, name] = self._categories[j][value]
            else:
                work.loc[rows, name] = value

    def _estimator_params(self, task: str, rng) -> dict:
        extra = self.classifier_params if task == "classification" else self.regressor_params
        # Dict merge: keys on the right override keys on the left, so the priority is
        # backend defaults < params < regressor_params / classifier_params.
        params = {**self._default_params, **(self.params or {}), **(extra or {})}
        if self.random_state is not None and not any(k in params for k in _SEED_ALIASES):
            # Each model gets its own seed, but the whole sequence of seeds is derived
            # from random_state, so the entire process is reproducible.
            params[self._seed_param] = int(rng.randint(np.iinfo(np.int32).max))
        return params

    def _fit_column(self, work: pd.DataFrame, j: int, missing: np.ndarray, rng) -> _ColumnModel:
        start = time.perf_counter()
        name = self._names[j]
        # Train only on rows where column j was ORIGINALLY known (~missing).
        # Never train on our own earlier predictions, or the model would repeat its mistakes.
        rows = np.flatnonzero(~missing)
        if self.max_samples is not None:
            # Subsample the training rows for speed on large data:
            # float = fraction, int = absolute count. Sorting keeps the original row order.
            n = (
                max(1, int(round(self.max_samples * len(rows))))
                if isinstance(self.max_samples, float)
                else int(self.max_samples)
            )
            if n < len(rows):
                rows = np.sort(rng.choice(rows, size=n, replace=False))

        target = work[name].iloc[rows]
        if self.categorical_mask_[j]:
            task = "classification"
            codes = target.cat.codes.to_numpy()
            # Codes can have gaps, e.g. {0, 2, 5}, when some categories are absent from
            # the training rows. XGBoost requires labels to be exactly 0..k-1, so we
            # renumber: y = position of the code in classes. Back again: classes[y].
            classes = np.unique(codes)
            if len(classes) == 1:
                return _ColumnModel(j, constant=classes[0])
            y = np.searchsorted(classes, codes)
        else:
            task = "regression"
            classes = None
            y = target.to_numpy(dtype="float64")
            if np.ptp(y) == 0:  # ptp = max - min; zero means all values are identical
                return _ColumnModel(j, constant=y[0])

        # Features are all columns except the target, with their CURRENT imputations.
        # This is the essence of the iterative approach: each round's features are
        # more accurate than the previous round's.
        X = self._prepare_features(work.drop(columns=name).iloc[rows])
        estimator = self._make_estimator(task, self._estimator_params(task, rng))
        # Names of the categorical features (excluding the target), needed by CatBoost's cat_features.
        cat_columns = [
            n for k, n in enumerate(self._names) if self.categorical_mask_[k] and k != j
        ]
        self._fit_estimator(estimator, X, y, cat_columns)
        if self.verbose > 1:
            print(
                f"    {task} for {self._label(j)!r} on {len(rows)} rows "
                f"({time.perf_counter() - start:.1f}s)"
            )
        return _ColumnModel(j, estimator=estimator, classes=classes)

    def _impute_column(self, work: pd.DataFrame, model: _ColumnModel, rows: np.ndarray) -> None:
        # Predict column j for the given rows (its missing cells) and write the result into work.
        j = model.index
        name = self._names[j]
        if model.constant is not None:
            predictions = np.full(int(rows.sum()), model.constant)
        else:
            X = self._prepare_features(work.drop(columns=name).loc[rows])
            if model.classes is not None:
                # predict_proba + argmax instead of predict: all three libraries return
                # probabilities in the same format, while predict differs (CatBoost returns shape (n, 1)).
                # Probability column k corresponds to code classes[k].
                proba = np.asarray(model.estimator.predict_proba(X))
                predictions = model.classes[proba.argmax(axis=1)]
            else:
                predictions = np.asarray(model.estimator.predict(X), dtype="float64").ravel()

        if self.categorical_mask_[j]:
            # Category code -> label ("A", "B"), because a Categorical stores labels.
            work.loc[rows, name] = self._categories[j][predictions.astype(int)]
        else:
            if self.clip and j in self._bounds:
                # Stay within the range seen in the data (no negative ages and the like).
                predictions = np.clip(predictions, *self._bounds[j])
            work.loc[rows, name] = predictions

    # ------------------------------------------------------------------ convergence & output

    def _values_at(self, work: pd.DataFrame, j: int, rows: np.ndarray) -> np.ndarray:
        # Current imputations of column j in its missing cells (codes for categories,
        # which are easier to compare). A snapshot used to measure convergence.
        col = work[self._names[j]]
        if self.categorical_mask_[j]:
            return col.cat.codes.to_numpy()[rows]
        return col.to_numpy()[rows]

    def _changes(self, previous: dict, work: pd.DataFrame, mask: np.ndarray) -> dict[int, float]:
        # How far the imputations moved during the round, per column.
        changes = {}
        for j, old in previous.items():
            if old.size == 0:  # the column was modelled but has no missing values
                continue
            new = self._values_at(work, j, mask[:, j])
            if self.categorical_mask_[j]:
                # Categories: share of missing cells whose label changed.
                changes[j] = float(np.mean(old != new))
            elif np.isnan(old).any():
                # With initial_strategy="none" these cells were NaN before round 1:
                # the change is undefined, so treat it as infinite (this can't be the last round).
                changes[j] = float("inf")
            else:
                # Numbers: RMSE between old and new imputations, divided by the column's std.
                # Normalising makes columns comparable: without it, a shift of 1000 in income
                # and 1 year in age can't be compared.
                changes[j] = float(np.sqrt(np.mean((new - old) ** 2)) / self._scales[j])
        return changes

    def _build_output(self, df: pd.DataFrame, work: pd.DataFrame, mask: np.ndarray, is_frame: bool):
        # The result is built from a copy of the ORIGINAL data, not from work: the original
        # columns, index and dtypes (category, boolean, Int64, str) are kept, and only
        # the cells that were NaN change.
        out = df.copy()
        for j, name in enumerate(self._names):
            rows = mask[:, j]
            if not rows.any():
                continue
            dtype = out.iloc[:, j].dtype
            is_number = is_numeric_dtype(dtype) and not is_bool_dtype(dtype)
            if self.categorical_mask_[j]:
                imputed = np.asarray(work[name].astype(object))[rows]
                if is_number:
                    # Numeric codes listed in categorical_features (e.g. Gender = 0/1):
                    # the labels are numbers, and pandas refuses to write an object
                    # array into a numeric column, so convert them back to floats.
                    imputed = imputed.astype("float64")
            else:
                imputed = work[name].to_numpy()[rows]
            # An integer column (e.g. Int64) can't hold 3.7, so round. If NaN remain
            # (the column had nothing to be filled from), leave it as is.
            if is_number and is_integer_dtype(dtype) and not np.isnan(imputed).any():
                imputed = np.round(imputed).astype("int64")
            out.iloc[rows, j] = imputed
        return out if is_frame else out.to_numpy()

    def _label(self, j: int):
        # Column name for messages and history_: the real name if there is one, else the position.
        if hasattr(self, "feature_names_in_"):
            return self.feature_names_in_[j]
        return j


def _observed_categories(series: pd.Series) -> pd.Index:
    # A column's unique observed values are its category list.
    # Going through object handles str, bool, category and numbers the same way.
    uniques = pd.Series(np.asarray(series.dropna().astype(object))).unique()
    categories = pd.Index(list(uniques))
    try:
        # Sorting makes the category codes independent of the row order in the data.
        return categories.sort_values()
    except TypeError:  # a mix of incomparable types (e.g. str and int): keep as is
        return categories
