import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from boost_imputers import (
    CatBoostImputer,
    LightGBMImputer,
    XGBoostImputer,
    imputation_error,
    make_missing,
)

FAST = {"n_estimators": 60, "max_depth": 3, "learning_rate": 0.1}
IMPUTERS = [CatBoostImputer, XGBoostImputer, LightGBMImputer]


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", message=".*did not converge.*")
        yield


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n = 600
    a = rng.normal(size=n)
    b = 2 * a + rng.normal(scale=0.2, size=n)
    c = np.sin(a) * 3 + rng.normal(scale=0.2, size=n)
    color = np.where(a > 0.5, "red", np.where(a < -0.5, "blue", "green"))
    flag = pd.array(b > 0, dtype="boolean")
    count = pd.array(np.round(np.abs(a) * 5).astype(int), dtype="Int64")
    return pd.DataFrame({"a": a, "b": b, "c": c, "color": color, "flag": flag, "count": count})


@pytest.fixture
def missing(data):
    return make_missing(data, frac=0.15, random_state=1)


def make(cls, **kwargs):
    kwargs.setdefault("max_iter", 3)
    return cls(params=FAST, random_state=0, **kwargs)


@pytest.mark.parametrize("cls", IMPUTERS)
def test_fit_transform_fills_everything_and_keeps_observed(cls, data, missing):
    X_missing, mask = missing
    out = make(cls).fit_transform(X_missing)

    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == list(data.columns)
    assert not out.isna().any().any()
    pd.testing.assert_frame_equal(out.where(~mask), X_missing.where(~mask))
    assert set(out["color"]) <= {"red", "blue", "green"}
    assert out["count"].dtype == "Int64"
    assert out["flag"].dtype == "boolean"


@pytest.mark.parametrize("cls", IMPUTERS)
def test_beats_mean_imputation(cls, data, missing):
    X_missing, mask = missing
    boosted = imputation_error(data, make(cls).fit_transform(X_missing), mask)

    numeric = ["a", "b", "c"]
    simple = X_missing.copy()
    simple[numeric] = SimpleImputer().fit_transform(X_missing[numeric])
    baseline = imputation_error(data[numeric], simple[numeric], mask[numeric])

    assert (boosted.loc[numeric, "nrmse"] < baseline["nrmse"]).all()
    assert boosted.loc["color", "error_rate"] < 0.3


@pytest.mark.parametrize("cls", IMPUTERS)
def test_transform_new_data(cls, data, missing):
    X_missing, _ = missing
    imputer = make(cls).fit(X_missing.iloc[:400])
    out = imputer.transform(X_missing.iloc[400:])
    assert out.index.equals(X_missing.index[400:])
    assert not out.isna().any().any()
    # Column order of a DataFrame does not matter at transform time.
    shuffled = imputer.transform(X_missing.iloc[400:, ::-1])
    pd.testing.assert_frame_equal(shuffled, out)


@pytest.mark.parametrize("cls", IMPUTERS)
def test_numpy_input_and_pipeline(cls, data, missing):
    X_missing, _ = missing
    X = X_missing[["a", "b", "c"]].to_numpy()
    out = make(cls).fit_transform(X)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64 and not np.isnan(out).any()

    pipe = make_pipeline(make(cls), LinearRegression()).fit(X, data["c"])
    assert pipe.score(X, data["c"]) > 0.5


@pytest.mark.parametrize("cls", IMPUTERS)
def test_clone_and_pickle(cls, missing):
    X_missing, _ = missing
    imputer = make(cls, initial_strategy="none").fit(X_missing)
    restored = pickle.loads(pickle.dumps(imputer))
    pd.testing.assert_frame_equal(restored.transform(X_missing), imputer.transform(X_missing))
    assert clone(imputer).get_params() == imputer.get_params()


def test_fit_transform_matches_transform(missing):
    X_missing, _ = missing
    imputer = make(LightGBMImputer)
    pd.testing.assert_frame_equal(imputer.fit_transform(X_missing), imputer.transform(X_missing))


def test_early_stopping_and_history(missing):
    X_missing, _ = missing
    imputer = make(LightGBMImputer, max_iter=20, tol=0.05).fit(X_missing)
    assert imputer.n_iter_ < 20
    assert len(imputer.history_) == imputer.n_iter_
    assert imputer.history_[-1]["change"] < 0.05
    assert set(imputer.history_[0]["columns"]) == set(X_missing.columns)
    assert len(imputer.imputation_sequence_) == imputer.n_iter_ * X_missing.shape[1]


def test_columns_and_categorical_features(missing):
    X_missing, mask = missing
    imputer = make(XGBoostImputer, columns=["a", "b"], categorical_features=["count", "color", "flag"])
    out = imputer.fit_transform(X_missing)
    assert imputer.imputation_order_ == sorted([0, 1], key=lambda j: mask.iloc[:, j].sum())
    assert imputer.categorical_mask_.tolist() == [False, False, False, True, True, True]
    assert not out.isna().any().any()


@pytest.mark.parametrize("cls", IMPUTERS)
def test_numeric_codes_as_categorical(cls, data):
    # A 0/1 code stored as float (it has NaN) and listed as categorical.
    X = data[["a", "b"]].assign(gender=(data["a"] > 0).astype(float))
    X_missing, _ = make_missing(X, frac=0.15, random_state=2)
    out = make(cls, categorical_features=["gender"]).fit_transform(X_missing)
    assert out["gender"].dtype == np.float64
    assert set(out["gender"]) == {0.0, 1.0}


def test_max_samples_and_clip(data, missing):
    X_missing, _ = missing
    out = make(CatBoostImputer, max_samples=0.5).fit_transform(X_missing)
    assert out["a"].between(X_missing["a"].min(), X_missing["a"].max()).all()


def test_fully_missing_column_is_left_alone(missing):
    X_missing, _ = missing
    X = X_missing.assign(empty=np.nan)
    with pytest.warns(UserWarning, match="no observed values"):
        out = make(LightGBMImputer).fit_transform(X)
    assert out["empty"].isna().all()
    assert not out.drop(columns="empty").isna().any().any()


def test_invalid_params(missing):
    X_missing, _ = missing
    with pytest.raises(ValueError, match="initial_strategy"):
        LightGBMImputer(initial_strategy="foo").fit(X_missing)
    with pytest.raises(ValueError, match="features"):
        make(LightGBMImputer).fit(X_missing).transform(X_missing[["a", "b"]].to_numpy())
