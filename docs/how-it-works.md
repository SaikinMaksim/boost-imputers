# How boost-imputers works

This document walks through the algorithm behind `CatBoostImputer`, `XGBoostImputer` and
`LightGBMImputer`: what happens during `fit`, how `transform` reuses the fitted models,
and why the code is written the way it is.

## The idea

Each column with missing values in turn becomes the **target**, and all other columns
become the **features**. A model is trained on the rows where the target is known and
predicts it where it is missing. The pass over the columns repeats several times: each
round sees better imputations from the round before.

This approach is called **MICE** (Multiple Imputation by Chained Equations). scikit-learn's
`IterativeImputer` works the same way. This package uses gradient boosting as the model
and handles categorical columns natively.

## A toy example

```
     age  income  city
0     25      30     A
1    NaN      50     B
2     40     NaN   NaN
3     35      45     A
```

**Step 0, initial fill.** `age` = 33.3 (mean), `income` = 41.7 (mean), `city` = A (mode).
The table now has no NaN, so models can be trained.

**Round 1.** Columns are visited in order `age → income → city`:

1. **age**: a regressor is trained on rows 0, 2, 3 (where age is known) with features
   `income` and `city`. In row 2 those features still hold the initial 41.7 and A.
   The model predicts age for row 1, and the value is written back immediately.
2. **income**: trained on rows 0, 1, 3. In row 1 the age is now the prediction from
   step 1, not the mean 33.3. The prediction goes into row 2.
3. **city**: a classifier is trained on rows 0, 1, 3 and predicts the city for row 2.
   It already sees the predicted `income`.

**Round 2 and later.** The same steps repeat, but the features now hold the previous
round's imputations instead of rough means, so quality improves from round to round.

Because each prediction is written back immediately, the next column in the same round
already sees the fresh values rather than last round's.

## Step by step through the code

All of the logic lives in [`base.py`](../src/boost_imputers/base.py), in
`BaseBoostingImputer._fit_transform`. The method is split into six commented steps.

### Step 1. Prepare the data

- **`_validate_input`** accepts a DataFrame or a NumPy array and turns it into a DataFrame.
  It remembers the column names and count so `transform` can check that it receives the
  same kind of data.
- **`_resolve_categorical`** decides which columns are categorical. By default these are
  `object`, string, `category` and `bool` columns. Override it with `categorical_features`:
  for example, a `Gender` column encoded as 0/1 should be marked categorical explicitly.
- **`_observed_categories`** fixes each categorical column's list of categories during `fit`.
  Models work with category codes, so "A" must get the same code in `transform` as it did
  in training. A category never seen during `fit` becomes NaN in the features.
- **`_encode`** builds the working copy `work`: numbers become `float64`, categories become
  pandas `category` dtype with the fixed list. Columns are renamed to `f0, f1, …` because
  LightGBM and XGBoost reject column names containing characters such as `[`, `<` or `"`.
- **`mask = df.isna()`** is the map of the original missing cells and the central data
  structure of the algorithm. It never changes and decides which cells may be overwritten.
  Observed values are never touched.

### Step 1b. Statistics

`_compute_statistics` stores, for each column:

- the **initial value**: mean, median, most frequent or constant for numbers; always the
  mode for categories;
- **min/max**, used when `clip=True` so the model can't impute an age of −5 or 300
  (trees rarely extrapolate, but this is a safety net);
- the **standard deviation**, used to normalise the convergence metric (see step 5).

### Step 2. Which columns, and in what order

- **`_resolve_targets`**: by default, a model is built for every column that has at least
  one NaN. A fully empty column has nothing to train on, so it is skipped with a warning.
- **`_order_targets`**: the default `ascending` order visits the columns with the fewest
  missing values first. Their imputations are the most reliable, and they improve the
  features for columns with more gaps.

### Step 3. Initial fill

`_initial_fill` writes the initial values into the missing cells. With
`initial_strategy="none"` the cells stay NaN instead: all three libraries handle missing
feature values natively, and sometimes that works better than plugging in the mean.

### Step 4. The main loop

```python
for it in range(max_iter):
    previous = imputed values before the round
    for j in imputation_order_:
        model = _fit_column(work, j, ...)   # train
        _impute_column(work, model, ...)    # predict and write back
        imputation_sequence_.append(model)  # keep for transform
    change = _changes(previous, work)       # how far everything moved
    if change < tol: break
```

**`_fit_column`** trains the model for one column:

- It uses only the rows where column `j` was **originally** known (`~mask[:, j]`), never the
  model's own earlier predictions. With `max_samples` set, it trains on a random subsample
  of those rows, which speeds things up on large data.
- `y` is column `j`; `X` is every other column, holding its current imputations. This is
  what makes the method iterative.
- **Categorical targets:** codes are renumbered to `0..k-1`. XGBoost requires consecutive
  labels, and the training rows may lack some categories (codes such as {0, 2, 5}).
  The `classes` array stores the mapping back.
- **Constant targets:** if a column has a single unique value, no model is trained and the
  constant is stored instead. Otherwise CatBoost fails with "All train targets are equal".
- **Parameters** (`_estimator_params`) are merged with this priority:
  backend defaults < `params` < `regressor_params` / `classifier_params`. If `random_state`
  is set, each model gets its own seed, derived from it deterministically.

**`_impute_column`** predicts the missing cells:

- **Regression:** `predict`, then clip to `[min, max]`.
- **Classification:** `predict_proba().argmax()` → `classes` → category label. This works
  the same way for all three libraries, whose `predict` methods return different shapes
  (CatBoost returns an array of shape `(n, 1)`).

### Step 5. Convergence

`_changes` compares the imputations before and after the round:

- **numeric column:** `RMSE(new − old) / std`. Normalising makes columns comparable:
  without it, a shift of 1000 in income and 1 year in age can't be put on the same scale;
- **categorical column:** the share of labels that changed;
- `change` is the mean over columns. It is recorded in `history_`, and the loop stops once
  `change < tol`.

Boosting almost never converges to zero, because each retrained model gives slightly
different predictions. The change levels off at a plateau instead (around 0.15–0.18 in the
benchmark). That is why, with `tol=1e-3`, you will often see a "did not converge" warning.
It is harmless: look at `history_` and tune `max_iter` or `tol` (for example `tol=0.05`).
On Kaggle-style tabular data, 3–5 rounds are usually enough.

### Step 6. Building the output

`_build_output` copies the original DataFrame and writes values only into the cells from
`mask`. It then restores the dtypes: codes back to category labels, `Int64` columns rounded
to whole numbers, `boolean`/`category` columns keep their dtypes. If the input was a NumPy
array, a NumPy array is returned.

## How `transform` works on new data

Nothing is trained in `transform`:

1. The data is encoded with the categories from `fit`, and a new `mask` is built.
2. The initial fill uses the statistics from `fit` (the training means, not the new data's).
3. Every model in `imputation_sequence_` is replayed in the same order: round 1 column a,
   round 1 column b, …, round N. Columns with no missing values in the new data are skipped.

**Why replay every round instead of only the last one.** The last round's models were
trained on features whose gaps were already well imputed. Feeding them mean-filled data
would give them a different feature distribution, and quality would drop. Replaying the
whole path reproduces exactly the same process, so `fit_transform(X)` equals
`fit(X).transform(X)` (a test checks this).

The cost is memory: every model is kept. The number of models is
`n_iter_ × number of columns with missing values`; for example, 10 columns × 10 rounds = 100 models.

## How the backends differ

The shared code calls three hooks: `_make_estimator`, `_prepare_features` and `_fit_estimator`.

| | Models | Categories | Notes |
|---|---|---|---|
| **CatBoost** ([`catboost.py`](../src/boost_imputers/catboost.py)) | `CatBoostRegressor/Classifier` | integer codes (−1 = NaN) + `cat_features` | CatBoost doesn't accept NaN in categorical features, hence the codes; `allow_writing_files=False` prevents a `catboost_info/` folder |
| **XGBoost** ([`xgboost.py`](../src/boost_imputers/xgboost.py)) | `XGBRegressor/Classifier` | `category` dtype, `enable_categorical=True` | categories are renamed to 0..k-1 because XGBoost only accepts str or int labels (it fails on `bool`) |
| **LightGBM** ([`lightgbm.py`](../src/boost_imputers/lightgbm.py)) | `LGBMRegressor/Classifier` | `category` dtype, natively | only `verbose=-1` |

All three libraries handle NaN in numeric features on their own; `initial_strategy="none"`
relies on this.

To add another library, subclass `BaseBoostingImputer` and implement `_make_estimator`
(see the README for a `HistGradientBoosting` example).

## Kaggle tips

### Prepare the frame

- **Drop the target and ID columns before imputing.** If you run `pd.concat([train, test])`
  with the target included, it is NaN for every test row and the imputer will "fill it in".
  The target then also acts as a feature for the other columns, so it leaks into their
  imputations. An `id` column is pure noise for the imputation models.
- **Impute train + test together** (and the original dataset, if the competition is built on
  one). The models see more rows to learn the relationships between columns. This is
  transductive and fine on Kaggle; in production, fit on training data only and call
  `transform` on new data.
- **Mark numeric codes as categorical.** Columns such as `Gender = 0/1` or
  `Education = 1..5` are detected as numeric. Pass them in `categorical_features` so they
  are imputed by a classifier and never get values like 0.37.

```python
features = [c for c in train.columns if c not in ("id", TARGET)]
full = pd.concat([train[features], test[features]], ignore_index=True)

imputer = LightGBMImputer(
    max_iter=5,
    tol=0.02,
    categorical_features=["Gender", "Education"],
    params={"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31},
    random_state=42,
    verbose=1,
)
full_imputed = imputer.fit_transform(full)
train_imp = full_imputed.iloc[: len(train)]
test_imp = full_imputed.iloc[len(train) :]
```

### Keep the missingness signal

The fact that a value is missing is often informative in itself. Add indicator columns
before imputing and give them to your downstream model:

```python
for col in features:
    if full[col].isna().any():
        full[f"{col}_was_missing"] = full[col].isna().astype("int8")
```

This snippet adds the indicators before imputing, so the imputation models can use them
too. To keep them out of the imputation, compute them from `full` but join them to the
imputed frame afterwards. Both are valid, so compare them in CV.

### Measure, don't guess

- **Benchmark the imputer on your own data.** Hide some observed values with
  `make_missing`, impute, and score with `imputation_error`. Compare against
  `max_iter=0`, which is plain mean/mode imputation:

  ```python
  from boost_imputers import imputation_error, make_missing

  hidden, mask = make_missing(full, frac=0.1, random_state=0)
  for name, imp in {"mean": LightGBMImputer(max_iter=0), "lgbm": imputer}.items():
      print(name, imputation_error(full, imp.fit_transform(hidden), mask)["nrmse"].mean())
  ```

  Cells that were already missing in `full` are never part of `mask`, so the score only
  uses values with a known answer.
- **Always confirm with the downstream CV score.** Better imputation error does not
  guarantee a better leaderboard score. Gradient-boosted downstream models handle NaN
  natively, so "leave NaN" is a strong baseline: compare it against imputed features
  with the same CV folds.

### Speed

- **Start with LightGBM.** It is usually the fastest of the three; switch to CatBoost when
  you have many categorical columns.
- **Use few rounds.** `max_iter=3`–`5` with `tol=0.01`–`0.05` captures most of the gain.
  Check `imputer.history_`: once `change` levels off, more rounds only cost time.
- **Use smaller models than for the final prediction.** `n_estimators` 200–500 with
  `learning_rate=0.05` is a good start. Imputation models don't need to be tuned to the
  last decimal.
- **Subsample big data.** `max_samples=50_000` (or a fraction such as `0.3`) trains each
  model on a sample of the observed rows.
- **Use the GPU** through the backend parameters, e.g.
  `CatBoostImputer(params={"task_type": "GPU"})` or
  `XGBoostImputer(params={"device": "cuda"})`.
- **Restrict the columns.** If only a few columns matter, pass `columns=[...]`: other
  columns still serve as features but get no models of their own.

### When imputation *is* the competition

Some competitions (for example Playground Series S3E15, "Feature Imputation with a Heat
Flux Dataset") ask you to predict the missing values of one column. Then:

- Model only that column with `columns=["target_col"]` and give it stronger settings
  through `regressor_params` or `classifier_params`. The other columns are still imputed
  with the initial fill, so they can serve as features.
- Better still, run a full imputer first to fill the other features, then a second, heavier
  one for the target column.
- **Ensemble the backends.** Average the imputations of CatBoost, XGBoost and LightGBM;
  their errors are only partly correlated, so the average is usually better than each one:

  ```python
  kw = dict(max_iter=5, params={"n_estimators": 500, "learning_rate": 0.05}, random_state=42)
  imputers = [CatBoostImputer(**kw), XGBoostImputer(**kw), LightGBMImputer(**kw)]
  results = [imp.fit_transform(full) for imp in imputers]
  full["target_col"] = sum(r["target_col"] for r in results) / len(results)
  ```

  Only the originally missing cells differ between the three results; observed values are
  identical, so the average leaves them unchanged.
- **Try `initial_strategy="none"`.** Keeping NaN in the features instead of the mean often
  helps when the missing values are not random.

### Reproducibility

Set `random_state`. Every model's seed is derived from it, so the same data and parameters
give the same imputations, and CV scores stay comparable between experiments.
