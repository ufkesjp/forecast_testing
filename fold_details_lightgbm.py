"""
fold_details_lightgbm.py

Generates a fold_details DataFrame using a single globally-trained
LightGBM model. The model is trained once on ALL series' training
data (pooled), then used for inference on each series individually.

This file is a standalone alternative to fold_details_only.py.
It produces the same output schema so results can be compared
directly.

Usage:
    from fold_details_lightgbm import generate_fold_details

    fold_details_df = generate_fold_details(
        df, date_col="date", id_col="item_id", value_col="demand",
        eval_window=52, min_train_weeks=52, n_workers=4,
    )

    # With attribute-based fallback for items with insufficient history:
    fold_details_df = generate_fold_details(
        df, date_col="date", id_col="item_id", value_col="demand",
        group_col="category",
        eval_window=52, min_train_weeks=52, n_workers=4,
    )

Output columns:
    [id_col, "model", "category", "fold", date_col, "actual",
     "forecast", "method"]

GLOBAL LIGHTGBM APPROACH
Unlike the per-series statistical models in fold_details_only.py,
this file trains a single LightGBM model on pooled training data
from ALL sufficient-history series. This lets the model learn
cross-series patterns (e.g., "series with high CV and low ADI tend
to spike in week 48") that per-series models cannot capture.

Pipeline flow:
    1. Split items into sufficient-history / short-history
    2. Build feature matrix across ALL sufficient-history series
    3. Train ONE global LightGBM regressor
    4. Distribute series to parallel workers for inference only
    5. Aggregate fallback for short-history items (global-avg or
       group-level, same as fold_details_only.py)

Feature engineering:
    - Lag features: lag_1, lag_4, lag_13, lag_26, lag_52
    - Rolling stats: rolling_mean_4, rolling_mean_13, rolling_std_4
    - Calendar: week_of_year, month
    - Series-level stats: historical_mean, historical_cv, adi,
      demand_ratio (fraction of nonzero periods)

Inference uses recursive multi-step prediction: predict week t,
append to history, rebuild features, predict week t+1, etc.
"""

__version__ = "1.0.0"

import numpy as np
import pandas as pd
import warnings
from typing import Dict, Callable, Any, List, Optional, Tuple, Union
from multiprocessing import Pool, cpu_count
from functools import partial

# Conditional imports

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# =============================================================
# Demand Classification (ADI / CV²)
# =============================================================

ADI_THRESHOLD = 1.32
CV2_THRESHOLD = 0.49

def classify_demand(history: np.ndarray) -> str:
    """
    Classify a demand series using the Syntetos-Boylan framework.

    Returns one of: 'smooth', 'erratic', 'intermittent', 'lumpy'.
    """
    nonzero_mask = history > 0
    nonzero_values = history[nonzero_mask]

    if len(nonzero_values) == 0:
        return "lumpy"

    nonzero_indices = np.where(nonzero_mask)[0]
    if len(nonzero_indices) <= 1:
        adi = float(len(history))
    else:
        intervals = np.diff(nonzero_indices).astype(float)
        adi = intervals.mean()

    mean_nz = nonzero_values.mean()
    if mean_nz > 0:
        cv2 = (nonzero_values.std() / mean_nz) ** 2
    else:
        cv2 = 0.0

    if adi <= ADI_THRESHOLD:
        return "erratic" if cv2 > CV2_THRESHOLD else "smooth"
    else:
        return "lumpy" if cv2 > CV2_THRESHOLD else "intermittent"

# =============================================================
# LightGBM Feature Engineering
# =============================================================

FEATURE_NAMES = [
    "lag_1", "lag_4", "lag_13", "lag_26", "lag_52",
    "rolling_mean_4", "rolling_mean_13", "rolling_std_4",
    "week_of_year", "month",
    "historical_mean", "historical_cv", "adi", "demand_ratio",
]

def _compute_series_stats(history: np.ndarray) -> Tuple[float, float, float, float]:
    """Compute static series-level features."""
    hist_mean = history.mean()
    nonzero = history[history > 0]
    if len(nonzero) > 1 and hist_mean > 0:
        hist_cv = nonzero.std() / nonzero.mean()
    else:
        hist_cv = 0.0

    nonzero_mask = history > 0
    nonzero_indices = np.where(nonzero_mask)[0]
    if len(nonzero_indices) <= 1:
        adi = float(len(history))
    else:
        adi = np.diff(nonzero_indices).astype(float).mean()

    demand_ratio = nonzero_mask.mean()
    return hist_mean, hist_cv, adi, demand_ratio


def _build_feature_row(values: np.ndarray, t: int,
                       week_of_year: int, month: int,
                       hist_mean: float, hist_cv: float,
                       adi: float, demand_ratio: float) -> np.ndarray:
    """
    Build a single feature row for time step t.
    values[:t+1] must be available (all history up to and including t).
    """
    def _safe_lag(offset):
        idx = t - offset
        return values[idx] if idx >= 0 else 0.0

    lag_1 = _safe_lag(1)
    lag_4 = _safe_lag(4)
    lag_13 = _safe_lag(13)
    lag_26 = _safe_lag(26)
    lag_52 = _safe_lag(52)

    # Rolling stats over the window ending at t (exclusive of t itself)
    def _rolling(window):
        start = max(0, t - window)
        chunk = values[start:t]
        return chunk if len(chunk) > 0 else np.array([0.0])

    roll_4 = _rolling(4)
    roll_13 = _rolling(13)
    rolling_mean_4 = roll_4.mean()
    rolling_mean_13 = roll_13.mean()
    rolling_std_4 = roll_4.std() if len(roll_4) > 1 else 0.0

    return np.array([
        lag_1, lag_4, lag_13, lag_26, lag_52,
        rolling_mean_4, rolling_mean_13, rolling_std_4,
        float(week_of_year), float(month),
        hist_mean, hist_cv, adi, demand_ratio,
    ])


def _build_training_features(history: np.ndarray,
                             dates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix X and target vector y from one series'
    training history.

    We start generating rows at t=52 (so lag_52 is always defined)
    to avoid excessive zero-padding.
    """
    n = len(history)
    min_t = 52  # need at least lag_52
    if n <= min_t:
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0)

    dates_ts = pd.DatetimeIndex(dates)
    weeks = dates_ts.isocalendar().week.astype(int).values
    months = dates_ts.month.values

    hist_mean, hist_cv, adi, demand_ratio = _compute_series_stats(
        history[:min_t]
    )

    rows = []
    targets = []
    for t in range(min_t, n):
        # Update series stats with expanding window
        if t > min_t and t % 26 == 0:
            hist_mean, hist_cv, adi, demand_ratio = _compute_series_stats(
                history[:t]
            )
        row = _build_feature_row(
            history, t, weeks[t], months[t],
            hist_mean, hist_cv, adi, demand_ratio,
        )
        rows.append(row)
        targets.append(history[t])

    return np.array(rows), np.array(targets)


def _train_global_lightgbm(df: pd.DataFrame, sufficient_ids,
                           date_col: str, id_col: str,
                           value_col: str, eval_window: int):
    """
    Train a single global LightGBM model on pooled training data
    from all sufficient-history series.

    Returns the fitted model, or None if lightgbm is unavailable.
    """
    if not HAS_LIGHTGBM:
        return None

    print("  Training global LightGBM model...")
    all_X = []
    all_y = []

    sufficient_df = df[df[id_col].isin(sufficient_ids)]
    for series_id, grp in sufficient_df.groupby(id_col):
        grp = grp.sort_values(date_col)
        values = grp[value_col].values.astype(float)
        dates = pd.to_datetime(grp[date_col]).values

        # Use only the training portion
        train_values = values[:-eval_window]
        train_dates = dates[:-eval_window]

        X, y = _build_training_features(train_values, train_dates)
        if len(X) > 0:
            all_X.append(X)
            all_y.append(y)

    if not all_X:
        print("    WARNING: No training data for LightGBM. Skipping.")
        return None

    X_train = np.vstack(all_X)
    y_train = np.concatenate(all_y)

    print(f"    Training data: {X_train.shape[0]:,} rows × "
          f"{X_train.shape[1]} features from "
          f"{len(all_X):,} series")

    model = lgb.LGBMRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_jobs=1,
        verbose=-1,
        random_state=42,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)

    print("    LightGBM training complete.")
    return model


def _lightgbm_predict_series(model, history: np.ndarray,
                             dates: np.ndarray,
                             horizon: int) -> np.ndarray:
    """
    Generate recursive multi-step forecasts for one series using
    the pre-trained global LightGBM model.
    """
    if model is None:
        return np.full(horizon, np.nan)

    dates_ts = pd.DatetimeIndex(dates)
    hist_mean, hist_cv, adi, demand_ratio = _compute_series_stats(history)

    # Build extended arrays for recursive prediction
    extended_values = np.concatenate([history, np.zeros(horizon)])
    last_date = dates_ts[-1]
    forecast_dates = pd.date_range(
        last_date + pd.Timedelta(weeks=1), periods=horizon, freq="W-SUN"
    )
    forecast_weeks = forecast_dates.isocalendar().week.astype(int).values
    forecast_months = forecast_dates.month.values

    n_hist = len(history)
    forecasts = np.zeros(horizon)

    for step in range(horizon):
        t = n_hist + step
        row = _build_feature_row(
            extended_values, t,
            forecast_weeks[step], forecast_months[step],
            hist_mean, hist_cv, adi, demand_ratio,
        )
        pred = model.predict(row.reshape(1, -1))[0]
        pred = max(pred, 0.0)
        forecasts[step] = pred
        extended_values[t] = pred

    return forecasts

# =============================================================
# Core Engine
# =============================================================

def _forecast_single_series_lgb(history: np.ndarray, dates: np.ndarray,
                                horizon: int,
                                lgb_model=None) -> Dict[str, np.ndarray]:
    """Run LightGBM on a single series using the pre-trained model."""
    results = {}
    preds = _lightgbm_predict_series(lgb_model, history, dates, horizon)
    results["lightgbm"] = preds
    return results

# =============================================================
# Aggregate Fallback for Insufficient-History Items
# =============================================================

def _build_aggregate_series(group_df: pd.DataFrame, date_col: str,
                            value_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build an aggregate time series by summing all items in a group
    across each date. Returns (values_array, dates_array) sorted by
    date.
    """
    agg = (
        group_df
        .groupby(date_col)[value_col]
        .sum()
        .sort_index()
        .reset_index()
    )
    return (
        agg[value_col].values.astype(float),
        pd.to_datetime(agg[date_col]).values,
    )


def _compute_item_shares(group_df: pd.DataFrame, date_col: str,
                         id_col: str, value_col: str,
                         short_ids: List, eval_window: int,
                         recency_weeks: int = 26,
                         new_item_share: float = 0.10
                         ) -> Dict:
    """
    Compute each short-history item's proportional share of the
    group-level aggregate forecast.
    """
    group_df = group_df.copy()
    group_df[date_col] = pd.to_datetime(group_df[date_col])

    group_df = group_df.sort_values(date_col)
    all_dates_sorted = group_df[date_col].unique()
    if len(all_dates_sorted) > eval_window:
        train_cutoff = all_dates_sorted[-eval_window]
        train_df = group_df[group_df[date_col] < train_cutoff]
    else:
        train_df = group_df

    if len(train_df) > 0:
        max_date = train_df[date_col].max()
        recency_cutoff = max_date - pd.Timedelta(weeks=recency_weeks)
        recent_df = train_df[train_df[date_col] >= recency_cutoff]
    else:
        recent_df = train_df

    short_set = set(short_ids)
    recent_demand = (
        recent_df[recent_df[id_col].isin(short_set)]
        .groupby(id_col)[value_col]
        .sum()
        .reindex(short_ids, fill_value=0.0)
    )

    has_history = recent_demand[recent_demand > 0]
    no_history = recent_demand[recent_demand == 0]

    shares = {}

    if len(no_history) > 0 and len(has_history) > 0:
        history_total = has_history.sum()
        if history_total > 0:
            for item_id, demand in has_history.items():
                shares[item_id] = (
                    (demand / history_total) * (1.0 - new_item_share)
                )
        else:
            per_item = (1.0 - new_item_share) / len(has_history)
            for item_id in has_history.index:
                shares[item_id] = per_item
        per_new = new_item_share / len(no_history)
        for item_id in no_history.index:
            shares[item_id] = per_new
    elif len(no_history) > 0:
        per_item = 1.0 / len(no_history)
        for item_id in no_history.index:
            shares[item_id] = per_item
    else:
        history_total = has_history.sum()
        if history_total > 0:
            for item_id, demand in has_history.items():
                shares[item_id] = demand / history_total
        else:
            per_item = 1.0 / len(has_history)
            for item_id in has_history.index:
                shares[item_id] = per_item

    return shares


def _generate_aggregate_fold_details(
        df: pd.DataFrame, short_items: pd.DataFrame,
        date_col: str, id_col: str, value_col: str, group_col: str,
        eval_window: int, min_train_weeks: int,
        lgb_model=None,
        new_item_share: float = 0.10,
) -> Tuple[List[dict], Dict[str, int]]:
    """
    Generate fold details for short-history items using aggregate
    fallback. For groups with sufficient aggregate history, trains
    LightGBM on the aggregate and allocates proportionally. Otherwise
    falls back to global average.
    """
    min_required = min_train_weeks + eval_window
    fold_details = []
    counts = {
        "aggregate_forecasted": 0,
        "global_avg_fallback": 0,
        "no_test_data": 0,
    }

    short_ids = short_items[id_col].unique()
    short_set = set(short_ids)

    if group_col is None or group_col not in df.columns:
        for series_id in short_ids:
            grp = df[df[id_col] == series_id].sort_values(date_col)
            n = len(grp)
            if n < eval_window:
                counts["no_test_data"] += 1
                continue

            all_values = grp[value_col].values.astype(float)
            all_dates = pd.to_datetime(grp[date_col]).values
            test_actual = all_values[-eval_window:]
            test_dates = all_dates[-eval_window:]
            mean_val = all_values[:-eval_window].mean() if n > eval_window else all_values.mean()
            forecast = np.full(eval_window, max(mean_val, 0.0))

            for i in range(eval_window):
                fold_details.append({
                    id_col: series_id,
                    "model": "lightgbm_global_avg",
                    "category": "insufficient_history",
                    "fold": 1,
                    date_col: test_dates[i],
                    "actual": test_actual[i],
                    "forecast": forecast[i],
                    "method": "global_avg_fallback",
                })
            counts["global_avg_fallback"] += 1

        return fold_details, counts

    # Group-based aggregate fallback
    short_item_groups = (
        df[df[id_col].isin(short_set)]
        .groupby(group_col)[id_col]
        .apply(lambda x: list(x.unique()))
        .to_dict()
    )

    for group_name, item_ids_in_group in short_item_groups.items():
        group_all = df[df[group_col] == group_name]
        agg_values, agg_dates = _build_aggregate_series(
            group_all, date_col, value_col
        )
        n_agg = len(agg_values)

        short_in_group = [sid for sid in item_ids_in_group
                          if sid in short_set]

        if n_agg >= min_required and lgb_model is not None:
            train_end_agg = n_agg - eval_window
            train_history_agg = agg_values[:train_end_agg]
            train_dates_agg = agg_dates[:train_end_agg]

            category = classify_demand(train_history_agg)
            agg_preds = _lightgbm_predict_series(
                lgb_model, train_history_agg, train_dates_agg, eval_window
            )

            shares = _compute_item_shares(
                group_all, date_col, id_col, value_col,
                short_in_group, eval_window,
                new_item_share=new_item_share,
            )

            for series_id in short_in_group:
                item_grp = (
                    df[df[id_col] == series_id]
                    .sort_values(date_col)
                )
                n_item = len(item_grp)

                if n_item < eval_window:
                    counts["no_test_data"] += 1
                    continue

                item_values = item_grp[value_col].values.astype(float)
                item_dates = pd.to_datetime(
                    item_grp[date_col]
                ).values
                test_actual = item_values[-eval_window:]
                test_dates = item_dates[-eval_window:]

                share = shares.get(series_id, 1.0 / max(len(short_in_group), 1))
                item_forecast = np.maximum(
                    agg_preds[:eval_window] * share, 0.0
                )

                for i in range(eval_window):
                    fold_details.append({
                        id_col: series_id,
                        "model": "lightgbm",
                        "category": category,
                        "fold": 1,
                        date_col: test_dates[i],
                        "actual": test_actual[i],
                        "forecast": item_forecast[i],
                        "method": "aggregate_fallback",
                    })
                counts["aggregate_forecasted"] += 1

        else:
            for series_id in short_in_group:
                item_grp = (
                    df[df[id_col] == series_id]
                    .sort_values(date_col)
                )
                n_item = len(item_grp)

                if n_item < eval_window:
                    counts["no_test_data"] += 1
                    continue

                item_values = item_grp[value_col].values.astype(float)
                item_dates = pd.to_datetime(
                    item_grp[date_col]
                ).values
                test_actual = item_values[-eval_window:]
                test_dates = item_dates[-eval_window:]

                train_vals = item_values[:-eval_window]
                mean_val = train_vals.mean() if len(train_vals) > 0 else item_values.mean()
                forecast = np.full(eval_window, max(mean_val, 0.0))

                for i in range(eval_window):
                    fold_details.append({
                        id_col: series_id,
                        "model": "lightgbm_global_avg",
                        "category": "insufficient_history",
                        "fold": 1,
                        date_col: test_dates[i],
                        "actual": test_actual[i],
                        "forecast": forecast[i],
                        "method": "global_avg_fallback",
                    })
                counts["global_avg_fallback"] += 1

    return fold_details, counts

# =============================================================
# Chunk Utility
# =============================================================

def _chunk_groups(groups: list, n_chunks: int) -> list:
    """Split a list into roughly equal chunks for worker distribution."""
    if n_chunks <= 0:
        return [groups]
    chunk_size = max(1, len(groups) // n_chunks)
    chunks = []
    for i in range(0, len(groups), chunk_size):
        chunks.append(groups[i:i + chunk_size])
    while len(chunks) > n_chunks:
        chunks[-2].extend(chunks[-1])
        chunks.pop()
    return chunks

# =============================================================
# Worker Function (inference only — model is pre-trained)
# =============================================================

def _fold_details_worker(chunk, date_col, id_col, value_col, eval_window,
                         min_train_weeks, lgb_model=None):
    """
    Worker function for parallel fold-detail generation.

    Unlike fold_details_only.py, this worker only performs INFERENCE
    using the pre-trained global LightGBM model. No per-series
    training happens here.
    """
    min_required = min_train_weeks + eval_window
    fold_details = []
    category_counts = {"smooth": 0, "erratic": 0,
                       "intermittent": 0, "lumpy": 0}

    for series_id, grp in chunk:
        grp = grp.sort_values(date_col).reset_index(drop=True)
        n = len(grp)

        if n < min_required:
            continue

        all_values = grp[value_col].values.astype(float)
        all_dates = pd.to_datetime(grp[date_col]).values

        train_end = n - eval_window
        train_history = all_values[:train_end]
        train_dates = all_dates[:train_end]
        test_actual = all_values[train_end:]
        test_dates = all_dates[train_end:]

        category = classify_demand(train_history)
        category_counts[category] += 1

        model_outputs = _forecast_single_series_lgb(
            train_history, train_dates, eval_window,
            lgb_model=lgb_model,
        )

        for model_name, preds in model_outputs.items():
            preds_trimmed = preds[:eval_window]
            for i in range(eval_window):
                fold_details.append({
                    id_col: series_id,
                    "model": model_name,
                    "category": category,
                    "fold": 1,
                    date_col: test_dates[i],
                    "actual": test_actual[i],
                    "forecast": preds_trimmed[i],
                    "method": "direct",
                })

    return fold_details, category_counts

# =============================================================
# Public API
# =============================================================

def generate_fold_details(
        df: pd.DataFrame,
        date_col: str = "date",
        id_col: str = "item_id",
        value_col: str = "demand",
        group_col: Optional[Union[str, List[str]]] = None,
        eval_window: int = 52,
        min_train_weeks: int = 52,
        new_item_share: float = 0.10,
        n_workers: int = None,
) -> pd.DataFrame:
    """
    Generate the fold_details DataFrame using a globally-trained
    LightGBM model.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with at least date, id, and value columns.
    date_col : str
        Name of the date column (weekly granularity expected).
    id_col : str
        Name of the series identifier column.
    value_col : str
        Name of the numeric value / demand column.
    group_col : str, list of str, or None
        Column(s) defining the product grouping for aggregate
        fallback. If None, short-history items get a global-average
        forecast.
    eval_window : int
        Number of weeks to hold out as the test set (default 52).
    min_train_weeks : int
        Minimum training history required for direct forecasting
        (default 52).
    new_item_share : float
        Fraction of group forecast reserved for zero-history items
        (default 0.10).
    n_workers : int or None
        Number of parallel workers. Defaults to cpu_count() - 1.

    Returns
    -------
    pd.DataFrame
        Columns: [id_col, "model", "category", "fold", date_col,
                  "actual", "forecast", "method"]
    """
    if not HAS_LIGHTGBM:
        raise ImportError(
            "lightgbm is required for this module. "
            "Install it with: pip install lightgbm"
        )

    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Handle composite group columns
    actual_group_col = None
    if group_col is not None:
        if isinstance(group_col, list):
            composite_name = "_".join(group_col)
            df[composite_name] = (
                df[group_col]
                .astype(str)
                .agg("||".join, axis=1)
            )
            actual_group_col = composite_name
        else:
            actual_group_col = group_col

    # Split items into sufficient-history and short-history
    min_required = min_train_weeks + eval_window
    series_lengths = df.groupby(id_col)[date_col].count()
    sufficient_ids = series_lengths[series_lengths >= min_required].index
    short_ids = series_lengths[series_lengths < min_required].index

    total_series = len(series_lengths)
    n_sufficient = len(sufficient_ids)
    n_short = len(short_ids)

    print(f"Generating fold details (LightGBM) for {total_series} series "
          f"with {n_workers} worker(s)...")
    print(f"  Sufficient history (>={min_required} weeks): "
          f"{n_sufficient} series → direct forecasting")
    print(f"  Short history (<{min_required} weeks): "
          f"{n_short} series → aggregate fallback"
          + (f" via '{group_col}'" if group_col else " (global avg)"))

    # ----- TRAIN GLOBAL LIGHTGBM MODEL -----
    lgb_model = _train_global_lightgbm(
        df, sufficient_ids, date_col, id_col, value_col, eval_window
    )

    # ----- DIRECT PATH: sufficient-history items (parallelized) -----
    direct_fold_details = []
    direct_category_counts = {"smooth": 0, "erratic": 0,
                              "intermittent": 0, "lumpy": 0}

    if n_sufficient > 0:
        sufficient_df = df[df[id_col].isin(sufficient_ids)]
        groups = [
            (series_id, grp)
            for series_id, grp in sufficient_df.groupby(id_col)
        ]
        chunks = _chunk_groups(groups, n_workers)

        worker_fn = partial(
            _fold_details_worker,
            date_col=date_col,
            id_col=id_col,
            value_col=value_col,
            eval_window=eval_window,
            min_train_weeks=min_train_weeks,
            lgb_model=lgb_model,
        )

        if n_workers == 1:
            results = [worker_fn(chunk) for chunk in chunks]
        else:
            with Pool(processes=n_workers) as pool:
                results = pool.map(worker_fn, chunks)

        for detail_list, counts in results:
            direct_fold_details.extend(detail_list)
            for cat, cnt in counts.items():
                direct_category_counts[cat] += cnt

    # ----- AGGREGATE FALLBACK PATH: short-history items -----
    agg_fold_details = []
    agg_counts = {
        "aggregate_forecasted": 0,
        "global_avg_fallback": 0,
        "no_test_data": 0,
    }

    if n_short > 0:
        short_items_df = df[df[id_col].isin(short_ids)]
        agg_fold_details, agg_counts = _generate_aggregate_fold_details(
            df=df,
            short_items=short_items_df,
            date_col=date_col,
            id_col=id_col,
            value_col=value_col,
            group_col=actual_group_col,
            eval_window=eval_window,
            min_train_weeks=min_train_weeks,
            lgb_model=lgb_model,
            new_item_share=new_item_share,
        )

    # ----- Combine results -----
    all_details = direct_fold_details + agg_fold_details
    fold_details_df = pd.DataFrame(all_details)

    # ----- Print summary -----
    print(f"\n  Demand Classification Distribution:")
    classified_total = sum(direct_category_counts.values())
    for cat in ["smooth", "erratic", "intermittent", "lumpy"]:
        cnt = direct_category_counts[cat]
        pct = cnt / classified_total * 100 if classified_total > 0 else 0
        print(f"    {cat:>13}: {cnt:>6} series ({pct:5.1f}%)")

    if n_short > 0:
        print(f"\n  Aggregate Fallback:")
        print(f"    Forecasted via group aggregate: "
              f"{agg_counts['aggregate_forecasted']}")
        print(f"    Global avg fallback (no group): "
              f"{agg_counts['global_avg_fallback']}")
        if agg_counts["no_test_data"] > 0:
            print(f"    Skipped (< {eval_window} weeks total, "
                  f"no test data):  {agg_counts['no_test_data']}")

    if len(fold_details_df) > 0:
        n_output_series = fold_details_df[id_col].nunique()
        n_direct = fold_details_df[
            fold_details_df["method"] == "direct"
        ][id_col].nunique()
        n_agg = fold_details_df[
            fold_details_df["method"] == "aggregate_fallback"
        ][id_col].nunique()
        n_global = fold_details_df[
            fold_details_df["method"] == "global_avg_fallback"
        ][id_col].nunique()
        print(f"\n  Output: {n_output_series} series, "
              f"{len(fold_details_df)} rows")
        print(f"    Direct: {n_direct} | Aggregate: {n_agg} | "
              f"Global fallback: {n_global}")
    else:
        print("\n  No fold details generated.")

    # Drop composite group column if we created one
    if (isinstance(group_col, list) and actual_group_col
            and actual_group_col in fold_details_df.columns):
        fold_details_df = fold_details_df.drop(
            columns=[actual_group_col], errors="ignore"
        )

    return fold_details_df

# =============================================================
# CLI entry point
# =============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate fold_details using global LightGBM model."
    )
    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument("-o", "--output", default="fold_details_lgb.csv",
                        help="Output CSV path (default: fold_details_lgb.csv)")
    parser.add_argument("--date-col", default="date",
                        help="Date column name (default: date)")
    parser.add_argument("--id-col", default="item_id",
                        help="Series ID column name (default: item_id)")
    parser.add_argument("--value-col", default="demand",
                        help="Value column name (default: demand)")
    parser.add_argument("--group-col", default=None,
                        help="Group column for aggregate fallback. "
                             "Use comma-separated for multiple columns.")
    parser.add_argument("--eval-window", type=int, default=52,
                        help="Test-set weeks (default: 52)")
    parser.add_argument("--min-train", type=int, default=52,
                        help="Minimum training weeks (default: 52)")
    parser.add_argument("--new-item-share", type=float, default=0.10,
                        help="Share of group forecast for zero-history "
                             "items (default: 0.10)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: auto)")

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    group_col = args.group_col
    if group_col and "," in group_col:
        group_col = [c.strip() for c in group_col.split(",")]

    fold_details_df = generate_fold_details(
        df,
        date_col=args.date_col,
        id_col=args.id_col,
        value_col=args.value_col,
        group_col=group_col,
        eval_window=args.eval_window,
        min_train_weeks=args.min_train,
        new_item_share=args.new_item_share,
        n_workers=args.workers,
    )

    fold_details_df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")
