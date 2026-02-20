"""
fold_details_statistical.py

Generates a fold_details DataFrame using two enhanced statistical
models: SBA (Syntetos-Boylan Approximation) and STL + SES.

This file is a standalone alternative to fold_details_only.py.
It produces the same output schema so results can be compared
directly.

Usage:
    from fold_details_statistical import generate_fold_details

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

MODELS

SBA (Syntetos-Boylan Approximation):
    Bias-corrected variant of Croston's method for intermittent
    demand. Separates demand into two components — non-zero demand
    sizes and inter-demand intervals — smooths each with SES, then
    applies the SBA bias correction factor: (1 - alpha_p/2).
    This corrects the upward bias inherent in Croston's original
    ratio estimator. Pure numpy implementation with optional numba
    acceleration.

    Routed to: intermittent, lumpy categories.

STL + SES:
    Robust seasonal-trend decomposition (STL via LOESS) followed
    by Simple Exponential Smoothing on the deseasonalized series.
    More robust than Holt-Winters on noisy data because the
    decomposition is non-parametric and STL's robust mode
    downweights outliers automatically. The seasonal component
    from the last cycle is tiled forward for re-seasonalization.
    Requires statsmodels.

    Routed to: smooth, erratic categories.

Both models receive all categories so every series gets forecasts
from both approaches, enabling direct comparison.
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
    from statsmodels.tsa.seasonal import STL
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# =============================================================
# Numba-Accelerated Smoothing Kernels
# =============================================================

if HAS_NUMBA:
    @njit(cache=True)
    def _ses_smooth_numba(values: np.ndarray, alpha: float) -> float:
        """Simple Exponential Smoothing — returns final level."""
        level = values[0]
        for i in range(1, len(values)):
            level = alpha * values[i] + (1.0 - alpha) * level
        return level

    @njit(cache=True)
    def _sba_smooth_numba(demand: np.ndarray, alpha_d: float,
                          alpha_p: float) -> Tuple[float, float]:
        """
        SBA smoothing loop.
        Smooths demand sizes (z) and inter-demand intervals (p)
        separately. Returns (smoothed_demand_size, smoothed_interval).
        """
        # Extract non-zero demands and intervals
        z_t = 0.0
        p_t = 0.0
        initialized = False

        period_counter = 0
        for t in range(len(demand)):
            period_counter += 1
            if demand[t] > 0.0:
                if not initialized:
                    z_t = demand[t]
                    p_t = float(period_counter)
                    initialized = True
                else:
                    z_t = alpha_d * demand[t] + (1.0 - alpha_d) * z_t
                    p_t = alpha_p * float(period_counter) + (1.0 - alpha_p) * p_t
                period_counter = 0

        return z_t, p_t

else:
    def _ses_smooth_numba(values: np.ndarray, alpha: float) -> float:
        level = values[0]
        for i in range(1, len(values)):
            level = alpha * values[i] + (1.0 - alpha) * level
        return level

    def _sba_smooth_numba(demand: np.ndarray, alpha_d: float,
                          alpha_p: float) -> Tuple[float, float]:
        z_t = 0.0
        p_t = 0.0
        initialized = False

        period_counter = 0
        for t in range(len(demand)):
            period_counter += 1
            if demand[t] > 0.0:
                if not initialized:
                    z_t = demand[t]
                    p_t = float(period_counter)
                    initialized = True
                else:
                    z_t = alpha_d * demand[t] + (1.0 - alpha_d) * z_t
                    p_t = alpha_p * float(period_counter) + (1.0 - alpha_p) * p_t
                period_counter = 0

        return z_t, p_t

# =============================================================
# Demand Classification (ADI / CV²)
# =============================================================

ADI_THRESHOLD = 1.32
CV2_THRESHOLD = 0.49

CATEGORY_MODELS = {
    "smooth":       ["sba", "stl_ses"],
    "erratic":      ["sba", "stl_ses"],
    "intermittent": ["sba", "stl_ses"],
    "lumpy":        ["sba", "stl_ses"],
}

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

def get_models_for_category(category: str) -> List[str]:
    """Return the list of model names assigned to a demand category."""
    return CATEGORY_MODELS.get(category, list(MODEL_REGISTRY.keys()))

# =============================================================
# Shared Seasonal Cache
# =============================================================

def _build_series_cache(history: np.ndarray, dates: np.ndarray,
                        horizon: int) -> Dict[str, Any]:
    """
    Precompute all shared datetime and seasonal data for one series.
    Called ONCE per series; results are passed to all models.
    """
    dates_ts = pd.DatetimeIndex(dates)
    week_numbers = dates_ts.isocalendar().week.astype(int).values
    years = dates_ts.year.values

    global_mean = history.mean()
    global_median = float(np.median(history))

    week_avg = {}
    week_median = {}
    week_means_raw = {}
    for w in range(1, 54):
        mask = week_numbers == w
        if mask.any():
            vals = history[mask]
            week_avg[w] = vals.mean()
            week_median[w] = float(np.median(vals))
            week_means_raw[w] = vals.mean()
        else:
            week_avg[w] = global_mean
            week_median[w] = global_median

    overall_mean_seasonal = (
        np.mean(list(week_means_raw.values())) if week_means_raw else 1.0
    )
    if overall_mean_seasonal > 0:
        seasonal_index = {
            w: m / overall_mean_seasonal for w, m in week_means_raw.items()
        }
    else:
        seasonal_index = {}

    start_date = dates_ts[-1] + pd.Timedelta(weeks=1)
    forecast_dates = pd.date_range(start_date, periods=horizon, freq="W-SUN")
    forecast_weeks = forecast_dates.isocalendar().week.astype(int).values

    return {
        "dates_ts": dates_ts,
        "week_numbers": week_numbers,
        "years": years,
        "global_mean": global_mean,
        "global_median": global_median,
        "week_avg": week_avg,
        "week_median": week_median,
        "week_means_raw": week_means_raw,
        "overall_mean_seasonal": overall_mean_seasonal,
        "seasonal_index": seasonal_index,
        "forecast_dates": forecast_dates,
        "forecast_weeks": forecast_weeks,
    }

# =============================================================
# Model Registry
# =============================================================

MODEL_REGISTRY: Dict[str, Callable] = {}

def register_model(name: str):
    """Decorator to register a model function in the global registry."""
    def decorator(func):
        MODEL_REGISTRY[name] = func
        return func
    return decorator

# =============================================================
# Model Implementations
# =============================================================

@register_model("sba")
def sba_forecast(history: np.ndarray, dates: np.ndarray,
                 horizon: int = 52, alpha_d: float = 0.1,
                 alpha_p: float = 0.1, **kwargs) -> np.ndarray:
    """
    Syntetos-Boylan Approximation (SBA) forecast.

    Bias-corrected Croston's method for intermittent demand.
    Smooths demand sizes and inter-demand intervals separately,
    then applies the SBA correction: forecast = (z/p) * (1 - alpha_p/2).
    """
    demand = history.astype(float)

    # Need at least one non-zero observation
    nonzero_mask = demand > 0
    if not nonzero_mask.any():
        return np.full(horizon, 0.0)

    z_t, p_t = _sba_smooth_numba(demand, alpha_d, alpha_p)

    if p_t <= 0.0:
        cache = kwargs.get("cache")
        mean_val = cache["global_mean"] if cache else history.mean()
        return np.full(horizon, max(mean_val, 0.0))

    # SBA bias correction
    sba_forecast_val = (z_t / p_t) * (1.0 - alpha_p / 2.0)
    sba_forecast_val = max(sba_forecast_val, 0.0)

    return np.full(horizon, sba_forecast_val)


@register_model("stl_ses")
def stl_ses_forecast(history: np.ndarray, dates: np.ndarray,
                     horizon: int = 52, seasonal_period: int = 52,
                     alpha: float = 0.2, **kwargs) -> np.ndarray:
    """
    STL Decomposition + Simple Exponential Smoothing.

    Decomposes the series into trend, seasonal, and residual
    components using STL (robust LOESS). Forecasts the
    deseasonalized series (trend + residual) with SES, then
    re-seasonalizes by tiling the last seasonal cycle.

    Requires statsmodels. Falls back to global mean if unavailable
    or if series is too short (< 2 full seasonal cycles).
    """
    cache = kwargs.get("cache")
    mean_val = cache["global_mean"] if cache else history.mean()
    fallback = np.full(horizon, max(mean_val, 0.0))

    if not HAS_STATSMODELS:
        return fallback

    n = len(history)
    if n < 2 * seasonal_period:
        return fallback

    # Guard against constant series (STL needs variance)
    if np.std(history) == 0:
        return fallback

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stl_result = STL(
                history,
                period=seasonal_period,
                robust=True,
            ).fit()

        seasonal = stl_result.seasonal
        deseasonalized = history - seasonal

        # Forecast the deseasonalized series with SES
        ses_level = _ses_smooth_numba(deseasonalized, alpha)

        # Re-seasonalize: tile last cycle of seasonal component
        last_cycle = seasonal[-seasonal_period:]

        # Align seasonal forecast to the correct week-of-year
        if cache:
            forecast_weeks = cache["forecast_weeks"]
            # Map the last cycle to week-of-year
            train_weeks = cache["week_numbers"]
            last_cycle_weeks = train_weeks[-seasonal_period:]

            # Build a week-of-year → seasonal value lookup from the
            # last cycle
            week_seasonal = {}
            for i, w in enumerate(last_cycle_weeks):
                week_seasonal[w] = last_cycle[i]

            seasonal_forecast = np.array([
                week_seasonal.get(w, 0.0) for w in forecast_weeks
            ])
        else:
            # Simple tiling if no cache
            reps = (horizon // seasonal_period) + 1
            seasonal_forecast = np.tile(last_cycle, reps)[:horizon]

        forecast = ses_level + seasonal_forecast
        forecast = np.maximum(forecast, 0.0)

        if not np.all(np.isfinite(forecast)):
            return fallback

        return forecast

    except Exception:
        return fallback

# =============================================================
# Core Engine
# =============================================================

def _forecast_single_series(history: np.ndarray, dates: np.ndarray,
                            horizon: int,
                            model_names: List[str] = None,
                            cache: Dict[str, Any] = None
                            ) -> Dict[str, np.ndarray]:
    """Run selected models on a single series."""
    names = model_names or list(MODEL_REGISTRY.keys())
    results = {}
    for name in names:
        func = MODEL_REGISTRY.get(name)
        if func is None:
            continue
        try:
            results[name] = func(history, dates, horizon, cache=cache)
        except Exception:
            results[name] = np.full(horizon, np.nan)
    return results

# =============================================================
# Aggregate Fallback for Insufficient-History Items
# =============================================================

def _build_aggregate_series(group_df: pd.DataFrame, date_col: str,
                            value_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build an aggregate time series by summing all items in a group
    across each date.
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
        new_item_share: float = 0.10,
) -> Tuple[List[dict], Dict[str, int]]:
    """
    Generate fold details for short-history items using aggregate
    fallback.
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
                    "model": "global_avg",
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

        if n_agg >= min_required:
            train_end_agg = n_agg - eval_window
            train_history_agg = agg_values[:train_end_agg]
            train_dates_agg = agg_dates[:train_end_agg]

            category = classify_demand(train_history_agg)
            model_names = get_models_for_category(category)
            cache = _build_series_cache(
                train_history_agg, train_dates_agg, eval_window
            )
            agg_outputs = _forecast_single_series(
                train_history_agg, train_dates_agg, eval_window,
                model_names=model_names, cache=cache,
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

                for model_name, agg_preds in agg_outputs.items():
                    item_forecast = np.maximum(
                        agg_preds[:eval_window] * share, 0.0
                    )
                    for i in range(eval_window):
                        fold_details.append({
                            id_col: series_id,
                            "model": model_name,
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
                        "model": "global_avg",
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
# Worker Function
# =============================================================

def _fold_details_worker(chunk, date_col, id_col, value_col, eval_window,
                         min_train_weeks):
    """
    Worker function for parallel fold-detail generation.
    Processes items with sufficient history using SBA and STL+SES.
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
        model_names = get_models_for_category(category)

        cache = _build_series_cache(train_history, train_dates, eval_window)

        model_outputs = _forecast_single_series(
            train_history, train_dates, eval_window,
            model_names=model_names, cache=cache,
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
    Generate the fold_details DataFrame using SBA and STL+SES
    models with demand-classification routing.

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
        fallback.
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

    numba_status = (
        "enabled"
        if HAS_NUMBA
        else "disabled (pip install numba for faster smoothing loops)"
    )
    statsmodels_status = (
        "enabled"
        if HAS_STATSMODELS
        else "disabled (pip install statsmodels for STL+SES)"
    )
    print(f"Generating fold details (SBA + STL+SES) for "
          f"{total_series} series with {n_workers} worker(s)...")
    print(f"  Numba JIT: {numba_status}")
    print(f"  Statsmodels STL: {statsmodels_status}")
    print(f"  Demand classification: ADI/CV² routing enabled")
    print(f"  Sufficient history (>={min_required} weeks): "
          f"{n_sufficient} series → direct forecasting")
    print(f"  Short history (<{min_required} weeks): "
          f"{n_short} series → aggregate fallback"
          + (f" via '{group_col}'" if group_col else " (global avg)"))

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
        n_models = len(CATEGORY_MODELS[cat])
        pct = cnt / classified_total * 100 if classified_total > 0 else 0
        print(f"    {cat:>13}: {cnt:>6} series ({pct:5.1f}%) "
              f"x {n_models} models = {cnt * n_models:>7} fits")

    total_direct_fits = sum(
        direct_category_counts[c] * len(CATEGORY_MODELS[c])
        for c in direct_category_counts
    )
    print(f"    {'TOTAL':>13}: {total_direct_fits:>6} model fits")

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
        description="Generate fold_details using SBA and STL+SES models."
    )
    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument("-o", "--output",
                        default="fold_details_statistical.csv",
                        help="Output CSV path "
                             "(default: fold_details_statistical.csv)")
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
