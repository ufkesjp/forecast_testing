"""
benchmark_forecasting_parallel.py
==================================
Parallelized version of the best-fit benchmark forecasting toolkit.

This module is fully self-contained — it includes all model definitions,
metric functions, evaluation logic, tournament selection, and the
end-to-end pipeline. Forecasting workloads are distributed across
multiple CPU cores using multiprocessing.Pool for significant speedups
on large datasets with thousands of series.

Usage:
    from benchmark_forecasting_parallel import best_fit_pipeline

    eval_df, best_fit_df, forecast_df, inactive_df = best_fit_pipeline(
        df, date_col="date", id_col="item_id", value_col="demand",
        n_workers=4,
    )
"""

__version__ = "2.0.0"

import numpy as np
import pandas as pd
import warnings
from typing import Tuple, Optional, Dict, Callable, List
from multiprocessing import Pool, cpu_count
from functools import partial

# Conditional imports for Tier 2 models (Holt-Winters, Theta)
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.forecasting.theta import ThetaModel
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# =============================================================
# Model Registry
# =============================================================

MODEL_REGISTRY: Dict[str, Callable] = {}


def register_model(name: str):
    '''Decorator to register a model function in the global registry.'''
    def decorator(func):
        MODEL_REGISTRY[name] = func
        return func
    return decorator


# =============================================================
# Inactive Item Detection
# =============================================================

def flag_inactive_items(df: pd.DataFrame, date_col: str, id_col: str,
                        value_col: str,
                        inactive_weeks: int = 26) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Flag and separate inactive items. Items with zero total demand in
    the most recent `inactive_weeks` weeks are flagged as inactive.
    '''
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    global_max_date = df[date_col].max()
    cutoff_date = global_max_date - pd.Timedelta(weeks=inactive_weeks)

    trailing = df[df[date_col] > cutoff_date]
    trailing_demand = trailing.groupby(id_col)[value_col].sum()

    all_ids = df[id_col].unique()
    trailing_demand = trailing_demand.reindex(all_ids, fill_value=0)

    inactive_ids = trailing_demand[trailing_demand == 0].index
    active_ids = trailing_demand[trailing_demand > 0].index

    inactive_records = []
    if len(inactive_ids) > 0:
        for item_id in inactive_ids:
            item_data = df[df[id_col] == item_id]
            nonzero = item_data[item_data[value_col] > 0]

            if len(nonzero) > 0:
                last_nonzero = nonzero[date_col].max()
                weeks_since = int((global_max_date - last_nonzero).days / 7)
            else:
                last_nonzero = pd.NaT
                weeks_since = np.nan

            inactive_records.append({
                id_col: item_id,
                "status": "inactive",
                "last_nonzero_date": last_nonzero,
                "weeks_since_demand": weeks_since,
                "total_history_weeks": len(item_data),
                "lifetime_total_qty": item_data[value_col].sum(),
            })

    inactive_df = pd.DataFrame(inactive_records)
    active_df = df[df[id_col].isin(active_ids)].copy()

    return active_df, inactive_df


# =============================================================
# Benchmark Models
# =============================================================

@register_model("seasonal_naive")
def seasonal_naive(history: np.ndarray, dates: np.ndarray,
                   horizon: int = 52, **kwargs) -> np.ndarray:
    '''Seasonal Naive: repeats the last 52 weeks of observed demand.'''
    last_year = history[-52:]
    if len(last_year) < 52:
        last_year = np.tile(history, (52 // len(history) + 1))[-52:]
    reps = (horizon // 52) + 1
    return np.tile(last_year, reps)[:horizon]


@register_model("weekly_hist_avg")
def weekly_historical_average(history: np.ndarray, dates: np.ndarray,
                               horizon: int = 52, **kwargs) -> np.ndarray:
    '''Weekly Historical Average: mean demand per ISO week-of-year.'''
    dates_ts = pd.DatetimeIndex(dates)
    week_numbers = dates_ts.isocalendar().week.astype(int).values

    global_mean = history.mean()
    week_avg = {}
    for w in range(1, 54):
        mask = week_numbers == w
        if mask.any():
            week_avg[w] = history[mask].mean()
        else:
            week_avg[w] = global_mean

    start_date = dates_ts[-1] + pd.Timedelta(weeks=1)
    forecast_dates = pd.date_range(start_date, periods=horizon, freq="W-SUN")
    forecast_weeks = forecast_dates.isocalendar().week.astype(int).values

    return np.array([week_avg.get(w, global_mean) for w in forecast_weeks])


@register_model("global_avg")
def global_average(history: np.ndarray, dates: np.ndarray,
                   horizon: int = 52, **kwargs) -> np.ndarray:
    '''Global Average: flat forecast equal to the overall mean.'''
    return np.full(horizon, history.mean())


@register_model("tsb")
def tsb_forecast(history: np.ndarray, dates: np.ndarray,
                 horizon: int = 52, alpha_d: float = 0.1,
                 alpha_p: float = 0.1, **kwargs) -> np.ndarray:
    '''Teunter-Syntetos-Babai (TSB) forecast for intermittent demand.'''
    demand = history.astype(float)
    nonzero_mask = demand > 0

    z_t = demand[nonzero_mask].mean() if nonzero_mask.any() else 0.0
    p_t = nonzero_mask.mean()

    for t in range(len(demand)):
        if demand[t] > 0:
            z_t = alpha_d * demand[t] + (1 - alpha_d) * z_t
            p_t = alpha_p + (1 - alpha_p) * p_t
        else:
            p_t = (1 - alpha_p) * p_t

    return np.full(horizon, max(p_t * z_t, 0.0))


@register_model("temp_agg_ses")
def temporal_agg_ses(history: np.ndarray, dates: np.ndarray,
                     horizon: int = 52, agg_periods: int = 4,
                     alpha: float = 0.2, **kwargs) -> np.ndarray:
    '''Temporal Aggregation + Simple Exponential Smoothing.'''
    values = history.astype(float)
    n = len(values)

    trim = n - (n % agg_periods)
    trimmed = values[-trim:]
    agg = trimmed.reshape(-1, agg_periods).sum(axis=1)

    level = agg[0]
    for v in agg[1:]:
        level = alpha * v + (1 - alpha) * level
    agg_forecast = max(level, 0.0)

    dates_ts = pd.DatetimeIndex(dates)
    week_numbers = dates_ts.isocalendar().week.astype(int).values

    week_means = {}
    for w in range(1, 54):
        mask = week_numbers == w
        if mask.any():
            week_means[w] = values[mask].mean()
    overall_mean = np.mean(list(week_means.values())) if week_means else 1.0
    seasonal_index = {w: m / overall_mean for w, m in week_means.items()} if overall_mean > 0 else {}

    start_date = dates_ts[-1] + pd.Timedelta(weeks=1)
    forecast_dates = pd.date_range(start_date, periods=horizon, freq="W-SUN")
    forecast_weeks = forecast_dates.isocalendar().week.astype(int).values

    raw = np.array([seasonal_index.get(w, 1.0) for w in forecast_weeks])

    weekly_forecast = np.zeros(horizon)
    for i in range(0, horizon, agg_periods):
        block = raw[i:i + agg_periods]
        block_sum = block.sum()
        if block_sum > 0:
            weekly_forecast[i:i + agg_periods] = (block / block_sum) * agg_forecast
        else:
            weekly_forecast[i:i + agg_periods] = agg_forecast / agg_periods

    return weekly_forecast


@register_model("weighted_seasonal_avg")
def weighted_seasonal_average(history: np.ndarray, dates: np.ndarray,
                               horizon: int = 52, decay: float = 0.8,
                               **kwargs) -> np.ndarray:
    '''Weighted Seasonal Average: recency-weighted mean per ISO week-of-year.'''
    dates_ts = pd.DatetimeIndex(dates)
    week_numbers = dates_ts.isocalendar().week.astype(int).values
    years = dates_ts.year.values

    unique_years = np.sort(np.unique(years))
    n_years = len(unique_years)
    year_rank = {yr: i for i, yr in enumerate(unique_years)}

    global_mean = history.mean()
    week_avg = {}

    for w in range(1, 54):
        mask = week_numbers == w
        if not mask.any():
            week_avg[w] = global_mean
            continue

        w_values = history[mask]
        w_years = years[mask]
        weights = np.array([decay ** (n_years - 1 - year_rank[yr]) for yr in w_years])

        weight_sum = weights.sum()
        if weight_sum > 0:
            week_avg[w] = np.average(w_values, weights=weights)
        else:
            week_avg[w] = global_mean

    start_date = dates_ts[-1] + pd.Timedelta(weeks=1)
    forecast_dates = pd.date_range(start_date, periods=horizon, freq="W-SUN")
    forecast_weeks = forecast_dates.isocalendar().week.astype(int).values

    return np.array([week_avg.get(w, global_mean) for w in forecast_weeks])


@register_model("seasonal_median")
def seasonal_median(history: np.ndarray, dates: np.ndarray,
                    horizon: int = 52, **kwargs) -> np.ndarray:
    '''Seasonal Median: median demand per ISO week-of-year.'''
    dates_ts = pd.DatetimeIndex(dates)
    week_numbers = dates_ts.isocalendar().week.astype(int).values

    global_median = np.median(history)
    week_med = {}
    for w in range(1, 54):
        mask = week_numbers == w
        if mask.any():
            week_med[w] = np.median(history[mask])
        else:
            week_med[w] = global_median

    start_date = dates_ts[-1] + pd.Timedelta(weeks=1)
    forecast_dates = pd.date_range(start_date, periods=horizon, freq="W-SUN")
    forecast_weeks = forecast_dates.isocalendar().week.astype(int).values

    return np.array([week_med.get(w, global_median) for w in forecast_weeks])


@register_model("seasonal_naive_blend")
def seasonal_naive_blend(history: np.ndarray, dates: np.ndarray,
                         horizon: int = 52, blend_weight: float = 0.5,
                         **kwargs) -> np.ndarray:
    '''Seasonal Naive + Weekly Average Blend.'''
    last_year = history[-52:]
    if len(last_year) < 52:
        last_year = np.tile(history, (52 // len(history) + 1))[-52:]
    reps = (horizon // 52) + 1
    snaive = np.tile(last_year, reps)[:horizon]

    dates_ts = pd.DatetimeIndex(dates)
    week_numbers = dates_ts.isocalendar().week.astype(int).values

    global_mean = history.mean()
    week_avg = {}
    for w in range(1, 54):
        mask = week_numbers == w
        if mask.any():
            week_avg[w] = history[mask].mean()
        else:
            week_avg[w] = global_mean

    start_date = dates_ts[-1] + pd.Timedelta(weeks=1)
    forecast_dates = pd.date_range(start_date, periods=horizon, freq="W-SUN")
    forecast_weeks = forecast_dates.isocalendar().week.astype(int).values
    whist = np.array([week_avg.get(w, global_mean) for w in forecast_weeks])

    return blend_weight * snaive + (1 - blend_weight) * whist


@register_model("holt_winters")
def holt_winters(history: np.ndarray, dates: np.ndarray,
                 horizon: int = 52, seasonal_periods: int = 52,
                 **kwargs) -> np.ndarray:
    '''Holt-Winters Exponential Smoothing (ETS) with additive trend and seasonality.'''
    if not HAS_STATSMODELS:
        return np.full(horizon, np.nan)

    n = len(history)
    fallback = np.full(horizon, history.mean())

    if n < 2 * seasonal_periods:
        return fallback
    if np.std(history) == 0:
        return fallback

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                history, trend="add", seasonal="add",
                seasonal_periods=seasonal_periods,
                initialization_method="estimated",
            )
            fit = model.fit(optimized=True, use_brute=True)
            forecast = fit.forecast(horizon)
            forecast = np.maximum(forecast, 0.0)
            if not np.all(np.isfinite(forecast)):
                return fallback
            return forecast
    except Exception:
        return fallback


@register_model("theta")
def theta_method(history: np.ndarray, dates: np.ndarray,
                 horizon: int = 52, seasonal_periods: int = 52,
                 **kwargs) -> np.ndarray:
    '''Theta Method forecast with seasonal decomposition.'''
    if not HAS_STATSMODELS:
        return np.full(horizon, np.nan)

    n = len(history)
    fallback = np.full(horizon, history.mean())

    if n < 2 * seasonal_periods:
        return fallback
    if np.std(history) == 0:
        return fallback

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            series = pd.Series(
                history,
                index=pd.date_range(
                    end=pd.Timestamp(dates[-1]), periods=n, freq="W-SUN"
                )
            )
            model = ThetaModel(series, period=seasonal_periods,
                               deseasonalize=True, method="auto")
            fit = model.fit()
            forecast = fit.forecast(horizon).values
            forecast = np.maximum(forecast, 0.0)
            if not np.all(np.isfinite(forecast)):
                return fallback
            return forecast
    except Exception:
        return fallback


@register_model("imapa")
def imapa(history: np.ndarray, dates: np.ndarray,
          horizon: int = 52, agg_levels: tuple = (1, 2, 4, 8, 13),
          alpha: float = 0.2, **kwargs) -> np.ndarray:
    '''IMAPA: Intermittent Multiple Aggregation Prediction Algorithm.'''
    values = history.astype(float)
    n = len(values)

    dates_ts = pd.DatetimeIndex(dates)
    week_numbers = dates_ts.isocalendar().week.astype(int).values

    week_means = {}
    for w in range(1, 54):
        mask = week_numbers == w
        if mask.any():
            week_means[w] = values[mask].mean()
    overall_mean = np.mean(list(week_means.values())) if week_means else 1.0
    seasonal_index = {w: m / overall_mean for w, m in week_means.items()} if overall_mean > 0 else {}

    start_date = dates_ts[-1] + pd.Timedelta(weeks=1)
    forecast_dates = pd.date_range(start_date, periods=horizon, freq="W-SUN")
    forecast_weeks = forecast_dates.isocalendar().week.astype(int).values
    raw_weights = np.array([seasonal_index.get(w, 1.0) for w in forecast_weeks])

    level_forecasts = []

    for level in agg_levels:
        if level > n:
            continue

        if level == 1:
            ses_level = values[0]
            for v in values[1:]:
                ses_level = alpha * v + (1 - alpha) * ses_level
            agg_forecast = max(ses_level, 0.0)
        else:
            trim = n - (n % level)
            trimmed = values[-trim:]
            agg = trimmed.reshape(-1, level).sum(axis=1)
            ses_level = agg[0]
            for v in agg[1:]:
                ses_level = alpha * v + (1 - alpha) * ses_level
            agg_forecast = max(ses_level, 0.0)

        weekly_forecast = np.zeros(horizon)
        block_size = max(level, 1)

        for i in range(0, horizon, block_size):
            block = raw_weights[i:i + block_size]
            block_sum = block.sum()
            if block_sum > 0:
                weekly_forecast[i:i + block_size] = (block / block_sum) * agg_forecast
            else:
                weekly_forecast[i:i + block_size] = agg_forecast / block_size

        level_forecasts.append(weekly_forecast)

    if not level_forecasts:
        return np.full(horizon, history.mean())

    return np.mean(level_forecasts, axis=0)


@register_model("damped_trend_seasonal")
def damped_trend_seasonal(history: np.ndarray, dates: np.ndarray,
                          horizon: int = 52, seasonal_periods: int = 52,
                          **kwargs) -> np.ndarray:
    '''Damped Trend Seasonal ETS — ETS(A,Ad,A).'''
    if not HAS_STATSMODELS:
        return np.full(horizon, np.nan)

    n = len(history)
    fallback = np.full(horizon, history.mean())

    if n < 2 * seasonal_periods:
        return fallback
    if np.std(history) == 0:
        return fallback

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                history, trend="add", damped_trend=True,
                seasonal="add", seasonal_periods=seasonal_periods,
                initialization_method="estimated",
            )
            fit = model.fit(optimized=True, use_brute=True)
            forecast = fit.forecast(horizon)
            forecast = np.maximum(forecast, 0.0)
            if not np.all(np.isfinite(forecast)):
                return fallback
            return forecast
    except Exception:
        return fallback


@register_model("linear_trend_seasonal")
def linear_trend_seasonal(history: np.ndarray, dates: np.ndarray,
                          horizon: int = 52, **kwargs) -> np.ndarray:
    '''Linear Trend + Seasonal Decomposition forecast.'''
    values = history.astype(float)
    n = len(values)

    dates_ts = pd.DatetimeIndex(dates)
    week_numbers = dates_ts.isocalendar().week.astype(int).values

    global_mean = values.mean()
    seasonal_idx = {}
    for w in range(1, 54):
        mask = week_numbers == w
        if mask.any():
            seasonal_idx[w] = values[mask].mean() - global_mean
        else:
            seasonal_idx[w] = 0.0

    seasonal_component = np.array([seasonal_idx.get(w, 0.0) for w in week_numbers])
    deseasonalized = values - seasonal_component

    t = np.arange(n, dtype=float)
    t_mean = t.mean()
    y_mean = deseasonalized.mean()

    cov_ty = np.sum((t - t_mean) * (deseasonalized - y_mean))
    var_t = np.sum((t - t_mean) ** 2)

    if var_t > 0:
        slope = cov_ty / var_t
        intercept = y_mean - slope * t_mean
    else:
        slope = 0.0
        intercept = y_mean

    future_t = np.arange(n, n + horizon, dtype=float)
    trend_forecast = intercept + slope * future_t

    start_date = dates_ts[-1] + pd.Timedelta(weeks=1)
    forecast_dates = pd.date_range(start_date, periods=horizon, freq="W-SUN")
    forecast_weeks = forecast_dates.isocalendar().week.astype(int).values
    future_seasonal = np.array([seasonal_idx.get(w, 0.0) for w in forecast_weeks])

    forecast = trend_forecast + future_seasonal
    forecast = np.maximum(forecast, 0.0)

    return forecast


# =============================================================
# Evaluation Metrics
# =============================================================

def calculate_wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    '''WAPE: sum(|actual - forecast|) / sum(|actual|).'''
    total_actual = np.sum(np.abs(actual))
    if total_actual == 0:
        return np.inf
    return np.sum(np.abs(actual - forecast)) / total_actual


def calculate_mase(actual: np.ndarray, forecast: np.ndarray,
                   in_sample: np.ndarray, seasonal_period: int = 52) -> float:
    '''MASE: mean(|actual - forecast|) / naive_mae.'''
    if len(in_sample) <= seasonal_period:
        return np.inf

    naive_mae = np.mean(np.abs(in_sample[seasonal_period:] - in_sample[:-seasonal_period]))
    if naive_mae == 0:
        return np.inf

    return np.mean(np.abs(actual - forecast)) / naive_mae


# =============================================================
# Core Engine: Forecast a Single Series with All Models
# =============================================================

def _forecast_single_series(history: np.ndarray, dates: np.ndarray,
                            horizon: int, model_names: list = None) -> Dict[str, np.ndarray]:
    '''Run all (or selected) registered models on a single series.'''
    names = model_names or list(MODEL_REGISTRY.keys())
    results = {}
    for name in names:
        try:
            results[name] = MODEL_REGISTRY[name](history, dates, horizon)
        except Exception:
            results[name] = np.full(horizon, np.nan)
    return results


# =============================================================
# Chunk Utility
# =============================================================

def _chunk_groups(groups: list, n_chunks: int) -> list:
    '''Split a list into roughly equal chunks for distribution across workers.'''
    if n_chunks <= 0:
        return [groups]
    chunk_size = max(1, len(groups) // n_chunks)
    chunks = []
    for i in range(0, len(groups), chunk_size):
        chunks.append(groups[i:i + chunk_size])
    # If we have more chunks than n_chunks, merge the last ones
    while len(chunks) > n_chunks:
        chunks[-2].extend(chunks[-1])
        chunks.pop()
    return chunks


# =============================================================
# Multiprocessing Worker Functions (module-level for pickling)
# =============================================================

def _eval_worker(chunk, date_col, id_col, value_col, eval_window,
                 n_folds, fold_spacing, min_train_weeks):
    '''
    Worker function for parallel evaluation. Takes a list of
    (series_id, group_dataframe) tuples and runs rolling-origin CV
    on each series.

    Returns (eval_results_list, fold_details_list, skipped_count).
    '''
    min_required = min_train_weeks + eval_window + (n_folds - 1) * fold_spacing
    skipped = 0
    eval_results = []
    fold_details = []

    for series_id, grp in chunk:
        grp = grp.sort_values(date_col).reset_index(drop=True)
        n = len(grp)

        if n < min_required:
            skipped += 1
            continue

        all_values = grp[value_col].values.astype(float)
        all_dates = pd.to_datetime(grp[date_col]).values

        fold_metrics = {name: {"wapes": [], "mases": [], "actuals": [], "forecasts": []}
                        for name in MODEL_REGISTRY}

        folds_evaluated = 0
        for fold_idx in range(n_folds):
            test_end = n - (fold_idx * fold_spacing)
            test_start = test_end - eval_window
            train_end = test_start

            if train_end < min_train_weeks:
                continue

            train_history = all_values[:train_end]
            train_dates = all_dates[:train_end]
            test_actual = all_values[test_start:test_end]

            model_outputs = _forecast_single_series(train_history, train_dates, eval_window)

            for model_name, preds in model_outputs.items():
                preds_trimmed = preds[:eval_window]
                wape = calculate_wape(test_actual, preds_trimmed)
                mase = calculate_mase(test_actual, preds_trimmed, train_history)

                fold_metrics[model_name]["wapes"].append(wape)
                fold_metrics[model_name]["mases"].append(mase)
                fold_metrics[model_name]["actuals"].append(np.sum(test_actual))
                fold_metrics[model_name]["forecasts"].append(np.sum(preds_trimmed))

                test_dates = all_dates[test_start:test_end]
                for i in range(eval_window):
                    fold_details.append({
                        id_col: series_id,
                        "model": model_name,
                        "fold": fold_idx + 1,
                        date_col: test_dates[i],
                        "actual": test_actual[i],
                        "forecast": preds_trimmed[i],
                    })

            folds_evaluated += 1

        if folds_evaluated == 0:
            skipped += 1
            continue

        for model_name, metrics in fold_metrics.items():
            if not metrics["wapes"]:
                continue

            wapes = np.array(metrics["wapes"])
            mases = np.array(metrics["mases"])

            wapes_clean = np.where(np.isinf(wapes), np.nan, wapes)
            mases_clean = np.where(np.isinf(mases), np.nan, mases)

            mean_wape = np.nanmean(wapes_clean) if not np.all(np.isnan(wapes_clean)) else np.inf
            mean_mase = np.nanmean(mases_clean) if not np.all(np.isnan(mases_clean)) else np.inf

            eval_results.append({
                id_col: series_id,
                "model": model_name,
                "wape": mean_wape,
                "mase": mean_mase,
                "total_actual": np.sum(metrics["actuals"]),
                "total_forecast": np.sum(metrics["forecasts"]),
                "n_folds": len(metrics["wapes"]),
                "wape_std": np.nanstd(wapes_clean) if len(wapes_clean) > 1 else 0.0,
                "fold_wapes": ",".join([f"{w:.4f}" if np.isfinite(w) else "inf"
                                        for w in wapes]),
            })

    return eval_results, fold_details, skipped


def _forecast_worker(chunk, date_col, id_col, value_col, horizon):
    '''
    Worker function for parallel forecast generation. Takes a list of
    (series_id, group_dataframe, winning_model_name) tuples and generates
    forecasts for each.

    Returns a list of DataFrames.
    '''
    steps = np.arange(1, horizon + 1)
    results = []

    for series_id, grp, winning_model in chunk:
        grp = grp.sort_values(date_col)
        history = grp[value_col].values.astype(float)
        dates = pd.to_datetime(grp[date_col]).values

        start = pd.Timestamp(dates[-1]) + pd.Timedelta(weeks=1)
        forecast_dates = pd.date_range(start, periods=horizon, freq="W-SUN")

        preds = _forecast_single_series(history, dates, horizon, [winning_model])

        chunk_df = pd.DataFrame({
            id_col: series_id,
            "model": winning_model,
            date_col: forecast_dates,
            "forecast": preds[winning_model],
            "step": steps,
        })
        results.append(chunk_df)

    return results


# =============================================================
# Best-Fit Tournament Selection
# =============================================================

def select_best_fit(eval_df: pd.DataFrame, id_col: str,
                    primary_metric: str = "wape",
                    secondary_metric: str = "mase",
                    primary_weight: float = 0.7,
                    secondary_weight: float = 0.3,
                    fallback_model: str = "seasonal_naive") -> pd.DataFrame:
    '''Tournament-style best-fit model selection per series.'''
    best_fits = []

    for series_id, grp in eval_df.groupby(id_col):
        grp = grp.copy()

        valid = grp[
            np.isfinite(grp[primary_metric]) & np.isfinite(grp[secondary_metric])
        ].copy()

        if len(valid) == 0:
            fallback_row = grp[grp["model"] == fallback_model]
            if len(fallback_row) == 0:
                fallback_row = grp.iloc[:1]
            row = fallback_row.iloc[0]
            best_fits.append({
                id_col: series_id,
                "best_model": row["model"],
                "composite_score": np.nan,
                primary_metric: row[primary_metric],
                secondary_metric: row[secondary_metric],
                "total_actual": row["total_actual"],
                "total_forecast": row["total_forecast"],
                "n_folds": row.get("n_folds", 0),
                "wape_std": row.get("wape_std", np.nan),
                "selection_method": "fallback",
            })
            continue

        for metric in [primary_metric, secondary_metric]:
            col = valid[metric]
            rng = col.max() - col.min()
            valid[f"{metric}_norm"] = (col - col.min()) / rng if rng > 0 else 0.0

        valid["composite_score"] = (
            primary_weight * valid[f"{primary_metric}_norm"]
            + secondary_weight * valid[f"{secondary_metric}_norm"]
        )

        min_score = valid["composite_score"].min()
        winners = valid[valid["composite_score"] == min_score]
        method = "tournament" if len(winners) == 1 else "tiebreak"
        winner = winners.iloc[0] if len(winners) == 1 else winners.loc[winners[primary_metric].idxmin()]

        best_fits.append({
            id_col: series_id,
            "best_model": winner["model"],
            "composite_score": winner["composite_score"],
            primary_metric: winner[primary_metric],
            secondary_metric: winner[secondary_metric],
            "total_actual": winner["total_actual"],
            "total_forecast": winner["total_forecast"],
            "n_folds": winner.get("n_folds", 0),
            "wape_std": winner.get("wape_std", np.nan),
            "selection_method": method,
        })

    return pd.DataFrame(best_fits)


# =============================================================
# Quick Summary Utility
# =============================================================

def summarize_results(best_fit_df: pd.DataFrame, eval_df: pd.DataFrame,
                      id_col: str,
                      inactive_df: pd.DataFrame = None) -> pd.DataFrame:
    '''Summary statistics for evaluation results.'''
    if inactive_df is not None and len(inactive_df) > 0:
        n_active = len(best_fit_df)
        n_inactive = len(inactive_df)
        n_total = n_active + n_inactive
        print(f"Activity Summary: {n_active} active ({n_active/n_total*100:.1f}%) | "
              f"{n_inactive} inactive ({n_inactive/n_total*100:.1f}%) | "
              f"{n_total} total")

    wins = best_fit_df["best_model"].value_counts().rename("wins")
    total = len(best_fit_df)

    valid_eval = eval_df[np.isfinite(eval_df["wape"]) & np.isfinite(eval_df["mase"])]

    medians = valid_eval.groupby("model").agg(
        median_wape=("wape", "median"),
        median_mase=("mase", "median"),
    )

    dw = valid_eval.groupby("model").apply(
        lambda g: (g["wape"] * g["total_actual"]).sum() / g["total_actual"].sum()
        if g["total_actual"].sum() > 0 else np.nan
    ).rename("demand_weighted_wape")

    if "wape_std" in valid_eval.columns:
        stability = valid_eval.groupby("model")["wape_std"].median().rename("median_wape_std")
    else:
        stability = pd.Series(dtype=float, name="median_wape_std")

    summary = pd.DataFrame({"wins": wins})
    summary["win_pct"] = (summary["wins"] / total * 100).round(1)
    summary = summary.join(medians).join(dw).join(stability)
    summary = summary.sort_values("wins", ascending=False).reset_index().rename(
        columns={"index": "model"}
    )

    return summary


# =============================================================
# Parallel Evaluation
# =============================================================

def evaluate_benchmarks_parallel(df: pd.DataFrame, date_col: str, id_col: str,
                                  value_col: str, eval_window: int = 13,
                                  n_folds: int = 3, fold_spacing: int = 13,
                                  min_train_weeks: int = 52,
                                  n_workers: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Parallel rolling-origin cross-validation evaluation.
    Distributes series across workers using multiprocessing.Pool.
    '''
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    groups = [(series_id, grp) for series_id, grp in df.groupby(id_col)]
    chunks = _chunk_groups(groups, n_workers)

    worker_fn = partial(
        _eval_worker,
        date_col=date_col,
        id_col=id_col,
        value_col=value_col,
        eval_window=eval_window,
        n_folds=n_folds,
        fold_spacing=fold_spacing,
        min_train_weeks=min_train_weeks,
    )

    if n_workers == 1:
        results = [worker_fn(chunk) for chunk in chunks]
    else:
        with Pool(processes=n_workers) as pool:
            results = pool.map(worker_fn, chunks)

    all_eval_results = []
    all_fold_details = []
    total_skipped = 0

    for eval_results, fold_details, skipped in results:
        all_eval_results.extend(eval_results)
        all_fold_details.extend(fold_details)
        total_skipped += skipped

    min_required = min_train_weeks + eval_window + (n_folds - 1) * fold_spacing
    if total_skipped:
        print(f"WARNING: Skipped {total_skipped} series with < {min_required} weeks of history "
              f"(need {min_train_weeks} train + {eval_window} eval + "
              f"{(n_folds - 1) * fold_spacing} spacing).")

    return pd.DataFrame(all_eval_results), pd.DataFrame(all_fold_details)


# =============================================================
# Parallel End-to-End Pipeline
# =============================================================

def best_fit_pipeline_parallel(df: pd.DataFrame, date_col: str, id_col: str,
                                value_col: str, horizon: int = 52,
                                eval_window: int = 13, n_folds: int = 3,
                                fold_spacing: int = 13,
                                inactive_weeks: int = 26,
                                primary_metric: str = "wape",
                                secondary_metric: str = "mase",
                                primary_weight: float = 0.7,
                                secondary_weight: float = 0.3,
                                n_workers: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    End-to-end best-fit forecasting pipeline with parallel processing.

    Steps:
        0. FILTER   — flag inactive items (single-process, already fast)
        1. EVALUATE — rolling-origin backtest (distributed across workers)
        2. SELECT   — tournament on fold-averaged metrics (single-process)
        3. FORECAST — generate final forecasts (distributed across workers)

    Returns (eval_df, best_fit_df, forecast_df, inactive_df).
    '''
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    total_items = df[id_col].nunique()
    print(f"Pipeline starting with {n_workers} workers...")

    # Step 0: Filter inactive items (single-process)
    if inactive_weeks > 0:
        print(f"Step 0/3: Filtering inactive items (zero demand in last {inactive_weeks} weeks)...")
        active_df, inactive_df = flag_inactive_items(
            df, date_col, id_col, value_col, inactive_weeks
        )
        n_inactive = len(inactive_df)
        n_active = active_df[id_col].nunique()
        print(f"  {n_active} active / {n_inactive} inactive out of {total_items} total items")
        if n_inactive > 0 and "weeks_since_demand" in inactive_df.columns:
            median_weeks = inactive_df["weeks_since_demand"].median()
            print(f"  Inactive median weeks since last demand: {median_weeks:.0f}")
    else:
        print("Step 0/3: Inactive filtering disabled (inactive_weeks=0)")
        active_df = df.copy()
        inactive_df = pd.DataFrame()

    if active_df[id_col].nunique() == 0:
        print("WARNING: No active items remain after filtering. Returning empty results.")
        empty = pd.DataFrame()
        return empty, empty, empty, inactive_df

    # Step 1: Parallel evaluation
    print(f"Step 1/3: Rolling-origin evaluation ({n_folds} folds x "
          f"{eval_window}-week window, {fold_spacing}-week spacing) "
          f"with {n_workers} workers...")
    eval_df, fold_details_df = evaluate_benchmarks_parallel(
        active_df, date_col, id_col, value_col,
        eval_window=eval_window, n_folds=n_folds, fold_spacing=fold_spacing,
        n_workers=n_workers,
    )
    n_series = eval_df[id_col].nunique()
    n_models = eval_df["model"].nunique()
    print(f"  Evaluated {n_series} series x {n_models} models")

    # Step 2: Tournament selection (single-process)
    print("Step 2/3: Running best-fit tournament on fold-averaged metrics...")
    best_fit_df = select_best_fit(
        eval_df, id_col, primary_metric, secondary_metric,
        primary_weight, secondary_weight
    )
    print(f"  Model distribution:")
    print(f"  {best_fit_df['best_model'].value_counts().to_string()}")
    print(f"  Selection methods: {best_fit_df['selection_method'].value_counts().to_dict()}")

    # Step 3: Parallel forecast generation
    print(f"Step 3/3: Generating {horizon}-week forecasts (winners only) "
          f"with {n_workers} workers...")
    winner_map = best_fit_df.set_index(id_col)["best_model"].to_dict()

    forecast_groups = []
    for series_id, grp in active_df.groupby(id_col):
        winning_model = winner_map.get(series_id)
        if winning_model is not None:
            forecast_groups.append((series_id, grp, winning_model))

    forecast_chunks = _chunk_groups(forecast_groups, n_workers)

    forecast_worker_fn = partial(
        _forecast_worker,
        date_col=date_col,
        id_col=id_col,
        value_col=value_col,
        horizon=horizon,
    )

    if n_workers == 1:
        forecast_results = [forecast_worker_fn(chunk) for chunk in forecast_chunks]
    else:
        with Pool(processes=n_workers) as pool:
            forecast_results = pool.map(forecast_worker_fn, forecast_chunks)

    all_forecast_dfs = []
    for result_list in forecast_results:
        all_forecast_dfs.extend(result_list)

    forecast_df = pd.concat(all_forecast_dfs, ignore_index=True) if all_forecast_dfs else pd.DataFrame()
    print(f"  Generated {len(forecast_df)} forecast rows for "
          f"{forecast_df[id_col].nunique() if len(forecast_df) > 0 else 0} series")

    return eval_df, best_fit_df, forecast_df, inactive_df


# =============================================================
# Convenience Alias
# =============================================================

best_fit_pipeline = best_fit_pipeline_parallel
