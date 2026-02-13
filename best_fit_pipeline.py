"""
benchmark_forecasting.py
========================
A best-fit benchmark forecasting toolkit for intermittent and lumpy
weekly demand series. Designed for large-scale use across thousands
of series with a 52-week forecast horizon.

Models Included:
    - Seasonal Naïve
    - Weekly Historical Average
    - Global Average (Flat)
    - Teunter-Syntetos-Babai (TSB)
    - Temporal Aggregation + Simple Exponential Smoothing (SES)

Evaluation Metrics:
    - WAPE (Weighted Absolute Percentage Error)
    - MASE (Mean Absolute Scaled Error)

Selection:
    - Tournament-style best-fit model selection per series using
      rolling-origin cross-validation with a configurable evaluation
      window (default 13 weeks / 3 months). Metrics are averaged
      across folds for robust model selection.

Pre-Processing:
    - Inactive item detection and exclusion. Items with zero demand
      in the trailing N weeks (default 26) are flagged as inactive
      and excluded from the forecasting pipeline.

Version History:
    v1.0.0  2026-02-13  Initial release. Five benchmark models, WAPE/MASE
                         evaluation, tournament selection, end-to-end pipeline.
    v1.1.0  2026-02-13  Refactored for efficiency: vectorized forecast
                         generation, eliminated per-row apply filtering,
                         consolidated model registry, added progress logging.
    v1.2.0  2026-02-13  Rolling-origin cross-validation with configurable
                         eval_window (default 13 weeks for 3-month WAPE).
                         Metrics averaged across folds for stable selection.
                         Decoupled eval_window from forecast horizon. Added
                         per-fold detail in evaluation output.
    v1.3.0  2026-02-13  Added inactive item detection. Items with zero demand
                         in the trailing window (default 26 weeks) are flagged
                         and excluded before benchmarking. Pipeline returns a
                         4-tuple including the inactive items DataFrame.

Author: [Your Name / Team]
License: [Your License]
"""

__version__ = "1.3.0"

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Callable, List

# =============================================================
# Model Registry
# =============================================================
# All models are registered here so they can be referenced by name
# throughout the evaluation, selection, and forecasting pipeline.
# To add a new benchmark, define the function and add it to MODEL_REGISTRY.

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
    Flag and Separate Inactive Items  [v1.3.0]
    ============================================
    Identifies items with zero total demand in the most recent
    `inactive_weeks` weeks and separates them from active items.
    Inactive items are excluded from the forecasting pipeline to
    avoid wasting compute on likely discontinued or dormant series.

    An item is considered INACTIVE if:
        - The sum of its demand over the last `inactive_weeks` weeks
          (based on the most recent date in the ENTIRE dataset) is
          exactly zero, OR
        - The item has no data rows at all in the trailing window.

    This uses the global max date across all series as the reference
    point, so all items are evaluated against the same calendar window.
    This prevents items that simply stopped receiving data rows from
    being missed.

    Inputs
    ------
    df : pd.DataFrame
        Long-format DataFrame, one row per series per week.

        Expected schema:
            - date_col  : datetime-like — week-ending Sunday dates
            - id_col    : str/categorical — unique series identifier
            - value_col : numeric — non-negative demand

    date_col : str
        Column name for week-ending dates.

    id_col : str
        Column name for series identifiers.

    value_col : str
        Column name for demand values.

    inactive_weeks : int, default 26
        Number of trailing weeks to check for activity. Items with
        zero total demand in this window are flagged inactive.
        26 weeks ≈ 6 months is a reasonable default; adjust based
        on your product lifecycle (e.g., 13 for faster-moving goods,
        52 for slow-moving capital equipment).

    Output
    ------
    Tuple of (active_df, inactive_df)

        active_df : pd.DataFrame
            Subset of `df` containing only rows for active items
            (same schema as input). Ready to pass into the pipeline.

        inactive_df : pd.DataFrame
            One row per inactive item with diagnostic details.

            Schema:
                - id_col               : str      — series identifier
                - "status"             : str      — always "inactive"
                - "last_nonzero_date"  : datetime — most recent date with
                                                    demand > 0 (NaT if never)
                - "weeks_since_demand" : int      — weeks between last nonzero
                                                    demand and the global max
                                                    date (NaN if never)
                - "total_history_weeks": int      — total weeks of data for item
                - "lifetime_total_qty" : float    — sum of all demand ever

            Example:
                | item_id | status   | last_nonzero_date | weeks_since_demand | ... |
                |---------|----------|-------------------|--------------------|-----|
                | SKU_099 | inactive | 2025-03-15        |                 47 | ... |
                | SKU_100 | inactive | NaT               |                NaN | ... |
    '''
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    global_max_date = df[date_col].max()
    cutoff_date = global_max_date - pd.Timedelta(weeks=inactive_weeks)

    # --- Trailing window demand per item ---
    trailing = df[df[date_col] > cutoff_date]
    trailing_demand = trailing.groupby(id_col)[value_col].sum()

    # --- All item IDs (including those with no trailing rows) ---
    all_ids = df[id_col].unique()
    trailing_demand = trailing_demand.reindex(all_ids, fill_value=0)

    inactive_ids = trailing_demand[trailing_demand == 0].index
    active_ids = trailing_demand[trailing_demand > 0].index

    # --- Build inactive diagnostics ---
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
    '''
    Seasonal Naïve Forecast  [v1.0.0]
    ==================================
    Repeats the last 52 weeks of observed demand as the forecast.
    This is the most common benchmark for weekly seasonal data and is
    often surprisingly hard to beat, especially for intermittent series
    with strong yearly patterns.

    If the history is shorter than 52 weeks, the available data is tiled
    (repeated) to fill a full 52-week seasonal cycle.

    Inputs
    ------
    history : np.ndarray of shape (n,)
        Historical demand values, ordered chronologically (oldest first).
        Numeric, non-negative.

    dates : np.ndarray of datetime64
        Week-ending Sunday dates aligned 1-to-1 with history.
        Not used by this model but accepted for interface consistency.

    horizon : int, default 52
        Number of future weekly periods to forecast.

    Output
    ------
    np.ndarray of shape (horizon,)
        Forecasted demand. Each value is the observed demand from the
        same relative week in the last year of history.
    '''
    last_year = history[-52:]
    if len(last_year) < 52:
        last_year = np.tile(history, (52 // len(history) + 1))[-52:]
    reps = (horizon // 52) + 1
    return np.tile(last_year, reps)[:horizon]


@register_model("weekly_hist_avg")
def weekly_historical_average(history: np.ndarray, dates: np.ndarray,
                               horizon: int = 52, **kwargs) -> np.ndarray:
    '''
    Weekly Historical Average Forecast  [v1.0.0]
    ==============================================
    Computes the average demand for each ISO week-of-year (1-53) across
    all years of available history, then maps each future forecast week
    to its corresponding week-of-year average. Preserves seasonal
    patterns while smoothing out year-to-year noise.

    Falls back to the global mean for any week-of-year with no history.

    Inputs
    ------
    history : np.ndarray of shape (n,)
        Historical demand values, ordered chronologically. Numeric.

    dates : np.ndarray of datetime64
        Week-ending Sunday dates aligned 1-to-1 with history.

    horizon : int, default 52
        Number of future weekly periods to forecast.

    Output
    ------
    np.ndarray of shape (horizon,)
        Forecasted demand. Each entry is the historical average demand
        for the ISO week-of-year that the forecast date falls in.
    '''
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
    '''
    Global Average (Flat) Forecast  [v1.0.0]
    =========================================
    Returns a constant forecast equal to the overall historical mean.
    Captures no seasonality or trend. Its primary purpose is as a
    sanity check: if a more complex model can't beat this, it isn't
    capturing useful structure.

    Inputs
    ------
    history : np.ndarray of shape (n,)
        Historical demand values, ordered chronologically. Numeric.

    dates : np.ndarray of datetime64
        Not used. Accepted for interface consistency.

    horizon : int, default 52
        Number of future weekly periods to forecast.

    Output
    ------
    np.ndarray of shape (horizon,)
        Every element is the arithmetic mean of `history`.
    '''
    return np.full(horizon, history.mean())


@register_model("tsb")
def tsb_forecast(history: np.ndarray, dates: np.ndarray,
                 horizon: int = 52, alpha_d: float = 0.1,
                 alpha_p: float = 0.1, **kwargs) -> np.ndarray:
    '''
    Teunter-Syntetos-Babai (TSB) Forecast  [v1.0.0]
    =================================================
    Designed for intermittent demand. Separately smooths two components:
        1. Demand size — exponential smoothing of non-zero demand values
        2. Demand probability — probability of demand occurring, which
           decays toward zero during consecutive zero-demand periods

    Forecast = smoothed_probability × smoothed_size

    The decay property makes TSB better than Croston's/SBA for items
    approaching obsolescence. Produces a flat (non-seasonal) forecast.

    Inputs
    ------
    history : np.ndarray of shape (n,)
        Historical demand values, ordered chronologically. Non-negative.

    dates : np.ndarray of datetime64
        Not used. Accepted for interface consistency.

    horizon : int, default 52
        Number of future weekly periods to forecast.

    alpha_d : float, default 0.1
        Smoothing parameter for demand size (0 < alpha_d < 1).

    alpha_p : float, default 0.1
        Smoothing parameter for demand probability (0 < alpha_p < 1).

    Output
    ------
    np.ndarray of shape (horizon,)
        Constant forecast: final smoothed_probability × smoothed_size,
        floored at 0.0.
    '''
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
    '''
    Temporal Aggregation + Simple Exponential Smoothing  [v1.0.0]
    ==============================================================
    Two-stage approach for intermittent weekly data:

    Stage 1 — Aggregate & Forecast:
        Weekly data is summed into larger buckets (default 4-week blocks).
        Aggregation reduces zero-inflation, making the series amenable to
        standard methods. SES forecasts one aggregated period ahead.

    Stage 2 — Disaggregate:
        The SES forecast is distributed back to weekly granularity using
        seasonal weights derived from historical week-of-year averages.
        Within each block, weights are normalized so the block total
        equals the SES forecast.

    Inputs
    ------
    history : np.ndarray of shape (n,)
        Historical demand values, ordered chronologically. Non-negative.

    dates : np.ndarray of datetime64
        Week-ending Sunday dates aligned 1-to-1 with history.

    horizon : int, default 52
        Number of future weekly periods to forecast.

    agg_periods : int, default 4
        Weeks per aggregation bucket (4 ≈ monthly, 13 ≈ quarterly).

    alpha : float, default 0.2
        SES smoothing parameter (0 < alpha < 1).

    Output
    ------
    np.ndarray of shape (horizon,)
        Weekly forecasted demand. Level from SES on aggregated series;
        weekly shape from historical seasonal proportions.
    '''
    values = history.astype(float)
    n = len(values)

    # --- Aggregate ---
    trim = n - (n % agg_periods)
    trimmed = values[-trim:]
    agg = trimmed.reshape(-1, agg_periods).sum(axis=1)

    # --- SES on aggregated ---
    level = agg[0]
    for v in agg[1:]:
        level = alpha * v + (1 - alpha) * level
    agg_forecast = max(level, 0.0)

    # --- Seasonal weights for disaggregation ---
    dates_ts = pd.DatetimeIndex(dates)
    week_numbers = dates_ts.isocalendar().week.astype(int).values

    week_means = {}
    for w in range(1, 54):
        mask = week_numbers == w
        if mask.any():
            week_means[w] = values[mask].mean()
    overall_mean = np.mean(list(week_means.values())) if week_means else 1.0
    seasonal_index = {w: m / overall_mean for w, m in week_means.items()} if overall_mean > 0 else {}

    # --- Build weekly forecast ---
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


# =============================================================
# Evaluation Metrics
# =============================================================

def calculate_wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    '''
    Weighted Absolute Percentage Error (WAPE)  [v1.0.0]
    ====================================================
    Total absolute error divided by total actual demand. Avoids
    division-by-zero on individual periods by aggregating first.

    Formula:  WAPE = sum(|actual - forecast|) / sum(|actual|)

    Inputs
    ------
    actual : np.ndarray of shape (n,)
        Observed demand values for the evaluation period. Non-negative.

    forecast : np.ndarray of shape (n,)
        Forecasted demand values, aligned 1-to-1 with actuals.

    Output
    ------
    float
        WAPE score (0.0 = perfect). Returns np.inf if sum of actuals is zero.
    '''
    total_actual = np.sum(np.abs(actual))
    if total_actual == 0:
        return np.inf
    return np.sum(np.abs(actual - forecast)) / total_actual


def calculate_mase(actual: np.ndarray, forecast: np.ndarray,
                   in_sample: np.ndarray, seasonal_period: int = 52) -> float:
    '''
    Mean Absolute Scaled Error (MASE)  [v1.0.0]
    =============================================
    Forecast accuracy relative to the in-sample seasonal naïve MAE.
    MASE < 1.0 means the model outperforms seasonal naïve; > 1.0
    means it does worse.

    Formula:
        naïve_mae = mean(|in_sample[t] - in_sample[t - seasonal_period]|)
        MASE      = mean(|actual - forecast|) / naïve_mae

    Inputs
    ------
    actual : np.ndarray of shape (n,)
        Observed demand for the holdout period. Non-negative.

    forecast : np.ndarray of shape (n,)
        Forecasted demand, aligned 1-to-1 with actuals.

    in_sample : np.ndarray of shape (m,)
        Training demand values (chronological). Must have length > seasonal_period.

    seasonal_period : int, default 52
        Seasonal cycle length in periods (52 for weekly/yearly).

    Output
    ------
    float
        MASE score. Returns np.inf if in-sample naïve MAE is zero or
        history is too short.
    '''
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
    '''
    Run all (or selected) registered models on a single series.  [v1.1.0]

    Inputs
    ------
    history : np.ndarray of shape (n,)
        Demand values, chronological.

    dates : np.ndarray of datetime64
        Corresponding week-ending dates.

    horizon : int
        Forecast horizon in weeks.

    model_names : list of str or None
        If provided, only run these models. If None, run all registered.

    Output
    ------
    dict of {model_name: np.ndarray of shape (horizon,)}
    '''
    names = model_names or list(MODEL_REGISTRY.keys())
    results = {}
    for name in names:
        try:
            results[name] = MODEL_REGISTRY[name](history, dates, horizon)
        except Exception as e:
            results[name] = np.full(horizon, np.nan)
    return results


# =============================================================
# Forecast Generation (All Models x All Series)
# =============================================================

def run_all_benchmarks(df: pd.DataFrame, date_col: str, id_col: str,
                       value_col: str, horizon: int = 52,
                       model_names: list = None) -> pd.DataFrame:
    '''
    Apply All Benchmark Models Across Multiple Series  [v1.1.0]
    =============================================================
    Iterates over every unique series in a long-format DataFrame,
    generates forecasts from each benchmark model, and returns a
    consolidated long-format DataFrame.

    Inputs
    ------
    df : pd.DataFrame
        Long-format DataFrame, one row per series per week.

        Expected schema:
            - date_col  : datetime-like — week-ending Sunday dates
            - id_col    : str/categorical — unique series identifier
            - value_col : numeric — non-negative demand quantity

        Example:
            | date       | item_id | demand |
            |------------|---------|--------|
            | 2023-01-08 | SKU_001 |      0 |
            | 2023-01-15 | SKU_001 |      5 |

    date_col : str
        Column name for week-ending dates.

    id_col : str
        Column name for series identifiers.

    value_col : str
        Column name for demand values.

    horizon : int, default 52
        Number of future weekly periods to forecast.

    model_names : list of str or None
        If provided, only run these models. If None, run all registered.

    Output
    ------
    pd.DataFrame
        Long-format, one row per series x model x forecast week.

        Schema:
            - id_col     : str      — series identifier
            - "model"    : str      — benchmark model name
            - date_col   : datetime — forecast week date
            - "forecast" : float    — predicted demand
            - "step"     : int      — forecast step (1 = first week ahead)
    '''
    names = model_names or list(MODEL_REGISTRY.keys())
    steps = np.arange(1, horizon + 1)
    chunks = []

    for series_id, grp in df.groupby(id_col):
        grp = grp.sort_values(date_col)
        history = grp[value_col].values.astype(float)
        dates = pd.to_datetime(grp[date_col]).values

        start = pd.Timestamp(dates[-1]) + pd.Timedelta(weeks=1)
        forecast_dates = pd.date_range(start, periods=horizon, freq="W-SUN")

        model_outputs = _forecast_single_series(history, dates, horizon, names)

        for model_name, preds in model_outputs.items():
            chunk = pd.DataFrame({
                id_col: series_id,
                "model": model_name,
                date_col: forecast_dates,
                "forecast": preds,
                "step": steps,
            })
            chunks.append(chunk)

    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


# =============================================================
# Rolling-Origin Cross-Validation Evaluation
# =============================================================

def evaluate_benchmarks(df: pd.DataFrame, date_col: str, id_col: str,
                        value_col: str, eval_window: int = 13,
                        n_folds: int = 3, fold_spacing: int = 13,
                        min_train_weeks: int = 52) -> pd.DataFrame:
    '''
    Rolling-Origin Cross-Validation Evaluation  [v1.2.0]
    =====================================================
    Evaluates all benchmark models using rolling-origin backtesting.
    Multiple evaluation folds are created by sliding the train/test
    cutoff backward through time, each using a fixed-width evaluation
    window. Metrics are computed per fold, then averaged across folds
    for stable model selection.

    This approach is aligned with 3-month WAPE evaluation: each fold
    tests on a 13-week window, and the average across folds prevents
    one anomalous quarter from dominating the selection.

    Fold Layout (example: n_folds=3, eval_window=13, fold_spacing=13)
    ------------------------------------------------------------------
    Given a series with 130 weeks of data:

        Fold 1 (most recent):
            Train: weeks 1-117    |  Test: weeks 118-130

        Fold 2:
            Train: weeks 1-104    |  Test: weeks 105-117

        Fold 3:
            Train: weeks 1-91     |  Test: weeks 92-104

    Each fold's training set is used to generate a 13-week forecast,
    which is scored against the corresponding 13-week test window.

    Series with insufficient history are skipped. The minimum required
    length is: min_train_weeks + eval_window + (n_folds - 1) * fold_spacing.

    Inputs
    ------
    df : pd.DataFrame
        Long-format DataFrame, one row per series per week.

        Expected schema:
            - date_col  : datetime-like — week-ending Sunday dates
            - id_col    : str/categorical — unique series identifier
            - value_col : numeric — non-negative demand

    date_col : str
        Column name for week-ending dates.

    id_col : str
        Column name for series identifiers.

    value_col : str
        Column name for demand values.

    eval_window : int, default 13
        Number of weeks in each evaluation (test) window. Set to 13
        for 3-month WAPE alignment. This is independent of the final
        forecast horizon.

    n_folds : int, default 3
        Number of rolling-origin folds. More folds = more stable
        evaluation but requires longer history. Recommended: 3-4.

    fold_spacing : int, default 13
        Number of weeks between the start of consecutive folds.
        Default 13 (one quarter) ensures folds don't overlap when
        eval_window=13 and captures different seasonal quarters.

    min_train_weeks : int, default 52
        Minimum number of training weeks required for the earliest
        fold. Set to 52 to ensure at least one full year of training
        data is always available.

    Output
    ------
    pd.DataFrame
        One row per series x model with metrics averaged across folds.

        Schema:
            - id_col           : str   — series identifier
            - "model"          : str   — benchmark model name
            - "wape"           : float — mean WAPE across folds
            - "mase"           : float — mean MASE across folds
            - "total_actual"   : float — sum of actual demand across all folds
            - "total_forecast" : float — sum of forecasted demand across all folds
            - "n_folds"        : int   — number of folds successfully evaluated
            - "wape_std"       : float — std dev of WAPE across folds (stability)
            - "fold_wapes"     : str   — comma-separated WAPE per fold (diagnostics)
    '''
    min_required = min_train_weeks + eval_window + (n_folds - 1) * fold_spacing
    skipped = 0
    eval_results = []

    for series_id, grp in df.groupby(id_col):
        grp = grp.sort_values(date_col).reset_index(drop=True)
        n = len(grp)

        if n < min_required:
            skipped += 1
            continue

        all_values = grp[value_col].values.astype(float)
        all_dates = pd.to_datetime(grp[date_col]).values

        # --- Build fold cutpoints ---
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

            folds_evaluated += 1

        if folds_evaluated == 0:
            skipped += 1
            continue

        # --- Average metrics across folds ---
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

    if skipped:
        print(f"WARNING: Skipped {skipped} series with < {min_required} weeks of history "
              f"(need {min_train_weeks} train + {eval_window} eval + "
              f"{(n_folds - 1) * fold_spacing} spacing).")

    return pd.DataFrame(eval_results)


# =============================================================
# Best-Fit Tournament Selection
# =============================================================

def select_best_fit(eval_df: pd.DataFrame, id_col: str,
                    primary_metric: str = "wape",
                    secondary_metric: str = "mase",
                    primary_weight: float = 0.7,
                    secondary_weight: float = 0.3,
                    fallback_model: str = "seasonal_naive") -> pd.DataFrame:
    '''
    Tournament-Style Best-Fit Model Selection  [v1.2.0]
    ====================================================
    For each series, selects the model with the lowest weighted
    composite score of two evaluation metrics (averaged across
    rolling-origin folds in v1.2.0+).

    Tournament Logic
    ----------------
    Per series:
        1. FILTER out models with inf/nan metrics.
        2. NORMALIZE both metrics to [0, 1] via min-max within-series.
        3. COMPOSITE = primary_weight x norm_primary + secondary_weight x norm_secondary
        4. SELECT the model with the lowest composite score.
        5. TIEBREAK: lowest raw primary metric wins.
        6. FALLBACK: if all models invalid, assign fallback_model.

    Inputs
    ------
    eval_df : pd.DataFrame
        Output of `evaluate_benchmarks()`. One row per series x model.

        Expected schema:
            - id_col               : str   — series identifier
            - "model"              : str   — model name
            - primary_metric col   : float — e.g. "wape" (averaged across folds)
            - secondary_metric col : float — e.g. "mase" (averaged across folds)
            - "total_actual"       : float
            - "total_forecast"     : float
            - "n_folds"            : int   — number of folds evaluated
            - "wape_std"           : float — fold-to-fold WAPE variability

    id_col : str
        Series identifier column name.

    primary_metric : str, default "wape"
        Primary evaluation metric (lower is better).

    secondary_metric : str, default "mase"
        Secondary evaluation metric (lower is better).

    primary_weight : float, default 0.7
        Weight for normalized primary metric.

    secondary_weight : float, default 0.3
        Weight for normalized secondary metric.

    fallback_model : str, default "seasonal_naive"
        Model assigned when no valid scores exist for a series.

    Output
    ------
    pd.DataFrame
        One row per series.

        Schema:
            - id_col              : str   — series identifier
            - "best_model"        : str   — winning model name
            - "composite_score"   : float — winning composite score
            - primary_metric      : float — winner's mean primary metric
            - secondary_metric    : float — winner's mean secondary metric
            - "total_actual"      : float — holdout actual demand (all folds)
            - "total_forecast"    : float — winner's total forecast (all folds)
            - "n_folds"           : int   — folds evaluated for winner
            - "wape_std"          : float — winner's WAPE std across folds
            - "selection_method"  : str   — "tournament", "tiebreak", or "fallback"
    '''
    best_fits = []

    for series_id, grp in eval_df.groupby(id_col):
        grp = grp.copy()

        # --- Filter invalid metrics ---
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

        # --- Min-max normalize within series ---
        for metric in [primary_metric, secondary_metric]:
            col = valid[metric]
            rng = col.max() - col.min()
            valid[f"{metric}_norm"] = (col - col.min()) / rng if rng > 0 else 0.0

        valid["composite_score"] = (
            primary_weight * valid[f"{primary_metric}_norm"]
            + secondary_weight * valid[f"{secondary_metric}_norm"]
        )

        # --- Select winner ---
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
# End-to-End Pipeline
# =============================================================

def best_fit_pipeline(df: pd.DataFrame, date_col: str, id_col: str,
                      value_col: str, horizon: int = 52,
                      eval_window: int = 13, n_folds: int = 3,
                      fold_spacing: int = 13,
                      inactive_weeks: int = 26,
                      primary_metric: str = "wape",
                      secondary_metric: str = "mase",
                      primary_weight: float = 0.7,
                      secondary_weight: float = 0.3) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    End-to-End Best-Fit Forecasting Pipeline  [v1.3.0]
    ====================================================
    Chains inactive filtering, rolling-origin evaluation, tournament
    selection, and final forecast generation. Recommended entry point
    for production use.

    Key change in v1.3.0: adds Step 0 to flag and exclude inactive
    items (zero demand in trailing `inactive_weeks`). Returns a 4-tuple
    including the inactive items DataFrame. Set inactive_weeks=0 to
    skip filtering and include all items.

    Steps:
        0. FILTER   — flag inactive items and remove from pipeline.
        1. EVALUATE — rolling-origin backtest on active items only.
        2. SELECT   — tournament on fold-averaged metrics.
        3. FORECAST — re-fit winner on FULL history, produce horizon-week forecast.

    Inputs
    ------
    df : pd.DataFrame
        Long-format DataFrame.

        Expected schema:
            - date_col  : datetime-like — week-ending Sunday dates
            - id_col    : str/categorical — unique series identifier
            - value_col : numeric — non-negative demand

    date_col : str
        Column name for week-ending dates.

    id_col : str
        Column name for series identifiers.

    value_col : str
        Column name for demand values.

    horizon : int, default 52
        Weeks to forecast into the future (final output).

    eval_window : int, default 13
        Weeks per evaluation fold (3 months). This is the window
        over which WAPE and MASE are computed for model selection.

    n_folds : int, default 3
        Number of rolling-origin folds for evaluation.

    fold_spacing : int, default 13
        Weeks between the start of consecutive folds.

    inactive_weeks : int, default 26
        Trailing weeks to check for activity. Items with zero demand
        in this window are excluded. Set to 0 to disable filtering
        and include all items.

    primary_metric : str, default "wape"
        Primary metric for tournament scoring.

    secondary_metric : str, default "mase"
        Secondary metric for tournament scoring.

    primary_weight : float, default 0.7
        Weight for primary metric in composite.

    secondary_weight : float, default 0.3
        Weight for secondary metric in composite.

    Output
    ------
    Tuple of (eval_df, best_fit_df, forecast_df, inactive_df)

        eval_df : pd.DataFrame
            Full evaluation results for all models x active series,
            metrics averaged across rolling-origin folds.
            See `evaluate_benchmarks` output schema.

        best_fit_df : pd.DataFrame
            Tournament results, one row per active series.
            See `select_best_fit` output schema.

        forecast_df : pd.DataFrame
            Final forward-looking forecasts (full horizon), containing
            only the winning model's output per active series.

            Schema:
                - id_col     : str      — series identifier
                - "model"    : str      — best-fit model name
                - date_col   : datetime — forecast week date
                - "forecast" : float    — predicted demand
                - "step"     : int      — forecast step (1-horizon)

        inactive_df : pd.DataFrame
            Inactive items excluded from the pipeline. One row per
            inactive item with diagnostics (last demand date, weeks
            since demand, lifetime quantity).
            See `flag_inactive_items` output schema.

    Example
    -------
        eval_df, best_fit_df, forecast_df, inactive_df = best_fit_pipeline(
            df, date_col="date", id_col="item_id", value_col="demand",
            eval_window=13, n_folds=3, inactive_weeks=26,
        )
        print(f"Active: {best_fit_df.shape[0]}, Inactive: {inactive_df.shape[0]}")
        print(best_fit_df["best_model"].value_counts())
    '''
    total_items = df[id_col].nunique()

    # Step 0: Filter inactive items
    if inactive_weeks > 0:
        print(f"Step 0/4: Filtering inactive items (zero demand in last {inactive_weeks} weeks)...")
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
        print("Step 0/4: Inactive filtering disabled (inactive_weeks=0)")
        active_df = df.copy()
        inactive_df = pd.DataFrame()

    if active_df[id_col].nunique() == 0:
        print("WARNING: No active items remain after filtering. Returning empty results.")
        empty = pd.DataFrame()
        return empty, empty, empty, inactive_df

    # Step 1: Evaluate with rolling-origin CV
    print(f"Step 1/4: Rolling-origin evaluation ({n_folds} folds x "
          f"{eval_window}-week window, {fold_spacing}-week spacing)...")
    eval_df = evaluate_benchmarks(
        active_df, date_col, id_col, value_col,
        eval_window=eval_window, n_folds=n_folds, fold_spacing=fold_spacing
    )
    n_series = eval_df[id_col].nunique()
    n_models = eval_df["model"].nunique()
    print(f"  Evaluated {n_series} series x {n_models} models")

    # Step 2: Tournament selection
    print("Step 2/4: Running best-fit tournament on fold-averaged metrics...")
    best_fit_df = select_best_fit(
        eval_df, id_col, primary_metric, secondary_metric,
        primary_weight, secondary_weight
    )
    print(f"  Model distribution:")
    print(f"  {best_fit_df['best_model'].value_counts().to_string()}")
    print(f"  Selection methods: {best_fit_df['selection_method'].value_counts().to_dict()}")

    # Step 3: Generate final forecasts — ONLY the winning model per series
    print(f"Step 3/4: Generating {horizon}-week forecasts (winners only, full history)...")
    winner_map = best_fit_df.set_index(id_col)["best_model"].to_dict()
    steps = np.arange(1, horizon + 1)
    chunks = []

    for series_id, grp in active_df.groupby(id_col):
        winning_model = winner_map.get(series_id)
        if winning_model is None:
            continue

        grp = grp.sort_values(date_col)
        history = grp[value_col].values.astype(float)
        dates = pd.to_datetime(grp[date_col]).values

        start = pd.Timestamp(dates[-1]) + pd.Timedelta(weeks=1)
        forecast_dates = pd.date_range(start, periods=horizon, freq="W-SUN")

        preds = _forecast_single_series(history, dates, horizon, [winning_model])

        chunk = pd.DataFrame({
            id_col: series_id,
            "model": winning_model,
            date_col: forecast_dates,
            "forecast": preds[winning_model],
            "step": steps,
        })
        chunks.append(chunk)

    forecast_df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    print(f"  Generated {len(forecast_df)} forecast rows for "
          f"{forecast_df[id_col].nunique()} series")

    return eval_df, best_fit_df, forecast_df, inactive_df


# =============================================================
# Quick Summary Utility
# =============================================================

def summarize_results(best_fit_df: pd.DataFrame, eval_df: pd.DataFrame,
                      id_col: str,
                      inactive_df: pd.DataFrame = None) -> pd.DataFrame:
    '''
    Summary Statistics for Evaluation Results  [v1.3.0]
    ====================================================
    Produces a concise summary table showing, for each model, its
    win count, median WAPE (averaged across folds), median MASE,
    demand-weighted WAPE, and median fold-to-fold WAPE stability.

    In v1.3.0, optionally accepts the inactive_df and prints a
    high-level activity summary before returning the model table.

    Inputs
    ------
    best_fit_df : pd.DataFrame
        Output of `select_best_fit()`.

    eval_df : pd.DataFrame
        Output of `evaluate_benchmarks()`.

    id_col : str
        Series identifier column name.

    inactive_df : pd.DataFrame or None, default None
        Output of `flag_inactive_items()`. If provided, prints
        an activity summary header before the model summary.

    Output
    ------
    pd.DataFrame
        One row per model.

        Schema:
            - "model"                : str   — model name
            - "wins"                 : int   — number of series won
            - "win_pct"              : float — percentage of total active series
            - "median_wape"          : float — median mean-WAPE across series
            - "median_mase"          : float — median mean-MASE across series
            - "demand_weighted_wape" : float — aggregate WAPE weighted by demand
            - "median_wape_std"      : float — median fold-to-fold WAPE std
                                               (lower = more stable model)
    '''
    # Activity summary
    if inactive_df is not None and len(inactive_df) > 0:
        n_active = len(best_fit_df)
        n_inactive = len(inactive_df)
        n_total = n_active + n_inactive
        print(f"Activity Summary: {n_active} active ({n_active/n_total*100:.1f}%) | "
              f"{n_inactive} inactive ({n_inactive/n_total*100:.1f}%) | "
              f"{n_total} total")

    # Win counts
    wins = best_fit_df["best_model"].value_counts().rename("wins")
    total = len(best_fit_df)

    # Median metrics from eval_df (across all series, not just wins)
    valid_eval = eval_df[np.isfinite(eval_df["wape"]) & np.isfinite(eval_df["mase"])]

    medians = valid_eval.groupby("model").agg(
        median_wape=("wape", "median"),
        median_mase=("mase", "median"),
    )

    # Demand-weighted WAPE
    dw = valid_eval.groupby("model").apply(
        lambda g: (g["wape"] * g["total_actual"]).sum() / g["total_actual"].sum()
        if g["total_actual"].sum() > 0 else np.nan
    ).rename("demand_weighted_wape")

    # Stability: median of per-series WAPE std across folds
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

# To run the whole pipeline 
eval_df, best_fit_df, forecast_df, inactive_df = best_fit_pipeline(
    df, date_col="date", id_col="item_id", value_col="demand",
    inactive_weeks=26,  # 6-month lookback
)