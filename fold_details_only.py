“””
generate_fold_details.py

Generates a fold_details DataFrame that shows the actual and forecasted
values for each week, for each series, for each model in the holdout
test set.

Usage:
from generate_fold_details import generate_fold_details

```
fold_details_df = generate_fold_details(
    df, date_col="date", id_col="item_id", value_col="demand",
    eval_window=52, min_train_weeks=52, n_workers=4,
)
```

The returned DataFrame has columns:
- <id_col>   : series identifier
- model      : name of the benchmark model
- fold       : fold number (always 1 for single-holdout)
- <date_col> : date of the test-set week
- actual     : observed value for that week
- forecast   : model’s predicted value for that week

## Performance Optimizations (v3.0.0)

This version builds on the v2 optimizations (shared seasonal cache,
numba JIT loops, constrained statsmodels) and adds demand-classification
routing for dramatically better scaling at 30,000+ series.

DEMAND CLASSIFICATION ROUTING (new in v3)
Enterprise ERP systems (SAP IBP, Oracle, Kinaxis) do not run every
model on every series. Instead, they classify each series by its
demand pattern and route it to a small set of appropriate models.
This version implements the standard ADI/CV² classification
framework:

```
- ADI (Average Demand Interval): mean number of periods between
  non-zero demand observations. Low ADI = frequent demand.
- CV² (Coefficient of Variation squared): (std / mean)² of the
  non-zero demand values. Low CV² = consistent order sizes.

These two metrics divide series into four categories:

┌─────────────────┬────────────────────┬─────────────────────┐
│                 │ Low CV² (≤ 0.49)   │ High CV² (> 0.49)   │
├─────────────────┼────────────────────┼─────────────────────┤
│ Low ADI (≤ 1.32)│ SMOOTH             │ ERRATIC             │
│ High ADI (> 1.32)│ INTERMITTENT      │ LUMPY               │
└─────────────────┴────────────────────┴─────────────────────┘

Each category is assigned a curated set of 2-3 models that are
known to perform well for that demand pattern, rather than running
all models exhaustively.
```

CURATED MODEL SETS (new in v3)
The full model registry has been reduced from 13 to 8 models.
Removed models and rationale:

```
- theta: Extremely slow (statsmodels optimization), rarely won
  the best-fit tournament, and performance is similar to
  holt_winters which is retained for smooth series.
- damped_trend_seasonal: Same slowness issue as theta. Damped
  trend is theoretically appealing but in practice the additive
  Holt-Winters already handles most seasonal series well, and the
  damped variant's extra parameters made fitting unreliable on
  noisy weekly data.
- linear_trend_seasonal: Prone to runaway forecasts — the linear
  trend component can extrapolate aggressively, producing
  unrealistic values at longer horizons. The weighted seasonal
  average captures trend adaptation more safely via recency
  weighting.
- seasonal_naive: Dominated by seasonal_naive_blend in virtually
  all evaluations. The blend variant averages seasonal_naive with
  weekly_hist_avg, which provides robustness without sacrificing
  the seasonal signal.

Category-to-model assignments:

SMOOTH (frequent, consistent):
    → weekly_hist_avg, weighted_seasonal_avg, holt_winters
    Rationale: These series have enough signal for trend and
    seasonality. Holt-Winters captures both; the two average-
    based models are fast fallbacks. Holt-Winters is only run
    here (not on intermittent/lumpy) since it requires dense
    data and is the slowest remaining model.

ERRATIC (frequent, volatile):
    → seasonal_median, weighted_seasonal_avg, seasonal_naive_blend
    Rationale: High variance makes mean-based methods unstable.
    Seasonal median is robust to outliers. Weighted seasonal avg
    adapts to recent level shifts. The blend provides a stable
    anchor mixing last year's actuals with historical averages.

INTERMITTENT (sparse, consistent):
    → tsb, imapa, temp_agg_ses
    Rationale: These are purpose-built for intermittent demand.
    TSB explicitly models demand probability and size separately.
    IMAPA aggregates across multiple temporal levels to smooth
    out zeros. Temporal aggregation SES works similarly.

LUMPY (sparse, volatile):
    → tsb, global_avg
    Rationale: The hardest category — complex models overfit
    badly on sparse, noisy data. TSB is the standard choice for
    intermittent demand. Global average provides a conservative
    fallback that avoids wild swings. Keeping the model count
    to 2 is intentional: more models just add noise here.

This routing reduces total model fits by approximately 75-80%
compared to running all models on all series (e.g., 30,000
series × 2.5 avg models = 75,000 fits vs. 30,000 × 13 = 390,000).
```

PRIOR OPTIMIZATIONS (carried forward from v2)
1. Shared seasonal cache: datetime parsing, ISO week computation,
week-of-year statistics, and forecast date ranges are computed
once per series and shared across all models via a cache dict.
2. Numba JIT loops: Exponential smoothing update loops in TSB,
temp_agg_ses, and IMAPA are compiled to machine code via numba
(with pure-Python fallback if numba is not installed).
3. Constrained statsmodels: Holt-Winters uses use_brute=False and
maxiter=50 to avoid expensive brute-force parameter search.
“””

**version** = “3.0.0”

import numpy as np
import pandas as pd
import warnings
from typing import Dict, Callable, Any, List, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial

# Conditional imports for Tier 2 models (Holt-Winters)

try:
from statsmodels.tsa.holtwinters import ExponentialSmoothing
HAS_STATSMODELS = True
except ImportError:
HAS_STATSMODELS = False

# Conditional import for numba JIT compilation

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
“”“Simple Exponential Smoothing — returns final level.”””
level = values[0]
for i in range(1, len(values)):
level = alpha * values[i] + (1.0 - alpha) * level
return level

```
@njit(cache=True)
def _tsb_smooth_numba(demand: np.ndarray, alpha_d: float,
                      alpha_p: float, z0: float,
                      p0: float) -> float:
    """TSB smoothing loop — returns p_t * z_t."""
    z_t = z0
    p_t = p0
    for t in range(len(demand)):
        if demand[t] > 0.0:
            z_t = alpha_d * demand[t] + (1.0 - alpha_d) * z_t
            p_t = alpha_p + (1.0 - alpha_p) * p_t
        else:
            p_t = (1.0 - alpha_p) * p_t
    result = p_t * z_t
    return result if result > 0.0 else 0.0
```

else:
def _ses_smooth_numba(values: np.ndarray, alpha: float) -> float:
level = values[0]
for i in range(1, len(values)):
level = alpha * values[i] + (1.0 - alpha) * level
return level

```
def _tsb_smooth_numba(demand: np.ndarray, alpha_d: float,
                      alpha_p: float, z0: float,
                      p0: float) -> float:
    z_t = z0
    p_t = p0
    for t in range(len(demand)):
        if demand[t] > 0.0:
            z_t = alpha_d * demand[t] + (1.0 - alpha_d) * z_t
            p_t = alpha_p + (1.0 - alpha_p) * p_t
        else:
            p_t = (1.0 - alpha_p) * p_t
    result = p_t * z_t
    return result if result > 0.0 else 0.0
```

# =============================================================

# Demand Classification (ADI / CV²)

# =============================================================

# Thresholds based on Syntetos-Boylan classification framework

ADI_THRESHOLD = 1.32
CV2_THRESHOLD = 0.49

# Model assignments per demand category

CATEGORY_MODELS = {
“smooth”:       [“weekly_hist_avg”, “weighted_seasonal_avg”, “holt_winters”],
“erratic”:      [“seasonal_median”, “weighted_seasonal_avg”, “seasonal_naive_blend”],
“intermittent”: [“tsb”, “imapa”, “temp_agg_ses”],
“lumpy”:        [“tsb”, “global_avg”],
}

def classify_demand(history: np.ndarray) -> str:
“””
Classify a demand series using the Syntetos-Boylan framework.

```
Parameters
----------
history : np.ndarray
    The demand time series (training portion).

Returns
-------
str
    One of: 'smooth', 'erratic', 'intermittent', 'lumpy'.
"""
nonzero_mask = history > 0
nonzero_values = history[nonzero_mask]

# If no nonzero demand at all, treat as lumpy (hardest case)
if len(nonzero_values) == 0:
    return "lumpy"

# ADI: average demand interval
# Count periods between consecutive nonzero observations
nonzero_indices = np.where(nonzero_mask)[0]
if len(nonzero_indices) <= 1:
    adi = float(len(history))  # effectively infinite interval
else:
    intervals = np.diff(nonzero_indices).astype(float)
    adi = intervals.mean()

# CV²: squared coefficient of variation of nonzero demand
mean_nz = nonzero_values.mean()
if mean_nz > 0:
    cv2 = (nonzero_values.std() / mean_nz) ** 2
else:
    cv2 = 0.0

# Classify
if adi <= ADI_THRESHOLD:
    return "erratic" if cv2 > CV2_THRESHOLD else "smooth"
else:
    return "lumpy" if cv2 > CV2_THRESHOLD else "intermittent"
```

def get_models_for_category(category: str) -> List[str]:
“”“Return the list of model names assigned to a demand category.”””
return CATEGORY_MODELS.get(category, list(MODEL_REGISTRY.keys()))

# =============================================================

# Shared Seasonal Cache

# =============================================================

def _build_series_cache(history: np.ndarray, dates: np.ndarray,
horizon: int) -> Dict[str, Any]:
“””
Precompute all shared datetime and seasonal data for one series.

```
Called ONCE per series; results are passed to all models via the
``cache`` keyword argument.
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
```

# =============================================================

# Model Registry

# =============================================================

MODEL_REGISTRY: Dict[str, Callable] = {}

def register_model(name: str):
“”“Decorator to register a model function in the global registry.”””
def decorator(func):
MODEL_REGISTRY[name] = func
return func
return decorator

# =============================================================

# Benchmark Models (8 curated models)

# =============================================================

@register_model(“weekly_hist_avg”)
def weekly_historical_average(history: np.ndarray, dates: np.ndarray,
horizon: int = 52, **kwargs) -> np.ndarray:
“”“Weekly Historical Average: mean demand per ISO week-of-year.”””
cache = kwargs.get(“cache”)
if cache:
week_avg = cache[“week_avg”]
global_mean = cache[“global_mean”]
forecast_weeks = cache[“forecast_weeks”]
else:
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
forecast_dates = pd.date_range(start_date, periods=horizon, freq=“W-SUN”)
forecast_weeks = forecast_dates.isocalendar().week.astype(int).values
return np.array([week_avg.get(w, global_mean) for w in forecast_weeks])

@register_model(“global_avg”)
def global_average(history: np.ndarray, dates: np.ndarray,
horizon: int = 52, **kwargs) -> np.ndarray:
“”“Global Average: flat forecast equal to the overall mean.”””
cache = kwargs.get(“cache”)
mean_val = cache[“global_mean”] if cache else history.mean()
return np.full(horizon, mean_val)

@register_model(“tsb”)
def tsb_forecast(history: np.ndarray, dates: np.ndarray,
horizon: int = 52, alpha_d: float = 0.1,
alpha_p: float = 0.1, **kwargs) -> np.ndarray:
“”“Teunter-Syntetos-Babai (TSB) forecast for intermittent demand.”””
demand = history.astype(float)
nonzero_mask = demand > 0
z0 = demand[nonzero_mask].mean() if nonzero_mask.any() else 0.0
p0 = nonzero_mask.mean()
result = _tsb_smooth_numba(demand, alpha_d, alpha_p, z0, p0)
return np.full(horizon, result)

@register_model(“temp_agg_ses”)
def temporal_agg_ses(history: np.ndarray, dates: np.ndarray,
horizon: int = 52, agg_periods: int = 4,
alpha: float = 0.2, **kwargs) -> np.ndarray:
“”“Temporal Aggregation + Simple Exponential Smoothing.”””
values = history.astype(float)
n = len(values)
trim = n - (n % agg_periods)
trimmed = values[-trim:]
agg = trimmed.reshape(-1, agg_periods).sum(axis=1)
agg_forecast = max(_ses_smooth_numba(agg, alpha), 0.0)

```
cache = kwargs.get("cache")
if cache:
    seasonal_index = cache["seasonal_index"]
    forecast_weeks = cache["forecast_weeks"]
else:
    dates_ts = pd.DatetimeIndex(dates)
    week_numbers = dates_ts.isocalendar().week.astype(int).values
    week_means = {}
    for w in range(1, 54):
        mask = week_numbers == w
        if mask.any():
            week_means[w] = values[mask].mean()
    overall_mean = np.mean(list(week_means.values())) if week_means else 1.0
    seasonal_index = (
        {w: m / overall_mean for w, m in week_means.items()}
        if overall_mean > 0 else {}
    )
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
```

@register_model(“weighted_seasonal_avg”)
def weighted_seasonal_average(history: np.ndarray, dates: np.ndarray,
horizon: int = 52, decay: float = 0.8,
**kwargs) -> np.ndarray:
“”“Weighted Seasonal Average: recency-weighted mean per ISO week-of-year.”””
cache = kwargs.get(“cache”)
if cache:
week_numbers = cache[“week_numbers”]
years = cache[“years”]
global_mean = cache[“global_mean”]
forecast_weeks = cache[“forecast_weeks”]
else:
dates_ts = pd.DatetimeIndex(dates)
week_numbers = dates_ts.isocalendar().week.astype(int).values
years = dates_ts.year.values
global_mean = history.mean()
start_date = dates_ts[-1] + pd.Timedelta(weeks=1)
forecast_dates = pd.date_range(start_date, periods=horizon, freq=“W-SUN”)
forecast_weeks = forecast_dates.isocalendar().week.astype(int).values

```
unique_years = np.sort(np.unique(years))
n_years = len(unique_years)
year_rank = {yr: i for i, yr in enumerate(unique_years)}

week_avg = {}
for w in range(1, 54):
    mask = week_numbers == w
    if not mask.any():
        week_avg[w] = global_mean
        continue
    w_values = history[mask]
    w_years = years[mask]
    weights = np.array([
        decay ** (n_years - 1 - year_rank[yr]) for yr in w_years
    ])
    weight_sum = weights.sum()
    if weight_sum > 0:
        week_avg[w] = np.average(w_values, weights=weights)
    else:
        week_avg[w] = global_mean

return np.array([week_avg.get(w, global_mean) for w in forecast_weeks])
```

@register_model(“seasonal_median”)
def seasonal_median(history: np.ndarray, dates: np.ndarray,
horizon: int = 52, **kwargs) -> np.ndarray:
“”“Seasonal Median: median demand per ISO week-of-year.”””
cache = kwargs.get(“cache”)
if cache:
week_med = cache[“week_median”]
global_median = cache[“global_median”]
forecast_weeks = cache[“forecast_weeks”]
else:
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
forecast_dates = pd.date_range(start_date, periods=horizon, freq=“W-SUN”)
forecast_weeks = forecast_dates.isocalendar().week.astype(int).values

```
return np.array([week_med.get(w, global_median) for w in forecast_weeks])
```

@register_model(“seasonal_naive_blend”)
def seasonal_naive_blend(history: np.ndarray, dates: np.ndarray,
horizon: int = 52, blend_weight: float = 0.5,
**kwargs) -> np.ndarray:
“”“Seasonal Naive + Weekly Average Blend.”””
last_year = history[-52:]
if len(last_year) < 52:
last_year = np.tile(history, (52 // len(history) + 1))[-52:]
reps = (horizon // 52) + 1
snaive = np.tile(last_year, reps)[:horizon]

```
cache = kwargs.get("cache")
if cache:
    week_avg = cache["week_avg"]
    global_mean = cache["global_mean"]
    forecast_weeks = cache["forecast_weeks"]
else:
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
```

@register_model(“holt_winters”)
def holt_winters(history: np.ndarray, dates: np.ndarray,
horizon: int = 52, seasonal_periods: int = 52,
**kwargs) -> np.ndarray:
“”“Holt-Winters ETS(A,A,A) — only used for smooth demand series.”””
if not HAS_STATSMODELS:
return np.full(horizon, np.nan)
n = len(history)
cache = kwargs.get(“cache”)
mean_val = cache[“global_mean”] if cache else history.mean()
fallback = np.full(horizon, mean_val)
if n < 2 * seasonal_periods:
return fallback
if np.std(history) == 0:
return fallback
try:
with warnings.catch_warnings():
warnings.simplefilter(“ignore”)
model = ExponentialSmoothing(
history, trend=“add”, seasonal=“add”,
seasonal_periods=seasonal_periods,
initialization_method=“estimated”,
)
fit = model.fit(optimized=True, use_brute=False, maxiter=50)
forecast = fit.forecast(horizon)
forecast = np.maximum(forecast, 0.0)
if not np.all(np.isfinite(forecast)):
return fallback
return forecast
except Exception:
return fallback

@register_model(“imapa”)
def imapa(history: np.ndarray, dates: np.ndarray,
horizon: int = 52, agg_levels: tuple = (1, 2, 4, 8, 13),
alpha: float = 0.2, **kwargs) -> np.ndarray:
“”“IMAPA: Intermittent Multiple Aggregation Prediction Algorithm.”””
values = history.astype(float)
n = len(values)

```
cache = kwargs.get("cache")
if cache:
    seasonal_index = cache["seasonal_index"]
    forecast_weeks = cache["forecast_weeks"]
else:
    dates_ts = pd.DatetimeIndex(dates)
    week_numbers = dates_ts.isocalendar().week.astype(int).values
    week_means = {}
    for w in range(1, 54):
        mask = week_numbers == w
        if mask.any():
            week_means[w] = values[mask].mean()
    overall_mean = (
        np.mean(list(week_means.values())) if week_means else 1.0
    )
    seasonal_index = (
        {w: m / overall_mean for w, m in week_means.items()}
        if overall_mean > 0 else {}
    )
    start_date = dates_ts[-1] + pd.Timedelta(weeks=1)
    forecast_dates = pd.date_range(start_date, periods=horizon, freq="W-SUN")
    forecast_weeks = forecast_dates.isocalendar().week.astype(int).values

raw_weights = np.array([seasonal_index.get(w, 1.0) for w in forecast_weeks])
level_forecasts = []

for level in agg_levels:
    if level > n:
        continue
    if level == 1:
        agg_forecast = max(_ses_smooth_numba(values, alpha), 0.0)
    else:
        trim = n - (n % level)
        trimmed = values[-trim:]
        agg = trimmed.reshape(-1, level).sum(axis=1)
        agg_forecast = max(_ses_smooth_numba(agg, alpha), 0.0)

    weekly_forecast = np.zeros(horizon)
    block_size = max(level, 1)
    for i in range(0, horizon, block_size):
        block = raw_weights[i:i + block_size]
        block_sum = block.sum()
        if block_sum > 0:
            weekly_forecast[i:i + block_size] = (
                (block / block_sum) * agg_forecast
            )
        else:
            weekly_forecast[i:i + block_size] = agg_forecast / block_size
    level_forecasts.append(weekly_forecast)

if not level_forecasts:
    mean_val = cache["global_mean"] if cache else history.mean()
    return np.full(horizon, mean_val)
return np.mean(level_forecasts, axis=0)
```

# =============================================================

# Core Engine

# =============================================================

def _forecast_single_series(history: np.ndarray, dates: np.ndarray,
horizon: int,
model_names: List[str] = None,
cache: Dict[str, Any] = None
) -> Dict[str, np.ndarray]:
“””
Run selected models on a single series.

```
If ``model_names`` is provided, only those models are run.
If ``cache`` is provided, it is forwarded to each model.
"""
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
```

# =============================================================

# Chunk Utility

# =============================================================

def _chunk_groups(groups: list, n_chunks: int) -> list:
“”“Split a list into roughly equal chunks for worker distribution.”””
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
“””
Worker function for parallel fold-detail generation.

```
For each series in the chunk:
  1. Hold out the last ``eval_window`` weeks as the test set.
  2. Classify the training history (ADI/CV²) to determine which
     models to run.
  3. Build a shared seasonal cache from the training history.
  4. Run only the assigned models for this series' demand category.
  5. Record actual vs. forecast for every test-set week and model.

Returns (fold_details_list, classification_counts_dict).
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

    # Classify demand pattern
    category = classify_demand(train_history)
    category_counts[category] += 1
    model_names = get_models_for_category(category)

    # Build cache once for all models on this series
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
            })

return fold_details, category_counts
```

# =============================================================

# Public API

# =============================================================

def generate_fold_details(
df: pd.DataFrame,
date_col: str = “date”,
id_col: str = “item_id”,
value_col: str = “demand”,
eval_window: int = 52,
min_train_weeks: int = 52,
n_workers: int = None,
) -> pd.DataFrame:
“””
Generate the fold_details DataFrame with demand-classification
routing.

```
For each series, the training history is classified into one of
four demand categories (smooth, erratic, intermittent, lumpy)
using the ADI/CV² framework. Only the models assigned to that
category are run, reducing total model fits by ~75-80%.

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
eval_window : int
    Number of weeks to hold out as the test set (default 52).
min_train_weeks : int
    Minimum training history required (default 52). Series with
    fewer than ``min_train_weeks + eval_window`` total weeks are
    skipped.
n_workers : int or None
    Number of parallel workers. Defaults to ``cpu_count() - 1``.

Returns
-------
pd.DataFrame
    Columns: [id_col, "model", "category", "fold", date_col,
              "actual", "forecast"]
    One row per (series x model x test-set week). The "category"
    column indicates the demand classification used for model
    routing.
"""
if n_workers is None:
    n_workers = max(1, cpu_count() - 1)

df = df.copy()
df[date_col] = pd.to_datetime(df[date_col])

groups = [(series_id, grp) for series_id, grp in df.groupby(id_col)]
total_series = len(groups)

chunks = _chunk_groups(groups, n_workers)

worker_fn = partial(
    _fold_details_worker,
    date_col=date_col,
    id_col=id_col,
    value_col=value_col,
    eval_window=eval_window,
    min_train_weeks=min_train_weeks,
)

numba_status = (
    "enabled"
    if HAS_NUMBA
    else "disabled (pip install numba for faster smoothing loops)"
)
print(f"Generating fold details for {total_series} series "
      f"with {n_workers} worker(s)...")
print(f"  Numba JIT: {numba_status}")
print(f"  Demand classification: ADI/CV² routing enabled")
print(f"  Model sets: smooth={CATEGORY_MODELS['smooth']}, "
      f"erratic={CATEGORY_MODELS['erratic']}, "
      f"intermittent={CATEGORY_MODELS['intermittent']}, "
      f"lumpy={CATEGORY_MODELS['lumpy']}")

if n_workers == 1:
    results = [worker_fn(chunk) for chunk in chunks]
else:
    with Pool(processes=n_workers) as pool:
        results = pool.map(worker_fn, chunks)

all_fold_details = []
total_counts = {"smooth": 0, "erratic": 0,
                "intermittent": 0, "lumpy": 0}

for detail_list, counts in results:
    all_fold_details.extend(detail_list)
    for cat, cnt in counts.items():
        total_counts[cat] += cnt

fold_details_df = pd.DataFrame(all_fold_details)

# Print classification summary
classified_total = sum(total_counts.values())
n_skipped = total_series - classified_total
print(f"\n  Demand Classification Summary:")
for cat in ["smooth", "erratic", "intermittent", "lumpy"]:
    cnt = total_counts[cat]
    n_models = len(CATEGORY_MODELS[cat])
    pct = cnt / classified_total * 100 if classified_total > 0 else 0
    print(f"    {cat:>13}: {cnt:>6} series ({pct:5.1f}%) "
          f"x {n_models} models = {cnt * n_models:>7} fits")

total_fits = sum(
    total_counts[c] * len(CATEGORY_MODELS[c]) for c in total_counts
)
naive_fits = classified_total * len(MODEL_REGISTRY)
reduction = (1 - total_fits / naive_fits) * 100 if naive_fits > 0 else 0
print(f"    {'TOTAL':>13}: {total_fits:>6} model fits "
      f"(vs {naive_fits} without routing, "
      f"{reduction:.0f}% reduction)")

if len(fold_details_df) > 0:
    n_series = fold_details_df[id_col].nunique()
    print(f"\n  Output: {n_series} series, "
          f"{len(fold_details_df)} rows")
if n_skipped > 0:
    min_required = min_train_weeks + eval_window
    print(f"  Skipped {n_skipped} series with < {min_required} "
          f"weeks of history")

return fold_details_df
```

# =============================================================

# CLI entry point

# =============================================================

if **name** == “**main**”:
import argparse

```
parser = argparse.ArgumentParser(
    description="Generate fold_details DataFrame (actual vs forecast "
                "per week/series/model) with demand classification."
)
parser.add_argument("input_csv", help="Path to input CSV file")
parser.add_argument("-o", "--output", default="fold_details.csv",
                    help="Output CSV path (default: fold_details.csv)")
parser.add_argument("--date-col", default="date",
                    help="Date column name (default: date)")
parser.add_argument("--id-col", default="item_id",
                    help="Series ID column name (default: item_id)")
parser.add_argument("--value-col", default="demand",
                    help="Value column name (default: demand)")
parser.add_argument("--eval-window", type=int, default=52,
                    help="Test-set weeks (default: 52)")
parser.add_argument("--min-train", type=int, default=52,
                    help="Minimum training weeks (default: 52)")
parser.add_argument("--workers", type=int, default=None,
                    help="Number of parallel workers (default: auto)")

args = parser.parse_args()

df = pd.read_csv(args.input_csv)

fold_details_df = generate_fold_details(
    df,
    date_col=args.date_col,
    id_col=args.id_col,
    value_col=args.value_col,
    eval_window=args.eval_window,
    min_train_weeks=args.min_train,
    n_workers=args.workers,
)

fold_details_df.to_csv(args.output, index=False)
print(f"\nSaved to {args.output}")
```
