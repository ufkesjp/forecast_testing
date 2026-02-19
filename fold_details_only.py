# “””
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
“””

**version** = “1.0.0”

import numpy as np
import pandas as pd
import warnings
from typing import Dict, Callable, Tuple
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
“”“Decorator to register a model function in the global registry.”””
def decorator(func):
MODEL_REGISTRY[name] = func
return func
return decorator

# =============================================================

# Benchmark Models

# =============================================================

@register_model(“seasonal_naive”)
def seasonal_naive(history: np.ndarray, dates: np.ndarray,
horizon: int = 52, **kwargs) -> np.ndarray:
last_year = history[-52:]
if len(last_year) < 52:
last_year = np.tile(history, (52 // len(history) + 1))[-52:]
reps = (horizon // 52) + 1
return np.tile(last_year, reps)[:horizon]

@register_model(“weekly_hist_avg”)
def weekly_historical_average(history: np.ndarray, dates: np.ndarray,
horizon: int = 52, **kwargs) -> np.ndarray:
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
return np.full(horizon, history.mean())

@register_model(“tsb”)
def tsb_forecast(history: np.ndarray, dates: np.ndarray,
horizon: int = 52, alpha_d: float = 0.1,
alpha_p: float = 0.1, **kwargs) -> np.ndarray:
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

@register_model(“temp_agg_ses”)
def temporal_agg_ses(history: np.ndarray, dates: np.ndarray,
horizon: int = 52, agg_periods: int = 4,
alpha: float = 0.2, **kwargs) -> np.ndarray:
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
forecast_dates = pd.date_range(start_date, periods=horizon, freq=“W-SUN”)
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

@register_model(“weighted_seasonal_avg”)
def weighted_seasonal_average(history: np.ndarray, dates: np.ndarray,
horizon: int = 52, decay: float = 0.8,
**kwargs) -> np.ndarray:
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
forecast_dates = pd.date_range(start_date, periods=horizon, freq=“W-SUN”)
forecast_weeks = forecast_dates.isocalendar().week.astype(int).values
return np.array([week_avg.get(w, global_mean) for w in forecast_weeks])

@register_model(“seasonal_median”)
def seasonal_median(history: np.ndarray, dates: np.ndarray,
horizon: int = 52, **kwargs) -> np.ndarray:
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
return np.array([week_med.get(w, global_median) for w in forecast_weeks])

@register_model(“seasonal_naive_blend”)
def seasonal_naive_blend(history: np.ndarray, dates: np.ndarray,
horizon: int = 52, blend_weight: float = 0.5,
**kwargs) -> np.ndarray:
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
forecast_dates = pd.date_range(start_date, periods=horizon, freq=“W-SUN”)
forecast_weeks = forecast_dates.isocalendar().week.astype(int).values
whist = np.array([week_avg.get(w, global_mean) for w in forecast_weeks])
return blend_weight * snaive + (1 - blend_weight) * whist

@register_model(“holt_winters”)
def holt_winters(history: np.ndarray, dates: np.ndarray,
horizon: int = 52, seasonal_periods: int = 52,
**kwargs) -> np.ndarray:
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
warnings.simplefilter(“ignore”)
model = ExponentialSmoothing(
history, trend=“add”, seasonal=“add”,
seasonal_periods=seasonal_periods,
initialization_method=“estimated”,
)
fit = model.fit(optimized=True, use_brute=True)
forecast = fit.forecast(horizon)
forecast = np.maximum(forecast, 0.0)
if not np.all(np.isfinite(forecast)):
return fallback
return forecast
except Exception:
return fallback

@register_model(“theta”)
def theta_method(history: np.ndarray, dates: np.ndarray,
horizon: int = 52, seasonal_periods: int = 52,
**kwargs) -> np.ndarray:
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
warnings.simplefilter(“ignore”)
series = pd.Series(
history,
index=pd.date_range(
end=pd.Timestamp(dates[-1]), periods=n, freq=“W-SUN”
)
)
model = ThetaModel(series, period=seasonal_periods,
deseasonalize=True, method=“auto”)
fit = model.fit()
forecast = fit.forecast(horizon).values
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
forecast_dates = pd.date_range(start_date, periods=horizon, freq=“W-SUN”)
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

@register_model(“damped_trend_seasonal”)
def damped_trend_seasonal(history: np.ndarray, dates: np.ndarray,
horizon: int = 52, seasonal_periods: int = 52,
**kwargs) -> np.ndarray:
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
warnings.simplefilter(“ignore”)
model = ExponentialSmoothing(
history, trend=“add”, damped_trend=True,
seasonal=“add”, seasonal_periods=seasonal_periods,
initialization_method=“estimated”,
)
fit = model.fit(optimized=True, use_brute=True)
forecast = fit.forecast(horizon)
forecast = np.maximum(forecast, 0.0)
if not np.all(np.isfinite(forecast)):
return fallback
return forecast
except Exception:
return fallback

@register_model(“linear_trend_seasonal”)
def linear_trend_seasonal(history: np.ndarray, dates: np.ndarray,
horizon: int = 52, **kwargs) -> np.ndarray:
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
forecast_dates = pd.date_range(start_date, periods=horizon, freq=“W-SUN”)
forecast_weeks = forecast_dates.isocalendar().week.astype(int).values
future_seasonal = np.array([seasonal_idx.get(w, 0.0) for w in forecast_weeks])
forecast = trend_forecast + future_seasonal
forecast = np.maximum(forecast, 0.0)
return forecast

# =============================================================

# Core Engine: Forecast a Single Series with All Models

# =============================================================

def _forecast_single_series(history: np.ndarray, dates: np.ndarray,
horizon: int, model_names: list = None) -> Dict[str, np.ndarray]:
“”“Run all (or selected) registered models on a single series.”””
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
  1. Hold out the last `eval_window` weeks as the test set.
  2. Train every registered model on the preceding history.
  3. Record actual vs. forecast for every test-set week and model.

Returns a list of dicts (one per week × model combination).
"""
min_required = min_train_weeks + eval_window
fold_details = []

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

    model_outputs = _forecast_single_series(
        train_history, train_dates, eval_window
    )

    for model_name, preds in model_outputs.items():
        preds_trimmed = preds[:eval_window]
        for i in range(eval_window):
            fold_details.append({
                id_col: series_id,
                "model": model_name,
                "fold": 1,
                date_col: test_dates[i],
                "actual": test_actual[i],
                "forecast": preds_trimmed[i],
            })

return fold_details
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
Generate the fold_details DataFrame.

```
For each active series, holds out the last `eval_window` weeks as a
test set, runs every registered benchmark model on the training
history, and records the actual and forecasted values for each
test-set week.

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
    Columns: [id_col, "model", "fold", date_col, "actual", "forecast"]
    One row per (series × model × test-set week).
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

print(f"Generating fold details for {total_series} series "
      f"with {n_workers} worker(s)...")

if n_workers == 1:
    results = [worker_fn(chunk) for chunk in chunks]
else:
    with Pool(processes=n_workers) as pool:
        results = pool.map(worker_fn, chunks)

all_fold_details = []
for detail_list in results:
    all_fold_details.extend(detail_list)

fold_details_df = pd.DataFrame(all_fold_details)

if len(fold_details_df) > 0:
    n_series = fold_details_df[id_col].nunique()
    n_models = fold_details_df["model"].nunique()
    n_skipped = total_series - n_series
    print(f"  {n_series} series x {n_models} models = "
          f"{len(fold_details_df)} rows")
    if n_skipped > 0:
        min_required = min_train_weeks + eval_window
        print(f"  Skipped {n_skipped} series with < {min_required} "
              f"weeks of history")
else:
    print("  No fold details generated (all series may have been "
          "too short).")

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
                "per week/series/model)."
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
print(f"Saved to {args.output}")
```