# Best-Fit Benchmark Forecasting Pipeline

A production-ready toolkit for automated best-fit model selection and forecasting of weekly demand series. Designed for large-scale deployment across thousands of SKUs with minimal manual tuning.

## Overview

This pipeline automates the process of forecasting weekly demand by:
1. **Filtering** inactive items to focus computational resources on active series
2. **Evaluating** multiple benchmark models using rolling-origin cross-validation
3. **Selecting** the best-performing model for each series via tournament-style ranking
4. **Forecasting** future demand using the winning model trained on full history

**Version**: 1.4.0

## Key Features

### 🎯 Automatic Model Selection
- Tournament-style selection picks the best model per series
- No one-size-fits-all approach — adapts to each item's unique demand pattern
- Composite scoring balances multiple evaluation metrics (WAPE + MASE)

### 📊 Eleven Benchmark Models
Each designed for different demand patterns:
- **Seasonal Naïve**: Repeats last year's pattern (strong for seasonal items)
- **Weekly Historical Average**: Multi-year seasonal averaging (smooths noise)
- **Global Average**: Flat baseline (sanity check)
- **TSB (Teunter-Syntetos-Babai)**: Intermittent demand specialist
- **Temporal Aggregation + SES**: Reduces zero-inflation via bucketing
- **Weighted Seasonal Average** *(v1.4.0)*: Recency-weighted week-of-year average
- **Seasonal Median** *(v1.4.0)*: Median demand per week-of-year (robust to spikes)
- **Seasonal Naïve Blend** *(v1.4.0)*: Tunable blend of seasonal naïve + weekly average
- **Holt-Winters / ETS** *(v1.4.0)*: Additive trend + seasonal smoothing (requires statsmodels)
- **Theta Method** *(v1.4.0)*: M3 competition winner, trend + seasonality (requires statsmodels)
- **IMAPA** *(v1.4.0)*: Multi-aggregation intermittent demand forecasting

### 🔄 Robust Evaluation
- **Rolling-origin cross-validation** with configurable folds (default: 3)
- **3-month evaluation windows** (13 weeks) aligned with business planning cycles
- **Fold-averaged metrics** prevent one anomalous period from dominating selection
- **Per-fold diagnostics** for model stability analysis

### 🚫 Inactive Item Detection
- Automatically identifies and excludes items with zero demand in trailing window (default: 26 weeks)
- Saves compute resources and prevents forecasting discontinued products
- Returns diagnostic details (last demand date, weeks since activity, lifetime volume)

### 📏 Comprehensive Metrics
- **WAPE** (Weighted Absolute Percentage Error): Industry-standard accuracy measure
- **MASE** (Mean Absolute Scaled Error): Benchmarks against seasonal naïve
- **Fold stability**: Tracks model consistency across evaluation periods

## Quick Start

### Basic Usage

```python
from best_fit_pipeline import best_fit_pipeline
import pandas as pd

# Load your demand data
df = pd.read_csv("demand_data.csv")

# Expected schema:
#   - date: datetime column with week-ending Sunday dates
#   - item_id: unique identifier for each SKU/series
#   - demand: non-negative numeric demand values

# Run the pipeline
eval_df, best_fit_df, forecast_df, inactive_df, fold_details = best_fit_pipeline(
    df,
    date_col="date",
    id_col="item_id",
    value_col="demand",
    horizon=52,          # Forecast 52 weeks ahead
    eval_window=13,      # Evaluate on 3-month windows
    n_folds=3,           # Use 3 rolling-origin folds
    inactive_weeks=26,   # Flag items with 6 months of zero demand
)

# View model selection results
print(best_fit_df["best_model"].value_counts())

# Export forecasts
forecast_df.to_csv("forecasts_52week.csv", index=False)
```

### Output DataFrames

The pipeline returns a 5-tuple of DataFrames for comprehensive analysis:

#### 1. `eval_df` — Full Evaluation Results
One row per series × model combination, showing fold-averaged performance:
- `item_id`: Series identifier
- `model`: Benchmark model name
- `wape`: Mean WAPE across folds
- `mase`: Mean MASE across folds
- `n_folds`: Number of folds evaluated
- `wape_std`: Fold-to-fold WAPE variability (lower = more stable)
- `fold_wapes`: Comma-separated WAPE per fold for diagnostics

#### 2. `best_fit_df` — Tournament Winners
One row per series, showing the selected model:
- `item_id`: Series identifier
- `best_model`: Winning model name
- `composite_score`: Tournament score (lower = better)
- `wape`: Winner's mean WAPE
- `mase`: Winner's mean MASE
- `selection_method`: "tournament", "tiebreak", or "fallback"

#### 3. `forecast_df` — Final Forecasts
One row per series × forecast week:
- `item_id`: Series identifier
- `model`: Winning model name
- `date`: Forecast week-ending date
- `forecast`: Predicted demand quantity
- `step`: Forecast step (1 = first week ahead, 52 = last week)

#### 4. `inactive_df` — Inactive Items
One row per inactive item (excluded from forecasting):
- `item_id`: Series identifier
- `status`: "inactive"
- `last_nonzero_date`: Most recent week with demand > 0
- `weeks_since_demand`: Weeks elapsed since last demand
- `total_history_weeks`: Total weeks of historical data
- `lifetime_total_qty`: Sum of all demand ever recorded

#### 5. `fold_details` — Week-Level Evaluation Details
One row per series × model × fold × week (granular diagnostic data):
- `item_id`: Series identifier
- `model`: Benchmark model name
- `fold`: Fold number (1, 2, 3, ...)
- `actual`: Actual demand value for that week
- `forecast`: Forecasted demand value for that week

**Use cases:**
- Inspect individual week-level forecast accuracy
- Identify which specific weeks/periods a model struggled with
- Calculate custom metrics beyond WAPE/MASE
- Visualize forecast vs. actual for specific series and folds
- Debug anomalous evaluation results

## Data Requirements

### Input Schema
Your DataFrame must contain:
- **Date column**: Week-ending dates (preferably Sundays) in datetime format
- **ID column**: Unique series/SKU identifiers (string or categorical)
- **Value column**: Non-negative demand quantities (numeric)

### Minimum History Requirements
- **Absolute minimum**: 52 weeks per series (one full year)
- **Recommended**: 78+ weeks for robust cross-validation
  - With default settings (3 folds, 13-week eval, 13-week spacing), minimum = 52 + 13 + (2 × 13) = 91 weeks
- Series with insufficient history are automatically skipped with a warning

### Data Quality Expectations
- **Complete date sequences**: No missing weeks (fill gaps with zeros if needed)
- **Consistent granularity**: All series must be weekly
- **Non-negative demand**: Negative values will cause undefined behavior
- **Week-ending alignment**: All dates should fall on the same day of week (e.g., all Sundays)

## Configuration Options

### Pipeline Parameters

```python
best_fit_pipeline(
    df,
    date_col,
    id_col,
    value_col,
    horizon=52,              # Forecast horizon in weeks
    eval_window=13,          # Weeks per evaluation fold (3 months)
    n_folds=3,               # Number of rolling-origin folds
    fold_spacing=13,         # Weeks between consecutive folds
    inactive_weeks=26,       # Trailing weeks to check for activity (0 = disable)
    primary_metric="wape",   # Primary tournament metric
    secondary_metric="mase", # Secondary tournament metric
    primary_weight=0.7,      # Weight for primary metric (0-1)
    secondary_weight=0.3,    # Weight for secondary metric (0-1)
)
```

### Tuning Recommendations

**For different planning cycles:**
- **Monthly planning**: `eval_window=4` (4 weeks)
- **Quarterly planning**: `eval_window=13` (3 months) — **default**
- **Annual planning**: `eval_window=52` (1 year)

**For different product lifecycles:**
- **Fast-moving consumer goods**: `inactive_weeks=13` (3 months)
- **Standard retail**: `inactive_weeks=26` (6 months) — **default**
- **Capital equipment / slow movers**: `inactive_weeks=52` (1 year)

**For more/less stable evaluation:**
- **More folds** (`n_folds=4-5`): More robust but requires longer history
- **Fewer folds** (`n_folds=2`): Faster, less stable, works with shorter history

## Model Details

### 1. Seasonal Naïve (`seasonal_naive`)
**Best for:** Items with strong yearly seasonality

Repeats the last 52 weeks of demand as the forecast. Surprisingly effective baseline that captures annual patterns (holidays, seasonality). Often hard to beat for seasonal items.

### 2. Weekly Historical Average (`weekly_hist_avg`)
**Best for:** Items with consistent seasonal patterns but year-to-year noise

Computes the average demand for each ISO week-of-year (1-53) across all available history. Smooths out one-time events while preserving seasonal shape.

### 3. Global Average (`global_avg`)
**Best for:** Flat, stable demand with no seasonality

Returns a constant forecast equal to the overall historical mean. Primary use is as a sanity check — if more complex models can't beat this, they're not adding value.

### 4. TSB (Teunter-Syntetos-Babai) (`tsb`)
**Best for:** Highly intermittent demand, items approaching obsolescence

Separately tracks demand probability and demand size using exponential smoothing. Probability decays toward zero during consecutive zero-demand periods, making it superior to Croston's method for obsolescence scenarios.

**Hyperparameters:**
- `alpha_d=0.1`: Demand size smoothing (0 < α < 1)
- `alpha_p=0.1`: Demand probability smoothing (0 < α < 1)

### 5. Temporal Aggregation + SES (`temp_agg_ses`)
**Best for:** Highly intermittent weekly demand with underlying seasonal patterns

**Two-stage approach:**
1. **Aggregate**: Sum weekly data into larger buckets (default 4-week blocks) to reduce zero-inflation
2. **Forecast**: Apply Simple Exponential Smoothing on aggregated series
3. **Disaggregate**: Distribute aggregated forecast back to weeks using historical seasonal weights

**Hyperparameters:**
- `agg_periods=4`: Weeks per aggregation bucket (4 ≈ monthly)
- `alpha=0.2`: SES smoothing parameter (0 < α < 1)

### 6. Weighted Seasonal Average (`weighted_seasonal_avg`)
**Best for:** Items with trending or shifting seasonal patterns

Like `weekly_hist_avg` but applies exponential decay weighting so more recent years count more. Adapts to demand that's growing, shrinking, or shifting over time.

**Hyperparameters:**
- `decay=0.8`: Exponential decay factor (lower = more recency bias, 1.0 = equal weighting)

### 7. Seasonal Median (`seasonal_median`)
**Best for:** Intermittent/lumpy demand where spikes bias the mean upward

Uses the median instead of the mean for each ISO week-of-year. Resists distortion from occasional large orders, producing a forecast that represents the "typical" week rather than the "average" week.

### 8. Seasonal Naïve Blend (`seasonal_naive_blend`)
**Best for:** Items where neither seasonal naïve nor weekly average alone performs well

Weighted blend of seasonal naïve and weekly historical average. Hedges between repeating last year exactly and averaging all years.

**Hyperparameters:**
- `blend_weight=0.5`: Weight for seasonal naïve component (0.0 = pure weekly avg, 1.0 = pure seasonal naïve)

### 9. Holt-Winters / ETS (`holt_winters`)
**Best for:** Items with clear trend and seasonal components; requires 104+ weeks of history

*Requires `statsmodels`.* Fits exponential smoothing with additive trend and additive seasonality. Automatically optimizes smoothing parameters. Falls back to global mean for sparse/short series.

**Hyperparameters:**
- `seasonal_periods=52`: Seasonal cycle length

### 10. Theta Method (`theta`)
**Best for:** Series with mixed trend and seasonality; M3 competition winner

*Requires `statsmodels`.* Decomposes the series into theta lines capturing trend and short-term dynamics, applies SES, and recombines with seasonal adjustment. Requires 104+ weeks.

**Hyperparameters:**
- `seasonal_periods=52`: Seasonal cycle length

### 11. IMAPA (`imapa`)
**Best for:** Intermittent demand — outperforms single-level temporal aggregation

Forecasts at multiple aggregation levels (weekly, bi-weekly, monthly, bi-monthly, quarterly) simultaneously and averages the disaggregated results. Different levels capture different aspects of the signal.

**Hyperparameters:**
- `agg_levels=(1, 2, 4, 8, 13)`: Aggregation levels in weeks
- `alpha=0.2`: SES smoothing parameter

**Reference:** Petropoulos & Kourentzes (2015), *Journal of the Operational Research Society*

## Evaluation Methodology

### Rolling-Origin Cross-Validation

The pipeline uses time-series cross-validation to simulate realistic forecasting:

```
Example: 130 weeks of history, 3 folds, 13-week eval window, 13-week spacing

Fold 1 (most recent):
  Train: weeks 1-117  →  Test: weeks 118-130

Fold 2:
  Train: weeks 1-104  →  Test: weeks 105-117

Fold 3:
  Train: weeks 1-91   →  Test: weeks 92-104
```

**Why multiple folds?**
- Single-fold backtests can be misleading if that period is anomalous
- Averaging across folds produces more stable, generalizable model selection
- Aligned with best practices in time-series forecasting research

### Tournament Selection

**Per series:**
1. **Filter**: Remove models with infinite/NaN metrics
2. **Normalize**: Scale WAPE and MASE to [0, 1] via min-max within-series
3. **Composite Score**: `0.7 × norm_WAPE + 0.3 × norm_MASE` (default weights)
4. **Winner**: Model with lowest composite score
5. **Tiebreak**: If tied, lowest raw WAPE wins
6. **Fallback**: If no valid models, default to `seasonal_naive`

**Why composite scoring?**
- WAPE measures forecast bias (total error vs. total demand)
- MASE measures relative skill (vs. seasonal naïve benchmark)
- Combining both prevents overfitting to a single metric

## Performance Considerations

### Computational Complexity
- **Time complexity**: O(S × M × F × W)
  - S = number of series
  - M = number of models (11)
  - F = number of folds (default 3)
  - W = evaluation window length (default 13)

### Optimization Tips
- **Parallel processing**: The pipeline is embarassingly parallel at the series level — wrap the groupby loop in a multiprocessing pool for 5-10× speedup on multi-core machines
- **Reduce folds**: Drop from 3 to 2 folds for 33% speedup (at cost of stability)
- **Filter more aggressively**: Increase `inactive_weeks` to exclude more items
- **Subset testing**: Run on a sample during development, full dataset in production

### Memory Usage
- **Low**: All operations are vectorized and avoid large intermediate DataFrames
- **Estimate**: ~50-100 MB for 10,000 series × 104 weeks of history
- Safe to run on standard laptops for datasets up to 50,000 series

## Interpreting Results

### Model Distribution Analysis
After running the pipeline, check which models won:

```python
print(best_fit_df["best_model"].value_counts())
```

**Typical distributions:**
- **Seasonal naïve dominates (50-70%)**: Strong seasonality in your data
- **TSB wins frequently (20-40%)**: High intermittency
- **Global average never wins**: Good sign — seasonality/structure exists
- **Global average wins often**: Warning — data may be too noisy or sparse

### Selection Method Breakdown
```python
print(best_fit_df["selection_method"].value_counts())
```

- **tournament**: Clear winner by composite score (ideal)
- **tiebreak**: Multiple models tied, broke tie via raw WAPE
- **fallback**: All models had invalid metrics (investigate these series)

### Stability Analysis
Models with lower `wape_std` are more consistent across evaluation folds:

```python
print(eval_df.groupby("model")["wape_std"].median())
```

High `wape_std` suggests the model is sensitive to the evaluation period — may be overfitting or volatile.

### Deep Dive with `fold_details`
The `fold_details` DataFrame provides week-level granularity for advanced diagnostics:

**Example 1: Identify which weeks are hardest to forecast**
```python
# Find weeks where all models struggled
weekly_errors = fold_details.groupby(["item_id", "fold"]).apply(
    lambda x: (x["actual"] - x["forecast"]).abs().mean()
)
worst_weeks = weekly_errors.nlargest(10)
print("Weeks with highest forecast errors across all models:")
print(worst_weeks)
```

**Example 2: Visualize forecast vs. actual for a specific series**
```python
import matplotlib.pyplot as plt

series_id = "SKU_12345"
model_name = "seasonal_naive"

# Get fold details for this series and model
data = fold_details[
    (fold_details["item_id"] == series_id) &
    (fold_details["model"] == model_name)
]

# Plot each fold
for fold_num in data["fold"].unique():
    fold_data = data[data["fold"] == fold_num]
    plt.figure(figsize=(10, 4))
    plt.plot(fold_data["actual"].values, label="Actual", marker="o")
    plt.plot(fold_data["forecast"].values, label="Forecast", marker="x")
    plt.title(f"{series_id} - {model_name} - Fold {fold_num}")
    plt.legend()
    plt.show()
```

**Example 3: Calculate custom metrics by fold**
```python
# Calculate RMSE per fold per model
from numpy import sqrt

fold_rmse = fold_details.groupby(["item_id", "model", "fold"]).apply(
    lambda x: sqrt(((x["actual"] - x["forecast"]) ** 2).mean())
).reset_index(name="rmse")

# Average RMSE across folds
avg_rmse = fold_rmse.groupby(["item_id", "model"])["rmse"].mean()
print("Average RMSE by series and model:")
print(avg_rmse.head(20))
```

**Example 4: Identify systematic bias patterns**
```python
# Check if models consistently over-forecast or under-forecast
bias = fold_details.groupby(["item_id", "model"]).apply(
    lambda x: (x["forecast"] - x["actual"]).mean()
).reset_index(name="mean_bias")

# Positive = over-forecasting, Negative = under-forecasting
print(bias[bias["mean_bias"].abs() > 5].sort_values("mean_bias", ascending=False))
```

## Troubleshooting

### "Skipped X series with insufficient history"
**Cause**: Series shorter than `min_train_weeks + eval_window + (n_folds - 1) × fold_spacing`

**Solutions:**
- Reduce `n_folds` (e.g., 3 → 2)
- Reduce `eval_window` (e.g., 13 → 8)
- Reduce `fold_spacing` (e.g., 13 → 8) — **warning**: may cause fold overlap
- Accept that short-history items are skipped

### "No active items remain after filtering"
**Cause**: `inactive_weeks` threshold is too aggressive, or all items truly are inactive

**Solutions:**
- Reduce `inactive_weeks` (e.g., 26 → 13)
- Set `inactive_weeks=0` to disable filtering
- Verify your data — are these items actually discontinued?

### All models produce NaN forecasts for some series
**Cause**: Unusual data (all zeros, constant values, extremely short history)

**Solutions:**
- Inspect the problematic series manually
- Pre-filter series with zero variance before running the pipeline
- These series will fall back to `seasonal_naive` by default

### WAPE/MASE is inf for many series
**Cause**: Evaluation period has zero total demand (WAPE) or in-sample seasonal naïve MAE is zero (MASE)

**Solutions:**
- This is expected for highly intermittent items
- Tournament selection automatically handles inf by filtering
- Review `selection_method` distribution — high "fallback" rate indicates an issue

## Version History

### v1.4.0 (2026-02-13)
- Added six new benchmark models: weighted seasonal average, seasonal median, seasonal naïve blend, Holt-Winters (ETS), Theta method, and IMAPA
- `statsmodels` is now an optional dependency for Tier 2 models (Holt-Winters, Theta)
- Tier 2 models gracefully return NaN when statsmodels is not installed (excluded from tournament)
- Total model count: 11

### v1.3.0 (2026-02-13)
- Added inactive item detection and filtering
- Pipeline now returns 5-tuple including `inactive_df` and `fold_details`
- `fold_details` provides week-level diagnostics for each model × fold × series
- Inactive items excluded from evaluation and forecasting
- Optional via `inactive_weeks` parameter (default 26, set to 0 to disable)

### v1.2.0 (2026-02-13)
- Implemented rolling-origin cross-validation
- Configurable `eval_window` (decoupled from forecast horizon)
- Metrics averaged across folds for stable selection
- Added per-fold diagnostics output
- Added fold stability tracking (`wape_std`)

### v1.1.0 (2026-02-13)
- Vectorized forecast generation for efficiency
- Consolidated model registry with decorator pattern
- Eliminated row-wise apply operations
- Added progress logging

### v1.0.0 (2026-02-13)
- Initial release
- Five benchmark models (seasonal naïve, weekly avg, global avg, TSB, temp agg + SES)
- Tournament-style model selection
- WAPE and MASE evaluation metrics
- End-to-end pipeline interface

## License & Author

**Author**: [Your Name / Team]
**License**: [Your License]

---

## Example Workflow

```python
import pandas as pd
from best_fit_pipeline import best_fit_pipeline, summarize_results

# 1. Load data
df = pd.read_csv("weekly_demand.csv", parse_dates=["date"])

# 2. Run pipeline
eval_df, best_fit_df, forecast_df, inactive_df, fold_details = best_fit_pipeline(
    df,
    date_col="date",
    id_col="sku_id",
    value_col="units_sold",
    horizon=52,
    eval_window=13,
    n_folds=3,
    inactive_weeks=26,
)

# 3. Summarize results
summary = summarize_results(best_fit_df, eval_df, id_col="sku_id", inactive_df=inactive_df)
print(summary)

# 4. Inspect inactive items
print(f"\nInactive items: {len(inactive_df)}")
print(inactive_df.sort_values("weeks_since_demand").head(10))

# 5. Export forecasts
forecast_df.to_csv("forecasts_next_year.csv", index=False)

# 6. Deep dive on specific series
series_id = "SKU_12345"
series_eval = eval_df[eval_df["sku_id"] == series_id]
print(f"\n{series_id} evaluation:")
print(series_eval[["model", "wape", "mase", "fold_wapes"]])

winner = best_fit_df[best_fit_df["sku_id"] == series_id].iloc[0]
print(f"\nWinner: {winner['best_model']} (WAPE={winner['wape']:.3f}, MASE={winner['mase']:.3f})")

# 7. Week-level diagnostics for a specific series and model
series_fold_data = fold_details[
    (fold_details["sku_id"] == series_id) &
    (fold_details["model"] == winner['best_model'])
]
print(f"\nWeek-level performance for {winner['best_model']}:")
print(series_fold_data.groupby("fold").apply(
    lambda x: pd.Series({
        "weeks": len(x),
        "total_actual": x["actual"].sum(),
        "total_forecast": x["forecast"].sum(),
        "mae": (x["actual"] - x["forecast"]).abs().mean()
    })
))
```

## Contributing

To add a new benchmark model:

1. Define the model function following the signature:
   ```python
   def my_model(history: np.ndarray, dates: np.ndarray,
                horizon: int = 52, **kwargs) -> np.ndarray:
       # Your forecasting logic
       return forecast_array
   ```

2. Register it using the decorator:
   ```python
   @register_model("my_model")
   def my_model(history, dates, horizon, **kwargs):
       # ...
   ```

3. The model will automatically participate in evaluation and tournament selection

## References

- Teunter, R. H., Syntetos, A. A., & Babai, M. Z. (2011). "Intermittent demand: Linking forecasting to inventory obsolescence." *European Journal of Operational Research*, 214(3), 606-615.
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts. [Chapter 5: Time series cross-validation](https://otexts.com/fpp3/tscv.html)
- Petropoulos, F. & Kourentzes, N. (2015). "Forecast combinations for intermittent demand." *Journal of the Operational Research Society*, 66(6), 914-924.
- Kolassa, S. (2016). "Evaluating predictive count data distributions in retail sales forecasting." *International Journal of Forecasting*, 32(3), 788-803.
