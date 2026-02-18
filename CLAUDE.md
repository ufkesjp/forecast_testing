# Project: Best-Fit Benchmark Forecasting Pipeline

## Purpose
Automated demand forecasting toolkit that runs a tournament of 13 benchmark models across 30,000+ SKUs, selects the best model per series, and generates week-ahead forecasts. Designed for large-scale production use with parallel processing.

## Architecture Note
The pipeline is split across two files intentionally: Python's `multiprocessing` requires functions to be defined in a standalone `.py` file (not a notebook) to be picklable and distributed across workers. The notebook imports from the `.py` file and serves as the interactive entry point.

## Key Files
- [benchmark_forecasting_parallel.py](benchmark_forecasting_parallel.py) — Core engine: all 13 models, evaluation logic, parallelization, pipeline orchestration. All functions live here to support multiprocessing.
- [best_fit_pipeline.ipynb](best_fit_pipeline.ipynb) — Interactive entry point: loads data, calls the pipeline, displays results. Does not define functions itself.
- [outlier_detection_demand_pattern.py](outlier_detection_demand_pattern.py) — Utility: classifies series into demand patterns (Smooth/Erratic/Intermittent/Lumpy) and flags outliers
- [README.md](README.md) — Full documentation: model descriptions, config options, output schemas, troubleshooting

## Tech Stack
- Python 3, pandas, numpy, multiprocessing
- Optional: statsmodels (for Holt-Winters, Theta, Damped Trend Seasonal models — degrades gracefully if absent)
- Jupyter notebooks for interactive use

## The 13 Models
Registered via `@register_model` decorator in `benchmark_forecasting_parallel.py`:
1. Seasonal Naive
2. Weekly Historical Average
3. Global Average
4. TSB (Teunter-Syntetos-Babai — intermittent demand)
5. Temporal Aggregation + SES
6. Weighted Seasonal Average
7. Seasonal Median
8. Seasonal Naive Blend
9. Holt-Winters / ETS *(requires statsmodels)*
10. Theta Method *(requires statsmodels)*
11. IMAPA (multi-aggregation intermittent)
12. Damped Trend Seasonal *(requires statsmodels)*
13. Linear Trend + Seasonal

## Pipeline Flow
```
Input data (date, item_id, demand)
  → flag_inactive_items()         # exclude zero-demand trailing window
  → evaluate_benchmarks_parallel() # holdout backtest (last N weeks)
  → select_best_fit()             # tournament: 70% WAPE + 30% MASE composite score
  → generate forecasts            # winning model retrained on full history
```

## Pipeline Output (5 DataFrames)
| DataFrame | Contents |
|---|---|
| `eval_df` | All model × series × metric evaluation results |
| `best_fit_df` | Tournament winners per series |
| `forecast_df` | Final point forecasts |
| `inactive_df` | Excluded items with diagnostic info |
| `fold_details` | Week-level actuals vs. forecasts for every model and series in the test period |

`fold_details` is particularly important — it is the primary diagnostic tool for analysts to inspect how each model performed week-by-week for each series during the 52-week holdout period.

## Typical Usage
```python
from benchmark_forecasting_parallel import best_fit_pipeline_parallel

eval_df, best_fit_df, forecast_df, inactive_df, fold_details = best_fit_pipeline_parallel(
    df, date_col="date", id_col="item_id", value_col="demand",
    horizon=52, eval_window=52, inactive_weeks=26, n_workers=4
)
```

## Data Requirements
- Columns: datetime (week-ending, preferably Sundays), item ID, non-negative demand
- Scale: 30,000+ series, each with ~4 years (~208 weeks) of training history
- Test period: 52 weeks (holdout split)
- Total data volume is large — performance and memory efficiency are important considerations

## Current Version
v2.0.0 — parallelized all major operations via `multiprocessing.Pool`
