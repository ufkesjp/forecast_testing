# Demand Forecasting Pipeline — How It Works

## What This Tool Does

This tool automatically generates weekly demand forecasts for every active product in your catalog. Instead of relying on a single forecasting method, it runs a **tournament of 13 different forecasting approaches** for each product, picks the one that performed best historically, and uses that winner to predict future demand.

The result: every product gets a forecast tailored to its own demand behavior — whether it's a steady seller, a seasonal item, or something that only sells occasionally.

---

## How It Works (Step by Step)

### 1. Remove Inactive Products
Products with zero sales in the last 6 months are flagged as inactive and set aside. This keeps the system focused on items that actually need forecasts and saves processing time.

Inactive items aren't deleted — they're reported separately so you can review them and confirm they should be excluded.

### 2. Test Every Forecasting Method
For each active product, the system holds back the most recent year of sales data as a "test period." It then asks each of the 13 forecasting methods to predict that year using only older data.

This is like giving each method a pop quiz on data it hasn't seen — the method that scores best on the quiz earns the right to make real predictions.

### 3. Pick the Winner
Each method is scored on two measures of accuracy:
- **How close were the total predictions to total actual demand?** (Are we in the right ballpark?)
- **Did this method beat a simple baseline?** (Is it actually adding value?)

These two scores are blended together (70% first measure, 30% second) to produce a single ranking. The top-ranked method wins.

### 4. Generate Forecasts
The winning method for each product is retrained on all available history (including the test period) and used to generate the final week-by-week demand forecast.

---

## The 13 Forecasting Methods

The methods range from simple baselines to more sophisticated statistical approaches. Each one is designed to handle a different type of demand pattern.

| Method | What It Does | Works Best For |
|---|---|---|
| **Seasonal Naive** | Repeats last year's weekly pattern | Products with strong seasonal cycles |
| **Weekly Historical Average** | Averages the same week across all years | Consistent seasonal items with some year-to-year noise |
| **Global Average** | Uses the overall average as a flat forecast | Stable, non-seasonal products (also acts as a sanity check) |
| **TSB** | Tracks both "how often" and "how much" separately | Products that sell infrequently or are winding down |
| **Temporal Aggregation + SES** | Groups weeks into monthly buckets before forecasting | Sparse weekly demand that's easier to see at the monthly level |
| **Weighted Seasonal Average** | Like the weekly average, but gives more weight to recent years | Products with shifting or trending demand |
| **Seasonal Median** | Uses the middle value instead of the average for each week | Products with occasional large spikes that would skew an average |
| **Seasonal Naive Blend** | Splits the difference between repeating last year and averaging all years | Products where neither approach alone is good enough |
| **Holt-Winters** | Statistical model that captures both trend and seasonality | Products with clear upward/downward trends and seasonal patterns |
| **Theta Method** | Award-winning competition method that balances trend and short-term signals | A strong all-around method for a wide range of patterns |
| **IMAPA** | Forecasts at multiple time scales and averages the results | Intermittent demand where single-scale methods struggle |
| **Damped Trend Seasonal** | Like Holt-Winters, but the trend gradually levels off over time | High-volume items where you don't want runaway trend projections |
| **Linear Trend + Seasonal** | Fits a straight-line trend and overlays seasonal patterns | Steady growth or decline with predictable seasonality |

You don't need to choose or configure these — the tournament handles selection automatically.

---

## What You Get Back

The pipeline produces five sets of results:

### Forecasts
The main output: a predicted demand quantity for each product for each future week. This is what feeds into planning, ordering, and allocation decisions.

### Best-Fit Selections
A summary of which forecasting method won the tournament for each product, along with its accuracy scores. Useful for understanding the overall composition of your forecast.

For example, if 60% of products were won by seasonal methods, that tells you seasonality is a dominant pattern in your catalog.

### Evaluation Results
Detailed accuracy scores for every method on every product. This is the full scorecard from the tournament — helpful if you want to understand why a particular method was chosen, or how close the runner-up was.

### Inactive Items Report
A list of products that were excluded due to inactivity, along with diagnostic details: when they last sold, how long they've been dormant, and their total lifetime volume. This helps you confirm that the right products were filtered out.

### Week-Level Details
The most granular output: for every product and every method, the actual vs. predicted demand for each week of the test period. This is the primary diagnostic tool for analysts who want to dig into specific products and understand exactly where a method did well or struggled.

---

## Key Concepts

### Why a Tournament?
No single forecasting method works best for every product. A seasonal method might be perfect for holiday items but terrible for a product with no seasonality. By running a competition and letting the data decide, each product gets the approach that actually fits its behavior.

### What Is "Holdout" Testing?
To fairly judge each method, we hide the most recent year of data and ask each method to predict it. Since we already know the real answer, we can objectively measure accuracy. This prevents the system from picking a method that just memorizes the past without actually being able to predict the future.

### Why Are Some Products Excluded?
Products with no recent sales are excluded because forecasting them wastes resources and can produce misleading results. These are typically discontinued items or products that have been out of stock. The inactive items report lets you verify this and take action if something was excluded by mistake.

### How Accurate Are the Forecasts?
Accuracy varies by product. Stable, high-volume items with clear seasonal patterns tend to forecast very well. Intermittent or highly volatile items are inherently harder to predict. The evaluation results and week-level details allow you to assess accuracy for any specific product or product group.

---

## Frequently Asked Questions

**Can I change how far ahead it forecasts?**
Yes. The forecast horizon is configurable — the default is 52 weeks (one year), but it can be set to any number of weeks.

**What if a product doesn't have enough history?**
Products need at least two years of weekly data to participate in the tournament (one year for training, one year for testing). Products with less history are skipped and noted in the output.

**What happens if all methods perform poorly for a product?**
If no method produces valid results, the system falls back to repeating last year's pattern (Seasonal Naive). The selection results will flag these cases so they can be reviewed.

**Can I add my own forecasting method?**
Yes. The system is designed to be extensible — new methods can be registered and will automatically participate in the tournament. This requires a developer to implement.

**How long does it take to run?**
The pipeline processes products in parallel across multiple CPU cores. Run time depends on the number of products and the amount of history, but the system is designed to handle 30,000+ products efficiently.
