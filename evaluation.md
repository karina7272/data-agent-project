# Data Agent Evaluation

This document summarizes the agent’s current performance and concrete areas for improvement.

## Per-Query Results

| # | Query Type    | Correct? | Latency (s) |
|---|---------------|:--------:|------------:|
| 1 | count         | No       | 0.000 |
| 2 | mean          | No       | 0.202 |
| 3 | groupby_mean  | No       | 0.000 |
| 4 | filter_count  | No       | 0.141 |
| 5 | python        | No       | 0.000 |

All five evaluation queries returned incorrect answers. Latencies were extremely low (average ≈ **0.069 s**), demonstrating that the agent is performant but not yet accurate.

## Scoring

Metrics computed with the **70% / 30%** weighting.

- **Accuracy (70%)**: **0.000** — fraction of correct (0/5)  
- **Speed score**: **0.936** — computed as `1 / (1 + avg_latency)` with avg latency ≈ 0.069 s  
- **Final score**: **0.281** — `0.70 × Accuracy + 0.30 × Speed`

The agent’s speed is excellent, but overall performance is dominated by the zero accuracy. The poor accuracy stems from incorrect or missing implementations for simple aggregations (**count**, **mean**, **groupby_mean**) and filtered queries. Ensuring the underlying functions parse the dataset correctly and apply the expected filters would markedly improve accuracy.

## Bonus Insights

**High missingness.** Columns such as *longitude* and *latitude* are entirely missing (100%), and *connecting_pipeline* has ~78.2% missing values. High missingness can bias computations and should be imputed or excluded. Other columns like *connecting_entity* and *county_name* also show notable gaps.

**Outliers.** **4,097** rows were flagged as outliers via z-score thresholding. These can skew means; prefer median or trimmed means, or filter outliers prior to aggregation.

**Collinearity.** Variance Inflation Factors indicate *rec_del_sign* and *scheduled_quantity* are nearly perfectly collinear (VIF ≈ 1.0), implying redundancy; downstream analyses should avoid using both.

**K-means clustering.** Three clusters were identified: two very large clusters (~8.9M and ~14.9M rows) and one tiny cluster (859 rows). The small cluster shows extremely high *scheduled_quantity* (~162M) and positive *rec_del_sign*; this likely reflects anomalous or special-purpose records and merits review.

## Recommendations

**Fix deterministic logic.** Correct the fallback path for *count / mean / groupby* and filter parsing. Add unit tests that mirror the scripted queries.

**Handle missing data.** Drop or impute columns with extreme missingness (e.g., 100% missing *longitude/latitude*; 78% missing *connecting_pipeline*). Investigate whether these fields are essential.

**Use robust aggregations.** Prefer **median** or **trimmed mean** when heavy tails are present; or exclude z-score outliers before computing means.

**Address multicollinearity.** Since *rec_del_sign* and *scheduled_quantity* are near-duplicates, drop one or reduce dimensionality before clustering or regression.

**Investigate the small cluster.** Validate whether those records are legitimate (e.g., emergency transfers) or data-entry errors; adjust downstream logic accordingly.

---

### Summary

- Accuracy is **0.0** on this scripted set; speed is **high** (avg latency ~0.07 s).  
- Final weighted score **0.281**.  
- Actionable data issues: missingness, outliers, multicollinearity, and a small anomalous cluster.  
- Implementing the fixes above should significantly improve accuracy without sacrificing speed.
