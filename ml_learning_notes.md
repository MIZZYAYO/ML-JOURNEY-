# ML Learning Notes: Salary Prediction

## Progress so far
1. NumPy: arrays, vectorization, indexing, sum/mean/std, boolean filtering
2. Pandas: DataFrames, selecting, filtering, sorting, read_csv, missing data, groupby
3. Visualization: bar chart, histogram, boxplot
4. Supervised ML: train/test split, LinearRegression, one-hot encoding (`get_dummies`)
5. Evaluation: cross-validated MAE, comparing "Age only" vs "Age + department"

## Regression metrics
A **metric** is a number that measures how well a model performs.

| Metric | Meaning | Better when |
|---|---|---|
| MAE | average of absolute misses, in salary units | lower |
| MSE | average of squared misses (squared units) | lower |
| RMSE | square root of MSE, in salary units, punishes big misses | lower |
| R² | 1 - (model's squared errors / "guess the average" squared errors) | closer to 1 |

- Miss (error) = actual - predicted
- RMSE ≈ MAE means errors are even. RMSE much bigger than MAE means a few big misses.
- `cross_val_score` with `neg_mean_absolute_error` returns negative values, so use `-scores.mean()`.

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae  = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2   = r2_score(y_test, predictions)
```

## Coefficients
A linear regression prediction = intercept + sum(feature value x coefficient).

```python
for name, coef in zip(feature_cols, model.coef_):
    print(f"{name}: {coef:,.0f}")
print(f"intercept: {model.intercept_:,.0f}")
```

Results on the 7-row dataset:

| Feature | Coefficient |
|---|---|
| age | -333 |
| department_Engineering | +29,889 |
| department_Sales | -8,944 |
| department_Marketing | -20,944 |
| intercept | 77,111 |

Example: 30-year-old in Marketing = 77,111 + (-20,944) + (-333.33 x 30) = 46,167

- Department columns are 0/1, so only the person's own department counts.
- Coefficients are found in `.fit()` by least squares: minimize total squared error on the training data.
- With one feature: slope = sum((x - x_mean)(y - y_mean)) / sum((x - x_mean)^2), intercept = y_mean - slope x x_mean.

## Caveats
- Only 7 rows remain after `dropna()`, so test scores (a 2-person test set) are unreliable.
- The negative age coefficient is a quirk of tiny data, not a real-world rule.
- Fill missing values with the mean, not 0, if you need to keep rows.

## Where we stopped
Just finished explaining R².

## Next steps
1. Classification: predict `active` with Logistic Regression (accuracy, precision, recall, F1)
2. Overfitting: why a great score on tiny data can't be trusted
3. Better models: Random Forest vs Linear Regression
