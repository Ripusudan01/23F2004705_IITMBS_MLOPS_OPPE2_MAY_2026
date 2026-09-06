# Deliverable 3 — Fairness Analysis

## Objective

Evaluate the fairness of the heart disease prediction model using Fairlearn, with **age** as the sensitive attribute.

The target variable is converted from:

- `no` → `0`
- `yes` → `1`

This binary representation is used for Fairlearn fairness metrics.

## Model

The same preprocessing and Logistic Regression training procedure from the official starter notebook was used.

- Missing rows are removed using `dropna()`.
- `gender` is factorized.
- Logistic Regression is tuned using `RandomizedSearchCV`.
- Solver: `liblinear`
- Best C: `0.615848211066026`
- Test accuracy: `0.9830508474576272` (~98.31%)

## Sensitive Attribute

The explicitly specified sensitive attribute for this deliverable is:

**Age**

Fairlearn evaluates the difference in model outcomes across the age groups present in the test set.

## Fairness Metrics

### 1. Selection Rate

Selection rate represents the proportion of samples receiving a positive prediction.

For this model, a positive prediction means:

`yes` → heart disease predicted.

The selection rate was calculated for each exact age.

The observed selection rates ranged from `0.0` to `1.0`.

Because many individual age groups contain only one to four samples, the selection rates for individual ages can be unstable.

### 2. Demographic Parity Difference

Observed result:

**1.0**

This indicates a large difference in positive prediction rates between the age groups in this test sample.

A value closer to zero indicates smaller differences in selection rates.

The result should be interpreted cautiously because the test dataset is small and many exact-age groups have very few observations.

### 3. Equalized Odds Difference

Observed result:

**1.0**

This indicates a large difference in the model's error-rate/true-positive-rate behavior across the age groups in this test sample.

Again, the result should be interpreted cautiously because individual age groups have small sample sizes.

## Age Group Analysis

To provide a more stable and interpretable comparison, ages were additionally grouped into:

- `<40`
- `40-49`
- `50-59`
- `60-69`
- `70+`

The observed selection rates were:

| Age Group | Selection Rate |
|---|---:|
| <40 | 1.0000 |
| 40-49 | 0.6316 |
| 50-59 | 0.5909 |
| 60-69 | 0.4545 |
| 70+ | 0.5000 |

The age-group analysis shows that positive prediction rates differ across age bands.

## Interpretation

The model achieves high predictive accuracy (~98.31%) on the test split, but the fairness metrics show disparities across age groups.

This demonstrates that **high predictive accuracy does not necessarily imply fairness across sensitive groups**.

The fairness results should not be interpreted as definitive evidence of population-level unfairness because this dataset is small and the exact-age groups contain few samples.

For a production system, fairness should be evaluated using a larger, representative validation dataset and monitored continuously after deployment.

## Generated Files

This analysis generates:

- `models/fairness_summary.csv`
- `models/selection_rate_by_age.csv`
- `models/fairness_metrics_by_age_group.csv`

## Reproducibility

The analysis can be reproduced using:

```bash
python src/fairness.py
