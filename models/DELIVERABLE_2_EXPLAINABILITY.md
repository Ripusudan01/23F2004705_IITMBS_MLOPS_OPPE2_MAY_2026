# Deliverable 2 — Model Explainability

## Objective

Use SHAP to identify the factors that have the least impact on the heart disease prediction model.

## Method

The model follows the preprocessing and training approach from the provided starter notebook:

1. Load `data.csv`.
2. Factorize the `gender` column.
3. Remove rows containing missing values.
4. Split the cleaned data into 80% training and 20% testing data.
5. Train Logistic Regression using `RandomizedSearchCV`.
6. Use SHAP to calculate feature contributions.
7. Calculate the mean absolute SHAP value for each feature.

## Model Result

- Cleaned samples: 293
- Best solver: `liblinear`
- Best C: `0.615848211066026`
- Test accuracy: `98.31%`

## SHAP Results

Features ranked from least to most impactful:

| Feature | Mean Absolute SHAP |
|---|---:|
| exang | 0.006960 |
| thal | 0.007379 |
| fbs | 0.011561 |
| gender | 0.011657 |
| ca | 0.139891 |
| slope | 0.248179 |
| restecg | 0.250865 |
| cp | 0.360391 |
| oldpeak | 1.035425 |
| chol | 1.090068 |
| age | 2.110840 |
| trestbps | 2.193408 |
| thalach | 3.665173 |
| sno | 28.667377 |

## Plain-English Interpretation

According to the SHAP analysis, `exang` (exercise-induced angina) has the smallest overall impact on the model's predictions. It is followed by `thal`, `fbs` (fasting blood sugar), and `gender`.

A lower mean absolute SHAP value means that the feature generally changes the model's predictions less than features with larger SHAP values.

Therefore, the least impactful features in this model are:

- Exercise-induced angina (`exang`)
- `thal`
- Fasting blood sugar (`fbs`)
- Gender

## Important Note About `sno`

The `sno` column is retained because it is included as a predictor by the provided starter notebook.

Although `sno` is not a clinical attribute, the official notebook includes it when creating the feature matrix. Therefore, it was retained to reproduce the provided model faithfully.

## Output Files

- `shap_feature_importance.csv`
- `shap_feature_importance.png`
- `shap_summary_plot.png`
