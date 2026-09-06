# Deliverable 7 — Input Drift Detection

## Objective

Compare the feature distribution of the training dataset with the distribution of the 100 randomly generated prediction inputs.

## Datasets

- Training/reference dataset: `data/data.csv`
- Current/generated prediction dataset: `data/random_100_predictions.csv`
- Training samples: 303
- Generated prediction samples: 100

## Method

Population Stability Index (PSI) was used to compare the training/reference distribution with the generated prediction distribution.

Interpretation:

- PSI < 0.10: No significant drift
- PSI 0.10–0.20: Moderate drift
- PSI >= 0.20: Significant drift

Categorical features were evaluated using categorical PSI and numerical features using PSI based on reference quantile bins.

## Results

| Feature | Method | PSI | Status |
|---|---|---:|---|
| sno | Numeric PSI | 7.574318 | Significant drift |
| age | Numeric PSI | 0.628357 | Significant drift |
| gender | Categorical PSI | 0.055059 | No significant drift |
| cp | Categorical PSI | 0.135341 | Moderate drift |
| trestbps | Numeric PSI | 1.359631 | Significant drift |
| chol | Numeric PSI | 1.129523 | Significant drift |
| fbs | Categorical PSI | 1.232470 | Significant drift |
| restecg | Categorical PSI | 0.810169 | Significant drift |
| thalach | Numeric PSI | 0.866951 | Significant drift |
| exang | Categorical PSI | 0.037328 | No significant drift |
| oldpeak | Numeric PSI | 1.929262 | Significant drift |
| slope | Categorical PSI | 0.425997 | Significant drift |
| ca | Categorical PSI | 0.699803 | Significant drift |
| thal | Categorical PSI | 1.761139 | Significant drift |

## Summary

- Significant drift: 11 features
- Moderate drift: 1 feature
- No significant drift: 2 features

The generated prediction data shows substantial distribution differences from the training data for most input features.

The largest PSI is observed for `sno` (7.574318). The `sno` field is retained because it is part of the original starter-model feature set, although it is an identifier-like feature rather than a clinical measurement.

## Conclusion

The drift analysis indicates that the generated prediction dataset is substantially different from the training distribution for most features. This demonstrates how PSI can be used as a production monitoring signal to detect changes in incoming model inputs.

If these differences represented real production traffic rather than a deliberately random test dataset, the model should be investigated and potentially retrained or recalibrated depending on the cause and impact of the drift.
