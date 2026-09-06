import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
    equalized_odds_difference,
)


# =========================================================
# 1. LOAD AND PREPROCESS DATA
# =========================================================

df = pd.read_csv("data.csv")

# Same preprocessing as the official starter notebook
df["gender"] = pd.factorize(df["gender"])[0]

# Remove rows containing missing values
cleaned_df = df.dropna()

# Features and target
X = cleaned_df.drop("target", axis=1)

# Convert target:
# no  -> 0
# yes -> 1
y = cleaned_df["target"].map({
    "no": 0,
    "yes": 1
})


# =========================================================
# 2. TRAIN / TEST SPLIT
# =========================================================

np.random.seed(42)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2
)


# =========================================================
# 3. TRAIN LOGISTIC REGRESSION
# =========================================================

log_reg_grid = {
    "C": np.logspace(-4, 4, 20),
    "solver": ["liblinear"]
}

rs_log_reg = RandomizedSearchCV(
    LogisticRegression(),
    param_distributions=log_reg_grid,
    cv=5,
    n_iter=20,
    verbose=True
)

rs_log_reg.fit(X_train, y_train)

model = rs_log_reg.best_estimator_


# =========================================================
# 4. MODEL PERFORMANCE
# =========================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print("\nBest parameters:")
print(rs_log_reg.best_params_)

print("\nTest accuracy:")
print(accuracy)


# =========================================================
# 5. CHECK ACTUAL AND PREDICTED LABELS
# =========================================================

print("\nActual labels:")
print(
    pd.Series(y_test)
    .map({0: "no", 1: "yes"})
    .value_counts()
)

print("\nPredicted labels:")
print(
    pd.Series(y_pred)
    .map({0: "no", 1: "yes"})
    .value_counts()
)


# =========================================================
# 6. AGE AS SENSITIVE ATTRIBUTE
# =========================================================

# The problem statement explicitly specifies AGE
# as the sensitive attribute.
sensitive_age = X_test["age"]


# =========================================================
# 7. SELECTION RATE BY EXACT AGE
# =========================================================

selection_rates = MetricFrame(
    metrics=selection_rate,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sensitive_age
)

print("\n" + "=" * 60)
print("SELECTION RATE BY EXACT AGE")
print("=" * 60)

print(selection_rates.by_group)


# =========================================================
# 8. DEMOGRAPHIC PARITY DIFFERENCE
# =========================================================

dp_difference = demographic_parity_difference(
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sensitive_age
)

print("\n" + "=" * 60)
print("DEMOGRAPHIC PARITY DIFFERENCE")
print("=" * 60)

print(dp_difference)


# =========================================================
# 9. EQUALIZED ODDS DIFFERENCE
# =========================================================

eo_difference = equalized_odds_difference(
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sensitive_age
)

print("\n" + "=" * 60)
print("EQUALIZED ODDS DIFFERENCE")
print("=" * 60)

print(eo_difference)


# =========================================================
# 10. CREATE AGE GROUPS
# =========================================================

def create_age_group(age):
    if age < 40:
        return "<40"
    elif age < 50:
        return "40-49"
    elif age < 60:
        return "50-59"
    elif age < 70:
        return "60-69"
    else:
        return "70+"


age_groups = X_test["age"].apply(create_age_group)


# =========================================================
# 11. FAIRNESS ANALYSIS BY AGE GROUP
# =========================================================

age_group_metrics = MetricFrame(
    metrics={
        "selection_rate": selection_rate
    },
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=age_groups
)

print("\n" + "=" * 60)
print("SELECTION RATE BY AGE GROUP")
print("=" * 60)

print(age_group_metrics.by_group)


# =========================================================
# 12. ADDITIONAL AGE-GROUP METRICS
# =========================================================

age_group_summary = age_group_metrics.by_group.copy()

age_group_summary["sample_count"] = (
    age_groups.value_counts()
    .reindex(age_group_summary.index)
)

age_group_summary = age_group_summary[
    ["sample_count", "selection_rate"]
]


# =========================================================
# 13. SAVE RESULTS
# =========================================================

os.makedirs("models", exist_ok=True)


# Required fairness summary
fairness_summary = pd.DataFrame({
    "metric": [
        "test_accuracy",
        "demographic_parity_difference",
        "equalized_odds_difference"
    ],
    "value": [
        accuracy,
        dp_difference,
        eo_difference
    ]
})

fairness_summary.to_csv(
    "models/fairness_summary.csv",
    index=False
)


# Selection rate for each exact age
selection_rates.by_group.to_csv(
    "models/selection_rate_by_age.csv",
    header=["selection_rate"]
)


# Selection rate and sample count for age groups
age_group_summary.to_csv(
    "models/fairness_metrics_by_age_group.csv"
)


# =========================================================
# 14. FINAL OUTPUT
# =========================================================

print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)

print("models/fairness_summary.csv")
print("models/selection_rate_by_age.csv")
print("models/fairness_metrics_by_age_group.csv")

print("\nFairness analysis completed successfully.")
