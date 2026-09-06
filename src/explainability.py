import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------
# 1. Load and preprocess data
# ---------------------------------------------------------
df = pd.read_csv("data.csv")

# Same preprocessing as the official starter notebook
df["gender"] = pd.factorize(df["gender"])[0]

cleaned_df = df.dropna()

X = cleaned_df.drop("target", axis=1)
y = cleaned_df["target"]

np.random.seed(42)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2
)


# ---------------------------------------------------------
# 2. Train the same model as the official notebook
# ---------------------------------------------------------
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

print("\nBest parameters:")
print(rs_log_reg.best_params_)

print("\nTest accuracy:")
print(rs_log_reg.score(X_test, y_test))


# ---------------------------------------------------------
# 3. SHAP explanation
# ---------------------------------------------------------
explainer = shap.Explainer(model, X_train)

shap_explanation = explainer(X_test)

shap_values = shap_explanation.values

# Handle possible multi-dimensional SHAP output
if shap_values.ndim == 3:
    # For binary classification, class index 1 is the positive class
    shap_values = shap_values[:, :, 1]


# ---------------------------------------------------------
# 4. Calculate mean absolute SHAP importance
# ---------------------------------------------------------
mean_abs_shap = np.abs(shap_values).mean(axis=0)

importance_df = pd.DataFrame({
    "feature": X_test.columns,
    "mean_abs_shap": mean_abs_shap
})

importance_df = importance_df.sort_values(
    "mean_abs_shap",
    ascending=True
)

print("\nSHAP feature importance - least to most impact:")
print(importance_df.to_string(index=False))


# ---------------------------------------------------------
# 5. Save results
# ---------------------------------------------------------
os.makedirs("models", exist_ok=True)

importance_df.to_csv(
    "models/shap_feature_importance.csv",
    index=False
)


# ---------------------------------------------------------
# 6. Plot feature importance
# ---------------------------------------------------------
plot_df = importance_df.sort_values(
    "mean_abs_shap",
    ascending=False
)

plt.figure(figsize=(10, 7))

plt.barh(
    plot_df["feature"],
    plot_df["mean_abs_shap"]
)

plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Feature")
plt.title("SHAP Feature Importance - Heart Disease Model")

plt.tight_layout()

plt.savefig(
    "models/shap_feature_importance.png",
    dpi=200
)

plt.close()

print("\nSaved:")
print("  models/shap_feature_importance.csv")
print("  models/shap_feature_importance.png")


# ---------------------------------------------------------
# 7. SHAP summary plot
# ---------------------------------------------------------
shap.summary_plot(
    shap_values,
    X_test,
    show=False
)

plt.tight_layout()

plt.savefig(
    "models/shap_summary_plot.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("  models/shap_summary_plot.png")
