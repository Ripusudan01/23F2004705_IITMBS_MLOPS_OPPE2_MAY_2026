import json
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split


DATA_PATH = "data.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "heart_disease_model.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")


# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
df = pd.read_csv(DATA_PATH)


# ---------------------------------------------------------
# 2. Same gender preprocessing as official notebook
# ---------------------------------------------------------
gender_values = df["gender"].unique().tolist()

gender_mapping = {
    str(value): int(index)
    for index, value in enumerate(gender_values)
}

df["gender"] = pd.factorize(df["gender"])[0]


# ---------------------------------------------------------
# 3. Remove missing values
# ---------------------------------------------------------
cleaned_df = df.dropna()


# ---------------------------------------------------------
# 4. Features and target
# ---------------------------------------------------------
X = cleaned_df.drop("target", axis=1)
y = cleaned_df["target"]


# ---------------------------------------------------------
# 5. Same train/test split as official notebook
# ---------------------------------------------------------
np.random.seed(42)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2
)


# ---------------------------------------------------------
# 6. Same Logistic Regression tuning
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


# ---------------------------------------------------------
# 7. Evaluate
# ---------------------------------------------------------
accuracy = model.score(X_test, y_test)

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED")
print("=" * 60)

print("\nBest parameters:")
print(rs_log_reg.best_params_)

print("\nTest accuracy:")
print(accuracy)

print("\nGender mapping:")
print(gender_mapping)


# ---------------------------------------------------------
# 8. Save model and metadata
# ---------------------------------------------------------
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, MODEL_PATH)

metadata = {
    "features": X.columns.tolist(),
    "gender_mapping": gender_mapping,
    "target_values": ["no", "yes"],
    "best_params": {
        "C": float(rs_log_reg.best_params_["C"]),
        "solver": rs_log_reg.best_params_["solver"]
    },
    "test_accuracy": float(accuracy)
}

with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=2)


print("\nSaved:")
print(MODEL_PATH)
print(METADATA_PATH)
