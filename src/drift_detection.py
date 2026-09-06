import numpy as np
import pandas as pd


TRAIN_FILE = "data/data.csv"
PREDICTION_FILE = "data/random_100_predictions.csv"

FEATURES = [
    "sno",
    "age",
    "gender",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

CATEGORICAL_FEATURES = [
    "gender",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]


def calculate_psi(reference, current, bins=10):
    """Calculate Population Stability Index."""
    reference = pd.Series(reference).dropna().astype(float)
    current = pd.Series(current).dropna().astype(float)

    if reference.empty or current.empty:
        return np.nan

    # Use reference quantiles as bins.
    edges = np.unique(
        np.quantile(
            reference,
            np.linspace(0, 1, bins + 1)
        )
    )

    if len(edges) < 2:
        return 0.0

    # Extend the boundaries.
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = ref_counts / len(reference)
    cur_pct = cur_counts / len(current)

    # Avoid log(0).
    ref_pct = np.clip(ref_pct, 1e-6, None)
    cur_pct = np.clip(cur_pct, 1e-6, None)

    psi = np.sum(
        (cur_pct - ref_pct)
        * np.log(cur_pct / ref_pct)
    )

    return float(psi)


def categorical_psi(reference, current):
    """Calculate PSI for categorical variables."""
    reference = pd.Series(reference).dropna().astype(str)
    current = pd.Series(current).dropna().astype(str)

    categories = sorted(
        set(reference.unique()) |
        set(current.unique())
    )

    ref_counts = reference.value_counts()
    cur_counts = current.value_counts()

    ref_pct = np.array(
        [ref_counts.get(c, 0) / len(reference) for c in categories]
    )

    cur_pct = np.array(
        [cur_counts.get(c, 0) / len(current) for c in categories]
    )

    ref_pct = np.clip(ref_pct, 1e-6, None)
    cur_pct = np.clip(cur_pct, 1e-6, None)

    psi = np.sum(
        (cur_pct - ref_pct)
        * np.log(cur_pct / ref_pct)
    )

    return float(psi)


def main():
    train = pd.read_csv(TRAIN_FILE)
    current = pd.read_csv(PREDICTION_FILE)

    # Convert gender to the same numerical representation
    # used by the model.
    gender_mapping = {
        "male": 0,
        "female": 1,
    }

    train["gender"] = (
        train["gender"]
        .map(gender_mapping)
    )

    current["gender"] = (
        current["gender"]
        .map(gender_mapping)
    )

    results = []

    for feature in FEATURES:

        if feature in CATEGORICAL_FEATURES:
            psi = categorical_psi(
                train[feature],
                current[feature]
            )
            method = "Categorical PSI"

        else:
            psi = calculate_psi(
                train[feature],
                current[feature]
            )
            method = "Numeric PSI"

        if pd.isna(psi):
            status = "Unable to calculate"
        elif psi < 0.10:
            status = "No significant drift"
        elif psi < 0.20:
            status = "Moderate drift"
        else:
            status = "Significant drift"

        results.append({
            "feature": feature,
            "method": method,
            "psi": round(psi, 6),
            "status": status,
        })

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "models/drift_results.csv",
        index=False
    )

    print("\nINPUT DRIFT ANALYSIS")
    print("=" * 70)
    print(results_df.to_string(index=False))

    print("\nInterpretation:")
    print("PSI < 0.10  -> No significant drift")
    print("0.10-0.20   -> Moderate drift")
    print("PSI >= 0.20 -> Significant drift")

    significant = (
        results_df["status"] == "Significant drift"
    ).sum()

    moderate = (
        results_df["status"] == "Moderate drift"
    ).sum()

    print()
    print(f"Significant drift features: {significant}")
    print(f"Moderate drift features: {moderate}")


if __name__ == "__main__":
    main()
