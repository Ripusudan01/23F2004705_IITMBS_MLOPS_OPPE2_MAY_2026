import json
import random
import time
from datetime import datetime, timezone

import pandas as pd
import requests


API_URL = "http://34.93.99.223/predict"
OUTPUT_FILE = "data/random_100_predictions.csv"

random.seed(42)


def generate_sample(sno):
    return {
        "sno": sno,
        "age": random.randint(29, 77),
        "gender": random.choice(["male", "female"]),
        "cp": random.randint(0, 3),
        "trestbps": round(random.uniform(90, 200), 1),
        "chol": round(random.uniform(125, 570), 1),
        "fbs": random.randint(0, 1),
        "restecg": random.randint(0, 2),
        "thalach": round(random.uniform(70, 200), 1),
        "exang": random.randint(0, 1),
        "oldpeak": round(random.uniform(0, 6.5), 1),
        "slope": random.randint(0, 2),
        "ca": random.randint(0, 3),
        "thal": random.randint(0, 3),
    }


def main():
    rows = []

    for i in range(1, 101):
        sample = generate_sample(i)

        request_time = datetime.now(timezone.utc).isoformat()

        try:
            response = requests.post(
                API_URL,
                json=sample,
                timeout=10
            )

            response.raise_for_status()
            result = response.json()

            rows.append({
                **sample,
                "prediction": result["prediction"],
                "probability": result["probability"],
                "timestamp": result["timestamp"],
                "http_status": response.status_code,
            })

            print(
                f"{i:03d}/100 "
                f"status={response.status_code} "
                f"prediction={result['prediction']} "
                f"probability={result['probability']}"
            )

        except Exception as exc:
            rows.append({
                **sample,
                "prediction": "ERROR",
                "probability": None,
                "timestamp": request_time,
                "http_status": None,
                "error": str(exc),
            })

            print(f"{i:03d}/100 ERROR: {exc}")

        time.sleep(0.05)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False)

    print()
    print(f"Saved {len(df)} rows to {OUTPUT_FILE}")
    print("Successful predictions:", (df["prediction"] != "ERROR").sum())


if __name__ == "__main__":
    main()
