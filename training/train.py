import argparse
import json
from pathlib import Path

import joblib
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train(version: str) -> dict:
    dataset = load_diabetes(as_frame=True)
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.2, random_state=42
    )
    estimator = LinearRegression() if version == "v0.1" else Ridge(alpha=10.0)
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    metrics = {
        "version": version,
        "model": type(estimator).__name__,
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "random_state": 42,
        "test_size": 0.2,
    }
    Path("models").mkdir(exist_ok=True)
    joblib.dump(pipeline, "models/model.joblib")
    Path("models/metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["v0.1", "v0.2"], required=True)
    args = parser.parse_args()
    train(args.version)
