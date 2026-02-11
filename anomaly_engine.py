from sklearn.ensemble import IsolationForest
import numpy as np


def run_anomaly_engine(X, method="Isolation Forest", params=None):

    if method == "Isolation Forest":

        contamination = params.get("contamination", 0.05)

        model = IsolationForest(
            contamination=contamination,
            random_state=42
        )

        model.fit(X)

        predictions = model.predict(X)

        # Convert: -1 = anomaly, 1 = normal
        anomaly_mask = predictions == -1

        scores = -model.decision_function(X)

        return {
            "anomaly_mask": anomaly_mask,
            "scores": scores
        }

    else:
        raise ValueError("Only Isolation Forest supported in this version.")
