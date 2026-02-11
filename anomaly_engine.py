from models.isolation_forest import run_isolation_forest
from models.autoencoder import run_autoencoder

def run_anomaly_engine(X, method, params):
    if method == "Isolation Forest":
        return run_isolation_forest(
            X,
            contamination=params.get("contamination", 0.05)
        )

    elif method == "Autoencoder":
        return run_autoencoder(
            X,
            epochs=params.get("epochs", 30),
            batch_size=params.get("batch_size", 32),
            anomaly_percentile=params.get("percentile", 95)
        )

    else:
        raise ValueError("Unknown anomaly detection method")
