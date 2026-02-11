import numpy as np

def explain_row(row, reference_stats, feature_columns, z_thresh=2.5):
    explanations = []

    for col in feature_columns:
        mean = reference_stats[col]["mean"]
        std = reference_stats[col]["std"]

        if std == 0:
            continue

        z = abs(row[col] - mean) / std

        if z >= z_thresh:
            explanations.append(
                f"{col} deviates by {z:.2f}σ from normal"
            )

    if explanations:
        return "; ".join(explanations)
    else:
        return "Unusual combination of features (no single dominant cause)"
