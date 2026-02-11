import pandas as pd
import json
import os

REGISTRY_PATH = "data/registry.json"

# ---------- Registry ----------
def load_registry():
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)

def list_builtin_datasets():
    registry = load_registry()
    return list(registry.keys())

# ---------- Loaders ----------
def load_builtin_dataset(name):
    registry = load_registry()
    if name not in registry:
        raise ValueError("Dataset not found in registry")

    info = registry[name]
    df = pd.read_csv(info["path"])

    return df, {
        "source": "builtin",
        "name": name,
        "type": info["type"],
        "description": info.get("description", "")
    }

def load_uploaded_dataset(file):
    df = pd.read_csv(file)
    return df, {
        "source": "upload",
        "name": file.name,
        "type": "unknown",
        "description": "User uploaded dataset"
    }

# ---------- Validation ----------
def basic_validation(df):
    issues = []

    if df.empty:
        issues.append("Dataset is empty")

    if df.shape[1] < 1:
        issues.append("No columns detected")

    return issues

# ---------- Feature inspection ----------
def inspect_columns(df):
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    categorical = df.select_dtypes(exclude=["number"]).columns.tolist()

    return {
        "numeric": numeric,
        "categorical": categorical
    }
