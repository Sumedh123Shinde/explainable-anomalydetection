import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def build_representation(df):
    """
    Converts any dataframe into a numeric feature matrix
    compatible with modern scikit-learn.
    """

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

    if not numeric_cols and not categorical_cols:
        raise ValueError("Dataset has no usable columns")

    numeric_pipeline = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    categorical_pipeline = Pipeline(
        steps=[(
            "encoder",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        )]
    )

    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ]
    )

    X = transformer.fit_transform(df)

    return X, {
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "total_features": X.shape[1]
    }
