"""
train_models.py

Train credit risk models on loan-level data and save artifacts for the dashboard.
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Keep only a subset of useful columns to start; you can expand later.
    # Adjust these based on the actual LendingClub schema you download.
    candidate_cols = [
        "loan_amnt",
        "term",
        "int_rate",
        "installment",
        "annual_inc",
        "dti",
        "grade",
        "sub_grade",
        "emp_length",
        "home_ownership",
        "purpose",
        "addr_state",
        "loan_status",
    ]
    existing = [c for c in candidate_cols if c in df.columns]
    df = df[existing].copy()
    return df


def prepare_labels(df: pd.DataFrame) -> pd.Series:
    """
    Create a binary default label from LendingClub-style loan_status column.
    You may need to tweak mappings depending on the file you use.
    """
    status = df["loan_status"].astype(str)

    default_like = [
        "Charged Off",
        "Default",
        "Late (16-30 days)",
        "Late (31-120 days)",
        "In Grace Period",
    ]
    y = status.isin(default_like).astype(int)
    return y


def build_pipeline(df: pd.DataFrame) -> Pipeline:
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [
        c for c in df.columns
        if c not in numeric_features + ["loan_status"]
    ]

    numeric_transformer = "passthrough"
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Start with a simple logistic regression model. You can add RandomForest, XGBoost, etc.
    clf = LogisticRegression(max_iter=1000)

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", clf),
        ]
    )
    return pipe


def main(data_path: str, output_dir: str = "artifacts"):
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(str(data_path))
    y = prepare_labels(df)
    X = df.drop(columns=["loan_status"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline(X)

    pipe.fit(X_train, y_train)
    y_pred_proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Validation ROC-AUC: {auc:.3f}")

    joblib.dump(pipe, output_dir / "credit_risk_model.joblib")
    joblib.dump(
        {"auc": float(auc), "columns": X.columns.tolist()},
        output_dir / "model_metadata.joblib",
    )
    print(f"Saved model and metadata to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Path to LendingClub CSV file")
    parser.add_argument("--output_dir", default="artifacts", help="Directory to save model artifacts")
    args = parser.parse_args()
    main(args.data_path, args.output_dir)
