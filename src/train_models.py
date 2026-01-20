# src/train_models.py

import argparse
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

EXPECTED_NUMERIC = [
    "loan_amnt",
    "int_rate",
    "annual_inc",
    "dti",
    "revol_util",
    "open_acc",
    "total_acc",
]

EXPECTED_CATEGORICAL = [
    "term",
    "grade",
    "home_ownership",
    "purpose",
    "verification_status",
]

TARGET_COLUMN = "loan_status"


def load_and_prepare_data(path: str) -> pd.DataFrame:
    print(f"📥 Loading data from: {path}")
    df = pd.read_csv(path, low_memory=False)

    print(f"✅ Loaded shape: {df.shape}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"'{TARGET_COLUMN}' column not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Make binary target: 1 = default / bad, 0 = fully paid / good
    df = df.copy()
    bad_statuses = [
        "Charged Off",
        "Default",
        "Does not meet the credit policy. Status:Charged Off",
        "Late (31-120 days)",
        "Late (16-30 days)",
    ]
    df["defaulted"] = df[TARGET_COLUMN].isin(bad_statuses).astype(int)

    # Pick the feature columns that actually exist in this CSV
    numeric_features = [c for c in EXPECTED_NUMERIC if c in df.columns]
    categorical_features = [c for c in EXPECTED_CATEGORICAL if c in df.columns]

    if not numeric_features and not categorical_features:
        raise ValueError(
            "None of the expected feature columns were found.\n"
            f"Expected numeric: {EXPECTED_NUMERIC}\n"
            f"Expected categorical: {EXPECTED_CATEGORICAL}\n"
            f"Actual: {list(df.columns)}"
        )

    used_cols = numeric_features + categorical_features + ["defaulted"]
    df = df[used_cols].dropna()

    print(f"🧹 After selecting columns & dropping NAs: {df.shape}")
    return df, numeric_features, categorical_features


def train_model(df: pd.DataFrame, numeric_features, categorical_features):
    X = df.drop(columns=["defaulted"])
    y = df["defaulted"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )

    print("✂️ Splitting train / test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🚂 Training model...")
    model.fit(X_train, y_train)

    print("📊 Evaluation on test set:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    return model


def main(data_path: str, output_path: str):
    df, num_feats, cat_feats = load_and_prepare_data(data_path)
    model = train_model(df, num_feats, cat_feats)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        "model": model,
        "numeric_features": num_feats,
        "categorical_features": cat_feats,
    }
    joblib.dump(payload, output_path)
    print(f"💾 Saved model artifact to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/loan.csv",
        help="Path to the LendingClub CSV file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="artifacts/credit_default_model.joblib",
        help="Where to save the trained model",
    )
    args = parser.parse_args()
    main(args.data_path, args.output_path)
