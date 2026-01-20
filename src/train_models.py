# src/train_models.py

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a credit risk default model on loan-level data."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to CSV file containing loan-level data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts",
        help="Directory to save the trained model.",
    )
    return parser.parse_args()


def create_default_label(df: pd.DataFrame) -> pd.Series:
    """
    Try to derive a binary default label.
    Assumes LendingClub-style 'loan_status' if present, otherwise
    expects a 'default' column already encoded as 0/1.
    """
    if "loan_status" in df.columns:
        status = df["loan_status"].astype(str)

        bad_statuses = [
            "Charged Off",
            "Default",
            "Late (31-120 days)",
            "Late (16-30 days)",
            "Does not meet the credit policy. Status:Charged Off",
        ]
        good_statuses = [
            "Fully Paid",
            "Current",
            "In Grace Period",
            "Does not meet the credit policy. Status:Fully Paid",
        ]

        y = status.apply(
            lambda s: 1
            if any(bad.lower() in s.lower() for bad in bad_statuses)
            else (0 if any(good.lower() in s.lower() for good in good_statuses) else np.nan)
        )
        y = y.dropna()
        # Align df to y index
        df = df.loc[y.index]
        return df, y.astype(int)

    elif "default" in df.columns:
        y = df["default"].astype(int)
        return df, y
    else:
        raise ValueError(
            "Could not find 'loan_status' or 'default' column. "
            "Please add one or adjust create_default_label()."
        )


def main():
    args = parse_args()

    # Load data
    df = pd.read_csv(args.data_path)

    # Define feature columns we care about (adjust to match your CSV)
    numeric_features = [
        "loan_amnt",
        "int_rate",
        "annual_inc",
        "dti",
        "revol_util",
        "open_acc",
        "total_acc",
    ]
    categorical_features = [
        "term",
        "grade",
        "home_ownership",
        "purpose",
        "verification_status",
    ]

    missing = [col for col in numeric_features + categorical_features if col not in df.columns]
    if missing:
        raise ValueError(
            f"Your data is missing these required columns: {missing}. "
            f"Either add them or edit train_models.py to match your dataset."
        )

    # Create target
    df, y = create_default_label(df)

    X = df[numeric_features + categorical_features].copy()

    # Preprocessing
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Classifier
    clf = GradientBoostingClassifier(random_state=42)

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"Validation ROC-AUC: {roc_auc:.3f}")

    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "credit_default_model.joblib")

    joblib.dump(
        {
            "model": model,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
        },
        model_path,
    )
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()

