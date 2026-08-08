"""
Train baseline models for 30-day readmission prediction.

Trains:
    1. Logistic Regression — interpretable baseline
    2. XGBoost — stronger baseline, feeds the SHAP explainability step

Usage:
    python ml/train.py --data data/processed/readmission_dataset.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb
import joblib

NUMERIC_FEATURES = [
    "length_of_stay_days",
    "total_claim_cost",
    "age_at_admit",
    "income",
    "encounters_prior_year",
    "n_conditions",
    "n_medications",
    "n_procedures",
]
CATEGORICAL_FEATURES = ["gender", "race", "ethnicity", "marital_status", "encounter_class"]
TARGET = "readmitted_30d"


def build_preprocessor():
    numeric = Pipeline([("scale", StandardScaler())])
    categorical = Pipeline(
        [("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def evaluate(name, y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    print(f"\n--- {name} ---")
    print(f"ROC-AUC: {auc:.3f}")
    print(f"Average Precision (PR-AUC): {ap:.3f}  (baseline = positive rate = {y_true.mean():.3f})")
    print(classification_report(y_true, y_pred, digits=3))
    return {"model": name, "roc_auc": auc, "pr_auc": ap}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/readmission_dataset.csv")
    parser.add_argument("--out-dir", default="ml/artifacts")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    df["marital_status"] = df["marital_status"].fillna("unknown")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    # Stratified split: critical given the 8.3% positive rate — a random split
    # could easily starve the test set of positive cases otherwise.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} ({y_train.mean():.1%} positive)")
    print(f"Test:  {len(X_test)} ({y_test.mean():.1%} positive)")

    results = []

    # --- Logistic Regression baseline ---
    logreg = Pipeline(
        [
            ("prep", build_preprocessor()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    logreg.fit(X_train, y_train)
    proba = logreg.predict_proba(X_test)[:, 1]
    results.append(evaluate("Logistic Regression", y_test, proba))
    joblib.dump(logreg, out_dir / "logreg_pipeline.joblib")

    # --- XGBoost ---
    prep = build_preprocessor()
    X_train_enc = prep.fit_transform(X_train)
    X_test_enc = prep.transform(X_test)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
    )
    xgb_model.fit(X_train_enc, y_train)
    proba_xgb = xgb_model.predict_proba(X_test_enc)[:, 1]
    results.append(evaluate("XGBoost", y_test, proba_xgb))

    joblib.dump(xgb_model, out_dir / "xgb_model.joblib")
    joblib.dump(prep, out_dir / "preprocessor.joblib")

    # Save test set for the evaluate.py / bias-audit step
    X_test.assign(readmitted_30d=y_test.values).to_csv(out_dir / "test_set.csv", index=False)
    np.save(out_dir / "xgb_test_proba.npy", proba_xgb)

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nArtifacts written to {out_dir}/")


if __name__ == "__main__":
    main()
