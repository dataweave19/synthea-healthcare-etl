"""
Bias / fairness audit for the readmission model.

Checks whether the model's error rates differ meaningfully across
demographic groups — race, gender, age band. In healthcare ML, a model
that is accurate on average but systematically worse for one group is
a real deployment risk, not a footnote.

Usage:
    python ml/evaluate.py --artifacts-dir ml/artifacts
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def age_band(age):
    if age < 18:
        return "0-17"
    if age < 40:
        return "18-39"
    if age < 65:
        return "40-64"
    return "65+"


def audit_group(df, group_col, proba_col="proba", label_col="readmitted_30d", threshold=0.5):
    rows = []
    for group, g in df.groupby(group_col):
        if len(g) < 10:
            # too few samples for a meaningful rate — flag rather than report
            rows.append(
                {
                    group_col: group,
                    "n": len(g),
                    "note": "sample too small (<10) for a reliable estimate",
                }
            )
            continue
        y_true = g[label_col]
        y_proba = g[proba_col]
        y_pred = (y_proba >= threshold).astype(int)

        auc = roc_auc_score(y_true, y_proba) if y_true.nunique() > 1 else float("nan")
        fnr = ((y_pred == 0) & (y_true == 1)).sum() / max((y_true == 1).sum(), 1)  # missed high-risk patients
        fpr = ((y_pred == 1) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1)  # false alarms
        rows.append(
            {
                group_col: group,
                "n": len(g),
                "positive_rate": round(y_true.mean(), 3),
                "roc_auc": round(auc, 3) if not np.isnan(auc) else None,
                "false_negative_rate": round(fnr, 3),
                "false_positive_rate": round(fpr, 3),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", default="ml/artifacts")
    args = parser.parse_args()

    art_dir = Path(args.artifacts_dir)
    test_df = pd.read_csv(art_dir / "test_set.csv")
    proba = np.load(art_dir / "xgb_test_proba.npy")
    test_df["proba"] = proba
    test_df["age_band"] = test_df["age_at_admit"].apply(age_band)

    print("=" * 60)
    print("BIAS / FAIRNESS AUDIT — XGBoost model")
    print("=" * 60)
    print(
        "\nfalse_negative_rate = share of ACTUAL readmissions the model missed"
        "\n  -> highest-stakes error in this context: a missed high-risk patient"
        "\n     gets no follow-up care."
        "\nfalse_positive_rate = share of non-readmissions flagged as high-risk"
        "\n  -> lower-stakes: means unnecessary follow-up resources spent."
    )

    for group_col in ["race", "gender", "age_band"]:
        print(f"\n--- By {group_col} ---")
        result = audit_group(test_df, group_col)
        print(result.to_string(index=False))

    print(
        "\nNote: this is a SYNTHETIC dataset from Synthea, generated with "
        "population statistics roughly modeled on Massachusetts census data. "
        "Any disparities found here reflect quirks of the synthetic generator "
        "and small subgroup sample sizes, NOT real clinical bias. The point of "
        "this audit is to demonstrate the *practice* — the same code and "
        "reasoning would apply to a real deployment, where such findings would "
        "need serious investigation before shipping."
    )


if __name__ == "__main__":
    main()
