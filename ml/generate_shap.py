from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "ml" / "artifacts"

MODEL_PATH = ARTIFACTS / "xgb_model.joblib"
PREPROCESSOR_PATH = ARTIFACTS / "preprocessor.joblib"
TEST_PATH = ARTIFACTS / "test_set.csv"

OUTPUT_IMPORTANCE = ARTIFACTS / "feature_importance.csv"
OUTPUT_SHAP = ARTIFACTS / "shap_importance.csv"

print("Loading model...")
model = joblib.load(MODEL_PATH)

print("Loading preprocessor...")
preprocessor = joblib.load(PREPROCESSOR_PATH)

print("Loading test data...")
test_df = pd.read_csv(TEST_PATH)

print("Test shape:", test_df.shape)

# Try to identify the target column.
target_candidates = [
    "readmitted",
    "readmission",
    "target",
    "y",
]

target_col = None

for col in target_candidates:
    if col in test_df.columns:
        target_col = col
        break

if target_col is not None:
    X_test = test_df.drop(columns=[target_col])
else:
    # If the saved test set contains only features, use it directly.
    X_test = test_df

print("Feature shape:", X_test.shape)

# Transform using the same preprocessing pipeline used during training.
X_transformed = preprocessor.transform(X_test)

# Convert sparse matrices if necessary.
if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

# Get transformed feature names.
try:
    feature_names = preprocessor.get_feature_names_out()
except Exception:
    feature_names = [
        f"feature_{i}" for i in range(X_transformed.shape[1])
    ]

feature_names = np.asarray(feature_names)

print("Transformed shape:", X_transformed.shape)
print("Number of feature names:", len(feature_names))

# ------------------------------------------------------------
# XGBoost feature importance
# ------------------------------------------------------------

if hasattr(model, "feature_importances_"):
    importance = model.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names[: len(importance)],
            "importance": importance,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    importance_df.to_csv(
        OUTPUT_IMPORTANCE,
        index=False,
    )

    print(
        f"Saved feature importance to: {OUTPUT_IMPORTANCE}"
    )

# ------------------------------------------------------------
# SHAP
# ------------------------------------------------------------

print("Calculating SHAP values...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_transformed)

# For binary classification, SHAP can return a list.
if isinstance(shap_values, list):
    shap_values = shap_values[-1]

mean_abs_shap = np.abs(shap_values).mean(axis=0)

shap_df = pd.DataFrame(
    {
        "feature": feature_names[: len(mean_abs_shap)],
        "mean_abs_shap": mean_abs_shap,
    }
).sort_values(
    "mean_abs_shap",
    ascending=False,
)

shap_df.to_csv(
    OUTPUT_SHAP,
    index=False,
)

print(
    f"Saved SHAP importance to: {OUTPUT_SHAP}"
)

print("\nTop 20 features:")
print(shap_df.head(20).to_string(index=False))

print("\nDone.")

