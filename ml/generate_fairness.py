from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "ml" / "artifacts"

MODEL_PATH = ARTIFACTS / "xgb_model.joblib"
PREPROCESSOR_PATH = ARTIFACTS / "preprocessor.joblib"
TEST_PATH = ARTIFACTS / "test_set.csv"

OUTPUT_PATH = ARTIFACTS / "fairness_metrics.csv"

print("Loading model...")
model = joblib.load(MODEL_PATH)

print("Loading preprocessor...")
preprocessor = joblib.load(PREPROCESSOR_PATH)

print("Loading test data...")
df = pd.read_csv(TEST_PATH)

print("Test data shape:", df.shape)
print("Columns:")
print(df.columns.tolist())


# ------------------------------------------------------------
# Identify target column
# ------------------------------------------------------------

target_candidates = [
    "readmitted_30d",
    "readmitted",
    "readmission",
    "readmission_30d",
    "readmission_30_day",
    "target",
    "y",
]

target_col = None

for col in target_candidates:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    raise ValueError(
        "Could not find the readmission target column. "
        "Check the columns printed above."
    )

print("Target column:", target_col)


# ------------------------------------------------------------
# Identify demographic columns
# ------------------------------------------------------------

def find_column(candidates):
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


gender_col = find_column([
    "gender",
    "Gender",
    "sex",
    "Sex",
])

race_col = find_column([
    "race",
    "Race",
])

age_col = find_column([
    "age",
    "Age",
    "age_years",
    "age_group",
])


print("Gender column:", gender_col)
print("Race column:", race_col)
print("Age column:", age_col)


# ------------------------------------------------------------
# Prepare features
# ------------------------------------------------------------

X = df.drop(columns=[target_col])
y = pd.to_numeric(df[target_col], errors="coerce")

valid = y.notna()

X = X.loc[valid].copy()
y = y.loc[valid].astype(int)

print("Valid samples:", len(y))


# ------------------------------------------------------------
# Generate predictions
# ------------------------------------------------------------

print("Transforming test data...")

X_transformed = preprocessor.transform(X)

if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

print("Generating predictions...")

y_pred = model.predict(X_transformed)

y_pred = np.asarray(y_pred).astype(int)


# ------------------------------------------------------------
# Create dataframe containing predictions
# ------------------------------------------------------------

analysis_df = df.loc[valid].copy()

analysis_df["_y_true"] = y.values
analysis_df["_y_pred"] = y_pred


# ------------------------------------------------------------
# Age groups
# ------------------------------------------------------------

if age_col is not None:

    age_numeric = pd.to_numeric(
        analysis_df[age_col],
        errors="coerce",
    )

    if age_numeric.notna().sum() > 0:

        analysis_df["_age_group"] = pd.cut(
            age_numeric,
            bins=[0, 18, 35, 50, 65, 200],
            labels=[
                "<18",
                "18-34",
                "35-49",
                "50-64",
                "65+",
            ],
            right=False,
        )

        age_col_for_analysis = "_age_group"

    else:
        age_col_for_analysis = age_col

else:
    age_col_for_analysis = None


# ------------------------------------------------------------
# Fairness metric calculation
# ------------------------------------------------------------

def calculate_metrics(group, group_name, group_value):

    y_true = group["_y_true"]
    y_pred = group["_y_pred"]

    n = len(group)

    positives = int(y_true.sum())
    predicted_positive = int(y_pred.sum())

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    tnr = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else np.nan
    )

    accuracy = (
        (tp + tn) / n
        if n > 0
        else np.nan
    )

    readmission_rate = (
        positives / n
        if n > 0
        else np.nan
    )

    return {
        "dimension": group_name,
        "group": str(group_value),
        "n": n,
        "actual_readmissions": positives,
        "predicted_readmissions": predicted_positive,
        "readmission_rate": readmission_rate,
        "accuracy": accuracy,
        "precision": precision,
        "tpr": tpr,
        "fnr": fnr,
        "fpr": fpr,
        "tnr": tnr,
    }


results = []


# ------------------------------------------------------------
# Gender fairness
# ------------------------------------------------------------

if gender_col is not None:

    print("\nCalculating gender fairness...")

    for value, group in analysis_df.groupby(
        gender_col,
        dropna=True,
    ):

        results.append(
            calculate_metrics(
                group,
                "Gender",
                value,
            )
        )


# ------------------------------------------------------------
# Race fairness
# ------------------------------------------------------------

if race_col is not None:

    print("\nCalculating race fairness...")

    for value, group in analysis_df.groupby(
        race_col,
        dropna=True,
    ):

        results.append(
            calculate_metrics(
                group,
                "Race",
                value,
            )
        )


# ------------------------------------------------------------
# Age fairness
# ------------------------------------------------------------

if age_col_for_analysis is not None:

    print("\nCalculating age-group fairness...")

    for value, group in analysis_df.groupby(
        age_col_for_analysis,
        dropna=True,
        observed=True,
    ):

        results.append(
            calculate_metrics(
                group,
                "Age Group",
                value,
            )
        )


# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

if not results:
    raise ValueError(
        "No demographic columns were found. "
        "Expected gender, race, and/or age."
    )

fairness_df = pd.DataFrame(results)

fairness_df = fairness_df.sort_values(
    ["dimension", "group"]
)

fairness_df.to_csv(
    OUTPUT_PATH,
    index=False,
)

print("\n==========================================")
print("FAIRNESS AUDIT")
print("==========================================")

print(
    fairness_df.to_string(
        index=False
    )
)

print("\nSaved:")
print(OUTPUT_PATH)
