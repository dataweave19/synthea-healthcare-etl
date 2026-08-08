
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# Synthea Healthcare ETL — Readmission Dashboard
# ============================================================

st.set_page_config(
    page_title="Synthea Healthcare Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "synthea"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "ml" / "artifacts"

# ---------- Known results from the completed pipeline ----------
# These are the results reported by the user's completed run.
DEFAULT_RESULTS = {
    "encounters": 1476,
    "readmissions": 122,
    "readmission_rate": 0.083,
    "logistic": {"roc_auc": 0.783, "pr_auc": 0.238},
    "xgboost": {"roc_auc": 0.826, "pr_auc": 0.333},
}

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #666;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        padding: 0.8rem 1rem;
        border-radius: 0.7rem;
        border: 1px solid rgba(128,128,128,.2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Helpers ----------
@st.cache_data
def load_csv(path):
    return pd.read_csv(path, low_memory=False)


def find_csv(name):
    candidates = [
        DATA_DIR / name,
        PROCESSED_DIR / name,
        ROOT / name,
    ]
    for p in candidates:
        if p.exists():
            return p

    # Search recursively for a matching filename.
    matches = list(ROOT.rglob(name))
    return matches[0] if matches else None


def load_first_available(names):
    for name in names:
        path = find_csv(name)
        if path:
            try:
                return load_csv(path), path
            except Exception:
                pass
    return None, None


def find_metrics():
    if not ARTIFACT_DIR.exists():
        return None

    names = [
        "metrics.json",
        "model_metrics.json",
        "evaluation_metrics.json",
        "results.json",
    ]
    for name in names:
        p = ARTIFACT_DIR / name
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
    return None


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


# ---------- Sidebar ----------
st.sidebar.title("🏥 Dashboard Controls")
st.sidebar.markdown("**Synthea Healthcare ETL**")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Model Performance",
        "Explainability",
        "Fairness",
        "Data Explorer",
    ],
)

st.sidebar.divider()
st.sidebar.caption(
    "Data are synthetically generated with Synthea. "
    "They should not be interpreted as real patient outcomes."
)

# ---------- Load data ----------
patients, patients_path = load_first_available(["patients.csv"])
encounters, encounters_path = load_first_available(["encounters.csv"])
observations, observations_path = load_first_available(["observations.csv"])
conditions, conditions_path = load_first_available(["conditions.csv"])

metrics_file = find_metrics()

# Use known pipeline results unless a compatible metrics file is available.
results = DEFAULT_RESULTS.copy()

if isinstance(metrics_file, dict):
    # Flexible support for common JSON layouts.
    results["encounters"] = metrics_file.get("encounters", results["encounters"])
    results["readmissions"] = metrics_file.get(
        "readmissions", results["readmissions"]
    )
    results["readmission_rate"] = metrics_file.get(
        "readmission_rate", results["readmission_rate"]
    )

    for model_key in ["logistic", "xgboost"]:
        if model_key in metrics_file and isinstance(metrics_file[model_key], dict):
            results[model_key].update(metrics_file[model_key])

# ============================================================
# Header
# ============================================================
st.markdown(
    '<div class="main-title">Synthea Healthcare Readmission Dashboard</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">ETL → Machine Learning → Explainability → Fairness</div>',
    unsafe_allow_html=True,
)

# ============================================================
# Overview
# ============================================================
if page == "Overview":

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Encounters", f"{results['encounters']:,}")
    c2.metric("30-Day Readmissions", f"{results['readmissions']:,}")
    c3.metric(
        "Readmission Rate",
        f"{results['readmission_rate'] * 100:.1f}%",
    )
    c4.metric(
        "Best ROC-AUC",
        f"{results['xgboost']['roc_auc']:.3f}",
        "XGBoost",
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Model Performance")

        model_df = pd.DataFrame(
            {
                "Model": ["Logistic Regression", "XGBoost"],
                "ROC-AUC": [
                    results["logistic"]["roc_auc"],
                    results["xgboost"]["roc_auc"],
                ],
                "PR-AUC": [
                    results["logistic"]["pr_auc"],
                    results["xgboost"]["pr_auc"],
                ],
            }
        )

        fig = px.bar(
            model_df,
            x="Model",
            y=["ROC-AUC", "PR-AUC"],
            barmode="group",
            range_y=[0, 1],
            title="Model comparison",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Readmission Distribution")

        outcome_df = pd.DataFrame(
            {
                "Outcome": ["No readmission", "30-day readmission"],
                "Patients": [
                    results["encounters"] - results["readmissions"],
                    results["readmissions"],
                ],
            }
        )

        fig = px.pie(
            outcome_df,
            names="Outcome",
            values="Patients",
            hole=0.45,
            title="Encounter outcome",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "The current completed pipeline contains 1,476 encounters and 122 "
        "30-day readmissions (8.3%). XGBoost is the stronger baseline model "
        "with ROC-AUC 0.826 and PR-AUC 0.333."
    )

# ============================================================
# Model Performance
# ============================================================
elif page == "Model Performance":

    st.header("Model Performance")

    model_df = pd.DataFrame(
        {
            "Model": ["Logistic Regression", "XGBoost"],
            "ROC-AUC": [
                results["logistic"]["roc_auc"],
                results["xgboost"]["roc_auc"],
            ],
            "PR-AUC": [
                results["logistic"]["pr_auc"],
                results["xgboost"]["pr_auc"],
            ],
        }
    )

    st.dataframe(
        model_df.style.format(
            {"ROC-AUC": "{:.3f}", "PR-AUC": "{:.3f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        fig = px.bar(
            model_df,
            x="Model",
            y="ROC-AUC",
            range_y=[0, 1],
            title="ROC-AUC",
            text_auto=".3f",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.bar(
            model_df,
            x="Model",
            y="PR-AUC",
            range_y=[0, 1],
            title="Precision-Recall AUC",
            text_auto=".3f",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Interpretation")

    improvement_roc = (
        results["xgboost"]["roc_auc"] - results["logistic"]["roc_auc"]
    )
    improvement_pr = (
        results["xgboost"]["pr_auc"] - results["logistic"]["pr_auc"]
    )

    st.markdown(
        f"""
        **XGBoost improves over Logistic Regression by:**

        - ROC-AUC: **{improvement_roc:+.3f}**
        - PR-AUC: **{improvement_pr:+.3f}**

        Because readmission is a relatively uncommon outcome, PR-AUC is
        particularly useful alongside ROC-AUC when evaluating the model.
        """
    )

    # If an ROC/PR CSV exists, use it.
    curves, curves_path = load_first_available(
        ["roc_pr_curves.csv", "model_curves.csv", "evaluation_curves.csv"]
    )

    if curves is not None:
        st.subheader("Stored Evaluation Curves")
        st.dataframe(curves.head(100), use_container_width=True)

# ============================================================
# Explainability
# ============================================================
elif page == "Explainability":

    st.header("Model Explainability")

    st.markdown(
        """
        This section is designed for the feature-importance and SHAP results
        generated by the ML pipeline. If an exported feature-importance or SHAP
        CSV exists in `ml/artifacts/`, the dashboard will visualize it.
        """
    )

    importance, importance_path = load_first_available(
        [
            "feature_importance.csv",
            "xgboost_feature_importance.csv",
            "shap_importance.csv",
            "shap_values.csv",
        ]
    )

    if importance is not None:
        st.caption(f"Loaded: `{importance_path.relative_to(ROOT)}`")

        # Try to identify feature and importance columns.
        feature_col = next(
            (
                c
                for c in importance.columns
                if c.lower() in ["feature", "features", "name"]
            ),
            None,
        )

        value_col = next(
            (
                c
                for c in importance.columns
                if any(
                    key in c.lower()
                    for key in ["importance", "mean_abs_shap", "shap"]
                )
                and c != feature_col
            ),
            None,
        )

        if feature_col and value_col:
            plot_df = importance[[feature_col, value_col]].copy()
            plot_df[value_col] = safe_numeric(plot_df[value_col])
            plot_df = plot_df.dropna().sort_values(value_col).tail(20)

            fig = px.bar(
                plot_df,
                x=value_col,
                y=feature_col,
                orientation="h",
                title="Top feature contributions",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(importance.head(100), use_container_width=True)

    else:
        st.warning(
            "No feature-importance/SHAP CSV was found yet. "
            "Run the SHAP analysis and export the results to "
            "`ml/artifacts/feature_importance.csv` or "
            "`ml/artifacts/shap_importance.csv`."
        )

# ============================================================
# Fairness
# ============================================================
elif page == "Fairness":

    st.header("Fairness Audit")

    fairness, fairness_path = load_first_available(
        [
            "fairness.csv",
            "fairness_metrics.csv",
            "fairness_audit.csv",
            "group_metrics.csv",
        ]
    )

    if fairness is None:
        st.warning(
            "No fairness CSV was found. Export the fairness audit from the "
            "ML pipeline to `ml/artifacts/fairness_metrics.csv`."
        )

        st.subheader("Known audit context")
        st.info(
            "The completed audit evaluated model behavior across gender, "
            "race, and age groups. Some groups had small sample sizes, so "
            "subgroup differences should be interpreted cautiously."
        )

    else:
        st.caption(f"Loaded: `{fairness_path.relative_to(ROOT)}`")
        st.dataframe(fairness, use_container_width=True, hide_index=True)

        # Detect a grouping column.
        group_col = next(
            (
                c
                for c in fairness.columns
                if c.lower()
                in ["group", "gender", "race", "age_group", "age"]
            ),
            None,
        )

        metric_options = [
            c
            for c in fairness.columns
            if any(
                key in c.lower()
                for key in [
                    "fnr",
                    "tpr",
                    "fpr",
                    "precision",
                    "recall",
                    "rate",
                ]
            )
        ]

        if group_col and metric_options:
            selected_metric = st.selectbox(
                "Fairness metric",
                metric_options,
            )

            plot_df = fairness.copy()
            plot_df[selected_metric] = safe_numeric(
                plot_df[selected_metric]
            )

            fig = px.bar(
                plot_df,
                x=group_col,
                y=selected_metric,
                title=f"{selected_metric} by subgroup",
                text_auto=".3f",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.warning(
            "Fairness results describe this synthetic dataset and model. "
            "They should not be interpreted as evidence about real-world "
            "healthcare disparities."
        )

# ============================================================
# Data Explorer
# ============================================================
elif page == "Data Explorer":

    st.header("Data Explorer")

    available = {}
    for filename in [
        "patients.csv",
        "encounters.csv",
        "conditions.csv",
        "observations.csv",
        "medications.csv",
        "procedures.csv",
        "claims.csv",
    ]:
        path = find_csv(filename)
        if path:
            available[filename] = path

    if not available:
        st.error(
            "No CSV files found. Expected them under `data/synthea/`."
        )
    else:
        selected = st.selectbox(
            "Select dataset",
            list(available.keys()),
        )

        df = load_csv(available[selected])

        st.write(
            f"**Rows:** {len(df):,}  |  **Columns:** {len(df.columns):,}"
        )

        search = st.text_input(
            "Filter rows by text (optional)",
            placeholder="e.g. diabetes, female, inpatient",
        )

        display_df = df

        if search:
            mask = display_df.astype(str).apply(
                lambda col: col.str.contains(
                    search, case=False, na=False
                )
            ).any(axis=1)
            display_df = display_df[mask]

        st.dataframe(
            display_df.head(500),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download displayed rows",
            display_df.head(500).to_csv(index=False),
            file_name=f"{selected.replace('.csv', '')}_preview.csv",
            mime="text/csv",
        )

# ---------- Footer ----------
st.divider()
st.caption(
    "Synthea Healthcare ETL Dashboard • Synthetic data • "
    "For research and demonstration purposes only"
)
