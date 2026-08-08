# Hospital Readmission Risk & Ops Intelligence Platform

Predicting 30-day patient readmission risk from hospital encounter data, with a
focus on the parts most portfolio projects skip: demographic bias auditing,
data governance documentation, and a path to production (RAG assistant + AWS
deployment).

Built on synthetic patient data — no real patient information is used anywhere
in this project.

## Problem

Unplanned 30-day readmissions are one of the most closely watched quality and
cost metrics in healthcare — <cite index="46-1">HCUP estimates unplanned 30-day readmissions cost the United States $41.3 billion, with roughly 18% of Medicare patients readmitted within 30 days of discharge</cite>. Hospitals that can flag high-risk patients *before* discharge can target
follow-up care and avoid both the human and financial cost of a preventable
readmission.

This project builds that flagging system end-to-end: from raw hospital data to
a risk model to (eventually) a deployed, explainable API a care team could
actually use.

## Data

Synthetic patient data generated with [Synthea](https://github.com/synthetichealth/synthea)
(MITRE, Apache-2.0) — realistic but entirely fake patient histories, so there
are zero privacy or compliance restrictions on this dataset. See `NOTICE` for
attribution.

Current dataset: 556 patients, ~28,700 encounters, spanning conditions,
medications, procedures, observations, and claims.

| Table | Rows | Description |
|---|---|---|
| `patients.csv` | 556 | demographics |
| `encounters.csv` | 28,665 | all visits (wellness, ambulatory, emergency, inpatient, etc.) |
| `conditions.csv` | 18,658 | diagnoses |
| `medications.csv` | 18,703 | prescriptions |
| `observations.csv` | 341,265 | vitals and labs |
| `procedures.csv` | — | procedures performed |
| `claims.csv`, `payers.csv` | — | billing/insurance data |

## Architecture

```
data/synthea (raw CSVs)
        │
        ▼
   etl/build_readmission_dataset.py   (DuckDB — encounter-level joins + labeling)
        │
        ▼
data/processed/readmission_dataset.csv
        │
        ├──► notebooks/  (EDA)
        │
        └──► ml/          (model training + SHAP explainability + bias audit)
                │
                ▼
        [ planned ] API + Streamlit dashboard
        [ planned ] RAG clinical assistant (Bedrock + OpenSearch/pgvector)
        [ planned ] AWS deployment (Glue, SageMaker, Bedrock)
```

sql/schema.sql defines the full Postgres schema if you want to load the data
into a real database rather than querying the CSVs directly (etl/load_to_postgres.py
handles the load).

## Results

*(to be filled in once the baseline model is trained — see `ml/`)*

Current label distribution: **8.3% of inpatient/emergency encounters** result
in another inpatient/emergency admission within 30 days (122 of 1,476
qualifying encounters) — consistent with published readmission rates, which
<cite index="48-1">typically run around 20% within 28-31 days across studies of medical-condition patients</cite>, though this synthetic cohort skews lower.

## How to run it

```bash
git clone https://github.com/dataweave19/synthea-healthcare-etl.git
cd synthea-healthcare-etl
pip install -r requirements.txt

# Build the model-ready dataset (no database required, runs on DuckDB in-memory)
python etl/build_readmission_dataset.py --data-dir data/synthea --out data/processed/readmission_dataset.csv
```

To load into Postgres instead of querying CSVs directly:

```bash
psql $DATABASE_URL -f sql/schema.sql
python etl/load_to_postgres.py --data-dir data/synthea
```

## Responsible AI / governance

Even on synthetic data, this project follows the practices a real clinical
deployment would require:
- **Bias audit** across race, gender, and age groups on model error rates (see `ml/evaluate.py`)
- **Explainability** via SHAP on every prediction, not just aggregate accuracy
- **Model card** documenting intended use and limitations (see `MODEL_CARD.md`)
- **Data governance plan** for what a real deployment would need — access control, audit logging, encryption (see `DATA_GOVERNANCE.md`)

## Roadmap

- [x] ETL pipeline: raw Synthea CSVs → model-ready encounter-level dataset
- [ ] Baseline model (logistic regression) + XGBoost, with SHAP explainability
- [ ] Bias/fairness audit across demographic groups
- [ ] Streamlit dashboard for readmission risk + hospital ops view
- [ ] RAG clinical assistant over synthetic clinical notes / protocol documents
- [ ] AWS deployment (S3, Glue, SageMaker, Bedrock)

## Attribution

Patient data generated using [Synthea](https://github.com/synthetichealth/synthea),
© The MITRE Corporation, licensed under Apache-2.0. See `NOTICE`.
