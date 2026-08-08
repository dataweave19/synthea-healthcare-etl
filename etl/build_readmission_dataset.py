"""
Build a model-ready, encounter-level dataset for 30-day readmission prediction
from Synthea data, either from raw CSVs (via DuckDB, no DB server needed) or
from a Postgres instance already loaded by load_to_postgres.py.

Output: data/processed/readmission_dataset.csv
    One row per inpatient/emergency encounter, with:
    - demographics (age at encounter, gender, race, marital status)
    - prior utilization (# encounters in last 12 months)
    - clinical load at this encounter (# conditions, # medications, # procedures)
    - cost fields
    - label: readmitted_30d (1/0)

Usage:
    python etl/build_readmission_dataset.py --data-dir data/raw
"""
import argparse
from pathlib import Path

import duckdb

QUERY = """
WITH admits AS (
    SELECT
        e.Id                AS encounter_id,
        e.PATIENT           AS patient_id,
        e.START              AS admit_time,
        e.STOP                AS discharge_time,
        e.ENCOUNTERCLASS       AS encounter_class,
        e.REASONDESCRIPTION      AS admit_reason,
        e.TOTAL_CLAIM_COST         AS total_claim_cost,
        e.PAYER_COVERAGE             AS payer_coverage
    FROM encounters e
    WHERE e.ENCOUNTERCLASS IN ('inpatient', 'emergency')
),
labeled AS (
    SELECT
        a.*,
        EXISTS (
            SELECT 1 FROM admits b
            WHERE b.patient_id = a.patient_id
              AND b.admit_time > a.discharge_time
              AND b.admit_time <= a.discharge_time + INTERVAL 30 DAY
        ) AS readmitted_30d
    FROM admits a
),
prior_utilization AS (
    SELECT
        l.encounter_id,
        (SELECT count(*) FROM encounters e2
         WHERE e2.PATIENT = l.patient_id
           AND e2.START < l.admit_time
           AND e2.START >= l.admit_time - INTERVAL 365 DAY) AS encounters_prior_year
    FROM labeled l
),
condition_counts AS (
    SELECT ENCOUNTER AS encounter_id, count(*) AS n_conditions
    FROM conditions GROUP BY 1
),
medication_counts AS (
    SELECT ENCOUNTER AS encounter_id, count(*) AS n_medications
    FROM medications GROUP BY 1
),
procedure_counts AS (
    SELECT ENCOUNTER AS encounter_id, count(*) AS n_procedures
    FROM procedures GROUP BY 1
)
SELECT
    l.encounter_id,
    l.patient_id,
    l.admit_time,
    l.discharge_time,
    date_diff('hour', l.admit_time, l.discharge_time) / 24.0 AS length_of_stay_days,
    l.encounter_class,
    l.admit_reason,
    l.total_claim_cost,
    l.payer_coverage,
    p.GENDER                                     AS gender,
    p.RACE                                          AS race,
    p.ETHNICITY                                        AS ethnicity,
    p.MARITAL                                             AS marital_status,
    date_diff('year', p.BIRTHDATE, l.admit_time)              AS age_at_admit,
    p.INCOME                                                     AS income,
    COALESCE(pu.encounters_prior_year, 0)                          AS encounters_prior_year,
    COALESCE(cc.n_conditions, 0)                                      AS n_conditions,
    COALESCE(mc.n_medications, 0)                                        AS n_medications,
    COALESCE(pc.n_procedures, 0)                                            AS n_procedures,
    l.readmitted_30d::int                                                     AS readmitted_30d
FROM labeled l
JOIN patients p ON p.Id = l.patient_id
LEFT JOIN prior_utilization pu ON pu.encounter_id = l.encounter_id
LEFT JOIN condition_counts cc ON cc.encounter_id = l.encounter_id
LEFT JOIN medication_counts mc ON mc.encounter_id = l.encounter_id
LEFT JOIN procedure_counts pc ON pc.encounter_id = l.encounter_id
ORDER BY l.admit_time
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw", help="Folder with Synthea CSVs")
    parser.add_argument("--out", default="data/processed/readmission_dataset.csv")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    for t in ["patients", "encounters", "conditions", "medications", "procedures"]:
        con.execute(f"CREATE TABLE {t} AS SELECT * FROM read_csv_auto('{data_dir}/{t}.csv')")

    df = con.execute(QUERY).fetchdf()
    df.to_csv(out_path, index=False)

    n = len(df)
    pos = df["readmitted_30d"].sum()
    print(f"Wrote {n:,} rows to {out_path}")
    print(f"Readmission rate: {pos}/{n} ({100*pos/n:.1f}%)")
    print(f"\nColumns: {list(df.columns)}")


if __name__ == "__main__":
    main()
