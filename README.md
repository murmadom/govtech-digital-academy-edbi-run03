# POI Data Pipeline

An end-to-end data engineering pipeline built on **Databricks** and **Apache Spark**, implementing the **medallion architecture** (Bronze → Silver → Gold) to ingest, clean, enrich, and serve Person-of-Interest (POI) data for analytical consumption.

## Architecture

```
S3 (CSV)
  │
  ▼
┌─────────────┐    ┌─────────────┐    ┌──────────────────────────┐
│   BRONZE    │───▶│   SILVER    │───▶│          GOLD            │
│  Raw append │    │ Deduplicated│    │  Star schema (Delta)     │
│  (audit)    │    │  QA-passed  │    │  ┌─ fact_poi_enriched    │
└─────────────┘    └─────────────┘    │  ├─ dim_poi              │
                                      │  ├─ dim_employer         │
                                      │  └─ dim_industry         │
                                      └──────────────────────────┘
```

## Data Sources

| Dataset | Records | Description |
| --- | --- | --- |
| `poi` | 10,000 | Person-of-Interest profiles (demographics, occupation, employer linkage) |
| `employer` | 2,700 | Employer registry (entity details, address, SSIC codes) |
| `industry` | 988 | SSIC industry classification hierarchy (section → sub-class) |
| `case` | — | Case records (ingested to bronze) |

## Project Structure

```
poi-data-pipeline/
├── README.md
├── notebooks/
│   ├── 01_data_ingestion.ipynb       # S3 → Bronze (raw CSV ingestion)
│   ├── 02_eda_qa.ipynb               # Bronze → Silver (EDA + QA rules)
│   ├── 03_data_transformation.ipynb  # Silver → Gold (enrichment + star schema)
│   └── 04_incremental_load.ipynb     # Incremental MERGE pipeline
├── utils/
│   └── poi_utils.py                  # Shared functions (clean, QA, enrich)
├── sql/
│   └── physical_model.sql            # Gold layer DDL / physical model
└── data/
    └── .gitkeep                      # Placeholder (no raw data committed)
```

## Pipeline Stages

### 1. Data Ingestion (`01_data_ingestion`)
- Loads CSV files from **AWS S3** (`s3://edbi-03/s3_teamG02/`)
- Writes raw data as **Delta tables** to `edbi_teamg02.bronze` with column mapping enabled
- Covers four datasets: `poi`, `employer`, `industry`, `case`

### 2. EDA & Quality Assurance (`02_eda_qa`)
- **Exploratory analysis**: distributions, missing values, cardinality, cross-variable relationships
- **QA rules** applied via `poi_utils.qa_poi()`:
  - Standardise string nulls (`null`, `nil`, `none`, `n/a`, etc.) → `NaN`
  - Remove rows with null NRIC (primary key)
  - Rename ambiguous column (`Integer` → `Year of Profile`)
  - Deduplicate by NRIC, keeping the latest `Year of Profile`
  - Validate DOB (birth year ≤ profile year)
- Result: 10,000 raw → **7,242 clean rows** written to `edbi_teamg02.silver`

### 3. Data Transformation (`03_data_transformation`)
- **Two-stage left join**: POI → Employer (on `uen_identifier`) → Industry (on `primary_ssic_code`)
- **Feature engineering**:

| Feature | Technique | Source |
| --- | --- | --- |
| `age` | Date arithmetic relative to `year_of_profile` | `dob` |
| `company_age` | Date arithmetic | `registration_incorporation_date` |
| `age_group` | Binning into 7 brackets (<18 to 65+) | `age` |
| `company_age_group` | Binning into 6 lifecycle stages | `company_age` |
| `poi_per_employer` | Group-level aggregation | `uen_identifier`, `nric` |
| `postal_district` | Mapping to 28 Singapore districts (D01–D28) | `postal_code` |
| `full_address` | Vectorised multi-field concatenation | address fields |
| `nationality_region` | \~200 nationalities → 13 regions | `nationality` |
| `occupation_sector` | 41 SSOC titles → 9 ISCO sectors | `occupation` |
| `occupation_skill_level` | Ordinal 1–4 skill tier | `occupation_sector` |

- **Boolean flags**: `has_employer`, `has_industry`, `is_unemployed`
- **Post-transformation validation**: 8 automated checks (row count, PK uniqueness, range checks, flag consistency)
- Output: **star schema** in `edbi_teamg02.gold` (1 fact + 3 dimension tables, partitioned Delta)

### 4. Incremental Load (`04_incremental_load`)
- **Dual-mode execution**: detects presence of `delta.csv` in S3
  - If found → full incremental pipeline (QA → Bronze append → Silver MERGE → Gold MERGE)
  - If not found → loads existing gold tables directly
- **MERGE logic**: upsert by NRIC; updates if new `Year of Profile` ≥ existing
- **Idempotent**: safe to re-run without side effects

## Gold Layer Schema

| Table | Records | Columns | Partition Key |
| --- | --- | --- | --- |
| `fact_poi_enriched` | 7,242 | 51 | `year_of_profile` |
| `dim_poi` | 7,242 | 19 | `year_of_profile` |
| `dim_employer` | 2,286 | 22 | `entity_status_description` |
| `dim_industry` | 988 | 11 | `section_code` |

## Shared Utilities (`poi_utils.py`)

| Function | Purpose |
| --- | --- |
| `clean_nulls(df)` | Replace string null equivalents with `NaN` across all object columns |
| `qa_poi(df)` | Apply full QA pipeline (null removal, dedup, DOB validation) |
| `enrich_poi(df_poi, df_employer, df_industry)` | Join, engineer features, add flags |
| `prepare_for_spark(df)` | Type conversion for pandas → Spark DataFrame compatibility |

## Tech Stack

- **Platform**: Databricks (Unity Catalog)
- **Compute**: Apache Spark + pandas (hybrid)
- **Storage**: Delta Lake on AWS S3
- **Languages**: Python, SQL
- **Key Libraries**: PySpark, pandas, NumPy, Matplotlib

## Key Design Decisions

- **Medallion architecture** ensures data lineage and audit trail (raw bronze preserved)
- **Left joins** preserve all POIs, even those without employer linkage (27.7% unlinked)
- **Vectorised pandas operations** for performance — no row-by-row iteration
- **Reusable utility module** (`poi_utils.py`) shared across all notebooks
- **Delta column mapping** (`delta.columnMapping.mode = name`) allows flexible schema evolution
- **Partitioned tables** enable predicate pushdown on common query filters
