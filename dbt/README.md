# dbt: Data Transformation Pipeline

This directory contains the **dbt** (data build tool) transformation models that create a **Star Schema** optimized for fake review detection analysis.

## What is dbt?

**dbt** is a SQL-based data transformation framework that:
- Transforms raw data into analytics-ready tables/views using pure SQL
- Manages data lineage and dependencies between models
- Documents data models and columns automatically
- Runs data quality tests (foreign keys, uniqueness, value ranges, etc.)
- Enables version control of transformations (like code, not SQL scripts)

**Why dbt for this project?**
Analytics transformations are complex SQL logic that should be:
- **Documented**: Every table, column has description and purpose
- **Tested**: Relationships, constraints, and business logic verified
- **Versioned**: Track changes to transformation logic over time
- **Reusable**: Models reference each other via `ref()`, not hardcoded table names
- **Auditable**: Data lineage shows which models depend on which sources

This demonstrates professional **Analytics Engineering** practices, central to modern data platforms.

## Star Schema Design

The dbt project creates a **Star Schema** for analytical queries:

```
                    ┌────────────────────┐
                    │   dim_restaurants  │
                    │  (Restaurant Info) │
                    │  restaurant_id (PK)│
                    └┬───────────────────┘
                     │
                     │ (FK)
       ┌─────────────┼────────────────┐
       │             │                │
       │    ┌────────▼──────────┐     │
       │    │   fact_reviews    │     │
       │    │   (All Reviews)   │     │
       │    │   review_id (PK)  │     │
       │    │   restaurant_id   │     │
       │    │   reviewer_user_id│     │
       │    │   rating_stars    │     │
       │    │   review_text     │     │
       │    │   ... measures    │     │
       │    └─────┬─────────────┘     │
       │          │ (FK)              │
       │    ┌─────▼──────────────┐    │
       └──→│ dim_reviewers      │    │
           │ (Reviewer Profile) │────┘
           │ reviewer_user_id(PK)│
           └────────────────────┘
```

### Model Layers

**1. Staging Models** (views)
- One-to-one transformations of raw data
- Deduplication and column normalization
- Add derived columns for analysis

Models:
- `stg_reviews`: All review records with derived features (detail_level, text_length, edit_delta_hours)
- `stg_reviewers`: Deduplicated reviewer profiles (one per reviewer_user_id)
- `stg_restaurants`: Deduplicated restaurant profiles (one per place_url)

**2. Mart Models** (tables)
- Optimized for specific analytical use cases
- Join-ready dimensions and facts
- PRIMARY KEYs and FOREIGN KEY constraints

Models:
- `dim_reviewers`: Reviewer dimension (PK: reviewer_user_id)
- `dim_restaurants`: Restaurant dimension (PK: restaurant_id / place_url)
- `fact_reviews`: Central fact table (PK: review_id, FK: reviewer_user_id, restaurant_id)

## Column Mappings

All column names are in English snake_case (translated from German in Phase 1: `scripts/preparation.py`).

Key mappings used in dbt models:

| German | English | Model | Purpose |
|--------|---------|-------|---------|
| Bewertungs-ID | review_id | fact_reviews | Primary key |
| Restaurant (Place) | place_url / restaurant_id | dim_restaurants | Restaurant identifier |
| Reviewer-ID | reviewer_user_id | dim_reviewers | Reviewer identifier |
| Sterne | rating_stars | fact_reviews | Review rating (1-5) |
| Bewertungstext | review_text | fact_reviews | Review text content |
| Erstellt am | timestamp_created_iso | fact_reviews | Creation timestamp (UTC) |
| Reviewer-ID | reviewer_user_id | dim_reviewers | Reviewer identifier |

See `scripts/preparation.py` COLUMN_MAP for complete mapping.

## Derived Columns (Added in Phase 1)

These columns are computed in Phase 1 (preparation.py) and available in raw_reviews:

| Column | Formula | Purpose |
|--------|---------|---------|
| review_date | CAST(timestamp_created_iso AS DATE) | Date dimension for grouping |
| review_text_length | LENGTH(review_text) | Text length analysis feature |
| edit_delta_hours | (timestamp_edited - timestamp_created) / 3600 | Edit behavior signal |
| reviewer_photo_review_ratio | reviewer_photo_count / reviewer_review_count | Engagement metric |
| review_detail_level | 0=stars, 1=text, 2=sub-ratings | Behavioral feature (Moon et al. 2021) |
| has_review_text | bool(review_text) | Boolean feature |
| has_sub_ratings | bool(sub_ratings) | Feature engineering |
| has_owner_response | bool(owner_response) | Response pattern |
| was_edited | bool(timestamp_edited) | Edit behavior |
| review_language | Normalized language code | Language features |

## How to Run dbt

### Prerequisites

```bash
# dbt is installed in requirements.txt
# DuckDB path: data/processed/reviews.duckdb (created by Phase 1)

# Verify dbt installation
dbt --version
```

### Running dbt Locally (Outside Docker)

```bash
# From project root
cd dbt

# Parse and validate all models (no SQL execution)
dbt parse

# Run all models (staging views + mart tables)
dbt run

# Run only staging models (views)
dbt run --select tag:staging

# Run only mart models (tables)
dbt run --select tag:marts

# Run dbt test suite (checks schema.yml constraints)
dbt test

# Generate documentation (creates docs/index.html)
dbt docs generate
dbt docs serve  # Starts local webserver
```

### Running dbt in Docker (via Prefect)

The `flows/pipeline.py` orchestrates dbt runs:

```python
@task(name="03_dbt_transform")
def run_dbt() -> None:
    subprocess_run(["dbt", "run", "--project-dir", "dbt/"], check=True)
    subprocess_run(["dbt", "test", "--project-dir", "dbt/"], check=False)
```

Full pipeline:
```bash
# From project root
docker-compose up pipeline   # Runs entire pipeline including dbt
```

## Data Quality Tests

All tests defined in `models/schema.yml`:

### Critical Tests (Hard Fail)
- `unique`: review_id, reviewer_user_id, restaurant_id
- `not_null`: review_id, reviewer_user_id, restaurant_id, rating_stars
- `accepted_values`: rating_stars ∈ {1, 2, 3, 4, 5}
- `relationships`: Foreign key integrity (FK → PK)
- `table_row_count_to_be_between`: Minimum 1000 rows

### Soft Tests (Warn Only)
- Language values in expected set
- Not tested in dbt but in `scripts/quality_checks_ge.py` (Great Expectations)

Run tests:
```bash
dbt test                    # Run all tests
dbt test --select stg_reviews  # Test single model
dbt test --select dim_restaurants  # Test single model
```

## Database Configuration

See `profiles.yml`:

```yaml
fake_review_detection:
  target: dev
  outputs:
    dev:
      type: duckdb
      path: '../data/processed/reviews.duckdb'
      schema: main
```

**Important**: Path is RELATIVE to dbt/ directory (`../data/processed/reviews.duckdb`)

No absolute paths to allow portability across machines/CI systems.

## Architecture References

- **Star Schema**: Kimball methodology for analytical databases
- **Data Build Tool**: dbt docs https://docs.getdbtcloud.com/
- **DuckDB**: Embedded OLAP database (no server setup required)
- **Lineage**: Models reference source via `{{ source() }}` and other models via `{{ ref() }}`

## Project Structure

```
dbt/
├── dbt_project.yml          # dbt configuration
├── profiles.yml             # Connection config (DuckDB path)
├── models/
│   ├── staging/             # Views (raw data with derived columns)
│   │   ├── stg_reviews.sql
│   │   ├── stg_reviewers.sql
│   │   └── stg_restaurants.sql
│   ├── marts/               # Tables (optimized for analysis)
│   │   ├── fact_reviews.sql
│   │   ├── dim_reviewers.sql
│   │   └── dim_restaurants.sql
│   └── schema.yml           # Model docs + data quality tests
├── analyses/                # SQL analytics (not part of pipeline)
├── tests/                   # Custom validation logic (Python/SQL)
├── seeds/                   # Static CSV data (lookup tables)
└── macros/                  # Reusable SQL templates

target/                       # Generated artifacts (git-ignored)
├── manifest.json            # Project lineage graph
├── run_results.json         # Test/run results
```

## Integration with Prefect Pipeline

The complete pipeline flow:

1. **Phase 1**: `scripts/preparation.py` → Load CSV, normalize columns → raw_reviews in DuckDB
2. **Phase 2**: `scripts/quality_checks_ge.py` -> Validate with Great Expectations
3. **Phase 3**: `dbt run` → Transform raw_reviews → Staging (views) → Marts (tables)
4. **Phase 3**: `dbt test` → Validate relationships and constraints
5. **Phase 4-7**: Heuristic, temporal, network, semantic scoring (Python)
6. **Phase 8**: Combine scores and export results

dbt's role: Transform raw denormalized data into dimensional tables optimized for analysis.

## Next Steps

Post-dbt (not implemented yet):
- Join analysis results from Phase 4-7 (scores) with fact_reviews
- Aggregate metrics for restaurant-level monitoring
- Build BI dashboards (Metabase, Tableau, etc.)

## Further Reading

- dbt Best Practices: https://docs.getdbtcloud.com/guides/best-practices
- Star Schema Design: Kimball, Ralph. "The Data Warehouse Toolkit"
- DuckDB: https://duckdb.org/docs/
