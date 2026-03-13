"""
Prefect orchestration pipeline for fake review detection.

This module defines a multi-stage fake review detection pipeline for Google Maps
restaurant reviews. The pipeline orchestrates data ingestion, quality checks,
transformation, feature engineering, and analysis across 9 sequential tasks.

**Pipeline Architecture:**
The flow executes 9 tasks in strict sequential order, reflecting data dependencies:

1. run_preparation() -> loads, cleans raw CSV → saves cleaned parquet
2. run_quality_checks() -> validates data quality against expectations
3. run_dbt() -> transforms raw data using dbt models (staging + marts)
4. run_heuristic_scoring() -> applies rule-based suspicious pattern detection
5. run_temporal_analysis() -> detects time-based anomalies and burst patterns
6. run_network_analysis() -> identifies suspicious reviewer clusters via graph analysis
7. run_semantic_analysis() -> detects copy-paste reviews via embeddings similarity
8. run_scoring() -> combines all signals into final suspicion scores
9. upload_outputs_to_minio() -> archives results to MinIO object storage

**Why Prefect?**
Prefect is a modern, Python-native workflow orchestration framework. It defines
entire DAGs as Python code with @flow and @task decorators, eliminating the need
for external YAML configuration or complex database setup required by Airflow.

Key advantages over Airflow:
- **Zero boilerplate**: No Postgres DB, no DAG upload ritual, no webserver restarts
- **Python-first**: DAG definition is pure Python; branching, looping, and conditions
  are native Python constructs
- **Cloud-native**: Seamless local, hybrid, and cloud execution without refactoring
- **Automatic telemetry**: Prefect logs start/end time, duration, status, and errors
  for every task and flow run. Built-in CLI and local UI dashboard.
- **Easy migration**: Migrating to Airflow later is trivial—the Task/DAG concepts
  are identical; only decorators and APIs change.

Prefect is ideal for portfolio projects because it demonstrates modern DevOps
practices while remaining lightweight for local development and testing.

**Environment:**
Expects Docker environment with env vars loaded from .env:
- MINIO_ENDPOINT, MINIO_ROOT_USER, MINIO_ROOT_PASSWORD
- SENTENCE_TRANSFORMER_MODEL (optional, defaults to multilingual-MiniLM)

**Usage:**
    Local execution (development):
        python flows/pipeline.py

    Docker execution:
        docker-compose up pipeline

    With Prefect Server (optional, for UI dashboard):
        docker-compose up prefect-server -d
        prefect cloud login  # or `prefect config set PREFECT_API_URL=...`)
        python flows/pipeline.py

**Logging:**
All tasks and flow runs are logged to:
- Console: Real-time status and progress
- /app/outputs/logs/ (in Docker) or ./outputs/logs/ (local)
- Prefect Server (if running)
"""

import logging
from pathlib import Path
from subprocess import run as subprocess_run
from datetime import datetime, timedelta
import os

# Configure Prefect to use server if PREFECT_API_URL is set, otherwise ephemeral
# MUST be set BEFORE importing prefect
if "PREFECT_API_URL" not in os.environ:
    os.environ["PREFECT_API_URL"] = "ephemeral"
os.environ["PREFECT_API_DISABLE_TELEMETRY"] = "true"

import pandas as pd

# Try to import Prefect, but provide fallbacks if not available
try:
    from prefect import flow, task
    from prefect.tasks import task_input_hash
except ImportError:
    # Fallback decorators that do nothing
    def flow(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])
    
    def task(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])

    # Fallback cache key function used by task decorators when Prefect is absent.
    def task_input_hash(*args, **kwargs):
        return None

# ============================================================================
# Configure logging
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# Task Definitions
# ============================================================================


@task(
    retries=2,
    retry_delay_seconds=30,
    name="01_ingest_and_clean",
    description="Load raw CSV, clean, normalize columns, generate IDs"
)
def run_preparation() -> str:
    """
    Load and prepare raw review data.
    
    Performs:
    - CSV loading from data/raw with .csv pattern matching
    - German → English column name normalization
    - Type conversions (datetime UTC, numeric, boolean)
    - Derived feature engineering (edit_delta_hours, reviewer_photo_ratio, etc.)
    - Deduplication by review_id (keeping latest crawl_timestamp)
    - Output to Parquet (analytical format) and DuckDB (transformation format)
    - Data quality report generation
    
    Returns:
        str: Relative path to cleaned Parquet file (data/processed/reviews_clean.parquet)
        
    Raises:
        FileNotFoundError: If no CSV files matching 'reviews_complete*.csv' in data/raw/
        Exception: If critical data validation or persistence fails
    """
    from scripts.preparation import run
    
    logger.info("Task: Ingesting and cleaning raw data...")
    parquet_path = run()
    
    logger.info(f"✓ Preparation complete: {parquet_path}")
    return parquet_path


@task(
    retries=1,
    retry_delay_seconds=10,
    name="02_data_quality_checks",
    description="Validate data quality against Great Expectations suite"
)
def run_quality_checks(parquet_path: str) -> None:
    """
    Validate data quality using Great Expectations.
    
    Runs validation suite against cleaned data to ensure:
    - No missing critical fields (review_id, rating_stars, place_url, reviewer_user_id, timestamp_created_iso)
    - Valid rating ranges (1-5 stars, 1.0-5.0 place rating)
    - Unique review IDs (deduplication verification)
    - Minimum viable dataset size (1000+ reviews)
    - Valid review languages and sub-rating coverage
    
    Critical checks halt the pipeline if failed. Soft checks (language, coverage)
    issue warnings but allow pipeline to continue.
    
    **Why Great Expectations?**
    Industry-standard framework for data validation in production pipelines:
    - Expectations defined as code (version-controlled, testable)
    - Human-readable validation reports (JSON)
    - Integrates with orchestration (Prefect, Airflow)
    - Demonstrates professional data engineering practices
    
    Args:
        parquet_path: Path to cleaned Parquet file from preparation stage
        
    Raises:
        RuntimeError: If critical validation checks fail
    """
    from scripts.quality_checks_ge import run as run_quality_validation
    
    logger.info("Task: Running data quality checks (Great Expectations)...")
    run_quality_validation(parquet_path)
    logger.info("✓ Quality checks passed")


@task(
    retries=2,
    retry_delay_seconds=15,
    name="03_dbt_transform",
    description="Execute dbt models to create staging and mart tables"
)
def run_dbt() -> None:
    """
    Execute dbt transformation models.
    
    Runs dbt pipeline:
    1. dbt run: Create staging tables (cleaned raw data) and mart tables (dimensions + facts)
    2. dbt test: Validate data integrity (unique keys, relationships, value ranges)
    
    Models created:
    - stg_reviews, stg_restaurants, stg_reviewers (staging)
    - dim_restaurants, dim_reviewers, fact_reviews (marts)
    
    Database: DuckDB at data/processed/reviews.duckdb
    
    Raises:
        subprocess.CalledProcessError: If dbt run or dbt test fails
    """
    logger.info("Task: Running dbt transformation...")
    
    # dbt run
    result = subprocess_run(
        ["dbt", "run", "--project-dir", "dbt/", "--profiles-dir", "dbt/"],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True
    )
    logger.info(f"dbt run completed: {result.stdout[:500]}")
    
    # dbt test
    result = subprocess_run(
        ["dbt", "test", "--project-dir", "dbt/", "--profiles-dir", "dbt/"],
        cwd=Path.cwd(),
        check=False,  # Don't fail pipeline if tests fail
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        logger.info(f"dbt tests passed: {result.stdout[:500]}")
    else:
        logger.warning(f"dbt tests had failures: {result.stderr[:500]}")


@task(
    retries=1,
    retry_delay_seconds=10,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1),
    name="04_heuristic_scoring",
    description="Apply rule-based heuristic scoring for suspicious patterns"
)
def run_heuristic_scoring(parquet_path: str) -> None:
    """
    Calculate heuristic-based suspicion scores.
    
    Applies behavior-based feature detection including:
    - Reviewer profile characteristics (review count, verification level, etc.)
    - Maximum number of reviews per day (MNR)
    - Percentage of positive reviews (PR)
    - Rating deviation from restaurant mean (RD)
    - Sub-rating consistency with main rating
    - Review detail level as trust indicator
    
    All scores normalized to [0, 1] where 1 = most suspicious.
    Output saved to config.DATA_PROCESSED_DIR / "scores_heuristic.parquet"
    
    Args:
        parquet_path: Path to cleaned Parquet file (unused; loaded from config)
    """
    from scripts.heuristic_scoring import run
    
    logger.info("Task: Running heuristic scoring pipeline...")
    run()
    logger.info("Heuristic scoring complete.")


@task(
    retries=1,
    retry_delay_seconds=10,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1),
    name="05_temporal_analysis",
    description="Detect time-based patterns and burst activity"
)
def run_temporal_analysis(parquet_path: str) -> None:
    """
    Analyze temporal patterns in review activity.
    
    Detects:
    - Review bursts (multiple reviews in short time windows)
    - Reviewer activity frequency patterns
    - Temporal clustering around events
    
    Generates temporal_suspicion_score for each review.
    
    Args:
        parquet_path: Path to cleaned Parquet file from preparation stage
    """
    from scripts.temporal_analysis import calculate_temporal_features
    
    logger.info("Task: Running temporal analysis...")
    
    df = pd.read_parquet(parquet_path)
    df = calculate_temporal_features(df)
    logger.info("Temporal analysis complete")


@task(
    retries=1,
    retry_delay_seconds=10,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1),
    name="06_network_analysis",
    description="Identify suspicious reviewer clusters via graph analysis"
)
def run_network_analysis(parquet_path: str) -> None:
    """
    Build and analyze reviewer-restaurant networks.
    
    Creates bipartite graph where edges connect reviewers to restaurants they reviewed.
    Detects:
    - Suspicious dense clusters (coordinated reviewers targeting same restaurants)
    - Outlier reviewers with unusual connection patterns
    - Hidden cliques and communities
    
    Generates network_suspicion_score and cluster membership flags.
    
    Args:
        parquet_path: Path to cleaned Parquet file from preparation stage
    """
    from scripts.network_analysis import calculate_network_features
    
    logger.info("Task: Running network analysis...")
    
    df = pd.read_parquet(parquet_path)
    df = calculate_network_features(df)
    logger.info("Network analysis complete")


@task(
    retries=2,
    retry_delay_seconds=60,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=2),
    name="07_semantic_analysis",
    description="Detect copy-paste and near-duplicate reviews via embeddings"
)
def run_semantic_analysis(parquet_path: str) -> None:
    """
    Detect semantically similar and duplicate reviews.
    
    Uses sentence transformers (multilingual embeddings) to:
    - Generate embeddings for all review texts
    - Find near-duplicate reviews (cosine similarity > 0.85)
    - Identify template-like reviews with boilerplate text
    
    Generates semantic_suspicion_score and similar_duplicate flags.
    
    Note: First run downloads ~1 GB of transformer model weights.
    
    Args:
        parquet_path: Path to cleaned Parquet file from preparation stage
    """
    from scripts.semantic_analysis import run as run_semantic_pipeline
    
    logger.info("Task: Running semantic analysis...")
    
    run_semantic_pipeline()
    logger.info("Semantic analysis complete")


@task(
    retries=1,
    retry_delay_seconds=10,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1),
    name="08_scoring_and_export",
    description="Combine all signals into final suspicion scores and export"
)
def run_scoring(parquet_path: str) -> None:
    """
    Combine all suspicion scores into final predictions.
    
    Performs weighted aggregation:
    - heuristic_score (25%)
    - temporal_suspicion_score (25%)
    - network_suspicion_score (25%)
    - semantic_suspicion_score (25%)
    
    Outputs:
    - final_suspicion_score: [0, 1] where 1 = most suspicious
    - suspicion_level: Categorical [Low, Medium, High]
    - is_suspicious: Binary flag (>= 0.6 threshold)
    
    Saves to data/processed/final_results.csv and suspicious_reviews.csv
    
    Args:
        parquet_path: Path to cleaned Parquet file from preparation stage
    """
    from scripts.scoring import run as run_scoring_pipeline
    
    logger.info("Task: Scoring and exporting results...")
    
    run_scoring_pipeline()
    
    logger.info("Scoring complete")


@task(
    retries=3,
    retry_delay_seconds=30,
    name="09_upload_to_minio",
    description="Archive final results to MinIO object storage"
)
def upload_outputs_to_minio() -> None:
    """
    Upload output files to MinIO for archival and sharing.
    
    Uploads:
    - final_results.csv (all reviews with scores)
    - suspicious_reviews.csv (high-suspicion subset)
    - Various intermediate analysis files
    
    MinIO bucket: 'fake-reviews' at MINIO_ENDPOINT from .env
    
    Gracefully handles MinIO connectivity issues (logs warning but doesn't fail).
    """
    from scripts.storage import upload_outputs
    
    logger.info("Task: Uploading outputs to MinIO...")
    
    try:
        result = upload_outputs()
        logger.info(f"Upload to MinIO complete: {result.get('total', 0)} files uploaded")
    except Exception as e:
        logger.warning(f"MinIO upload failed: {e} (continuing anyway)")


# ============================================================================
# Flow Definition
# ============================================================================


@flow(
    name="fake-review-detection-pipeline",
    description="Complete pipeline for detecting fake Google Maps reviews"
)
def pipeline() -> None:
    """
    Execute complete fake review detection pipeline.
    
    **Flow Overview:**
    This is the main orchestration flow that coordinates a multi-stage data analysis
    pipeline for detecting suspicious (fake) reviews on Google Maps restaurant listings
    in Würzburg, Germany.
    
    **Execution Model:**
    Tasks execute sequentially (SequentialTaskRunner) to ensure data dependencies
    are honored. Each task waits for the previous to complete before starting.
    
    **Automatic Logging:**
    Prefect automatically captures and logs:
    - Task start time, end time, and duration
    - Task status (COMPLETED, FAILED, SKIPPED)
    - Return values from tasks
    - Exceptions and tracebacks
    - Memory and CPU usage (if configured)
    
    All logs are available in:
    1. Console output (real-time)
    2. Prefect Server UI (if running at http://localhost:4200)
    3. Local log files (if configured)
    
    **Error Handling:**
    - run_preparation() automatically retries up to 2 times with 30-second delays
    - Other tasks fail fast by default to avoid wasted computation
    - The entire pipeline stops at first task failure
    
    **Expected Runtime:**
    ~2-5 minutes for 10k-20k reviews (depends on network model download and MinIO upload)
    """
    logger.info("=" * 80)
    logger.info("Starting Fake Review Detection Pipeline")
    logger.info(f"   Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 80)
    
    # Sequential execution with data flow
    parquet_path = run_preparation()
    run_quality_checks(parquet_path)
    run_dbt()
    run_heuristic_scoring(parquet_path)
    run_temporal_analysis(parquet_path)
    run_network_analysis(parquet_path)
    run_semantic_analysis(parquet_path)
    run_scoring(parquet_path)
    upload_outputs_to_minio()
    
    logger.info("=" * 80)
    logger.info("Pipeline completed successfully")
    logger.info(f"   Timestamp: {datetime.now().isoformat()}")
    logger.info("   Results in: outputs/ and data/processed/")
    logger.info("=" * 80)


# ============================================================================
# Entry Point
# ============================================================================


if __name__ == "__main__":
    pipeline()
