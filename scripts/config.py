"""
Central configuration for Fake Review Detection project.
All paths, constants, and environment variables are defined here.
This serves as the single source of truth for the entire project.
"""

import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# ============================================================================
# Path Configuration
# ============================================================================

# Project root (only place where root is calculated)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

# Database and storage
DUCKDB_PATH = DATA_PROCESSED_DIR / "reviews.duckdb"
PARQUET_PATH = DATA_PROCESSED_DIR / "reviews_clean.parquet"

# Output directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# dbt directories
DBT_DIR = PROJECT_ROOT / "dbt"
DBT_PROFILES_DIR = DBT_DIR

# Expectations
EXPECTATIONS_DIR = PROJECT_ROOT / "expectations"

# ============================================================================
# Create directories if they don't exist
# ============================================================================

for directory in [
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    PLOTS_DIR,
    REPORTS_DIR,
    EXPECTATIONS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MinIO / S3 Configuration
# ============================================================================

# MinIO settings (for local development use defaults, for production use .env)
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ROOT_USER = os.environ.get("MINIO_ROOT_USER", "")
MINIO_ROOT_PASSWORD = os.environ.get("MINIO_ROOT_PASSWORD", "")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "fake-reviews")
MINIO_SECURE = os.environ.get("MINIO_SECURE", "false").lower() == "true"

# ============================================================================
# Model Configuration
# ============================================================================

# Sentence Transformer model
SENTENCE_TRANSFORMER_MODEL = os.environ.get(
    "SENTENCE_TRANSFORMER_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
)

# ============================================================================
# Analysis Thresholds
# ============================================================================

# Heuristic scoring thresholds
MIN_REVIEW_LENGTH = 10
MAX_REVIEW_LENGTH = 5000
SUSPICIOUS_RATING_THRESHOLD = 1.5  # Standard deviation from restaurant mean

# Network analysis thresholds
MIN_REVIEWER_CONNECTIONS = 3
SUSPICIOUS_CLUSTER_SIZE = 5

# Semantic similarity threshold
SIMILARITY_THRESHOLD = 0.85

# ============================================================================
# Logging Configuration
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

# Log configuration on import
logger.info(f"Project root: {PROJECT_ROOT}")
logger.info(f"DuckDB path: {DUCKDB_PATH}")
logger.info(f"MinIO endpoint: {MINIO_ENDPOINT}")
