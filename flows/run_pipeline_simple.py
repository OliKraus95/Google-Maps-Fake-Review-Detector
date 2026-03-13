#!/usr/bin/env python3
"""
Simple pipeline runner without Prefect dependency.

Executes the first 3 phases of the fake review detection pipeline:
1. Data Preparation
2. Quality Checks  
3. dbt Transformations
"""

import logging
import subprocess
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress verbose logs
logging.getLogger("great_expectations").setLevel(logging.WARNING)
logging.getLogger("great_expectations._docs_decorators").setLevel(logging.WARNING)


def main():
    """Run pipeline phases sequentially."""
    logger.info("=" * 80)
    logger.info("FAKE REVIEW DETECTION PIPELINE")
    logger.info("=" * 80)
    
    try:
        # Phase 1: Data Preparation
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 1: Data Preparation")
        logger.info("=" * 80)
        from scripts.preparation import run as run_preparation
        run_preparation()
        logger.info("✅ Phase 1 complete\n")
        
        # Phase 2: Quality Checks
        logger.info("=" * 80)
        logger.info("PHASE 2: Quality Checks")
        logger.info("=" * 80)
        from scripts.quality_checks_ge import run as run_quality_checks
        run_quality_checks("/app/data/processed/reviews_clean.parquet")
        logger.info("✅ Phase 2 complete\n")
        
        # Phase 3: dbt Transformations
        logger.info("=" * 80)
        logger.info("PHASE 3: dbt Transformations")
        logger.info("=" * 80)
        
        logger.info("\n[DBT RUN]")
        result = subprocess.run(
            ["dbt", "run", "--project-dir", "/app/dbt", "--profiles-dir", "/app/dbt"],
            cwd="/app"
        )
        if result.returncode != 0:
            raise RuntimeError(f"dbt run failed with exit code {result.returncode}")
        
        logger.info("\n[DBT TEST]")
        result = subprocess.run(
            ["dbt", "test", "--project-dir", "/app/dbt", "--profiles-dir", "/app/dbt"],
            cwd="/app"
        )
        if result.returncode != 0:
            logger.warning(f"dbt test had failures (exit code {result.returncode})")
        
        logger.info("✅ Phase 3 complete\n")
        
        # Summary
        logger.info("=" * 80)
        logger.info("✅ PIPELINE COMPLETE!")
        logger.info("=" * 80)
        logger.info("\nNext steps:")
        logger.info("  - View results in Jupyter: notebooks/02_run_full_pipeline.ipynb")
        logger.info("  - Query DuckDB: data/processed/reviews.duckdb")
        logger.info("  - Check outputs: outputs/")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
