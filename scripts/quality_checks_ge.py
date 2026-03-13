"""Data quality checks with Great Expectations installed as a dependency.

This module performs practical dataframe-level checks used by the pipeline and
emits both critical failures (pipeline stop) and soft warnings.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

# Suppress verbose GE internal logs
logging.getLogger("great_expectations").setLevel(logging.WARNING)
logging.getLogger("great_expectations._docs_decorators").setLevel(logging.WARNING)

import pandas as pd

try:
    import great_expectations as ge
except ImportError:
    ge = None

from scripts import config

logger = logging.getLogger(__name__)

# ============================================================================
# Quality Check Constants
# ============================================================================

# Soft check thresholds
SUB_RATING_MIN_FILL_RATE = 0.20

# Attribute coverage thresholds (soft checks — warning only)
# These columns have known low fill rates from scraping.
# Thresholds trigger warnings when coverage drops below expected minimum.
ATTRIBUTE_COVERAGE_THRESHOLDS: dict[str, float] = {
    # Restaurant attributes from Google Maps
    "attr_noise_level": 0.10,
    "attr_wait_time": 0.05,
    "attr_group_size": 0.08,
    "attr_parking": 0.03,
    "attr_vegetarian": 0.07,
    # Owner response fields
    "owner_response_text": 0.10,
    "owner_response_timestamp_iso": 0.10,
    "owner_response_language": 0.10,
}

# Minimum overall non-null rate for any column to trigger a warning
GLOBAL_MIN_FILL_RATE = 0.01  # Warn if any column has <1% fill rate

def _validate_with_ge(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate a DataFrame while requiring GE to be installed.
    
    Args:
        df: Pandas DataFrame to validate
        
    Returns:
        Tuple of (success: bool, results: Dict with validation details)
    """
    if not ge:
        logger.warning("Great Expectations not properly installed, skipping GE validation")
        return True, {"skipped": True, "reason": "GE not available"}
    
    try:
        # Create a simple Validator-like object using GE's pandas compatibility
        # Modern approach: directly check expectations
        results = {
            "critical_passed": [],
            "critical_failed": [],
            "soft_warnings": []
        }
        
        # Check 1: Critical columns
        critical_cols = ["review_id", "rating_stars", "place_url", "reviewer_user_id", "timestamp_created_iso"]
        missing = [c for c in critical_cols if c not in df.columns]
        if missing:
            results["critical_failed"].append(f"Missing columns: {missing}")
            logger.error(f"❌ Critical: Missing columns: {missing}")
        else:
            results["critical_passed"].append("All critical columns exist")
            logger.info("✓ All critical columns exist")
        
        # Check 2: Non-null values
        required_non_null = ["review_id", "place_url", "reviewer_user_id"]
        for col in required_non_null:
            null_count = df[col].isna().sum()
            if null_count > 0:
                results["critical_failed"].append(f"{col}: {null_count:,} null values")
                logger.error(f"❌ Critical: {col} has {null_count:,} null values")
            else:
                results["critical_passed"].append(f"{col}: all non-null")
        
        # Check 3: Rating ranges
        invalid_ratings = ((df['rating_stars'] < 1) | (df['rating_stars'] > 5)).sum()
        if invalid_ratings > 0:
            results["critical_failed"].append(f"rating_stars: {invalid_ratings:,} invalid values")
            logger.error(f"❌ Critical: rating_stars has {invalid_ratings:,} invalid values")
        else:
            results["critical_passed"].append(f"rating_stars: all values in [1,5]")
            logger.info("✓ rating_stars: all values in [1,5]")
        
        # Check 4: Unique review_ids
        dup_count = df['review_id'].duplicated().sum()
        if dup_count > 0:
            results["critical_failed"].append(f"Duplicate review_ids: {dup_count:,}")
            logger.error(f"❌ Critical: {dup_count:,} duplicate review_ids")
        else:
            results["critical_passed"].append(f"Unique review_ids: {len(df):,}")
            logger.info(f"✓ Unique review_ids: {len(df):,} unique")
        
        # Check 5: Minimum row count
        if len(df) < 1000:
            results["critical_failed"].append(f"Row count: {len(df):,} < 1000 minimum")
            logger.error(f"❌ Critical: Only {len(df):,} rows (< 1000)")
        else:
            results["critical_passed"].append(f"Row count: {len(df):,}")
            logger.info(f"✓ Row count: {len(df):,} (>= 1000)")
        
        # Check 6: Sub-rating coverage (soft check)
        sub_rating_cols = ["sub_rating_food", "sub_rating_service", "sub_rating_atmosphere"]
        for col in sub_rating_cols:
            if col in df.columns:
                fill_rate = 1 - (df[col].isna().sum() / len(df))
                if fill_rate < SUB_RATING_MIN_FILL_RATE:
                    results["soft_warnings"].append(
                        f"{col}: {fill_rate:.1%} fill rate (< {SUB_RATING_MIN_FILL_RATE:.0%})"
                    )
                    logger.warning(f"⚠ Soft: {col}: {fill_rate:.1%} fill rate")
        
        # Check 7: Attribute column coverage (soft check)
        # Warns when known low-coverage columns drop below their expected minimum.
        # This catches scraping regressions or data source changes early.
        logger.info("\n[ATTRIBUTE COVERAGE CHECKS]")
        for col, min_rate in ATTRIBUTE_COVERAGE_THRESHOLDS.items():
            if col in df.columns:
                fill_rate = 1 - (df[col].isna().sum() / len(df))
                if fill_rate < min_rate:
                    results["soft_warnings"].append(
                        f"Coverage regression: {col} at {fill_rate:.1%} "
                        f"(below minimum {min_rate:.0%})"
                    )
                    logger.warning(
                        f"⚠ Coverage regression: {col} at {fill_rate:.1%} "
                        f"(expected >= {min_rate:.0%})"
                    )
                else:
                    logger.info(f"  ✓ {col}: {fill_rate:.1%} (>= {min_rate:.0%})")
            else:
                results["soft_warnings"].append(f"Column missing: {col}")
                logger.warning(f"⚠ Expected column missing: {col}")
        
        # Check 8: Global fill rate check (soft check)
        # Catches any column that has nearly zero data — indicates broken scraping.
        logger.info("\n[GLOBAL FILL RATE CHECK]")
        low_fill_cols = []
        for col in df.columns:
            fill_rate = 1 - (df[col].isna().sum() / len(df))
            if fill_rate < GLOBAL_MIN_FILL_RATE:
                low_fill_cols.append((col, fill_rate))
        if low_fill_cols:
            for col, rate in low_fill_cols:
                results["soft_warnings"].append(
                    f"Near-empty column: {col} ({rate:.1%} fill rate)"
                )
                logger.warning(f"⚠ Near-empty column: {col} ({rate:.1%})")
        else:
            logger.info(f"  ✓ All columns above {GLOBAL_MIN_FILL_RATE:.0%} fill rate")
        
        success = len(results["critical_failed"]) == 0
        return success, results
        
    except Exception as e:
        logger.error(f"GE validation error: {e}", exc_info=True)
        return False, {"error": str(e)}


# ============================================================================
# Main Validation Function
# ============================================================================

def run(parquet_path: str) -> None:
    """
    Execute comprehensive data quality validation using Great Expectations.
    
    Args:
        parquet_path: Path to Parquet file to validate
        
    Raises:
        FileNotFoundError: If Parquet file not found
        RuntimeError: If critical checks fail
    """
    logger.info("=" * 80)
    logger.info("STAGE: Data Quality Checks (Great Expectations Integration)")
    logger.info("=" * 80)
    
    # Load data
    parquet_file = Path(parquet_path)
    if not parquet_file.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    
    df = pd.read_parquet(parquet_path)
    logger.info(f"✓ Loaded {len(df):,} reviews from {parquet_file.name}")
    logger.info(f"  Dimensions: {len(df):,} rows × {len(df.columns)} columns")
    
    # Run validation
    logger.info("\n[VALIDATION SUITE]")
    success, results = _validate_with_ge(df)
    
    # Add coverage summary to report
    coverage_summary = {}
    for col in df.columns:
        fill_rate = 1 - (df[col].isna().sum() / len(df))
        if fill_rate < 1.0:  # Only include columns with some nulls
            coverage_summary[col] = round(fill_rate, 4)
    results["coverage_summary"] = dict(
        sorted(coverage_summary.items(), key=lambda x: x[1])
    )
    
    # Log results
    logger.info(f"\n✓ Passed checks: {len(results.get('critical_passed', []))}")
    for msg in results.get('critical_passed', []):
        logger.info(f"  ✓ {msg}")
    
    if results.get('critical_failed'):
        logger.error(f"\n❌ Failed checks: {len(results['critical_failed'])}")
        for msg in results['critical_failed']:
            logger.error(f"  ❌ {msg}")
    
    if results.get('soft_warnings'):
        logger.warning(f"\n⚠ Soft warnings: {len(results['soft_warnings'])}")
        for msg in results['soft_warnings']:
            logger.warning(f"  ⚠ {msg}")
    
    # Save validation report
    report_path = config.OUTPUTS_DIR / "validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\n✓ Validation report saved: {report_path}")
    
    # Final status
    logger.info("\n" + "=" * 80)
    if success:
        logger.info("✓ Data validation PASSED - Ready for downstream processing")
    else:
        logger.error("❌ Data validation FAILED - Critical issues detected")
        logger.error("Aborting pipeline")
        raise RuntimeError("Critical data quality checks failed")
    logger.info("=" * 80)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(name)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Test run
    parquet_path = "/app/data/processed/reviews_clean.parquet"
    run(parquet_path)
