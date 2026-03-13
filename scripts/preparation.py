"""
Data preparation module for fake review detection pipeline.

This module handles End-to-End data preparation:
1. CSV ingestion from raw data directory with German column normalization
2. Type conversions and validation
3. Feature engineering (derived columns for analysis)
4. Deduplication of multi-scrape runs
5. Persistence to Parquet (analytical format) and DuckDB (query format)
6. Data quality reporting

**Design Notes:**
- German column names are normalized to English snake_case for consistency
- Derived features represent behavioral patterns known to correlate with fake reviews:
  - review_detail_level: deeper reviews more likely authentic (Moon et al. 2021)
  - edit_delta_hours: editing behavior suggests legitimate user engagement
  - reviewer_photo_review_ratio: ratio indicates reviewer investment (Lim et al. 2010)
- Timestamps are stored in UTC for timezone-agnostic analysis
- Deduplication prioritizes latest scrape to capture review updates
"""

import csv
import logging
import os
from pathlib import Path
from typing import Union

import duckdb
import pandas as pd

from scripts import config

logger = logging.getLogger(__name__)

SENSITIVE_EXAMPLE_COLUMNS = {
    "reviewer_name",
    "reviewer_user_id",
    "reviewer_avatar_url",
    "reviewer_profile_url",
    "review_text",
    "owner_response_text",
}


def _sanitize_example_value(column_name: str, example: object) -> object:
    """Mask sensitive sample values in quality reports for safe sharing."""
    if pd.isna(example):
        return example

    anonymize_enabled = os.environ.get("ANONYMIZE_EXPORTS", "true").lower() in {
        "1", "true", "yes", "on"
    }
    if not anonymize_enabled:
        return example

    if column_name in SENSITIVE_EXAMPLE_COLUMNS:
        return "[REDACTED]"

    return example

# ============================================================================
# Column Mapping: German → English (snake_case)
# ============================================================================

COLUMN_MAP = {
    "Restaurant": "place_name",
    "Google Maps URL": "place_url",
    "Restaurant-Bewertung": "place_overall_rating",
    "Anzahl Bewertungen": "place_total_reviews",
    "Bewertungs-ID": "review_id",
    "Sterne": "rating_stars",
    "Bewertungstext": "review_text",
    "Sprache": "review_language",
    "Fotos": "review_images_count",
    "Erstellt am": "timestamp_created_iso",
    "Bearbeitet am": "timestamp_edited_iso",
    "Reviewer": "reviewer_name",
    "Reviewer-ID": "reviewer_user_id",
    "Avatar-URL": "reviewer_avatar_url",
    "Reviewer-Bewertungen": "reviewer_review_count",
    "Reviewer-Fotos": "reviewer_photo_count",
    "Reviewer-Profil-URL": "reviewer_profile_url",
    "Local Guide": "reviewer_is_local_guide",
    "Reviewer-Level": "reviewer_level",
    "Badge": "reviewer_badge",
    "Quelle": "review_source",
    "Quelle-Bewertung": "review_source_rating",
    "Essen (Bewertung)": "sub_rating_food",
    "Service (Bewertung)": "sub_rating_service",
    "Ambiente (Bewertung)": "sub_rating_atmosphere",
    "Mahlzeit": "attr_meal_type",
    "Preisgruppe": "attr_price_range",
    "Geräuschpegel": "attr_noise_level",
    "Verzehr": "attr_service_type",
    "Wartezeit": "attr_wait_time",
    "Gruppengröße": "attr_group_size",
    "Parkmöglichkeiten": "attr_parking",
    "Vegetarische Gerichte": "attr_vegetarian",
    "Restauranterantwort": "owner_response_text",
    "Antwort am": "owner_response_timestamp_iso",
    "Antwort-Sprache": "owner_response_language",
    "Scrapingzeit": "crawl_timestamp",
}


# ============================================================================
# Step 1.1: CSV Loading & Column Normalization
# ============================================================================


def _load_raw_csvs() -> pd.DataFrame:
    """
    Load and concatenate all raw CSV files matching pattern "reviews_complete*.csv".
    
    Reading parameters:
    - encoding: utf-8-sig (handles BOM/byte-order mark)
    - sep: ";" (semicolon-delimited as per data spec)
    - quoting: csv.QUOTE_ALL (strict quoting)
    
    Returns:
        Concatenated DataFrame from all matching CSV files.
        
    Raises:
        FileNotFoundError: If no matching CSV files found in DATA_RAW_DIR.
    """
    # Try reviews_complete pattern first, then merged_reviews pattern
    csv_files = sorted(config.DATA_RAW_DIR.glob("reviews_complete*.csv"))
    if not csv_files:
        csv_files = sorted(config.DATA_RAW_DIR.glob("merged_reviews*.csv"))
    
    if not csv_files:
        msg = f"No CSV files matching 'reviews_complete*.csv' or 'merged_reviews*.csv' in {config.DATA_RAW_DIR}"
        logger.error(msg)
        raise FileNotFoundError(msg)
    
    logger.info(f"Found {len(csv_files)} CSV file(s) to load")
    
    dfs = []
    for filepath in csv_files:
        logger.info(f"Loading: {filepath.name}")
        try:
            df = pd.read_csv(
                filepath,
                sep=";",
                encoding="utf-8-sig",
                quoting=csv.QUOTE_ALL,
                dtype=str  # Load all as strings initially for flexibility
            )
            logger.info(f"  ✓ {len(df)} records loaded")
            dfs.append(df)
        except Exception as e:
            logger.error(f"  ✗ Failed to load: {e}")
            raise
    
    # Concatenate if multiple files
    if len(dfs) == 1:
        df = dfs[0]
    else:
        df = pd.concat(dfs, ignore_index=True)
        dup_count = df.duplicated(subset=["Bewertungs-ID"], keep="first").sum()
        df = df.drop_duplicates(subset=["Bewertungs-ID"], keep="first")
        logger.info(f"Concatenated {len(dfs)} files; removed {dup_count} cross-file duplicates")
    
    return df


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename all columns from German to English using COLUMN_MAP.
    
    Logs a warning for any unexpected columns that don't have a mapping.
    
    Args:
        df: DataFrame with original German column names
        
    Returns:
        DataFrame with English column names (snake_case)
    """
    # Rename columns that exist in COLUMN_MAP
    rename_dict = {col: COLUMN_MAP[col] for col in df.columns if col in COLUMN_MAP}
    df = df.rename(columns=rename_dict)
    
    logger.info(f"Renamed {len(rename_dict)}/{len(df.columns)} columns")
    
    # Log unexpected columns
    expected_cols = set(COLUMN_MAP.values())
    actual_cols = set(df.columns)
    unexpected = actual_cols - expected_cols
    
    if unexpected:
        logger.warning(f"Unexpected columns (not in COLUMN_MAP): {unexpected}")
    
    return df


# ============================================================================
# Step 1.2: Type Conversions & Derived Columns
# ============================================================================


def _convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to appropriate dtypes.
    
    Handles:
    - Timestamps → pandas datetime (UTC)
    - Numeric columns → float (with coercion for invalid values)
    - Boolean → bool
    
    Args:
        df: DataFrame with string dtypes
        
    Returns:
        DataFrame with properly typed columns
    """
    # Timestamp columns to datetime (UTC)
    timestamp_cols = [
        "timestamp_created_iso",
        "timestamp_edited_iso",
        "owner_response_timestamp_iso",
        "crawl_timestamp"
    ]
    for col in timestamp_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
            logger.info(f"  {col}: converted to datetime[UTC]")
    
    # Numeric columns
    numeric_cols = [
        "rating_stars",
        "place_overall_rating",
        "place_total_reviews",
        "review_images_count",
        "reviewer_review_count",
        "reviewer_photo_count",
        "reviewer_level",
        "sub_rating_food",
        "sub_rating_service",
        "sub_rating_atmosphere"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    if any(col in df.columns for col in numeric_cols):
        logger.info(f"  Converted {sum(col in df.columns for col in numeric_cols)} numeric columns")
    
    # Boolean column
    if "reviewer_is_local_guide" in df.columns:
        df["reviewer_is_local_guide"] = df["reviewer_is_local_guide"].isin(
            ["True", "true", "TRUE", "1", "yes", "Yes", True]
        )
        logger.info("  reviewer_is_local_guide: converted to bool")
    
    return df


def _create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived columns for behavioral analysis.
    
    Features created (references to literature):
    - edit_delta_hours: Time between creation and edit (behavior signal)
    - reviewer_photo_review_ratio: Ratio of photos to reviews (engagement metric)
    - review_language: Normalized language code
    - has_review_text: Boolean presence of review text
    - has_sub_ratings: Boolean presence of sub-category ratings
    - has_owner_response: Boolean presence of owner response
    - was_edited: Boolean if review was edited
    - review_detail_level: 0=minimal, 1=text-only, 2=detailed (Moon et al. 2021)
    
    Args:
        df: DataFrame with converted types from _convert_types()
        
    Returns:
        DataFrame with additional derived columns
    """
    # place_id: stable Google place identifier extracted from place_url
    # and canonicalize place_url per place_id to enforce 1:1 mapping.
    if "place_url" in df.columns:
        unique_urls_before = df["place_url"].nunique(dropna=True)
        df["place_id"] = df["place_url"].astype(str).str.extract(r"!1s([^!]+)", expand=False)
        unique_place_ids = df["place_id"].nunique(dropna=True)

        canonical_urls = df.groupby("place_id")["place_url"].agg(
            lambda x: x.value_counts().index[0]
        )
        original_place_url = df["place_url"]
        df["place_url"] = df["place_id"].map(canonical_urls).fillna(original_place_url)
        unique_urls_after = df["place_url"].nunique(dropna=True)

        # Verify 1:1 mapping between place_id and place_url after normalization.
        non_1to1_place_ids = (
            df.dropna(subset=["place_id", "place_url"])
            .groupby("place_id")["place_url"]
            .nunique()
            .gt(1)
            .sum()
        )

        logger.info(f"  Unique place_urls before normalization: {unique_urls_before}")
        logger.info(f"  Unique place_ids: {unique_place_ids}")
        logger.info(f"  Unique place_urls after normalization: {unique_urls_after}")
        if non_1to1_place_ids > 0:
            logger.warning(
                f"  Non-1:1 mapping remains for {non_1to1_place_ids} place_id(s) after normalization"
            )
        else:
            logger.info("  place_id-place_url mapping is 1:1 after normalization")

    # edit_delta_hours: (timestamp_edited - timestamp_created) in hours
    if "timestamp_created_iso" in df.columns and "timestamp_edited_iso" in df.columns:
        edit_delta = (df["timestamp_edited_iso"] - df["timestamp_created_iso"]).dt.total_seconds() / 3600
        df["edit_delta_hours"] = edit_delta
        logger.info("  Created: edit_delta_hours")
    
    # reviewer_photo_review_ratio: reviewer_photo_count / reviewer_review_count
    if "reviewer_photo_count" in df.columns and "reviewer_review_count" in df.columns:
        df["reviewer_photo_review_ratio"] = (
            df["reviewer_photo_count"] / df["reviewer_review_count"].replace(0, pd.NA)
        ).fillna(0)
        logger.info("  Created: reviewer_photo_review_ratio")
    
    # review_language: normalize and fill.empty
    if "review_language" in df.columns:
        df["review_language"] = (
            df["review_language"]
            .fillna("unknown")
            .str.strip()
            .str.lower()
            .replace("", "unknown")
        )
        logger.info("  Created: review_language (normalized)")
    
    # has_review_text: boolean presence of non-empty text
    if "review_text" in df.columns:
        df["has_review_text"] = (
            df["review_text"].notna() & 
            (df["review_text"].astype(str).str.strip().str.len() > 0)
        )
        logger.info("  Created: has_review_text")
    
    # has_sub_ratings: boolean if any sub-rating exists
    sub_rating_cols = ["sub_rating_food", "sub_rating_service", "sub_rating_atmosphere"]
    existing_sub_ratings = [col for col in sub_rating_cols if col in df.columns]
    if existing_sub_ratings:
        df["has_sub_ratings"] = df[existing_sub_ratings].notna().any(axis=1)
        logger.info("  Created: has_sub_ratings")
    
    # has_owner_response: boolean presence of owner response
    if "owner_response_text" in df.columns:
        df["has_owner_response"] = (
            df["owner_response_text"].notna() &
            (df["owner_response_text"].astype(str).str.strip().str.len() > 0)
        )
        logger.info("  Created: has_owner_response")
    
    # was_edited: boolean if timestamp_edited exists
    if "timestamp_edited_iso" in df.columns:
        df["was_edited"] = df["timestamp_edited_iso"].notna()
        logger.info("  Created: was_edited")
    
    # review_detail_level: categorical [0, 1, 2]
    # 0 = minimal (only stars)
    # 1 = text but no sub-ratings
    # 2 = has sub-ratings (detailed feature breakdown)
    if "has_review_text" in df.columns and "has_sub_ratings" in df.columns:
        df["review_detail_level"] = (
            df["has_sub_ratings"].astype(int) * 2 +
            (~df["has_sub_ratings"] & df["has_review_text"]).astype(int)
        )
        logger.info("  Created: review_detail_level")
    
    return df


# ============================================================================
# Step 1.3: Deduplication
# ============================================================================


def _deduplicate_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate reviews, keeping the most recent scrape.
    
    Deduplication by review_id, keeping the record with latest crawl_timestamp.
    Useful when the data was scrapped multiple times over days/weeks.
    
    Args:
        df: DataFrame to deduplicate
        
    Returns:
        Deduplicated DataFrame
    """
    if "review_id" not in df.columns:
        logger.warning("review_id column not found; skipping deduplication")
        return df
    
    initial_count = len(df)
    
    # Sort by crawl_timestamp descending, keep first (latest)
    if "crawl_timestamp" in df.columns:
        df = df.sort_values("crawl_timestamp", ascending=False, na_position="last")
    
    df = df.drop_duplicates(subset=["review_id"], keep="first")
    removed_count = initial_count - len(df)
    
    if removed_count > 0:
        logger.info(f"Deduplication: removed {removed_count} duplicate reviews "
                    f"({removed_count/initial_count*100:.1f}%)")
    
    return df


# ============================================================================
# Step 1.4: Output to Parquet & DuckDB
# ============================================================================


def _save_parquet(df: pd.DataFrame) -> Path:
    """
    Save DataFrame to Parquet format.
    
    Parquet is chosen because:
    - Columnar storage: efficient for analytical queries
    - Compression: ~90% smaller than CSV
    - Schema preservation: type information persists
    - Speed: ~10x faster to load than CSV
    
    Args:
        df: DataFrame to save
        
    Returns:
        Path to saved Parquet file
    """
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(
        config.PARQUET_PATH,
        compression="snappy",
        index=False
    )
    
    size_mb = config.PARQUET_PATH.stat().st_size / 1024 / 1024
    try:
        rel_path = config.PARQUET_PATH.relative_to(Path.cwd())
        logger.info(f"Saved Parquet: {rel_path} ({size_mb:.1f} MB)")
    except ValueError:
        # If relative path fails (e.g., when running from different directory), use absolute
        logger.info(f"Saved Parquet: {config.PARQUET_PATH} ({size_mb:.1f} MB)")
    
    return config.PARQUET_PATH


def _save_duckdb(df: pd.DataFrame) -> None:
    """
    Load DataFrame into DuckDB as table "raw_reviews".
    
    DuckDB is used for:
    - OLAP queries (SQL dialect familiar to analysts)
    - dbt integration (ELT transformation pipeline)
    - Schema validation (CHECK constraints, ForeignKey tests)
    
    Args:
        df: DataFrame to load
    """
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    con = duckdb.connect(str(config.DUCKDB_PATH))
    con.execute("CREATE OR REPLACE TABLE raw_reviews AS SELECT * FROM df")
    con.close()
    
    logger.info(f"Loaded DuckDB table: raw_reviews ({len(df)} rows)")


# ============================================================================
# Step 1.5: Data Quality Report
# ============================================================================


def _generate_quality_report(df: pd.DataFrame) -> None:
    """
    Calculate fill rates and save data quality report.
    
    Creates a CSV with columns:
    - column_name
    - fill_rate (0.0 to 1.0)
    - dtype
    - null_count
    - example_value
    
    Logs columns with <20% coverage as warnings.
    
    Args:
        df: DataFrame to analyze
    """
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    quality_rows = []
    for col in df.columns:
        fill_rate = 1 - (df[col].isna().sum() / len(df))
        null_count = df[col].isna().sum()
        example = df[col].dropna().iloc[0] if null_count < len(df) else None
        example = _sanitize_example_value(col, example)
        
        quality_rows.append({
            "column_name": col,
            "fill_rate": fill_rate,
            "null_count": null_count,
            "dtype": str(df[col].dtype),
            "example_value": example
        })
        
        if fill_rate < 0.2:
            logger.warning(f"LOW COVERAGE: {col} ({fill_rate:.1%})")
    
    quality_df = pd.DataFrame(quality_rows)
    quality_path = config.OUTPUTS_DIR / "data_quality_report.csv"
    quality_df.to_csv(quality_path, index=False)
    
    try:
        rel_path = quality_path.relative_to(Path.cwd())
        logger.info(f"Quality report: {rel_path}")
    except ValueError:
        # If relative path fails (e.g., when running from different directory), use absolute
        logger.info(f"Quality report: {quality_path}")


# ============================================================================
# Main Run Function
# ============================================================================


def run() -> str:
    """
    Execute complete data preparation pipeline.
    
    **Pipeline stages:**
    1. Load raw CSV files from config.DATA_RAW_DIR
    2. Normalize German column names to English
    3. Convert types (timestamps, numerics, booleans)
    4. Create derived features for analysis
    5. Deduplicate reviews by review_id
    6. Output to Parquet (analytical) and DuckDB (transformation)
    7. Generate data quality report
    
    **Returns:**
        str: Relative path to Parquet file for next pipeline stage
        
    **Raises:**
        FileNotFoundError: If no input CSV files found
        Exception: If critical processing fails
    """
    logger.info("=" * 80)
    logger.info("STAGE: Data Preparation")
    logger.info("=" * 80)
    
    try:
        # 1.1: Load raw CSVs
        logger.info("\n[1.1] Loading raw CSV files...")
        df = _load_raw_csvs()
        logger.info(f"Total records loaded: {len(df):,}")
        
        # 1.1: Normalize columns
        logger.info("\n[1.1] Normalizing column names...")
        df = _normalize_column_names(df)
        logger.info(f"DataFrame columns: {len(df.columns)}")
        
        # 1.2: Convert types
        logger.info("\n[1.2] Converting data types...")
        df = _convert_types(df)
        
        # 1.2: Create derived features
        logger.info("\n[1.2] Creating derived features...")
        df = _create_derived_features(df)
        
        # 1.3: Deduplicate
        logger.info("\n[1.3] Deduplicating reviews...")
        df = _deduplicate_reviews(df)
        logger.info(f"Records after dedup: {len(df):,}")
        
        # 1.3.5: Filter invalid reviews (missing critical fields)
        logger.info("\n[1.3.5] Filtering invalid reviews...")
        initial_count = len(df)
        
        # Remove reviews without reviewer_user_id (anonymous/system reviews)
        if "reviewer_user_id" in df.columns:
            df = df[df["reviewer_user_id"].notna()].copy()
            removed = initial_count - len(df)
            if removed > 0:
                logger.warning(f"Removed {removed} reviews without reviewer_user_id")
        
        logger.info(f"Valid records after filtering: {len(df):,}")
        
        # 1.4: Save to Parquet and DuckDB
        logger.info("\n[1.4] Saving outputs...")
        parquet_path = _save_parquet(df)
        _save_duckdb(df)
        
        # 1.5: Generate quality report
        logger.info("\n[1.5] Generating quality report...")
        _generate_quality_report(df)
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ Data preparation complete")
        logger.info("=" * 80)
        
        # Return relative path as string, or absolute if relative fails
        try:
            return str(parquet_path.relative_to(Path.cwd()))
        except ValueError:
            return str(parquet_path)
        
    except Exception as e:
        logger.error(f"Data preparation failed: {e}", exc_info=True)
        raise


# ============================================================================
# Testing / Standalone Execution
# ============================================================================


if __name__ == "__main__":
    parquet_path = run()
    print(f"✓ Preparation complete: {parquet_path}")
