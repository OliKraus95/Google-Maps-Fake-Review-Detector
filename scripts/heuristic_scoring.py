"""
Heuristic scoring module for fake review detection based on reviewer behavior.

This module calculates suspicion scores for Google Maps reviews using
behavior-based features proven more discriminative than linguistic analysis
(Mukherjee et al. 2013; Lim et al. 2010). Each review receives a normalized
suspicion score [0, 1] where 0 = unsuspicious and 1 = maximally suspicious.

Features implemented:
  - Reviewer profile characteristics (review count, level, local guide, photo ratio, language)
  - Review count tier scoring: Data-driven thresholds derived from percentile
    analysis of 29,823 unique reviewers. Replaces log normalization (reference
    point 500 from Lim et al. 2010) which lacked discrimination on Google Maps
    data. Tiers align with empirical percentile breakpoints:
    1 review = 1.0 (P10), 2-4 = 0.7 (P25), 5-14 = 0.4 (P50),
    15-50 = 0.15 (P75), 51-130 = 0.05 (P90), 130+ = 0.0.
  - Maximum number of reviews per day (MNR): empirically strong predictor
  - Percentage of positive reviews (PR): Binary extreme-value approach (platform-adapted)
  - Rating deviation (RD) + sub-rating consistency: plausibility analysis
  - Sub-rating/main-rating discrepancy detection
  - Review detail level as trust indicator

IMPORTANT METHODOLOGICAL NOTE:
  Sub-rating homogeneity (std deviation of aspect ratings) is only computed
  for reviewers with ≥2 reviews containing sub-ratings. This avoids the
  statistical artifact where single-review users appear "perfectly consistent"
  (std=0 with only one value, falsely indicating no variance).
  
  For reviewers with <2 sub-rating reviews, RD score is based solely on
  deviation from restaurant mean, without the homogeneity component.

REMOVED FEATURES:
  - Profile picture detection (s_no_pic): Removed because Google Maps generates
    unique avatar URLs for all accounts. Auto-generated avatars (initial letter +
    color) cannot be distinguished from real uploaded photos via URL structure.
    Empirical result: 0/37,813 reviews had null avatar URLs → zero signal.

PLATFORM-SPECIFIC ADAPTATIONS:
  PR (Percentage Positive Reviews) uses binary extreme-value approach instead of
  continuous scoring methods (absolute threshold, Z-score, percentile rank).
  
  Google Maps exhibits stark bimodal distribution of reviewer behavior:
  - 89.4% of reviewers give exclusively positive reviews (pct=1.0)
  - 8.9% give exclusively negative reviews (pct=0.0)
  - Only 1.7% have mixed behavior
  
  Continuous approaches failed (max 13 unique values across 16k reviewers):
  - Absolute threshold (Mukherjee 2013: 80%): most reviewers exceed it
  - Z-score: distribution compressed to ±0.3 std, poor discrimination
  - Percentile rank: produces 13 unique values from continuous distribution
  
  Binary approach exploits the bimodal shape:
    - pct=1.0 (all positive): score=0.3 (weak suspicion, reduced from 0.6 because
        82% of Google Maps reviewers exhibit this pattern - it is normal platform
        behavior rather than a reliable fake signal)
  - pct=0.0 (all negative): score=1.0 (high suspicion, rare/visible spam signal)
  - pct=mixed: score=0.0 (normal behavior)
  
    Negative bombing (all negative) is weighted higher because it is statistically rare
  and therefore more suspicious given the platform's positivity bias.

References:
  - Mukherjee et al. (2013): Yelp filtered reviews
  - Lim et al. (2010): Amazon review analysis
  - Savage et al. (2015): Rating deviation studies
  - Moon et al. (2021): Sub-rating differentiation
  - Duma et al. (2023): Consistency analysis of fake reviews
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from scripts import config

logger = logging.getLogger(__name__)

# ============================================================================
# Heuristic Scoring Constants
# ============================================================================

# Reviewer Profile Weights
# Based on empirical discriminative power for fake review detection
# Note: Profile picture detection (s_no_pic) was removed after empirical analysis
# showed Google Maps generates unique avatar URLs for ALL accounts (including those
# without uploaded photos). The auto-generated avatars (initial letter + color +
# optional Local Guide badge) are indistinguishable from real photos via URL alone.
# See: https://lh3.googleusercontent.com/a/ACg8oc... (real photo) vs
#      https://lh3.googleusercontent.com/a-/ALV-Uj... (also potentially real)
# Result: reviewer_avatar_url.isna() was always 0 — zero discriminative power.
REVIEWER_PROFILE_WEIGHTS = {
    "s_review_count": 0.35,      # Few reviews strongly indicate suspicious behavior
    "s_level": 0.20,             # Higher Local Guide level = more trustworthy
    "s_local_guide": 0.25,       # Verification status indicates real user
    "s_photo_ratio": 0.15,       # Spammers post few photos relative to reviews
    "s_language": 0.05,          # Non-German for Würzburg = weak signal
}

# Review Count Scoring Tiers (data-driven, based on percentile analysis)
# Empirical distribution of 29,823 unique reviewers (global Google Maps count):
#   P10=1, P25=4, P50=14, P75=48, P90=129, P95=221, P99=566, Max=6927
#
# Tier boundaries align with natural percentile breakpoints:
# - 1 review: One-time accounts, highest fake risk (P10)
# - 2-4: Casual users / tourists (P10-P25)
# - 5-14: Regular users (P25-P50, median reviewer)
# - 15-50: Active reviewers (P50-P75)
# - 50+: Power users / Local Guides (above P75)
REVIEW_COUNT_TIERS: list[tuple[int, float]] = [
    (1, 1.0),     # Single review — maximum suspicion (P10)
    (4, 0.7),     # 2-4 reviews — high suspicion (P10-P25)
    (14, 0.4),    # 5-14 reviews — moderate suspicion (P25-P50)
    (50, 0.15),   # 15-50 reviews — low suspicion (P50-P75)
    (130, 0.05),  # 51-130 reviews — very low suspicion (P75-P90)
]
# 130+ reviews → 0.0 (power users, essentially unsuspicious)

# Migration note: The original log normalization (1 - log(1+count)/log(1+500))
# from Lim et al. (2010) was designed for Amazon product reviews where reviewers
# commonly post hundreds of reviews. On Google Maps restaurant data, the global
# review count distribution is compressed differently (median=14, P90=129),
# and log normalization assigned similar scores (0.7-0.9) to the majority of
# reviewers regardless of actual count. Tier-based scoring provides clear
# separation: a 1-review account (score=1.0) is now clearly distinguished from
# a median reviewer with 14 reviews (score=0.4).

# Maximum Number of Reviews per Day (MNR) Scoring (Lim et al. 2010)
# Lim et al. show 75% of spammers post 6+ reviews/day; 10+ is heavily suspicious
MNR_SCALE = 9  # 1 review/day = 0, 10+ reviews/day = 1

# Percentage Positive Reviews (PR) Scoring (Mukherjee et al. 2013)
# Binary extreme-value approach adapted for Google Maps:
# Original (Mukherjee 2013): 80% positive threshold for suspicion
# Google Maps reality: 82% of reviewers are 100% positive
# Adaptation: Score reduced to 0.3 for 100%-positive (platform normal)
# while keeping 1.0 for 0%-positive (rare negative-bombing)

# Rating Deviation Scoring (Savage et al. 2015, Moon et al. 2021)
# Spammers deviate significantly from restaurant mean and lack sub-rating logic
RD_DEVIATION_SCALE = 4  # Stars scale 1-5, normalize by ~4
RD_HOMOGENEITY_WEIGHT = 0.5  # Weight for sub-rating homogeneity component
RD_DEVIATION_WEIGHT = 0.5    # Weight for deviation component
MAX_SUB_RATING_STD = 2.0  # Normalize std deviation by 2 (max practical spread)

# Sub-Rating Consistency Scoring (Duma et al. 2023)
# Spammers cannot logically differentiate between aspect ratings
SUB_RATING_DISCREPANCY_SCALE = 2.0  # Stars 1-5, ±2 is max discrepancy

# Review Detail Level Scoring
# Level 0 = only stars (no effort) = suspicious
# Level 1 = with text (moderate effort)
# Level 2 = with sub-ratings (high effort/legitimacy)
DETAIL_LEVEL_MAX = 2

# Local Guide Level Normalization
LOCAL_GUIDE_LEVEL_MAX = 10  # Google uses 0-10 scale


def _calculate_reviewer_profile_score(df: pd.DataFrame) -> pd.Series:
    """
    Calculate normalized reviewer profile suspicion score.

    Combines five components: review count (tier-based scoring using percentile-derived
    thresholds), verification level, local guide status, photo review ratio, and review language.

    Args:
        df: DataFrame with columns [reviewer_review_count, reviewer_level,
            reviewer_is_local_guide, reviewer_photo_review_ratio,
            reviewer_avatar_url, review_language]

    Returns:
        Series of profile scores [0, 1] per review; index matches df.index.
    """
    scores = pd.DataFrame(index=df.index)

    # Component 1: Few reviews = suspicious (tier-based, data-driven)
    # Uses percentile-derived tiers instead of log normalization for better
    # discrimination between one-time accounts and casual users.
    review_counts = df["reviewer_review_count"]
    s_review_count = pd.Series(0.0, index=df.index)
    
    # Apply tiers from highest threshold down (first match wins)
    for max_count, score in reversed(REVIEW_COUNT_TIERS):
        s_review_count = s_review_count.where(
            review_counts > max_count, score
        )
    # Reviewers above highest tier threshold get 0.0 (already initialized)
    
    scores["s_review_count"] = s_review_count

    # Component 2: Missing or low local guide level = suspicious
    scores["s_level"] = (
        1 - (df["reviewer_level"].fillna(LOCAL_GUIDE_LEVEL_MAX) / LOCAL_GUIDE_LEVEL_MAX)
    ).clip(0, 1)

    # Component 3: Not a local guide = suspicious
    scores["s_local_guide"] = (~df["reviewer_is_local_guide"]).astype(float)

    # Component 4: Low photo-to-review ratio = suspicious
    scores["s_photo_ratio"] = (
        1 - (df["reviewer_photo_review_ratio"] / 2.0).clip(0, 1)
    )

    # Component 5: Non-German language for Würzburg = weak signal
    scores["s_language"] = (df["review_language"] != "de").astype(float) * 0.5

    # Weighted sum (already normalized since all components [0, 1])
    weights = REVIEWER_PROFILE_WEIGHTS
    profile_score = (
        scores["s_review_count"] * weights["s_review_count"]
        + scores["s_level"] * weights["s_level"]
        + scores["s_local_guide"] * weights["s_local_guide"]
        + scores["s_photo_ratio"] * weights["s_photo_ratio"]
        + scores["s_language"] * weights["s_language"]
    )

    return profile_score


def _calculate_mnr_score(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Maximum Number of Reviews (MNR) per day suspicion score.

    MNR is one of the most discriminative single features: Lim et al. (2010)
    show 75% of spammers post 6+ reviews per day.

    Args:
        df: DataFrame with columns [reviewer_user_id, timestamp_created_iso]

    Returns:
        Series of MNR scores [0, 1] per review; index matches df.index.
    """
    df_temp = df.copy()
    df_temp["review_date"] = df_temp["timestamp_created_iso"].dt.date

    # Per reviewer per day: count reviews
    reviews_per_day = (
        df_temp.groupby(["reviewer_user_id", "review_date"]).size().reset_index(name="count")
    )

    # Per reviewer: find max
    mnr_per_reviewer = (
        reviews_per_day.groupby("reviewer_user_id")["count"].max().reset_index()
    )
    mnr_per_reviewer.columns = ["reviewer_user_id", "mnr"]

    # Normalized score: 1 review = 0, 10+ reviews = 1
    mnr_per_reviewer["mnr_score"] = (
        ((mnr_per_reviewer["mnr"] - 1) / MNR_SCALE).clip(0, 1)
    )

    # Merge back to original dataframe
    mnr_scores = df.copy()[["reviewer_user_id"]].merge(
        mnr_per_reviewer[["reviewer_user_id", "mnr_score"]], on="reviewer_user_id"
    )

    return mnr_scores["mnr_score"].values


def _calculate_pr_score(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Percentage of Positive Reviews (PR) suspicion score using binary extreme-value approach.

    Uses a binary extreme-value approach instead of a continuous score.
    Empirical analysis of this dataset showed a strongly bimodal distribution:
    - 89.4% of reviewers post only positive reviews (pct=1.0)
    - 8.9% post only negative reviews (pct=0.0)
    - 1.7% are mixed

    Continuous approaches (absolute threshold from Mukherjee 2013,
    z-score, and percentile rank) did not provide enough discrimination
    (max. 13 unique score values for ~16k reviewers).
    
    Scoring logic:
        - pct_positive == 1.0 → pr_score = 0.3 (100% positive = weak suspicion,
            common on Google Maps - 82% of reviewers exhibit this pattern.
            Reduced from 0.6 to avoid score inflation for normal platform behavior.)
        - pct_positive == 0.0 -> pr_score = 1.0 (all-negative = high suspicion,
            classic negative-bombing pattern; rarer and more visible)
        - mixed reviews → pr_score = 0.0 (normal differentiated behavior)
    
    Negative-bombing reviewers (0% positive) are weighted higher (1.0)
    than positive-only reviewers (0.3) because all-negative behavior is rarer
    on Google Maps and therefore more suspicious.

    Methodological deviation is documented in the README section on
    platform-specific adaptations.

    Args:
        df: DataFrame with columns [reviewer_user_id, rating_stars]

    Returns:
        Series of PR scores [0, 1] per review; index matches df.index.
    """
    # Step 1: Compute pct_positive per reviewer.
    positive_mask = df["rating_stars"] >= 4
    df_temp = df[["reviewer_user_id"]].copy()
    df_temp["is_positive"] = positive_mask
    
    pct_positive = (
        df_temp.groupby("reviewer_user_id")["is_positive"]
        .mean()
        .reset_index()
    )
    pct_positive.columns = ["reviewer_user_id", "pct_positive"]

    # Step 2: Assign binary extreme-value score.
    # Scoring adapted for Google Maps positivity bias.
    # 82% of reviewers give exclusively positive reviews - this is normal
    # platform behavior, not a suspicion signal. Score reduced from 0.6 to 0.3
    # to avoid dominating the final suspicion score.
    #
    # Combined with review count for better discrimination:
    # A reviewer with 1 review and 100% positive is less suspicious than
    # a reviewer with 50 reviews and 100% positive - the latter has a
    # genuine track record. PR alone cannot distinguish these cases,
    # but the reduced weight (0.3) limits the damage from false positives.
    pct_positive["pr_score"] = 0.0  # Default: mixed reviews (normal behavior)
    pct_positive.loc[pct_positive["pct_positive"] == 1.0, "pr_score"] = 0.3
    pct_positive.loc[pct_positive["pct_positive"] == 0.0, "pr_score"] = 1.0  # all-negative reviews

    # Step 3: Merge back to original dataframe
    pr_scores = df.copy()[["reviewer_user_id"]].merge(
        pct_positive[["reviewer_user_id", "pr_score"]], on="reviewer_user_id"
    )

    # Step 4: Fill NaN with 0.0.
    pr_scores["pr_score"] = pr_scores["pr_score"].fillna(0.0)

    return pr_scores["pr_score"].values


def _calculate_rating_deviation_score(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Rating Deviation (RD) suspicion score.

    Combines absolute rating deviation from restaurant mean with sub-rating
    consistency (spammers cannot logically differentiate aspect ratings).
    
    IMPORTANT: Sub-rating homogeneity is only calculated for reviewers with ≥2
    reviews containing sub-ratings. Single-review users would have std=0 (only
    one value → no variance), falsely indicating "perfect consistency".
    Reviews from such reviewers use deviation score only (no homogeneity component).
    
    References: Savage et al. (2015), Moon et al. (2021), Duma et al. (2023).

    Args:
        df: DataFrame with columns [rating_stars, place_overall_rating,
            reviewer_user_id, sub_rating_food, sub_rating_service,
            sub_rating_atmosphere]

    Returns:
        Series of RD scores [0, 1] per review; index matches df.index.
    """
    # Part 1: Absolute deviation from restaurant mean
    deviation = np.abs(df["rating_stars"] - df["place_overall_rating"])
    deviation_score = (deviation / RD_DEVIATION_SCALE).clip(0, 1)

    # Part 2: Sub-rating reviewer homogeneity
    # ONLY for reviewers with >=2 reviews containing sub-ratings
    # Calculate intra-review std for each review
    sub_cols = ["sub_rating_food", "sub_rating_service", "sub_rating_atmosphere"]
    intra_review_std = df[sub_cols].std(axis=1, skipna=True)

    # Per reviewer: count how many reviews with sub-ratings they have
    df_temp = df[["reviewer_user_id"]].copy()
    df_temp["intra_std"] = intra_review_std
    df_temp["has_any_subrating"] = df[sub_cols].notna().any(axis=1)
    
    reviewer_subrating_counts = (
        df_temp.groupby("reviewer_user_id")["has_any_subrating"].sum().reset_index()
    )
    reviewer_subrating_counts.columns = ["reviewer_user_id", "subrating_count"]

    # Per reviewer: mean of intra-review stds (only if >=2 sub-rating reviews)
    reviewer_homogeneity = (
        df_temp.groupby("reviewer_user_id")["intra_std"]
        .mean()
        .reset_index()
    )
    reviewer_homogeneity.columns = ["reviewer_user_id", "mean_std"]

    # Join with counts and set NaN for reviewers with <2 sub-rating reviews
    reviewer_homogeneity = reviewer_homogeneity.merge(
        reviewer_subrating_counts, on="reviewer_user_id"
    )
    reviewer_homogeneity.loc[
        reviewer_homogeneity["subrating_count"] < 2, "mean_std"
    ] = np.nan

    # Normalize: low std (inconsistency) = suspicious, high std (varied) = good
    # NaN means insufficient data to assess homogeneity
    reviewer_homogeneity["homogeneity_score"] = (
        1 - (reviewer_homogeneity["mean_std"] / MAX_SUB_RATING_STD).clip(0, 1)
    )

    # Merge back
    homogeneity = df.copy()[["reviewer_user_id"]].merge(
        reviewer_homogeneity[["reviewer_user_id", "homogeneity_score"]],
        on="reviewer_user_id",
    )

    # Combined RD score with conditional logic:
    # - If homogeneity_score is available: weighted average (50% deviation, 50% homogeneity)
    # - If NaN (insufficient sub-rating data): use deviation score only
    has_homogeneity = homogeneity["homogeneity_score"].notna()
    
    rd_score = np.where(
        has_homogeneity,
        deviation_score * RD_DEVIATION_WEIGHT
        + homogeneity["homogeneity_score"].fillna(0.0).values * RD_HOMOGENEITY_WEIGHT,
        deviation_score,  # Only deviation if homogeneity unavailable
    )

    return rd_score


def _calculate_consistency_score(df: pd.DataFrame) -> pd.Series:
    """
    Calculate sub-rating consistency suspicion score.

    Detects illogical gaps between aspect ratings and overall star rating.
    Only applies to reviews with sub-ratings; others score 0.0.
    Reference: Duma et al. (2023).

    Args:
        df: DataFrame with columns [rating_stars, has_sub_ratings,
            sub_rating_food, sub_rating_service, sub_rating_atmosphere]

    Returns:
        Series of consistency scores [0, 1] per review; index matches df.index.
    """
    consistency_scores = np.zeros(len(df))

    # Only score reviews with sub-ratings
    has_sub = df["has_sub_ratings"].astype(bool)

    if has_sub.any():
        sub_cols = ["sub_rating_food", "sub_rating_service", "sub_rating_atmosphere"]
        sub_means = df.loc[has_sub, sub_cols].mean(axis=1, skipna=True)

        discrepancy = np.abs(
            df.loc[has_sub, "rating_stars"].values - sub_means.values
        )

        consistency_scores[has_sub] = (
            (discrepancy / SUB_RATING_DISCREPANCY_SCALE).clip(0, 1)
        )

    return pd.Series(consistency_scores, index=df.index)


def _calculate_detail_level_score(df: pd.DataFrame) -> pd.Series:
    """
    Calculate review detail level trust score.

    Level 0 (only stars, no effort) = 1.0 (suspicious)
    Level 1 (+ text, moderate effort) = 0.5
    Level 2 (+ sub-ratings, high effort) = 0.0 (trustworthy)

    Args:
        df: DataFrame with column [review_detail_level]

    Returns:
        Series of detail level scores [0, 1] per review; index matches df.index.
    """
    detail_score = (
        (DETAIL_LEVEL_MAX - df["review_detail_level"]) / DETAIL_LEVEL_MAX
    ).clip(0, 1)

    return detail_score


def run() -> None:
    """
    Run heuristic scoring pipeline end-to-end.

    1. Load clean parquet from config.PARQUET_PATH
    2. Calculate all suspicion scores (profile, MNR, PR, RD, consistency, detail-level)
    3. Save scores + original review_id to config.DATA_PROCESSED_DIR / "scores_heuristic.parquet"
    4. Log summary statistics
    """
    logger.info("Starting heuristic scoring pipeline...")

    # Load clean parquet
    input_path = config.PARQUET_PATH
    if not input_path.exists():
        logger.error(f"Input parquet not found: {input_path}")
        raise FileNotFoundError(f"Missing input: {input_path}")

    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df)} reviews from {input_path}")

    # Validate required columns
    required_cols = [
        "review_id",
        "reviewer_user_id",
        "rating_stars",
        "place_overall_rating",
        "reviewer_review_count",
        "reviewer_level",
        "reviewer_is_local_guide",
        "reviewer_photo_review_ratio",
        "reviewer_avatar_url",
        "review_language",
        "sub_rating_food",
        "sub_rating_service",
        "sub_rating_atmosphere",
        "has_sub_ratings",
        "review_detail_level",
        "timestamp_created_iso",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        logger.error(f"Missing columns: {missing}")
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure timestamp is datetime
    if df["timestamp_created_iso"].dtype == "object":
        df["timestamp_created_iso"] = pd.to_datetime(df["timestamp_created_iso"])

    # Calculate all scores
    logger.info("Calculating reviewer profile score...")
    df["reviewer_profile_score"] = _calculate_reviewer_profile_score(df)

    logger.info("Calculating MNR (max reviews per day) score...")
    df["mnr_score"] = _calculate_mnr_score(df)

    logger.info("Calculating PR (percentage positive) score...")
    df["pr_score"] = _calculate_pr_score(df)

    logger.info("Calculating RD (rating deviation) score...")
    df["rating_deviation_score"] = _calculate_rating_deviation_score(df)

    logger.info("Calculating consistency score...")
    df["consistency_score"] = _calculate_consistency_score(df)

    logger.info("Calculating detail level score...")
    df["detail_level_score"] = _calculate_detail_level_score(df)

    # Select output columns
    score_cols = [
        "review_id",
        "reviewer_profile_score",
        "mnr_score",
        "pr_score",
        "rating_deviation_score",
        "consistency_score",
        "detail_level_score",
    ]
    output_df = df[score_cols].copy()

    # Save to parquet
    output_path = config.DATA_PROCESSED_DIR / "scores_heuristic.parquet"
    output_df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(output_df)} score records to {output_path}")

    # Log summary statistics
    logger.info("=" * 70)
    logger.info("HEURISTIC SCORING SUMMARY STATISTICS")
    logger.info("=" * 70)
    for col in score_cols[1:]:  # Skip review_id
        mean_val = output_df[col].mean()
        median_val = output_df[col].median()
        std_val = output_df[col].std()
        logger.info(
            f"{col:30s} | Mean: {mean_val:.4f} | Median: {median_val:.4f} | Std: {std_val:.4f}"
        )
    logger.info("=" * 70)

    logger.info("Heuristic scoring pipeline completed successfully.")


if __name__ == "__main__":
    run()
