"""Temporal analysis for fake review detection.

This module implements temporal burst detection and co-bursting analysis,
which are among the most robust methods for identifying coordinated review
campaigns (Fei et al. 2013; Wang et al. 2022). Since genuine reviews typically
arrive randomly distributed over time, sudden bursts of activity are
statistically suspicious.

Reviews within burst windows that exhibit high rating homogeneity (especially
five-star ratings) and a high proportion of text-less reviews are classic
signals of purchased campaigns (Li et al. 2017). Co-bursting—when the same
reviewer groups appear simultaneously at different restaurants—is considered
near-certain evidence of professional spam networks (Ye et al. 2016; Fei et al. 2013).

PLATFORM-SPECIFIC ADAPTATIONS (Google Maps):
  Standard burst quality metrics (absolute pct_five_stars, pct_no_text) perform
  poorly on Google Maps data due to the platform's inherent positivity bias:
  the average restaurant has 73% five-star reviews (std=0.12, n=121 restaurants).
  
  Absolute scoring assigned a mean burst_suspicion_score of 0.55 — meaning
  virtually all bursts appeared moderately suspicious, regardless of whether
  they reflected genuine popularity spikes or purchased campaigns.
  
  This module uses RELATIVE burst scoring instead: the burst's five-star
  percentage is compared to the restaurant's historical baseline. A burst is
  only suspicious if its rating pattern deviates significantly from the
  restaurant's normal distribution. This eliminates false positives from
  restaurants that are simply popular.
  
  Additionally, a minimum daily review count (BURST_MIN_DAILY_REVIEWS=4) filters
  out low-volume days that reach high Z-scores purely from low baseline traffic.
  Empirical analysis showed days with 1-3 reviews have identical rating patterns
  (pct_five_stars=0.72-0.74) to non-burst days; only 6+ review days show
  meaningfully different patterns (pct_five_stars=0.85).

References:
    Fei, G., Mukherjee, A., Liu, B., Hsu, M., Castellanos, M., & Ghosh, R. (2013).
        Exploiting burstiness in reviews for review spammer detection.
    Li, H., Chen, Z., Mukherjee, A., Liu, B., & Shao, J. (2017).
        Analyzing and detecting opinion spam on a large-scale dataset via temporal
        and spatial patterns.
    Wang, X., et al. (2022). Temporal patterns in review spam detection.
    Ye, J., Kumar, S., & Akoglu, L. (2016). Temporal opinion spam detection
        by multivariate point process.
"""

import logging
from urllib.parse import unquote

import pandas as pd
import numpy as np
import duckdb
from joblib import Parallel, delayed

from scripts import config

# Configure logging
logger = logging.getLogger(__name__)

# Temporal analysis parameters
ROLLING_WINDOW_DAYS = 14  # Window size for rolling average
BURST_Z_THRESHOLD = 2.5   # Z-score threshold (p < 0.01)
BURST_MIN_DAILY_REVIEWS = 4  # Minimum reviews per day to qualify as burst candidate

# Empirical justification: Analysis of 30,244 restaurant-days showed that days
# with 1-3 reviews exhibit identical rating patterns to non-burst days
# (pct_five_stars: 0.72 vs 0.74). Only days with 6+ reviews show meaningfully
# different patterns (pct_five_stars: 0.85). Threshold of 4 is a conservative
# lower bound that filters noise while retaining signal.
# Without this filter, 27.1% of all reviews were flagged as "in burst" —
# far too many to be meaningful as a suspicion signal.

BURST_FIVE_STAR_WEIGHT = 0.6
BURST_NO_TEXT_WEIGHT = 0.4
CO_BURST_MIN_SHARED_REVIEWERS = 2  # Minimum shared reviewers for co-bursting
CO_BURST_MAX_DAY_GAP = 3  # Maximum days apart for temporal overlap
ATTR_MIN_FILL_RATE = 0.2  # Minimum 20% fill rate for attribute analysis

# Dining attribute columns for homogeneity analysis
ATTR_COLUMNS = [
    "attr_meal_type",
    "attr_price_range",
    "attr_noise_level",
    "attr_service_type",
    "attr_wait_time",
    "attr_group_size"
]


def _extract_restaurant_name(url: str) -> str:
    """Extract restaurant name from Google Maps URL.

    Args:
        url: Google Maps URL containing restaurant name.

    Returns:
        Extracted restaurant name (URL-decoded) or truncated URL if extraction fails.
    """
    try:
        # Google Maps URLs typically have format: .../data=.../1s<NAME>!2s...
        if "/data=" in url:
            parts = url.split("/data=")
            if len(parts) > 1:
                data_part = parts[1]
                # Look for !1s pattern (name marker)
                if "!1s" in data_part:
                    name_start = data_part.find("!1s") + 3
                    name_end = data_part.find("!", name_start)
                    if name_end > name_start:
                        raw_name = data_part[name_start:name_end]
                        return unquote(raw_name)
        # Fallback: return truncated URL
        return url[-50:] if len(url) > 50 else url
    except Exception:
        return url[-50:] if len(url) > 50 else url


def _aggregate_daily_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate reviews to daily timeseries per restaurant.

    Creates a complete date range for each restaurant with zero-filled missing days.
    This prevents bias in rolling statistics from irregular sampling.

    Args:
        df: DataFrame with reviews containing timestamp_created_iso and place_url.

    Returns:
        DataFrame with daily aggregates per restaurant, indexed by (place_url, review_date).
    """
    logger.info("Aggregating daily timeseries per restaurant...")

    # Use DuckDB for fast initial aggregation
    try:
        conn = duckdb.connect(':memory:')
        conn.register('reviews_df', df)
        
        query = """
            SELECT 
                place_url,
                CAST(timestamp_created_iso AS DATE) as review_date,
                COUNT(*) as review_count,
                AVG(rating_stars) as mean_stars,
                AVG(CASE WHEN rating_stars = 5 THEN 1.0 ELSE 0.0 END) as pct_five_stars,
                AVG(CASE WHEN has_review_text = false THEN 1.0 ELSE 0.0 END) as pct_no_text
            FROM reviews_df
            GROUP BY place_url, review_date
            ORDER BY place_url, review_date
        """
        daily = conn.execute(query).df()
        conn.close()
        
        # Convert review_date to date objects for consistency with pandas
        daily["review_date"] = pd.to_datetime(daily["review_date"]).dt.date
        
        logger.info(f"DuckDB aggregation complete: {len(daily)} restaurant-day combinations")
    except Exception as e:
        logger.warning(f"DuckDB aggregation failed, falling back to pandas: {e}")
        # Fallback to pandas
        df = df.copy()
        df["review_date"] = pd.to_datetime(df["timestamp_created_iso"]).dt.date
        agg_dict = {
            "review_id": "count",
            "rating_stars": ["mean", lambda x: (x == 5).mean()],
            "has_review_text": lambda x: (~x).mean()
        }
        daily = df.groupby(["place_url", "review_date"]).agg(agg_dict)
        daily.columns = ["review_count", "mean_stars", "pct_five_stars", "pct_no_text"]
        daily = daily.reset_index()

    # Add pct_edited if column exists (pandas-based, low cost)
    if "was_edited" in df.columns:
        df_copy = df.copy()
        df_copy["review_date"] = pd.to_datetime(df_copy["timestamp_created_iso"]).dt.date
        edited_agg = df_copy.groupby(["place_url", "review_date"])["was_edited"].mean().reset_index()
        daily = daily.merge(edited_agg, on=["place_url", "review_date"], how="left")
        daily.rename(columns={"was_edited": "pct_edited"}, inplace=True)

    # Fill missing days with zeros for each restaurant
    complete_daily = []
    for place_url, group in daily.groupby("place_url"):
        min_date = group["review_date"].min()
        max_date = group["review_date"].max()

        # Create complete date range
        date_range = pd.date_range(start=min_date, end=max_date, freq='D').date
        complete_dates = pd.DataFrame({
            "place_url": place_url,
            "review_date": date_range
        })

        # Merge and fill missing values
        merged = complete_dates.merge(group, on=["place_url", "review_date"], how="left")
        fill_values = {
            "review_count": 0,
            "mean_stars": 0,
            "pct_five_stars": 0,
            "pct_no_text": 0
        }
        if "pct_edited" in merged.columns:
            fill_values["pct_edited"] = 0
        merged = merged.fillna(fill_values)

        complete_daily.append(merged)

    result = pd.concat(complete_daily, ignore_index=True)
    logger.info(f"Created timeseries with {len(result)} restaurant-days")

    return result


def _detect_bursts_for_restaurant(place_url: str, group: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Detect burst windows for a single restaurant (parallel worker).

    Note: Days require a minimum of BURST_MIN_DAILY_REVIEWS reviews to qualify
    as burst candidates. This prevents low-traffic restaurants from generating
    false burst signals when a single additional review causes a high Z-score.

    Args:
        place_url: Restaurant URL identifier.
        group: Daily timeseries for this restaurant.

    Returns:
        Tuple of (processed dataframe, number of unique bursts).
    """
    group = group.sort_values("review_date").copy()

    # Rolling statistics (shift to avoid look-ahead bias)
    group["rolling_mean"] = (
        group["review_count"]
        .rolling(window=ROLLING_WINDOW_DAYS, min_periods=3)
        .mean()
        .shift(1)
    )
    group["rolling_std"] = (
        group["review_count"]
        .rolling(window=ROLLING_WINDOW_DAYS, min_periods=3)
        .std()
        .shift(1)
    )

    # Z-score (handle division by zero)
    group["z_score"] = 0.0
    mask = group["rolling_std"] > 0
    group.loc[mask, "z_score"] = (
        (group.loc[mask, "review_count"] - group.loc[mask, "rolling_mean"])
        / group.loc[mask, "rolling_std"]
    )

    # Identify burst days
    # Require minimum review volume to qualify as burst
    # Days with few reviews can have high Z-scores purely from low baselines
    group["is_burst_day"] = (
        (group["z_score"] > BURST_Z_THRESHOLD) &
        (group["review_count"] >= BURST_MIN_DAILY_REVIEWS)
    )

    # Group consecutive burst days into burst windows (cumsum trick)
    group["burst_id"] = (~group["is_burst_day"]).cumsum() * group["is_burst_day"]

    unique_bursts = group[group["burst_id"] > 0]["burst_id"].nunique() if (group["burst_id"] > 0).any() else 0
    
    return group, unique_bursts


def _detect_bursts(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Identify burst windows using Z-score analysis.

    A burst is defined as a day where review count exceeds the rolling mean
    by more than BURST_Z_THRESHOLD standard deviations. Consecutive burst days
    are grouped into burst windows.

    Args:
        daily_df: Daily aggregated timeseries per restaurant.

    Returns:
        DataFrame with burst indicators and IDs.
    """
    logger.info("Detecting burst windows...")

    # Prepare restaurant groups
    restaurant_groups = [(place_url, group) for place_url, group in daily_df.groupby("place_url")]
    
    # Parallel burst detection
    results = Parallel(n_jobs=-1, backend='threading')(
        delayed(_detect_bursts_for_restaurant)(place_url, group)
        for place_url, group in restaurant_groups
    )
    
    burst_results = []
    total_bursts = 0
    for (group, unique_bursts), (place_url, _) in zip(results, restaurant_groups):
        burst_results.append(group)
        if unique_bursts > 0:
            total_bursts += unique_bursts
            restaurant_name = _extract_restaurant_name(place_url)
            num_burst_days = (group["burst_id"] > 0).sum()
            logger.info(f"  {restaurant_name}: {unique_bursts} burst window(s), {num_burst_days} burst days")

    result = pd.concat(burst_results, ignore_index=True)
    logger.info(f"Total burst windows detected: {total_bursts}")

    return result


def _calculate_burst_quality(df: pd.DataFrame, daily_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate suspicion scores for burst windows using relative deviation.

    Burst quality is assessed by comparing the burst's rating distribution
    against the restaurant's historical baseline. This relative approach
    avoids false positives from Google Maps' inherent positivity bias
    (mean 73% five-star reviews across 121 restaurants).

    A burst at a restaurant with normally 80% five-star reviews that shows
    85% five-star reviews is unremarkable. The same burst at a restaurant
    with normally 50% five-star reviews is highly suspicious.

    Components:
    - five_star_deviation: burst pct_five_stars minus restaurant baseline (weight: 0.6)
    - no_text_deviation: burst pct_no_text minus restaurant baseline (weight: 0.4)
    Both clipped to [0, 1] — only positive deviations (more suspicious than normal)
    contribute to the score.

    Args:
        df: Original review DataFrame.
        daily_df: Daily timeseries with burst indicators.

    Returns:
        Tuple of (burst_windows DataFrame, reviews DataFrame with burst_id).
    """
    logger.info("Calculating burst quality scores...")

    # Merge burst info back to reviews
    df = df.copy()
    df["review_date"] = pd.to_datetime(df["timestamp_created_iso"]).dt.date
    df = df.merge(
        daily_df[["place_url", "review_date", "burst_id", "is_burst_day"]],
        on=["place_url", "review_date"],
        how="left"
    )
    df["burst_id"] = df["burst_id"].fillna(0)
    df["is_burst_day"] = df["is_burst_day"].fillna(False)

    # Calculate burst window statistics
    burst_windows = (
        df[df["burst_id"] > 0]
        .groupby(["place_url", "burst_id"])
        .agg(
            burst_review_count=("review_id", "count"),
            burst_pct_five_stars=("rating_stars", lambda x: (x == 5).mean()),
            burst_pct_no_text=("has_review_text", lambda x: (~x).mean())
        )
        .reset_index()
    )

    # Calculate restaurant baselines for relative scoring
    # A burst is only suspicious if its rating pattern DEVIATES from the
    # restaurant's normal pattern, not if it has high five-star rates in general.
    restaurant_baselines = (
        df.groupby("place_url")
        .agg(
            baseline_pct_five_stars=("rating_stars", lambda x: (x == 5).mean()),
            baseline_pct_no_text=("has_review_text", lambda x: (~x).mean()),
        )
        .reset_index()
    )

    burst_windows = burst_windows.merge(restaurant_baselines, on="place_url", how="left")

    # Relative deviation: how much does the burst deviate from normal?
    # Positive values = burst is MORE five-star-heavy / text-less than normal
    # Clipped to [0, 1] — only positive deviations are suspicious
    burst_windows["five_star_deviation"] = (
        burst_windows["burst_pct_five_stars"] - burst_windows["baseline_pct_five_stars"]
    ).clip(0, 1)

    burst_windows["no_text_deviation"] = (
        burst_windows["burst_pct_no_text"] - burst_windows["baseline_pct_no_text"]
    ).clip(0, 1)

    # Combined score using relative deviations
    burst_windows["burst_suspicion_score"] = (
        burst_windows["five_star_deviation"] * BURST_FIVE_STAR_WEIGHT +
        burst_windows["no_text_deviation"] * BURST_NO_TEXT_WEIGHT
    )

    logger.info(f"Calculated quality scores for {len(burst_windows)} burst windows")
    logger.info(f"Restaurant baseline pct_five_stars: mean={restaurant_baselines['baseline_pct_five_stars'].mean():.2f}")
    logger.info(f"Mean five_star_deviation in bursts: {burst_windows['five_star_deviation'].mean():.3f}")
    logger.info(f"Mean no_text_deviation in bursts: {burst_windows['no_text_deviation'].mean():.3f}")

    return burst_windows, df


def _calculate_attribute_homogeneity(
    df: pd.DataFrame,
    burst_windows: pd.DataFrame
) -> pd.DataFrame:
    """Calculate attribute homogeneity within burst windows.

    High homogeneity of dining attributes (meal type, price range, etc.) within
    a burst window corresponds to the Maximal Clique Structure (MCS) concept
    applied to structured data (Mukherjee et al. 2013).

    Args:
        df: Review DataFrame with burst_id and attribute columns.
        burst_windows: Burst window statistics.

    Returns:
        Updated burst_windows DataFrame with homogeneity scores.
    """
    logger.info("Calculating dining attribute homogeneity...")

    # Check which attribute columns have sufficient fill rate
    available_attrs = []
    for col in ATTR_COLUMNS:
        if col in df.columns:
            fill_rate = df[col].notna().mean()
            if fill_rate >= ATTR_MIN_FILL_RATE:
                available_attrs.append(col)
                logger.info(f"  Using {col} (fill rate: {fill_rate:.1%})")
            else:
                logger.info(f"  Skipping {col} (fill rate: {fill_rate:.1%} < {ATTR_MIN_FILL_RATE:.1%})")

    if not available_attrs:
        logger.warning("No attribute columns with sufficient fill rate, skipping homogeneity analysis")
        burst_windows["burst_attr_homogeneity"] = 0.0
        burst_windows["burst_suspicion_score_adjusted"] = burst_windows["burst_suspicion_score"]
        return burst_windows

    # Calculate homogeneity per burst window
    homogeneity_scores = []

    for _, burst in burst_windows.iterrows():
        burst_reviews = df[
            (df["place_url"] == burst["place_url"]) &
            (df["burst_id"] == burst["burst_id"])
        ]

        # Create attribute tuple for each review
        attr_tuples = burst_reviews[available_attrs].apply(tuple, axis=1)

        # Remove rows with all NaN attributes
        valid_attrs = attr_tuples[attr_tuples.apply(lambda x: not all(pd.isna(v) for v in x))]

        if len(valid_attrs) == 0:
            homogeneity = 0.0
        else:
            # Calculate mode frequency
            mode_count = valid_attrs.value_counts().iloc[0]
            homogeneity = mode_count / len(valid_attrs)

        homogeneity_scores.append(homogeneity)

    burst_windows["burst_attr_homogeneity"] = homogeneity_scores

    # Adjust suspicion score with attribute homogeneity
    burst_windows["burst_suspicion_score_adjusted"] = (
        burst_windows["burst_suspicion_score"] * 0.85 +
        burst_windows["burst_attr_homogeneity"] * 0.15
    )

    logger.info(f"Calculated attribute homogeneity for {len(burst_windows)} bursts")

    return burst_windows


def _check_coburst_for_restaurant_pair(
    restaurant1: str,
    restaurant2: str,
    burst_dict: dict
) -> tuple[set, int]:
    """Check co-bursting between two restaurants (parallel worker).

    Compares all burst windows of restaurant1 against all burst windows
    of restaurant2 to find shared reviewers in temporally overlapping bursts.

    Args:
        restaurant1: First restaurant URL.
        restaurant2: Second restaurant URL.
        burst_dict: Dictionary mapping restaurant_url to list of burst windows.

    Returns:
        Tuple of (set of co-bursting reviewers, number of co-burst pairs found).
    """
    if restaurant1 not in burst_dict or restaurant2 not in burst_dict:
        return set(), 0
    
    bursts1 = burst_dict[restaurant1]
    bursts2 = burst_dict[restaurant2]
    
    co_burst_reviewers_local = set()
    co_burst_count = 0
    
    # Check all burst combinations between these two restaurants
    for burst1 in bursts1:
        for burst2 in bursts2:
            # Check temporal overlap (within CO_BURST_MAX_DAY_GAP days)
            gap1 = abs((burst1["min_date"] - burst2["max_date"]).days)
            gap2 = abs((burst2["min_date"] - burst1["max_date"]).days)
            gap3 = abs((burst1["min_date"] - burst2["min_date"]).days)
            gap4 = abs((burst1["max_date"] - burst2["max_date"]).days)
            min_gap = min(gap1, gap2, gap3, gap4)

            # Check for overlap
            overlap = not (burst1["max_date"] < burst2["min_date"] or 
                          burst2["max_date"] < burst1["min_date"])
            
            if not overlap and min_gap > CO_BURST_MAX_DAY_GAP:
                continue

            # Check shared reviewers
            shared = burst1["reviewers"] & burst2["reviewers"]
            if len(shared) >= CO_BURST_MIN_SHARED_REVIEWERS:
                co_burst_reviewers_local.update(shared)
                co_burst_count += 1
    
    return co_burst_reviewers_local, co_burst_count


def _detect_co_bursting(df: pd.DataFrame, burst_windows: pd.DataFrame) -> pd.DataFrame:
    """Detect co-bursting patterns across restaurants.

    Co-bursting occurs when the same reviewers appear in burst windows at
    different restaurants within a short time period. This is near-certain
    evidence of professional spam networks (Ye et al. 2016; Fei et al. 2013).

    Args:
        df: Review DataFrame with burst_id and review_date.
        burst_windows: Burst window statistics.

    Returns:
        DataFrame with reviewer_user_id and in_co_burst flag.
    """
    logger.info("Detecting co-bursting patterns...")

    # Extract reviewer sets per burst window with date ranges, grouped by restaurant
    burst_windows_by_restaurant = {}
    for _, burst in burst_windows.iterrows():
        burst_reviews = df[
            (df["place_url"] == burst["place_url"]) &
            (df["burst_id"] == burst["burst_id"])
        ]

        if len(burst_reviews) == 0:
            continue

        reviewer_set = set(burst_reviews["reviewer_user_id"].unique())
        min_date = burst_reviews["review_date"].min()
        max_date = burst_reviews["review_date"].max()

        burst_info = {
            "burst_id": burst["burst_id"],
            "reviewers": reviewer_set,
            "min_date": pd.to_datetime(min_date),
            "max_date": pd.to_datetime(max_date)
        }
        
        place_url = burst["place_url"]
        if place_url not in burst_windows_by_restaurant:
            burst_windows_by_restaurant[place_url] = []
        burst_windows_by_restaurant[place_url].append(burst_info)

    # Generate restaurant pairs (not burst-window pairs!)
    from itertools import combinations
    restaurant_ids = list(burst_windows_by_restaurant.keys())
    pairs = list(combinations(restaurant_ids, 2))
    
    logger.info(f"Checking {len(pairs)} restaurant pairs for co-bursting...")

    # Parallel co-burst detection across restaurant pairs
    results = Parallel(n_jobs=-1, backend='threading')(
        delayed(_check_coburst_for_restaurant_pair)(r1, r2, burst_windows_by_restaurant)
        for r1, r2 in pairs
    )
    
    # Collect results
    co_burst_reviewers = set()
    co_burst_pairs = 0
    
    for (shared, coburst_count), (r1, r2) in zip(results, pairs):
        if coburst_count > 0:
            co_burst_reviewers.update(shared)
            co_burst_pairs += coburst_count
            name1 = _extract_restaurant_name(r1)
            name2 = _extract_restaurant_name(r2)
            logger.info(
                f"  Co-burst detected: {len(shared)} shared reviewers between "
                f"{name1} and {name2} ({coburst_count} burst pairs)"
            )

    logger.info(
        f"Found {co_burst_pairs} co-burst pairs affecting "
        f"{len(co_burst_reviewers)} reviewers"
    )

    # Create result DataFrame
    if len(co_burst_reviewers) > 0:
        result = pd.DataFrame({
            "reviewer_user_id": list(co_burst_reviewers),
            "in_co_burst": True
        })
    else:
        result = pd.DataFrame(columns=["reviewer_user_id", "in_co_burst"])

    return result


def _build_temporal_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute temporal signals for each review.

    This function runs all temporal analysis stages and returns one row per
    ``review_id`` with temporal indicators.

    Args:
        df: Cleaned reviews DataFrame.

    Returns:
        DataFrame with columns:
            - review_id
            - in_burst
            - burst_suspicion_score
            - burst_attr_homogeneity
            - in_co_burst
    """
    # 3.1: Aggregate daily timeseries
    daily_df = _aggregate_daily_timeseries(df)

    # 3.2: Detect burst windows
    daily_df = _detect_bursts(daily_df)

    # 3.3: Calculate burst quality scores
    burst_windows, df_with_burst = _calculate_burst_quality(df, daily_df)

    # 3.4: Calculate attribute homogeneity
    burst_windows = _calculate_attribute_homogeneity(df_with_burst, burst_windows)

    # Merge burst scores back to reviews
    df_with_burst = df_with_burst.merge(
        burst_windows[[
            "place_url",
            "burst_id",
            "burst_suspicion_score_adjusted",
            "burst_attr_homogeneity",
        ]],
        on=["place_url", "burst_id"],
        how="left",
    )

    df_with_burst["in_burst"] = df_with_burst["burst_id"] > 0
    df_with_burst["burst_suspicion_score"] = df_with_burst[
        "burst_suspicion_score_adjusted"
    ].fillna(0.0)
    df_with_burst["burst_attr_homogeneity"] = df_with_burst[
        "burst_attr_homogeneity"
    ].fillna(0.0)

    # 3.5: Detect co-bursting
    co_burst_df = _detect_co_bursting(df_with_burst, burst_windows)
    df_with_burst = df_with_burst.merge(co_burst_df, on="reviewer_user_id", how="left")
    df_with_burst["in_co_burst"] = df_with_burst["in_co_burst"].fillna(False)

    output_df = df_with_burst[
        [
            "review_id",
            "in_burst",
            "burst_suspicion_score",
            "burst_attr_homogeneity",
            "in_co_burst",
        ]
    ].copy()

    return output_df


def calculate_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate temporal fake-review signals and persist score parquet.

    This is the entrypoint used by ``flows/pipeline.py`` in Phase 5.

    Args:
        df: Cleaned reviews DataFrame loaded from ``config.PARQUET_PATH``.

    Returns:
        Input DataFrame enriched with temporal feature columns.
    """
    logger.info("Running temporal feature calculation...")

    temporal_scores = _build_temporal_scores(df)

    output_path = config.DATA_PROCESSED_DIR / "scores_temporal.parquet"
    temporal_scores.to_parquet(output_path, index=False)
    logger.info(f"Saved temporal scores to {output_path}")

    enriched = df.merge(temporal_scores, on="review_id", how="left")
    enriched["in_burst"] = enriched["in_burst"].fillna(False)
    enriched["in_co_burst"] = enriched["in_co_burst"].fillna(False)
    enriched["burst_suspicion_score"] = enriched["burst_suspicion_score"].fillna(0.0)
    enriched["burst_attr_homogeneity"] = enriched["burst_attr_homogeneity"].fillna(0.0)

    return enriched


def run() -> None:
    """Run temporal analysis pipeline.

    Performs burst detection, quality scoring, attribute homogeneity analysis,
    and co-bursting detection. Saves results to scores_temporal.parquet.
    """
    logger.info("=" * 80)
    logger.info("Starting Temporal Analysis")
    logger.info("=" * 80)

    # Load data
    logger.info(f"Loading data from {config.PARQUET_PATH}")
    df = pd.read_parquet(config.PARQUET_PATH)
    logger.info(f"Loaded {len(df)} reviews")

    output_df = _build_temporal_scores(df)

    # Save results
    output_path = config.DATA_PROCESSED_DIR / "scores_temporal.parquet"
    output_df.to_parquet(output_path, index=False)
    logger.info(f"Saved temporal scores to {output_path}")

    # Log summary statistics
    logger.info("=" * 80)
    logger.info("Temporal Analysis Summary")
    logger.info("=" * 80)
    logger.info(f"Reviews in bursts: {output_df['in_burst'].sum()} ({output_df['in_burst'].mean():.1%})")
    logger.info(f"Reviews in co-bursts: {output_df['in_co_burst'].sum()} ({output_df['in_co_burst'].mean():.1%})")
    
    burst_reviews = output_df[output_df['in_burst']]
    if len(burst_reviews) > 0:
        logger.info(f"Mean burst suspicion score: {burst_reviews['burst_suspicion_score'].mean():.3f}")
        logger.info(f"Mean attribute homogeneity: {burst_reviews['burst_attr_homogeneity'].mean():.3f}")
    else:
        logger.info("No bursts detected in dataset")
    
    logger.info("=" * 80)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    run()
