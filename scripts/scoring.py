"""Scoring module for fake review detection.

Combines evidence from heuristic, temporal, network, and semantic analyses into
a single weighted suspicion score. The score is literature-backed and reflects
the relative reliability of different detection methods.

Weight hierarchy (by detection method reliability):
1. Network analysis (35%): Content-independent, reveals coordinated behavior.
    Hardest to manipulate. (Rayana & Akoglu 2015)
2. Temporal analysis (15%): Burst detection with relative scoring.
    (Fei et al. 2013)
3. Behavioral heuristics (35%): Rating deviation, consistency, review patterns.
    (Lim et al. 2010, Mukherjee et al. 2013, Savage et al. 2015)
4. Semantic/NLP (15%): Text similarity, fact contradictions, sentiment.
    Lowest reliability. (Mohawesh et al. 2021, Ott et al. 2011)

Suspicion threshold is calculated dynamically as the 90th percentile of the
score distribution, adapting to changes in scoring methodology. Literature
suggests 10-30% fake review rates (Mukherjee et al. 2013, Jindal & Liu 2008).

Generates comprehensive visualizations and summary tables for reporting.
"""

import logging
import os
import hashlib
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from scripts import config

# Configure logging
logger = logging.getLogger(__name__)

_HIGH_SUSPICION_THRESHOLD: float | None = None
_VISUALIZATION_THRESHOLD: float | None = None

# Literature-backed score weights (sum must equal 1.0)
SCORE_WEIGHTS = {
    # --- Network & Temporal (50% total) ---
    # Highest reliability: content-independent, reveals coordinated behavior
    # (Rayana & Akoglu 2015, Fei et al. 2013)
    "network_score": 0.35,                   # Network analysis - strongest signal
    "burst_suspicion_score": 0.15,           # Temporal bursts - reduced from 0.20

    # --- Behavioral / Heuristic (35% total) ---
    # Second most reliable: reviewer behavior patterns
    # (Lim et al. 2010, Mukherjee et al. 2013, Savage et al. 2015)
    "rating_deviation_score": 0.10,          # Deviation from restaurant mean - raised
    "consistency_score": 0.08,               # Sub-rating consistency - slight raise
    "mnr_score": 0.07,                       # Max reviews per day
    "pr_score": 0.05,                        # Percentage positive - reduced due to positivity bias
    "detail_level_score": 0.05,              # Review detail level

    # --- Semantic / NLP (15% total) ---
    # Lowest reliability but complementary signal
    # (Mohawesh et al. 2021, Ott et al. 2011)
    "mcs_score": 0.06,                       # Maximum Content Similarity - raised
    "fact_compatibility_score": 0.03,        # Text-attribute contradictions - reduced
    "attr_template_score": 0.02,             # Disabled but kept for compatibility
    "sentiment_star_inconsistency": 0.02,    # Sentiment-rating mismatch - slight raise
    "sentiment_subrating_inconsistency": 0.02,  # Sentiment-subrating mismatch
}

# Validate weights sum to 1.0
assert abs(sum(SCORE_WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

# Weight calibration notes (based on empirical score distributions):
# - network_score (mean 0.143): Primary driver, good right-skew distribution
# - burst_suspicion_score (mean 0.006): Low after relative scoring fix,
#   but high weight justified because non-zero values are strong signals
# - pr_score (mean 0.581): Poor discrimination on Google Maps (positivity bias),
#   weight intentionally low despite high mean
# - fact_compatibility_score (mean 0.0004): Near-zero signal, weight minimal
# - attr_template_score (mean 0.0): Disabled, weight kept for compatibility

# Suspicion thresholds
# Suspicion threshold is calculated dynamically as the 90th percentile
# of the score distribution. This adapts to changes in scoring methodology
# and dataset composition, unlike a static threshold which becomes stale
# after pipeline modifications.
#
# Literature basis: Empirical fake review rates range from 10-30%
# (Mukherjee et al. 2013, Jindal & Liu 2008). The 90th percentile
# (top 10%) is a conservative lower bound.
#
# The threshold is computed at runtime in _calculate_suspicion_scores()
# and logged for reproducibility.
HIGH_SUSPICION_PERCENTILE = 90  # Top 10% flagged as high suspicion
VISUALIZATION_PERCENTILE = 95   # Top 5% shown in network graph


def _anonymize_sensitive_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Anonymize reviewer-related personal identifiers for exported artifacts."""
    anonymize_enabled = os.environ.get("ANONYMIZE_EXPORTS", "true").lower() in {
        "1", "true", "yes", "on"
    }
    if not anonymize_enabled:
        return df

    salt = os.environ.get("ANONYMIZATION_SALT", "change-me")
    out = df.copy()

    def _hash_token(value: object, prefix: str) -> str | object:
        if pd.isna(value):
            return value
        digest = hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()[:16]
        return f"{prefix}_{digest}"

    if "reviewer_user_id" in out.columns:
        out["reviewer_user_id"] = out["reviewer_user_id"].map(
            lambda v: _hash_token(v, "uid")
        )

    if "reviewer_name" in out.columns:
        if "reviewer_user_id" in out.columns:
            out["reviewer_name"] = out["reviewer_user_id"].map(
                lambda v: f"reviewer_{str(v).split('_')[-1][:10]}" if pd.notna(v) else v
            )
        else:
            out["reviewer_name"] = out["reviewer_name"].map(
                lambda v: _hash_token(v, "reviewer")
            )

    # Remove direct profile identifiers entirely from exported files.
    for col in ["reviewer_profile_url", "reviewer_avatar_url"]:
        if col in out.columns:
            out[col] = pd.NA

    return out


def _load_all_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Load and merge all score parquets.

    Uses LEFT JOIN on review_id to preserve all reviews even if some
    scoring phases failed. Missing scores are filled with 0.0.

    Args:
        df: Base review DataFrame.

    Returns:
        DataFrame with all scores merged.
    """
    logger.info("Loading all score parquets...")

    result = df[["review_id"]].copy()

    # Load heuristic scores
    heuristic_path = config.DATA_PROCESSED_DIR / "scores_heuristic.parquet"
    if heuristic_path.exists():
        heuristic_df = pd.read_parquet(heuristic_path)
        result = result.merge(heuristic_df, on="review_id", how="left")
        logger.info(f"Loaded heuristic scores: {len(heuristic_df)} reviews")
    else:
        logger.warning(f"Heuristic scores not found: {heuristic_path}")

    # Load temporal scores
    temporal_path = config.DATA_PROCESSED_DIR / "scores_temporal.parquet"
    if temporal_path.exists():
        temporal_df = pd.read_parquet(temporal_path)
        result = result.merge(temporal_df, on="review_id", how="left")
        logger.info(f"Loaded temporal scores: {len(temporal_df)} reviews")
    else:
        logger.warning(f"Temporal scores not found: {temporal_path}")

    # Load network scores
    network_path = config.DATA_PROCESSED_DIR / "scores_network.parquet"
    if network_path.exists():
        network_df = pd.read_parquet(network_path)
        result = result.merge(network_df, on="review_id", how="left")
        logger.info(f"Loaded network scores: {len(network_df)} reviews")
    else:
        logger.warning(f"Network scores not found: {network_path}")

    # Load semantic scores
    semantic_path = config.DATA_PROCESSED_DIR / "scores_semantic.parquet"
    if semantic_path.exists():
        semantic_df = pd.read_parquet(semantic_path)
        result = result.merge(semantic_df, on="review_id", how="left")
        logger.info(f"Loaded semantic scores: {len(semantic_df)} reviews")
    else:
        logger.warning(f"Semantic scores not found: {semantic_path}")

    # Fill missing scores with 0.0
    for weight_key in SCORE_WEIGHTS.keys():
        if weight_key not in result.columns:
            result[weight_key] = 0.0
        result[weight_key] = result[weight_key].fillna(0.0)

    return result


def _calculate_suspicion_scores(df_scores: pd.DataFrame) -> pd.DataFrame:
    """Calculate weighted suspicion scores.

    Combines all score components using literature-backed weights.

    Args:
        df_scores: DataFrame with all component scores.

    Returns:
        DataFrame with suspicion_score added.
    """
    logger.info("Calculating weighted suspicion scores...")

    df_scores = df_scores.copy()

    # Calculate suspicion score as weighted sum
    df_scores["suspicion_score"] = 0.0

    for component, weight in SCORE_WEIGHTS.items():
        df_scores["suspicion_score"] += df_scores[component] * weight

    # Clip to [0, 1]
    df_scores["suspicion_score"] = df_scores["suspicion_score"].clip(0, 1)

    logger.info(f"Calculated suspicion scores for {len(df_scores)} reviews")
    logger.info(f"  Mean: {df_scores['suspicion_score'].mean():.3f}")
    logger.info(f"  Median: {df_scores['suspicion_score'].median():.3f}")
    logger.info(f"  Max: {df_scores['suspicion_score'].max():.3f}")

    # Calculate dynamic threshold from score distribution
    high_suspicion_threshold = df_scores["suspicion_score"].quantile(
        HIGH_SUSPICION_PERCENTILE / 100
    )
    visualization_threshold = df_scores["suspicion_score"].quantile(
        VISUALIZATION_PERCENTILE / 100
    )

    # Store thresholds on DataFrame for downstream use
    df_scores.attrs["high_suspicion_threshold"] = high_suspicion_threshold
    df_scores.attrs["visualization_threshold"] = visualization_threshold

    logger.info(f"  High suspicion threshold (P{HIGH_SUSPICION_PERCENTILE}): {high_suspicion_threshold:.3f}")
    logger.info(f"  Visualization threshold (P{VISUALIZATION_PERCENTILE}): {visualization_threshold:.3f}")
    high_count = (df_scores["suspicion_score"] > high_suspicion_threshold).sum()
    logger.info(f"  High suspicion reviews: {high_count} ({high_count / len(df_scores) * 100:.1f}%)")

    return df_scores


def _plot_score_distribution(df_scores: pd.DataFrame) -> None:
    """Plot histogram of suspicion scores.

    Shows distribution and highlights high suspicion threshold.
    """
    logger.info("Creating score distribution plot...")

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid", palette="muted")

    plt.hist(df_scores["suspicion_score"], bins=50, alpha=0.7, color="steelblue", edgecolor="black")
    plt.axvline(
        _HIGH_SUSPICION_THRESHOLD,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"High Suspicion (P{HIGH_SUSPICION_PERCENTILE}: {_HIGH_SUSPICION_THRESHOLD:.3f})"
    )

    plt.xlabel("Suspicion Score", fontsize=12, fontweight="bold")
    plt.ylabel("Number of Reviews", fontsize=12, fontweight="bold")
    plt.title("Distribution of Review Suspicion Scores", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.tight_layout()

    output_path = config.PLOTS_DIR / "01_score_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved plot to {output_path}")
    plt.close()


def _plot_restaurant_ranking(df: pd.DataFrame) -> None:
    """Plot top restaurants by mean suspicion score.

    Horizontal bar chart ranked by mean suspicion score.
    """
    logger.info("Creating restaurant ranking plot...")

    # Calculate mean scores by restaurant
    restaurant_scores = (
        df.groupby("place_name")["suspicion_score"]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
        .head(20)
    )

    # Create color map based on scores
    colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(restaurant_scores)))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.set_theme(style="whitegrid", palette="muted")

    bars = ax.barh(restaurant_scores["place_name"], restaurant_scores["mean"], color=colors)
    ax.set_xlabel("Mean Suspicion Score", fontsize=12, fontweight="bold")
    ax.set_title("Top 20 Restaurants by Mean Suspicion Score", fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    # Add count labels
    for i, (idx, row) in enumerate(restaurant_scores.iterrows()):
        ax.text(row["mean"] + 0.01, i, f"n={int(row['count'])}", va="center", fontsize=9)

    plt.tight_layout()

    output_path = config.PLOTS_DIR / "02_restaurant_ranking.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved plot to {output_path}")
    plt.close()


def _plot_age_vs_score(df: pd.DataFrame) -> None:
    """Plot review age vs. suspicion score.

    Shows temporal relationship between review age and suspicion.
    """
    logger.info("Creating age vs score plot...")

    df_plot = df[["timestamp_created_iso", "suspicion_score"]].copy()
    df_plot["timestamp_created_iso"] = pd.to_datetime(df_plot["timestamp_created_iso"])
    df_plot["age_days"] = (datetime.now(timezone.utc) - df_plot["timestamp_created_iso"]).dt.days

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid", palette="muted")

    plt.scatter(df_plot["age_days"], df_plot["suspicion_score"], alpha=0.3, s=20, color="steelblue")
    plt.xlabel("Review Age (days)", fontsize=12, fontweight="bold")
    plt.ylabel("Suspicion Score", fontsize=12, fontweight="bold")
    plt.title("Review Age vs. Suspicion Score", fontsize=14, fontweight="bold")

    plt.tight_layout()

    output_path = config.PLOTS_DIR / "03_age_vs_score.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved plot to {output_path}")
    plt.close()


def _plot_score_components_heatmap(df: pd.DataFrame) -> None:
    """Plot score component breakdown by restaurant.

    Heatmap showing which signals dominate for suspicious restaurants.
    """
    logger.info("Creating score components heatmap...")

    # Get top 20 restaurants by mean suspicion
    top_restaurants = (
        df.groupby("place_name")["suspicion_score"]
        .mean()
        .nlargest(20)
        .index
    )

    df_top = df[df["place_name"].isin(top_restaurants)].copy()

    # Calculate mean component scores per restaurant
    component_cols = [k for k in SCORE_WEIGHTS.keys() if k in df_top.columns]
    component_means = df_top.groupby("place_name")[component_cols].mean()
    component_means = component_means.loc[top_restaurants]  # Preserve order

    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid", palette="muted")

    sns.heatmap(
        component_means,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        cbar_kws={"label": "Mean Score"},
        linewidths=0.5
    )

    plt.title("Score Component Breakdown by Restaurant (Top 20)", fontsize=14, fontweight="bold")
    plt.xlabel("Score Component", fontsize=12, fontweight="bold")
    plt.ylabel("Restaurant", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    output_path = config.PLOTS_DIR / "04_score_components_heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved plot to {output_path}")
    plt.close()


def _plot_network_graph(df: pd.DataFrame) -> None:
    """Plot reviewer-restaurant network for suspicious reviews.

    Network visualization showing connections for reviews above threshold.
    """
    logger.info("Creating network graph plot...")

    # Try to load existing graph
    graph_path = config.OUTPUTS_DIR / "reviewer_restaurant_graph.graphml"
    if graph_path.exists():
        try:
            G = nx.read_graphml(graph_path)
            logger.info("Loaded pre-computed graph")
        except Exception as e:
            logger.warning(f"Could not load graph: {e}")
            G = None
    else:
        G = None

    # If no graph exists, build it from suspicious reviews
    if G is None:
        logger.info("Building graph from suspicious reviews...")
        G = nx.Graph()

        # Filter suspicious reviews
        suspicious = df[df["suspicion_score"] > _VISUALIZATION_THRESHOLD].copy()

        # Add nodes and edges
        for _, row in suspicious.iterrows():
            reviewer_node = f"r_{row['reviewer_user_id']}"
            restaurant_node = f"p_{row['place_url']}"

            G.add_node(reviewer_node, node_type="reviewer", name=row["reviewer_name"])
            G.add_node(restaurant_node, node_type="restaurant", name=row["place_name"])
            G.add_edge(reviewer_node, restaurant_node, suspicion_score=row["suspicion_score"])

    # Separate nodes by type
    reviewer_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "reviewer"]
    restaurant_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "restaurant"]

    # Calculate layout
    n_nodes = G.number_of_nodes()
    if n_nodes > 500:
        # Graph is too large for a readable spring layout; plot top nodes only.
        top_nodes = sorted(G.nodes(), 
                           key=lambda x: G.degree(x), 
                           reverse=True)[:200]
        G = G.subgraph(top_nodes).copy()
        # Recompute node groups after reducing the graph.
        reviewer_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "reviewer"]
        restaurant_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "restaurant"]
    
    logger.info(f"Plotting network graph with {G.number_of_nodes()} nodes")
    pos = nx.spring_layout(G, k=2, iterations=20, seed=42)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.set_theme(style="white", palette="muted")

    # Get suspicion scores for coloring
    if len(reviewer_nodes) > 0:
        reviewer_suspicions = []
        for node in reviewer_nodes:
            # Find max suspicion among edges from this reviewer
            edges = G.edges(node, data=True)
            max_susp = max([d.get("suspicion_score", 0) for _, _, d in edges]) if edges else 0
            reviewer_suspicions.append(max_susp)

        # Color map: green (0) to red (1)
        colors = plt.cm.RdYlGn_r(np.array(reviewer_suspicions))
    else:
        colors = ["gray"] * len(reviewer_nodes)

    # Draw restaurants (large gray circles)
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=restaurant_nodes,
        node_color="lightgray",
        node_size=3000,
        node_shape="o",
        ax=ax
    )

    # Draw reviewers (small colored circles)
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=reviewer_nodes,
        node_color=colors if len(reviewer_nodes) > 0 else "gray",
        node_size=500,
        node_shape="o",
        cmap=None,
        ax=ax
    )

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax)

    # Add labels (only for restaurants due to crowding)
    # Prefer human-readable place_name and never display full URL node IDs.
    place_name_by_url = {}
    if {"place_url", "place_name"}.issubset(df.columns):
        place_name_by_url = (
            df[["place_url", "place_name"]]
            .dropna()
            .drop_duplicates(subset=["place_url"])
            .set_index("place_url")["place_name"]
            .to_dict()
        )

    def _short_label(node_id: str) -> str:
        node_data = G.nodes[node_id]
        # 1) Explicit label from graph metadata.
        label = node_data.get("name") or node_data.get("place_name")
        if isinstance(label, str) and label.lower().startswith(("http://", "https://", "www.")):
            label = None
        # 2) Fallback via place_url mapping.
        if not label and node_id.startswith("p_"):
            place_url = node_id[2:]
            label = place_name_by_url.get(place_url)
        # 3) Last fallback: generic placeholder, never the raw URL.
        if not label:
            label = "restaurant"
        label = str(label)
        if len(label) > 34:
            return label[:31] + "..."
        return label

    restaurant_labels = {n: _short_label(n) for n in restaurant_nodes}
    nx.draw_networkx_labels(
        G,
        pos,
        labels=restaurant_labels,
        font_size=8,
        font_weight="bold",
        ax=ax
    )

    plt.title("Reviewer-Restaurant Network (Suspicious Reviews)", fontsize=14, fontweight="bold")
    plt.axis("off")

    plt.tight_layout()

    output_path = config.PLOTS_DIR / "05_reviewer_graph.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved plot to {output_path}")
    plt.close()


def _export_top_suspicious_reviews(df: pd.DataFrame) -> None:
    """Export top 50 most suspicious reviews as CSV.

    Args:
        df: Full review DataFrame with suspicion scores.
    """
    logger.info("Exporting top 50 suspicious reviews...")

    output_df = (
        df[["place_name", "reviewer_name", "rating_stars", "review_text",
            "suspicion_score", "network_score", "burst_suspicion_score",
            "mnr_score", "pr_score", "mcs_score", "timestamp_created_iso"]]
        .nlargest(50, "suspicion_score")
        .copy()
    )

    # Keep full review text in final exports.
    output_df["review_text"] = output_df["review_text"].fillna("")

    output_path = config.OUTPUTS_DIR / "top50_suspicious.csv"
    output_df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Saved to {output_path}")


def _export_restaurant_summary(df: pd.DataFrame) -> None:
    """Export restaurant-level summary statistics.

    Args:
        df: Full review DataFrame with suspicion scores.
    """
    logger.info("Exporting restaurant summary...")

    burst_cols = [c for c in ["in_burst", "in_co_burst"] if c in df.columns]

    agg_dict = {
        "total_reviews": ("review_id", "count"),
        "mean_suspicion_score": ("suspicion_score", "mean"),
        "pct_high_suspicion": (
            "suspicion_score",
            lambda x: (x > _HIGH_SUSPICION_THRESHOLD).mean()
        ),
    }

    if "in_burst" in burst_cols:
        agg_dict["pct_in_burst"] = ("in_burst", lambda x: x.fillna(False).mean())

    if "in_co_burst" in burst_cols:
        agg_dict["pct_in_co_burst"] = ("in_co_burst", lambda x: x.fillna(False).mean())

    restaurant_summary = (
        df.groupby(["place_name", "place_url"])
        .agg(**agg_dict)
        .reset_index()
        .sort_values("mean_suspicion_score", ascending=False)
    )

    if "pct_in_burst" not in restaurant_summary.columns:
        restaurant_summary["pct_in_burst"] = 0.0

    if "pct_in_co_burst" not in restaurant_summary.columns:
        restaurant_summary["pct_in_co_burst"] = 0.0

    output_path = config.OUTPUTS_DIR / "restaurant_summary.csv"
    restaurant_summary.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Saved to {output_path}")


def _export_all_reviews_with_scores(df: pd.DataFrame) -> None:
    """Export all reviews with all score columns.

    Args:
        df: Full review DataFrame with suspicion scores.
    """
    logger.info("Exporting all reviews with scores...")

    output_path = config.OUTPUTS_DIR / "reviews_with_scores.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Saved to {output_path}")


def run() -> None:
    """Run scoring pipeline.

    Combines all score components into a weighted suspicion score,
    generates visualizations and summary tables.
    """
    logger.info("=" * 80)
    logger.info("Starting Scoring Pipeline")
    logger.info("=" * 80)

    # Load base reviews
    logger.info(f"Loading base reviews from {config.PARQUET_PATH}")
    df = pd.read_parquet(config.PARQUET_PATH)
    logger.info(f"Loaded {len(df)} reviews")

    # Load all scores
    df_scores = _load_all_scores(df)

    # Calculate suspicion scores
    df_scores = _calculate_suspicion_scores(df_scores)

    high_suspicion_threshold = df_scores.attrs.get(
        "high_suspicion_threshold",
        df_scores["suspicion_score"].quantile(HIGH_SUSPICION_PERCENTILE / 100)
    )
    visualization_threshold = df_scores.attrs.get(
        "visualization_threshold",
        df_scores["suspicion_score"].quantile(VISUALIZATION_PERCENTILE / 100)
    )

    # Set module-level thresholds for use in plotting/export functions
    global _HIGH_SUSPICION_THRESHOLD, _VISUALIZATION_THRESHOLD
    _HIGH_SUSPICION_THRESHOLD = high_suspicion_threshold
    _VISUALIZATION_THRESHOLD = visualization_threshold

    # Merge back with original data
    merge_cols = ["review_id", "suspicion_score"] + list(SCORE_WEIGHTS.keys())
    if "reviewer_profile_score" in df_scores.columns:
        merge_cols.append("reviewer_profile_score")

    df = df.merge(df_scores[merge_cols], on="review_id", how="left")

    # Create plots directory
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using plots directory: {config.PLOTS_DIR}")

    # 6.2: Generate visualizations
    logger.info("Generating visualizations...")
    _plot_score_distribution(df_scores)
    _plot_restaurant_ranking(df)
    _plot_age_vs_score(df)
    _plot_score_components_heatmap(df)
    _plot_network_graph(df)

    # 6.3: Generate output tables
    logger.info("Generating output tables...")
    df_public = _anonymize_sensitive_columns(df)
    _export_top_suspicious_reviews(df_public)
    _export_restaurant_summary(df_public)
    _export_all_reviews_with_scores(df_public)

    # Save scored dataset
    output_scores_path = config.DATA_PROCESSED_DIR / "reviews_with_suspicion_scores.parquet"
    df_public.to_parquet(output_scores_path, index=False)
    logger.info(f"Saved scored dataset to {output_scores_path}")

    # Export suspicious reviews (high suspicion threshold)
    suspicious_path = config.OUTPUTS_DIR / "suspicious_reviews.csv"
    suspicious_df = df_public[df_public['suspicion_score'] > _HIGH_SUSPICION_THRESHOLD]
    suspicious_df.to_csv(suspicious_path, index=False, encoding="utf-8")
    logger.info(f"Saved {len(suspicious_df)} suspicious reviews to {suspicious_path}")

    # Summary statistics
    logger.info("=" * 80)
    logger.info("Scoring Summary")
    logger.info("=" * 80)
    logger.info(f"Total reviews: {len(df):,}")
    logger.info(f"Mean suspicion score: {df['suspicion_score'].mean():.3f}")
    logger.info(f"Median suspicion score: {df['suspicion_score'].median():.3f}")
    logger.info(f"Max suspicion score: {df['suspicion_score'].max():.3f}")

    high_suspicious = (df["suspicion_score"] > _HIGH_SUSPICION_THRESHOLD).sum()
    logger.info(
        f"High suspicion (P{HIGH_SUSPICION_PERCENTILE}, >{_HIGH_SUSPICION_THRESHOLD:.3f}): "
        f"{high_suspicious} ({high_suspicious / len(df) * 100:.1f}%)"
    )

    # Top 5 restaurants
    top5_restaurants = (
        df.groupby("place_name")["suspicion_score"].mean().nlargest(5)
    )
    logger.info("Top 5 Restaurants by Mean Suspicion Score:")
    for i, (name, score) in enumerate(top5_restaurants.items(), 1):
        logger.info(f"  {i}. {name}: {score:.3f}")

    logger.info("=" * 80)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    run()
