"""Network analysis for fake review detection.

Network-based methods are considered the most robust category of fake review
detection. The combination of metadata (text, timestamps, ratings) with the
relational structure in a network is significantly harder to manipulate than
pure text analysis, as it is content-independent and reveals coordinated
behavior that remains hidden from text analysis. Since professional spammers
often operate in groups and leave certain relational patterns in their connections
to different places, these methods cannot be circumvented by clever text formulation.

SpEagle (Rayana & Akoglu 2015) uses a reviewer-review-restaurant graph with Belief
Propagation and achieves significantly better results in real datasets like Yelp—
which is structurally very similar to Google Maps—than isolated text models. Here
we implement a simplified but methodologically related variant.

PLATFORM-SPECIFIC ADAPTATIONS (Google Maps, Wuerzburg dataset):
    The original SpEagle framework uses Belief Propagation on large-scale
    datasets (millions of reviews). This implementation uses a simplified
    but methodologically related approach adapted to a local dataset
    (37,813 reviews, 29,823 reviewers, 121 restaurants).

    Key adaptations:
    - Degree scoring replaced with degree x homogeneity interaction term:
        Raw degree has no discriminative power (67% of reviewers have degree=1,
        max=23). The interaction term captures the SpEagle insight that
        coordinated campaigns involve reviewers visiting MULTIPLE restaurants
        with IDENTICAL ratings.
    - Rating homogeneity requires minimum 3 local reviews. With 67% of
        reviewers having only 1 review in the dataset, unconditional homogeneity
        calculation produces an artifact (mean=0.965, virtually no discrimination).
    - Co-reviewer threshold raised from 2 to 3 shared restaurants. At 121
        restaurants, sharing 2 is common by chance (31.7% of reviewers affected).
    - Co-burst weight reduced from 0.20 to 0.05: The signal is high-confidence
        but fires for only 0.03% of reviews (11/37,813). High weight on a
        near-zero signal penalizes the entire score distribution.

The bipartite graph structure captures the relationships between reviewers and
restaurants through review edges. By analyzing degree distributions, rating
homogeneity, burst patterns, and co-reviewer connections, we can identify
coordinated spam campaigns that would be invisible to content-based approaches.

References:
    Rayana, S., & Akoglu, L. (2015). Collective opinion spam detection: Bridging
        review networks and metadata. In KDD '15.
"""

import logging
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
import networkx as nx

from scripts import config

# Configure logging
logger = logging.getLogger(__name__)

# Network analysis parameters
NETWORK_WEIGHTS = {
    "s_degree_homogeneity": 0.20,  # Interaction: degree x rating homogeneity
    "s_homogeneity": 0.20,         # Rating homogeneity (with min-review filter)
    "s_burst_ratio": 0.25,         # Proportion of burst edges
    "s_co_reviewer": 0.30,         # Co-reviewer count (strongest network signal)
    "s_co_burst": 0.05,            # In co-burst (rare but high-confidence)
}

# Weight rationale:
# - s_co_reviewer gets highest weight (0.30) because co-reviewing patterns are
#   the core SpEagle signal and affect 31.7% of reviewers - good discrimination.
# - s_burst_ratio (0.25) bridges temporal and network analysis.
# - s_degree_homogeneity (0.20) captures coordinated campaigns (SpEagle-inspired).
# - s_homogeneity (0.20) detects rating manipulation (with min-review filter).
# - s_co_burst (0.05) is near-certain evidence but fires for only 0.03% of
#   reviews. Keeping it at low weight preserves the signal without penalizing
#   the 99.97% of reviews where it contributes nothing.

CO_REVIEWER_MIN_SHARED = 3   # Minimum shared restaurants for co-reviewer detection

# Empirical justification: With 121 restaurants in the dataset, sharing 2
# restaurants is common by chance - 31.7% of reviewers have co-reviewers at
# threshold 2. Raising to 3 filters coincidental overlaps (e.g. tourists
# visiting the same popular spots) while retaining genuine coordination signals.
MAX_RESTAURANTS = 121        # Actual restaurant count in dataset


def _build_bipartite_graph(df: pd.DataFrame) -> nx.MultiGraph:
    """Build bipartite reviewer-restaurant graph.

    Creates a MultiGraph to handle multiple reviews from the same reviewer to
    the same restaurant. Uses prefixes 'r_' for reviewers and 'p_' for places
    to avoid ID collisions.

    Args:
        df: Review DataFrame with all necessary columns.

    Returns:
        NetworkX MultiGraph with reviewer and restaurant nodes.
    """
    logger.info("Building bipartite reviewer-restaurant graph...")

    G = nx.MultiGraph()

    # Add reviewer nodes
    reviewer_cols = [
        "reviewer_user_id", "reviewer_name", "reviewer_review_count",
        "reviewer_is_local_guide", "reviewer_profile_score", "mnr_score",
        "pr_score", "in_co_burst"
    ]
    reviewers = df[reviewer_cols].drop_duplicates(subset=["reviewer_user_id"])

    for _, row in reviewers.iterrows():
        node_id = f"r_{row['reviewer_user_id']}"
        G.add_node(
            node_id,
            node_type="reviewer",
            reviewer_user_id=row["reviewer_user_id"],
            reviewer_name=row["reviewer_name"],
            reviewer_review_count=row["reviewer_review_count"],
            is_local_guide=row["reviewer_is_local_guide"],
            reviewer_profile_score=row["reviewer_profile_score"],
            mnr_score=row["mnr_score"],
            pr_score=row["pr_score"],
            in_co_burst=row["in_co_burst"]
        )

    logger.info(f"Added {len(reviewers)} reviewer nodes")

    # Add restaurant nodes
    restaurant_cols = ["place_url", "place_name", "place_overall_rating"]
    restaurants = df[restaurant_cols].drop_duplicates(subset=["place_url"])

    for _, row in restaurants.iterrows():
        node_id = f"p_{row['place_url']}"
        G.add_node(
            node_id,
            node_type="restaurant",
            place_url=row["place_url"],
            place_name=row["place_name"],
            place_overall_rating=row["place_overall_rating"]
        )

    logger.info(f"Added {len(restaurants)} restaurant nodes")

    # Add review edges
    for _, row in df.iterrows():
        reviewer_id = f"r_{row['reviewer_user_id']}"
        restaurant_id = f"p_{row['place_url']}"

        G.add_edge(
            reviewer_id,
            restaurant_id,
            review_id=row["review_id"],
            rating_stars=row["rating_stars"],
            timestamp_created_iso=row["timestamp_created_iso"],
            in_burst=row["in_burst"],
            burst_suspicion_score=row["burst_suspicion_score"],
            in_co_burst=row["in_co_burst"]
        )

    logger.info(f"Added {len(df)} review edges")
    logger.info(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return G


def _calculate_graph_metrics(G: nx.MultiGraph, df: pd.DataFrame) -> pd.DataFrame:
    """Calculate graph-based metrics for each reviewer.

    Computes degree (number of distinct restaurants), PageRank, rating
    homogeneity, five-star percentage, and burst edge ratio.

    Note: Rating homogeneity uses a neutral default (0.5) for reviewers with
    fewer than 3 reviews in the local dataset. This avoids the statistical
    artifact where single-review users appear "perfectly homogeneous" (std=0)
    and receive maximum suspicion scores despite insufficient data.

    Args:
        G: Bipartite reviewer-restaurant graph.
        df: Review DataFrame for rating analysis.

    Returns:
        DataFrame with metrics per reviewer_user_id.
    """
    logger.info("Calculating graph metrics...")

    metrics = []

    # Get all reviewer nodes
    reviewer_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "reviewer"]

    # Calculate PageRank for the entire graph
    logger.info("Computing PageRank...")
    pagerank = nx.pagerank(G)

    for reviewer_node in reviewer_nodes:
        reviewer_user_id = G.nodes[reviewer_node]["reviewer_user_id"]

        # Degree: number of DISTINCT restaurants (neighbors in MultiGraph)
        neighbors = list(G.neighbors(reviewer_node))
        degree = len(neighbors)

        # Get all review edges for this reviewer
        all_edges = []
        for neighbor in neighbors:
            # MultiGraph: get all edges between reviewer and restaurant
            edges = G.get_edge_data(reviewer_node, neighbor)
            if edges:
                for key, edge_data in edges.items():
                    all_edges.append(edge_data)

        if len(all_edges) == 0:
            continue

        # Rating homogeneity requires a minimum sample size.
        ratings = [e["rating_stars"] for e in all_edges]
        # With <3 reviews, variance is unstable and creates artifacts.
        MIN_REVIEWS_FOR_HOMOGENEITY = 3
        if len(ratings) < MIN_REVIEWS_FOR_HOMOGENEITY:
            rating_homogeneity = 0.5  # Neutral: insufficient data
        else:
            rating_std = np.std(ratings)
            rating_homogeneity = 1 - (rating_std / 4.0)
            rating_homogeneity = max(0.0, rating_homogeneity)  # Clip negative values

        # Percentage of five-star reviews
        pct_five_star = sum(1 for r in ratings if r == 5) / len(ratings)

        # Burst edge ratio
        burst_edges = sum(1 for e in all_edges if e["in_burst"])
        burst_edge_ratio = burst_edges / len(all_edges)

        metrics.append({
            "reviewer_user_id": reviewer_user_id,
            "degree": degree,
            "pagerank": pagerank[reviewer_node],
            "rating_homogeneity": rating_homogeneity,
            "pct_five_star": pct_five_star,
            "burst_edge_ratio": burst_edge_ratio
        })

    result = pd.DataFrame(metrics)
    logger.info(f"Calculated metrics for {len(result)} reviewers")

    return result


def _detect_co_reviewers(df: pd.DataFrame) -> pd.DataFrame:
    """Detect co-reviewers who share multiple restaurants.

    Two reviewers are co-reviewers if they review at least CO_REVIEWER_MIN_SHARED
    restaurants together. Uses an inverted index for efficient pair detection.

    Args:
        df: Review DataFrame.

    Returns:
        DataFrame with co_reviewer_count per reviewer_user_id.
    """
    logger.info("Detecting co-reviewers...")

    # Build inverted index: restaurant -> set of reviewer IDs
    restaurant_reviewers = defaultdict(set)
    for _, row in df.iterrows():
        restaurant_reviewers[row["place_url"]].add(row["reviewer_user_id"])

    # Count shared restaurants per reviewer pair
    pair_shared_count = Counter()

    for place_url, reviewer_set in restaurant_reviewers.items():
        reviewers = list(reviewer_set)
        # Generate all pairs for this restaurant
        for i in range(len(reviewers)):
            for j in range(i + 1, len(reviewers)):
                r1, r2 = reviewers[i], reviewers[j]
                # Use sorted tuple as key for consistency
                pair_key = tuple(sorted([r1, r2]))
                pair_shared_count[pair_key] += 1

    # Filter pairs with sufficient shared restaurants
    co_reviewer_pairs = [
        pair for pair, count in pair_shared_count.items()
        if count >= CO_REVIEWER_MIN_SHARED
    ]

    logger.info(f"Found {len(co_reviewer_pairs)} co-reviewer pairs (>= {CO_REVIEWER_MIN_SHARED} shared restaurants)")

    # Count co-reviewers per reviewer
    co_reviewer_count = Counter()
    for r1, r2 in co_reviewer_pairs:
        co_reviewer_count[r1] += 1
        co_reviewer_count[r2] += 1

    # Create result DataFrame
    result = pd.DataFrame({
        "reviewer_user_id": list(co_reviewer_count.keys()),
        "co_reviewer_count": list(co_reviewer_count.values())
    })

    logger.info(f"Identified co-reviewers for {len(result)} reviewers")
    if len(result) > 0:
        logger.info(f"  Mean co-reviewer count: {result['co_reviewer_count'].mean():.1f}")
        logger.info(f"  Max co-reviewer count: {result['co_reviewer_count'].max()}")

    return result


def _calculate_network_scores(
    metrics_df: pd.DataFrame,
    co_reviewer_df: pd.DataFrame
) -> pd.DataFrame:
    """Calculate final network suspicion scores.

    Normalizes all metrics to [0, 1] and combines them using NETWORK_WEIGHTS.

    Score components:
    - s_degree_homogeneity: Interaction of degree and rating homogeneity.
      Captures coordinated campaigns (many restaurants, identical ratings).
      Neither metric is meaningful alone on this dataset.
    - s_homogeneity: Rating consistency across reviews (min 3 reviews required).
    - s_burst_ratio: Proportion of reviews posted during temporal bursts.
    - s_co_reviewer: Number of reviewers sharing 3+ restaurants (percentile-normalized).
    - s_co_burst: Binary flag for co-bursting (rare, high-confidence signal).

    Args:
        metrics_df: Graph metrics per reviewer.
        co_reviewer_df: Co-reviewer counts per reviewer.

    Returns:
        DataFrame with normalized scores and final network_score.
    """
    logger.info("Calculating network suspicion scores...")

    # Merge co-reviewer counts
    df = metrics_df.merge(co_reviewer_df, on="reviewer_user_id", how="left")
    df["co_reviewer_count"] = df["co_reviewer_count"].fillna(0)

    # Normalize metrics to [0, 1]
    # Interaction term: degree x homogeneity (SpEagle-inspired)
    # High degree alone is not suspicious (power users review many places).
    # High homogeneity alone is unreliable (single-review users always score 1.0).
    # But high degree COMBINED with high homogeneity indicates coordinated
    # campaigns: reviewers who visit many restaurants and give identical ratings.
    # Degree normalized by max observed degree instead of total restaurants.
    df["s_homogeneity"] = df["rating_homogeneity"].clip(0, 1)
    degree_normalized = (df["degree"] / df["degree"].max()).clip(0, 1)
    df["s_degree_homogeneity"] = (degree_normalized * df["s_homogeneity"]).clip(0, 1)
    df["s_burst_ratio"] = df["burst_edge_ratio"].clip(0, 1)

    # Percentile-based normalization for co-reviewer count.
    # Distribution is extremely right-skewed (median=0, P75=4, max=540).
    # Linear /10 normalization saturates too quickly.
    # Using P90 (~20) as reference point: co_reviewer_count >= 20 -> score 1.0
    CO_REVIEWER_NORM_REFERENCE = 20  # Approximately P90 of distribution
    df["s_co_reviewer"] = (df["co_reviewer_count"] / CO_REVIEWER_NORM_REFERENCE).clip(0, 1)

    # Note: in_co_burst will be merged from original data, not available here yet
    # We'll add s_co_burst in the main function after merging

    # Calculate preliminary network score (without co_burst component)
    score_components = ["s_degree_homogeneity", "s_homogeneity", "s_burst_ratio", "s_co_reviewer"]
    df["network_score_partial"] = 0.0

    for component in score_components:
        weight = NETWORK_WEIGHTS[component]
        df["network_score_partial"] += df[component] * weight

    logger.info(f"Calculated network scores for {len(df)} reviewers")
    logger.info(f"  Mean network score (partial): {df['network_score_partial'].mean():.3f}")
    logger.info(f"  Max network score (partial): {df['network_score_partial'].max():.3f}")

    return df


def calculate_network_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate network-based fake-review signals and persist score parquet.

    This is the entrypoint used by ``flows/pipeline.py`` in Phase 6.
    Loads heuristic and temporal scores, then builds a bipartite graph to analyze
    reviewer-restaurant relationships.

    Args:
        df: Cleaned reviews DataFrame loaded from ``config.PARQUET_PATH``.

    Returns:
        Input DataFrame enriched with network feature columns.
    """
    logger.info("=" * 80)
    logger.info("Starting Network Analysis")
    logger.info("=" * 80)

    # Load heuristic scores
    heuristic_path = config.DATA_PROCESSED_DIR / "scores_heuristic.parquet"
    logger.info(f"Loading heuristic scores from {heuristic_path}")
    heuristic_df = pd.read_parquet(heuristic_path)

    # Load temporal scores
    temporal_path = config.DATA_PROCESSED_DIR / "scores_temporal.parquet"
    logger.info(f"Loading temporal scores from {temporal_path}")
    temporal_df = pd.read_parquet(temporal_path)

    # Merge all data
    df = df.merge(heuristic_df, on="review_id", how="left")
    df = df.merge(temporal_df, on="review_id", how="left")

    logger.info(f"Processing {len(df)} reviews with all scores merged")

    # 4.1: Build bipartite graph
    G = _build_bipartite_graph(df)

    # 4.2: Calculate graph metrics
    metrics_df = _calculate_graph_metrics(G, df)

    # 4.3: Detect co-reviewers
    co_reviewer_df = _detect_co_reviewers(df)

    # 4.4: Calculate network scores
    scores_df = _calculate_network_scores(metrics_df, co_reviewer_df)

    # Merge network scores back to reviews
    enriched = df.merge(
        scores_df[["reviewer_user_id", "network_score_partial", "degree", "pagerank",
                   "rating_homogeneity", "pct_five_star", "burst_edge_ratio", "co_reviewer_count",
                   "s_degree_homogeneity"]],
        on="reviewer_user_id",
        how="left"
    )

    # Add co_burst component to final network score
    enriched["s_co_burst"] = enriched["in_co_burst"].astype(float)
    enriched["network_score"] = (
        enriched["network_score_partial"] +
        enriched["s_co_burst"] * NETWORK_WEIGHTS["s_co_burst"]
    )

    # Prepare output
    output_df = enriched[[
        "review_id",
        "network_score",
        "degree",
        "pagerank",
        "rating_homogeneity",
        "pct_five_star",
        "burst_edge_ratio",
        "co_reviewer_count",
        "s_degree_homogeneity"
    ]].copy()

    # Save results
    output_path = config.DATA_PROCESSED_DIR / "scores_network.parquet"
    output_df.to_parquet(output_path, index=False)
    logger.info(f"Saved network scores to {output_path}")

    # Export graph for visualization
    try:
        graphml_path = config.OUTPUTS_DIR / "reviewer_restaurant_graph.graphml"

        for node, data in G.nodes(data=True):
            for key, value in list(data.items()):
                if hasattr(value, 'isoformat'):
                    data[key] = str(value)

        for u, v, data in G.edges(data=True):
            for key, value in list(data.items()):
                if hasattr(value, 'isoformat'):
                    data[key] = str(value)

        nx.write_graphml(G, graphml_path)
        logger.info(f"Exported graph to {graphml_path}")
    except Exception as e:
        logger.warning(f"Could not export graph: {e}")

    # Log summary statistics
    logger.info("=" * 80)
    logger.info("Network Analysis Summary")
    logger.info("=" * 80)
    logger.info(f"Mean network score: {output_df['network_score'].mean():.3f}")
    logger.info(f"Median network score: {output_df['network_score'].median():.3f}")
    logger.info(f"Max network score: {output_df['network_score'].max():.3f}")
    logger.info(f"Reviews with network_score > 0.5: {(output_df['network_score'] > 0.5).sum()} ({(output_df['network_score'] > 0.5).mean():.1%})")
    logger.info(f"Mean degree: {output_df['degree'].mean():.1f}")
    logger.info(f"Mean rating homogeneity: {output_df['rating_homogeneity'].mean():.3f}")
    logger.info(f"Mean burst edge ratio: {output_df['burst_edge_ratio'].mean():.3f}")
    logger.info(f"Reviewers with co-reviewers: {(output_df['co_reviewer_count'] > 0).sum()}")
    logger.info(f"Mean s_degree_homogeneity: {output_df.get('s_degree_homogeneity', pd.Series([0])).mean():.3f}")
    logger.info(f"Reviewers with degree >= 3: {(output_df['degree'] >= 3).sum()} ({(output_df['degree'] >= 3).mean():.1%})")
    logger.info("=" * 80)

    return enriched


def run() -> None:
    """Run network analysis pipeline (standalone).

    Loads base data and calls calculate_network_features() to perform
    the full network analysis.
    """
    # Load base data
    logger.info(f"Loading data from {config.PARQUET_PATH}")
    df = pd.read_parquet(config.PARQUET_PATH)

    # Call the main calculation function (which loads scores internally)
    calculate_network_features(df)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    run()
