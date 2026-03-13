"""Semantic analysis for fake review detection.

NLP methods are less reliable than behavioral and network-based approaches when
dealing with smart human spammers. Modern transformer models like RoBERTa can
detect subtle linguistic patterns invisible to humans and are nearly perfect at
distinguishing machine-generated text (e.g., GPT-2 or GPT-4) from human text.
However, their reliability drops significantly with professional human spammers
who imitate the style of genuine reviews-achieving only ~68% accuracy in real
datasets vs. >90% on artificial test data. Error rates are particularly high for
short texts (<100 words) (Mohawesh et al. 2021).

Classic approaches based on word frequencies and stylistics work well for "cheaply"
produced fake reviews (e.g., via Amazon Mechanical Turk) but often fail against
real professional spammers who imitate the vocabulary of genuine users (Ott et al.
2011). NLP features usefully complement behavioral and network-based methods but
do not replace them.

References:
    Mohawesh, R., et al. (2021). Fake reviews detection: A survey.
    Mukherjee, A., et al. (2013). What yelp fake review filter might be doing?
    Ott, M., et al. (2011). Finding deceptive opinion spam by any stretch of
        the imagination.
    Moon, S., et al. (2021). Examining the consistency between review text and
        numeric ratings.

LIMITATIONS AND ADAPTATIONS:
    - Attribute template scoring (Jaccard similarity on dining attributes) is
      disabled due to insufficient attribute fill rates in scraped Google Maps
      data (best: price_range at 41%). Restaurant-specific attributes like
      price_range do not indicate reviewer template reuse. MCS on text
      embeddings provides superior template detection for this dataset.
    - Sentiment consistency uses tightened thresholds (only 1-star with positive
      sentiment or 5-star with negative sentiment). Analysis of 1,169 flagged
      inconsistencies showed 31% were 4-star reviews - typically mixed experiences
      ("food good, service bad") misclassified by the sentiment model, not genuine
      contradictions. This reduces false positives at the cost of some recall.
    - Fact compatibility scoring found only 16 contradictions (0.04% of reviews)
      due to low attribute fill rates and conservative keyword matching. The
      method is retained as it demonstrates the concept for portfolio purposes,
      but contributes minimal signal to the final score.
    - All NLP-based scores are weighted low in the final aggregation (total 14%)
      reflecting their lower reliability compared to behavioral and network-based
      methods (Mohawesh et al. 2021).
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from scripts import config

# Configure logging
logger = logging.getLogger(__name__)

# Semantic analysis parameters
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
SENTIMENT_MODEL = "oliverguhr/german-sentiment-bert"
MCS_THRESHOLD = 0.85         # Above this: near-identical text
MIN_TEXT_LENGTH = 20         # Minimum length for sentiment analysis
ATTR_COLUMNS = [
    "attr_meal_type", "attr_price_range", "attr_noise_level",
    "attr_service_type", "attr_wait_time", "attr_group_size",
    "attr_parking", "attr_vegetarian"
]

# Fact compatibility rules
FACT_RULES = [
    {
        "keywords": ["lange wartezeit", "ewig gewartet", "lange warten", "viel wartezeit"],
        "field": "attr_wait_time",
        "contradicts": ["Keine Wartezeit", "Unter 15 Min."],
        "description": "Long wait time vs no/short wait"
    },
    {
        "keywords": ["teuer", "überteuert", "preislich hoch", "nicht billig"],
        "field": "attr_price_range",
        "contradicts": ["Unter 10 €", "10–20 €"],
        "description": "Expensive vs low price range"
    },
    {
        "keywords": ["billig", "günstig", "preiswert", "gutes preis"],
        "field": "attr_price_range",
        "contradicts": ["40–60 €", "Über 60 €"],
        "description": "Cheap vs high price range"
    },
    {
        "keywords": ["sehr laut", "lärm", "zu laut", "extrem laut"],
        "field": "attr_noise_level",
        "contradicts": ["Ruhig"],
        "description": "Very loud vs quiet"
    },
    {
        "keywords": ["sehr ruhig", "angenehm still", "ruhige atmosphäre"],
        "field": "attr_noise_level",
        "contradicts": ["Laut", "Sehr laut"],
        "description": "Very quiet vs loud"
    },
    {
        "keywords": ["kein parkplatz", "parken schwierig", "parkplatzsuche"],
        "field": "attr_parking",
        "contradicts": ["Kostenlose Parkplätze", "Eigener Parkplatz"],
        "description": "No parking vs available parking"
    },
    {
        "keywords": ["kein vegetarisch", "nichts vegetarisches", "für vegetarier ungeeignet"],
        "field": "attr_vegetarian",
        "contradicts": ["Viele vegetarische Gerichte"],
        "description": "No vegetarian vs many vegetarian options"
    }
]


def _compute_or_load_embeddings(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Compute or load cached text embeddings.

    Caches embeddings to avoid recomputation. Only processes German reviews
    with text content.

    Args:
        df: Review DataFrame.

    Returns:
        Tuple of (embeddings array, review_ids array).
    """
    embedding_path = config.DATA_PROCESSED_DIR / "embeddings.npy"
    review_ids_path = config.DATA_PROCESSED_DIR / "embedding_review_ids.npy"
    parquet_mtime = config.PARQUET_PATH.stat().st_mtime

    # Check if cache is valid
    if (embedding_path.exists() and review_ids_path.exists() and
            embedding_path.stat().st_mtime > parquet_mtime):
        logger.info("Using cached embeddings")
        embeddings = np.load(embedding_path)
        review_ids = np.load(review_ids_path, allow_pickle=True)
        logger.info(f"Loaded {len(embeddings)} cached embeddings")
        return embeddings, review_ids

    # Filter for German reviews with text
    text_df = df[
        (df["review_language"] == "de") &
        (df["has_review_text"] == True)
    ].copy()

    if len(text_df) == 0:
        logger.warning("No German text reviews found for embeddings")
        return np.array([]), np.array([])

    logger.info(f"Computing embeddings for {len(text_df)} German reviews...")

    # Detect device
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {'GPU (cuda)' if device == 'cuda' else 'CPU'}")

    # Load model
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    logger.info(f"Loaded embedding model: {EMBEDDING_MODEL}")

    # Generate embeddings
    texts = text_df["review_text"].tolist()
    review_ids = text_df["review_id"].values

    embeddings = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Cache results
    np.save(embedding_path, embeddings)
    np.save(review_ids_path, review_ids)
    logger.info(f"Cached embeddings to {embedding_path}")

    return embeddings, review_ids


def _calculate_mcs_scores(df: pd.DataFrame, embeddings: np.ndarray, review_ids: np.ndarray) -> pd.DataFrame:
    """Calculate Maximum Content Similarity (MCS) scores.

    MCS detects spammers who reuse text templates across different locations.
    Mukherjee et al. (2013) showed this was part of a feature set achieving
    ~86% accuracy while pure text classification failed.

    Args:
        df: Review DataFrame.
        embeddings: Text embeddings array.
        review_ids: Corresponding review IDs.

    Returns:
        DataFrame with review_id and mcs_score.
    """
    logger.info("Calculating MCS (Maximum Content Similarity) scores...")

    # Create mapping from review_id to embedding index
    review_id_to_idx = {rid: idx for idx, rid in enumerate(review_ids)}

    # Merge review_ids with reviewer info
    embedding_df = pd.DataFrame({"review_id": review_ids})
    embedding_df = embedding_df.merge(
        df[["review_id", "reviewer_user_id"]],
        on="review_id"
    )

    mcs_scores = []

    # Group by reviewer
    for reviewer_id, group in embedding_df.groupby("reviewer_user_id"):
        if len(group) == 1:
            # Single review: no comparison possible
            for rid in group["review_id"]:
                mcs_scores.append({"review_id": rid, "mcs_score": 0.0})
            continue

        # Get embeddings for this reviewer
        indices = [review_id_to_idx[rid] for rid in group["review_id"]]
        reviewer_embeddings = embeddings[indices]

        # Compute similarity matrix
        sim_matrix = cosine_similarity(reviewer_embeddings)

        # Set diagonal to 0 (ignore self-similarity)
        np.fill_diagonal(sim_matrix, 0)

        # MCS = max similarity to any other review by same reviewer
        max_sims = sim_matrix.max(axis=1)

        for rid, mcs in zip(group["review_id"], max_sims):
            mcs_scores.append({"review_id": rid, "mcs_score": float(mcs)})

    result = pd.DataFrame(mcs_scores)
    high_mcs = (result["mcs_score"] > MCS_THRESHOLD).sum()
    logger.info(f"Found {high_mcs} reviews with MCS > {MCS_THRESHOLD} (quasi-identical text)")

    return result


def _calculate_attr_template_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate attribute template scores using Jaccard similarity.

    NOTE: This function is currently disabled in the pipeline because empirical
    analysis showed insufficient signal on the Wuerzburg dataset (mean score 0.027,
    75th percentile 0.0). The low attribute fill rates (best: price_range at 41%)
    and the restaurant-specific nature of most attributes make Jaccard similarity
    ineffective for detecting reviewer template reuse. MCS on text embeddings
    provides superior template detection.

    Preserved for documentation and potential reactivation with richer data.

    Detects reviewers using identical dining attribute combinations across
    different restaurants (MCS on structured data).

    Args:
        df: Review DataFrame.

    Returns:
        DataFrame with review_id and attr_template_score.
    """
    logger.info("Calculating attribute template scores...")

    template_scores = []

    # Group by reviewer
    for reviewer_id, group in df.groupby("reviewer_user_id"):
        if len(group) == 1:
            # Single review: no template possible
            for rid in group["review_id"]:
                template_scores.append({"review_id": rid, "attr_template_score": 0.0})
            continue

        # Calculate Jaccard for all pairs
        reviews = group[["review_id", "place_url"] + ATTR_COLUMNS].copy()
        max_jaccard_per_review = {}

        for i, row1 in reviews.iterrows():
            max_jaccard = 0.0

            for j, row2 in reviews.iterrows():
                if i >= j or row1["place_url"] == row2["place_url"]:
                    continue  # Same review or same restaurant

                # Get non-NaN attributes
                attrs1 = set()
                attrs2 = set()
                for col in ATTR_COLUMNS:
                    val1 = row1[col]
                    val2 = row2[col]
                    if pd.notna(val1):
                        attrs1.add((col, val1))
                    if pd.notna(val2):
                        attrs2.add((col, val2))

                # Calculate Jaccard
                if len(attrs1) == 0 and len(attrs2) == 0:
                    jaccard = 0.0
                else:
                    intersection = attrs1 & attrs2
                    union = attrs1 | attrs2
                    jaccard = len(intersection) / len(union) if len(union) > 0 else 0.0

                max_jaccard = max(max_jaccard, jaccard)

            max_jaccard_per_review[row1["review_id"]] = max_jaccard

        for rid, score in max_jaccard_per_review.items():
            template_scores.append({"review_id": rid, "attr_template_score": score})

    result = pd.DataFrame(template_scores)
    logger.info(f"Calculated template scores for {len(result)} reviews")

    return result


def _calculate_fact_compatibility_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate fact compatibility scores.

    Detects contradictions between review text and structured attributes.
    Spammers without real visit experience often produce such inconsistencies
    (Moon et al. 2021).

    Args:
        df: Review DataFrame.

    Returns:
        DataFrame with review_id and fact_compatibility_score.
    """
    logger.info("Calculating fact compatibility scores...")

    scores = []

    for _, row in df.iterrows():
        if not row["has_review_text"] or pd.isna(row["review_text"]):
            scores.append({"review_id": row["review_id"], "fact_compatibility_score": 0.0})
            continue

        text_lower = str(row["review_text"]).lower()

        applicable_rules = 0
        contradictions = 0

        for rule in FACT_RULES:
            field_value = row.get(rule["field"])

            # Skip if field is NaN
            if pd.isna(field_value):
                continue

            # Check if any keyword matches
            keyword_match = any(kw in text_lower for kw in rule["keywords"])

            if keyword_match:
                applicable_rules += 1

                # Check for contradiction
                if any(contradict in str(field_value) for contradict in rule["contradicts"]):
                    contradictions += 1

        # Score: proportion of contradictions among applicable rules
        if applicable_rules > 0:
            score = contradictions / applicable_rules
        else:
            score = 0.0

        scores.append({"review_id": row["review_id"], "fact_compatibility_score": score})

    result = pd.DataFrame(scores)
    contradictory = (result["fact_compatibility_score"] > 0).sum()
    logger.info(f"Found {contradictory} reviews with fact contradictions")

    return result


def _calculate_sentiment_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate sentiment consistency scores.

    Detects inconsistencies between sentiment and star ratings. Has lowest
    reliability, so weighted low in final score (Ott et al. 2011).

        Threshold rationale: Only extreme contradictions are flagged (positive
        sentiment + 1 star, negative sentiment + 5 stars). Intermediate cases
        (e.g., negative sentiment + 4 stars) produce too many false positives
        because sentiment models classify mixed reviews ("food good, service bad")
        as negative, while 4 stars is a reasonable overall rating for such
        experiences. Empirical analysis showed 31% of flagged inconsistencies
        came from 4-star reviews.

        Limitations:
        - German sentiment model (oliverguhr/german-sentiment-bert) has limited
            accuracy on short texts (<100 words) and mixed-sentiment reviews.
        - Binary positive/negative classification loses nuance of mixed reviews.
        - Only German-language reviews are analyzed; non-German reviews receive 0.0.
        - Sentiment analysis is the least reliable detection method in this pipeline
            (Ott et al. 2011: ~68% accuracy on real data vs >90% on artificial data).

    Args:
        df: Review DataFrame.

    Returns:
        DataFrame with review_id, sentiment_star_inconsistency, sentiment_subrating_inconsistency.
    """
    logger.info("Calculating sentiment consistency scores...")

    # Import here to avoid loading model if not needed
    try:
        import torch
        from transformers import pipeline
        try:
            from datasets import Dataset
            from transformers.pipelines.pt_utils import KeyDataset
            USE_HF_DATASET = True
        except ImportError:
            USE_HF_DATASET = False
            logger.info("HuggingFace datasets/KeyDataset not available, using list-based batching")

        device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Using device: {'GPU (cuda)' if device == 0 else 'CPU'}")
        
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=SENTIMENT_MODEL,
            device=device,
            top_k=None
        )
        logger.info(f"Loaded sentiment model: {SENTIMENT_MODEL}")
    except Exception as e:
        logger.warning(f"Could not load sentiment model: {e}. Skipping sentiment analysis.")
        return pd.DataFrame({
            "review_id": df["review_id"],
            "sentiment_star_inconsistency": 0.0,
            "sentiment_subrating_inconsistency": 0.0
        })

    # Filter for German text reviews with sufficient length
    text_df = df[
        (df["review_language"] == "de") &
        (df["has_review_text"] == True) &
        (df["review_text"].str.len() > MIN_TEXT_LENGTH)
    ].copy()

    if len(text_df) == 0:
        logger.warning("No eligible reviews for sentiment analysis")
        return pd.DataFrame({
            "review_id": df["review_id"],
            "sentiment_star_inconsistency": 0.0,
            "sentiment_subrating_inconsistency": 0.0
        })

    logger.info(f"Analyzing sentiment for {len(text_df)} reviews...")

    sentiments = []
    if USE_HF_DATASET:
        # Process using HuggingFace Dataset for optimized GPU batching.
        # The transformers pipeline accepts Dataset objects which enable
        # prefetching and parallel data loading, avoiding the sequential
        # processing warning ("using pipelines sequentially on GPU").
        hf_dataset = Dataset.from_dict({"text": text_df["review_text"].tolist()})
        key_dataset = KeyDataset(hf_dataset, "text")

        try:
            for result in sentiment_pipeline(
                key_dataset,
                truncation=True,
                max_length=512,
                batch_size=128,
            ):
                if isinstance(result, list):
                    label = max(result, key=lambda x: x["score"])["label"]
                else:
                    label = result["label"]
                sentiments.append(label.lower())
        except Exception as e:
            logger.error(f"Sentiment pipeline failed: {e}")
            sentiments = ["neutral"] * len(text_df)
    else:
        # Fallback: list-based batching when datasets is not available.
        batch_size = 128
        texts = text_df["review_text"].tolist()

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                results = sentiment_pipeline(batch, truncation=True, max_length=512)
                for result in results:
                    if isinstance(result, list):
                        label = max(result, key=lambda x: x["score"])["label"]
                    else:
                        label = result["label"]
                    sentiments.append(label.lower())
            except Exception as e:
                logger.warning(f"Error processing batch {i}: {e}")
                sentiments.extend(["neutral"] * len(batch))

    # Ensure output length always matches input length
    if len(sentiments) != len(text_df):
        logger.warning(
            "Sentiment result length mismatch (%s vs %s). Filling with neutral labels.",
            len(sentiments),
            len(text_df),
        )
        sentiments = ["neutral"] * len(text_df)

    text_df["sentiment"] = sentiments

    # Calculate inconsistencies
    inconsistencies = []

    for _, row in text_df.iterrows():
        sentiment = row["sentiment"]
        rating_stars = row["rating_stars"]

        # Star inconsistency - only flag CLEAR contradictions
        # 4-star reviews with negative sentiment are often mixed experiences
        # ("food good, service bad") and not genuine contradictions.
        # Empirical analysis: 31% of false positives came from 4-star reviews.
        # Tightened thresholds: negative requires 5 stars, positive requires 1 star.
        star_incon = (
            (sentiment == "positive" and rating_stars == 1) or
            (sentiment == "negative" and rating_stars == 5)
        )

        # Subrating inconsistency - same tightened logic
        # 3-star reviews with sentiment mismatch are typically mixed reviews
        # misclassified by the sentiment model, not genuine contradictions.
        subrating_incon = False
        if pd.notna(row.get("sub_rating_service")):
            sub_rating = row["sub_rating_service"]
            subrating_incon = (
                (sentiment == "positive" and sub_rating == 1) or
                (sentiment == "negative" and sub_rating == 5)
            )

        inconsistencies.append({
            "review_id": row["review_id"],
            "sentiment_star_inconsistency": float(star_incon),
            "sentiment_subrating_inconsistency": float(subrating_incon)
        })

    result = pd.DataFrame(inconsistencies)
    logger.info(f"Found {result['sentiment_star_inconsistency'].sum():.0f} star inconsistencies")
    logger.info(f"Found {result['sentiment_subrating_inconsistency'].sum():.0f} subrating inconsistencies")

    # Merge with full dataset (fill missing with 0.0)
    full_result = df[["review_id"]].merge(result, on="review_id", how="left").fillna(0.0)

    return full_result


def run() -> None:
    """Run semantic analysis pipeline.

    Computes text embeddings, MCS scores, attribute template scores,
    fact compatibility scores, and sentiment consistency checks.
    Saves results to scores_semantic.parquet.
    """
    logger.info("=" * 80)
    logger.info("Starting Semantic Analysis")
    logger.info("=" * 80)

    # Load data
    logger.info(f"Loading data from {config.PARQUET_PATH}")
    df = pd.read_parquet(config.PARQUET_PATH)
    logger.info(f"Loaded {len(df)} reviews")

    # 5.1: Compute or load embeddings
    embeddings, embedding_review_ids = _compute_or_load_embeddings(df)

    # 5.2: Calculate MCS scores
    if len(embeddings) > 0:
        mcs_df = _calculate_mcs_scores(df, embeddings, embedding_review_ids)
    else:
        mcs_df = pd.DataFrame({"review_id": df["review_id"], "mcs_score": 0.0})

    # 5.3: Attribute template scores - DISABLED
    # Empirical analysis showed this score is ineffective on this dataset:
    # - Mean score: 0.027, 75th percentile: 0.0 (virtually no signal)
    # - Attribute fill rates too low (meal_type 32%, price_range 41%, rest <26%)
    # - Restaurant-specific attributes (e.g. price_range) do not indicate
    #   reviewer template reuse - the same restaurant always has the same price
    # - MCS score on text embeddings already captures text template reuse more
    #   effectively
    # - O(n^2) computation per reviewer is not scalable
    # The function _calculate_attr_template_scores() is preserved in code for
    # documentation and potential future use with richer attribute data.
    attr_template_df = pd.DataFrame({
        "review_id": df["review_id"],
        "attr_template_score": 0.0
    })
    logger.info("Attribute template scoring disabled (insufficient attribute coverage)")

    # 5.4: Calculate fact compatibility scores
    fact_compat_df = _calculate_fact_compatibility_scores(df)

    # 5.5: Calculate sentiment consistency
    sentiment_df = _calculate_sentiment_consistency(df)

    # Merge all scores
    output_df = (
        df[["review_id"]]
        .merge(mcs_df, on="review_id", how="left")
        .merge(attr_template_df, on="review_id", how="left")
        .merge(fact_compat_df, on="review_id", how="left")
        .merge(sentiment_df, on="review_id", how="left")
    )

    # Fill any missing values with 0.0
    output_df = output_df.fillna(0.0)

    # Save results
    output_path = config.DATA_PROCESSED_DIR / "scores_semantic.parquet"
    output_df.to_parquet(output_path, index=False)
    logger.info(f"Saved semantic scores to {output_path}")

    # Log summary statistics
    logger.info("=" * 80)
    logger.info("Semantic Analysis Summary")
    logger.info("=" * 80)
    logger.info(f"Mean MCS score: {output_df['mcs_score'].mean():.3f}")
    logger.info(f"High MCS (>{MCS_THRESHOLD}): {(output_df['mcs_score'] > MCS_THRESHOLD).sum()}")
    logger.info(f"Mean attr template score: {output_df['attr_template_score'].mean():.3f}")
    logger.info(f"Mean fact compatibility score: {output_df['fact_compatibility_score'].mean():.3f}")
    logger.info(f"Sentiment-star inconsistencies: {output_df['sentiment_star_inconsistency'].sum():.0f}")
    logger.info(f"Sentiment-subrating inconsistencies: {output_df['sentiment_subrating_inconsistency'].sum():.0f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    run()
