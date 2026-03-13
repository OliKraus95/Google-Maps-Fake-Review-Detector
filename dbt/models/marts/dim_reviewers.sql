{{ config(materialized='table') }}

-- Dimension table for reviewers
-- Contains all unique reviewers with their attributes and behavior patterns
-- Used in fact_reviews as the reviewer dimension

-- Primary Key: reviewer_user_id
-- Grain: One row per unique reviewer

select
    reviewer_user_id,
    reviewer_name,
    reviewer_review_count,
    reviewer_photo_count,
    reviewer_photo_review_ratio,
    reviewer_is_local_guide,
    reviewer_level,
    reviewer_badge,
    reviewer_avatar_url,
    reviewer_profile_url
from {{ ref('stg_reviewers') }}
