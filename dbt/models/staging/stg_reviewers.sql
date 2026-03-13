{{ config(materialized='view') }}

-- Staging model for reviewers
-- Deduplicates reviewers from raw_reviews (keeping latest crawl_timestamp snapshot)
-- Each reviewer_user_id appears once with their most recent profile snapshot

with reviewer_dedup as (
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
        reviewer_profile_url,
        crawl_timestamp,
        row_number() over (partition by reviewer_user_id order by crawl_timestamp desc) as rn
    from raw_reviews
    where reviewer_user_id is not null
)

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
from reviewer_dedup
where rn = 1
