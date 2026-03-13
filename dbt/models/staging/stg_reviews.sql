{{ config(materialized='view') }}

-- Staging model for reviews
-- Extracts all reviews from raw_reviews with derived columns
-- This is a 1:1 view of raw data after Phase 1 preparation

-- Derived columns added in Phase 1:
-- - review_date: CAST(timestamp_created_iso AS DATE)
-- - review_text_length: LENGTH(review_text)
-- All timestamp columns are UTC
-- All columns are in English snake_case (mapped from German in Phase 1)

select
    -- Primary key
    review_id,
    
    -- Foreign keys to dimensions
    place_url,              -- Restaurant identifier
    reviewer_user_id,       -- Reviewer identifier
    
    -- Review content and ratings
    review_text,
    rating_stars,
    review_language,
    sub_rating_food,
    sub_rating_service,
    sub_rating_atmosphere,
    
    -- Derived content features
    cast(timestamp_created_iso as date) as review_date,
    length(coalesce(review_text, '')) as review_text_length,
    has_review_text,
    has_sub_ratings,
    review_detail_level,
    
    -- Review metadata
    review_images_count,
    was_edited,
    edit_delta_hours,
    
    -- Reviewer profile snapshot at review time
    reviewer_name,
    reviewer_review_count,
    reviewer_photo_count,
    reviewer_photo_review_ratio,
    reviewer_is_local_guide,
    reviewer_level,
    reviewer_badge,
    reviewer_avatar_url,
    reviewer_profile_url,
    
    -- Restaurant snapshot at review time
    place_name,
    place_overall_rating,
    place_total_reviews,
    place_url,
    
    -- Owner response
    has_owner_response,
    owner_response_text,
    owner_response_timestamp_iso,
    owner_response_language,
    
    -- Attributes and context
    attr_meal_type,
    attr_price_range,
    attr_noise_level,
    attr_service_type,
    attr_wait_time,
    attr_group_size,
    attr_parking,
    attr_vegetarian,
    
    -- Admin/sources
    review_source,
    review_source_rating,
    crawl_timestamp,
    
    -- Timestamps
    timestamp_created_iso as review_created_at,
    timestamp_edited_iso as review_edited_at

from raw_reviews
where review_id is not null
