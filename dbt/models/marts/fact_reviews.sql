{{ config(materialized='table') }}

-- Fact table for reviews
-- Central fact table with all reviews, measures, and foreign keys to dimensions
-- Each row is one review

-- Primary Key: review_id
-- Foreign Keys:
--   - restaurant_id → dim_restaurants.restaurant_id (place_url)
--   - reviewer_user_id → dim_reviewers.reviewer_user_id
-- Grain: One row per review

select
    -- Primary key
    review_id,
    
    -- Foreign keys to dimensions
    place_url as restaurant_id,
    reviewer_user_id,
    
    -- Review content and ratings (measures)
    review_text,
    review_text_length,
    rating_stars,
    review_language,
    sub_rating_food,
    sub_rating_service,
    sub_rating_atmosphere,
    review_detail_level,
    
    -- Review behavior features (measures)
    has_review_text,
    has_sub_ratings,
    has_owner_response,
    was_edited,
    edit_delta_hours,
    
    -- Review metadata
    review_images_count,
    review_date,
    review_created_at,
    review_edited_at,
    
    -- Restaurant snapshot
    place_name,
    place_overall_rating,
    place_total_reviews,
    
    -- Reviewer snapshot
    reviewer_name,
    reviewer_review_count,
    reviewer_photo_count,
    reviewer_photo_review_ratio,
    reviewer_is_local_guide,
    reviewer_level,
    reviewer_badge,
    
    -- Owner response metadata
    owner_response_text,
    owner_response_timestamp_iso,
    owner_response_language,
    
    -- Context attributes
    attr_meal_type,
    attr_price_range,
    attr_noise_level,
    attr_service_type,
    attr_wait_time,
    attr_group_size,
    attr_parking,
    attr_vegetarian,
    
    -- Admin/source metadata
    review_source,
    review_source_rating,
    crawl_timestamp
    
from {{ ref('stg_reviews') }}
