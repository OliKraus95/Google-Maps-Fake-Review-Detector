{{ config(materialized='table') }}

-- Dimension table for restaurants
-- Contains all unique restaurants with their attributes
-- Used in fact_reviews as the restaurant dimension

-- Primary Key: restaurant_id (= place_url)
-- Grain: One row per unique restaurant

select
    place_url as restaurant_id,
    place_name,
    place_overall_rating,
    place_total_reviews
from {{ ref('stg_restaurants') }}
