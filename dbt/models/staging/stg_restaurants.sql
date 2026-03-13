{{ config(materialized='view') }}

-- Staging model for restaurants
-- Deduplicates restaurants from raw_reviews (keeping latest crawl_timestamp snapshot)
-- Each place_url (restaurant ID) appears once with their most recent profile snapshot

with restaurant_dedup as (
    select
        place_url,
        place_name,
        place_overall_rating,
        place_total_reviews,
        crawl_timestamp,
        row_number() over (partition by place_url order by crawl_timestamp desc) as rn
    from raw_reviews
    where place_url is not null
)

select
    place_url,
    place_name,
    place_overall_rating,
    place_total_reviews
from restaurant_dedup
where rn = 1
