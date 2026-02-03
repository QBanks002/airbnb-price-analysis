-- Airbnb Listings Table Schema
CREATE TABLE IF NOT EXISTS airbnb_listings (
    id BIGINT PRIMARY KEY,
    neighbourhood TEXT,
    room_type TEXT,
    price NUMERIC,
    minimum_nights INTEGER,
    number_of_reviews INTEGER,
    reviews_per_month NUMERIC,
    availability_365 INTEGER,
    calculated_host_listings_count INTEGER,
    accommodates INTEGER,
    bedrooms INTEGER,
    bathrooms_text TEXT,
    review_scores_rating NUMERIC,
    host_is_superhost TEXT
);
