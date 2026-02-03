import os
import pandas as pd
import psycopg2
from sqlalchemy import create_engine

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

DB_USER = os.getenv('POSTGRES_USER', 'airbnb')
DB_PASS = os.getenv('POSTGRES_PASSWORD', 'airbnb_pass')
DB_HOST = os.getenv('POSTGRES_HOST', 'localhost')
DB_PORT = os.getenv('POSTGRES_PORT', '5432')
DB_NAME = os.getenv('POSTGRES_DB', 'airbnb_db')

CSV_URL = "https://data.insideairbnb.com/united-states/tx/austin/2025-09-16/visualisations/listings.csv"

# Columns to match schema
COLUMNS = [
    "id", "neighbourhood", "room_type", "price",
    "minimum_nights", "number_of_reviews", "reviews_per_month",
    "availability_365", "calculated_host_listings_count",
    "accommodates", "bedrooms", "bathrooms_text",
    "review_scores_rating", "host_is_superhost"
]

def main():
    print("Loading data from CSV URL...")
    df = pd.read_csv(CSV_URL)
    df = df[[c for c in COLUMNS if c in df.columns]].copy()
    df["price"] = df["price"].replace(r"[\$,]", "", regex=True).astype(float)
    print(f"Loaded {len(df):,} rows.")

    # Create SQLAlchemy engine
    url = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(url)

    print("Uploading to database...")
    df.to_sql("airbnb_listings", engine, if_exists="replace", index=False)
    print("Upload complete.")

if __name__ == "__main__":
    main()
