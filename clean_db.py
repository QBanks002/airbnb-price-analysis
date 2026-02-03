import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv('POSTGRES_USER', 'airbnb')
DB_PASS = os.getenv('POSTGRES_PASSWORD', 'airbnb_pass')
DB_HOST = os.getenv('POSTGRES_HOST', 'localhost')
DB_PORT = os.getenv('POSTGRES_PORT', '5432')
DB_NAME = os.getenv('POSTGRES_DB', 'airbnb_db')

def clean_airbnb_table():
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        host=DB_HOST,
        port=DB_PORT
    )
    cur = conn.cursor()
    print("Cleaning airbnb_listings table...")
    # Remove out-of-range prices
    cur.execute("""
        DELETE FROM airbnb_listings
        WHERE price < 10 OR price > 1000
    """)
    # Remove listings with zero availability
    cur.execute("""
        DELETE FROM airbnb_listings
        WHERE availability_365 <= 0
    """)
    # Fill NULL reviews_per_month with 0
    cur.execute("""
        UPDATE airbnb_listings
        SET reviews_per_month = 0
        WHERE reviews_per_month IS NULL
    """)
    # Fill NULL review_scores_rating with median
    cur.execute("""
        WITH median AS (
            SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY review_scores_rating) AS med
            FROM airbnb_listings
            WHERE review_scores_rating IS NOT NULL
        )
        UPDATE airbnb_listings
        SET review_scores_rating = (SELECT med FROM median)
        WHERE review_scores_rating IS NULL
    """)
    # Remove rows missing critical fields
    cur.execute("""
        DELETE FROM airbnb_listings
        WHERE price IS NULL OR minimum_nights IS NULL OR availability_365 IS NULL
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("Database cleaning complete.")

if __name__ == "__main__":
    clean_airbnb_table()
