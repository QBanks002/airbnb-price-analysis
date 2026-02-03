"""
Multiple Linear Regression Analysis of Airbnb Listing Prices
Austin, Texas
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

sns.set_style("whitegrid")
DATA_URL = (
    "https://data.insideairbnb.com/united-states/tx/austin/"
    "2025-09-16/visualisations/listings.csv"
)
RANDOM_STATE = 42

# -------------------------------------------------------------------
# Data Loading
# -------------------------------------------------------------------

def load_data(url: str) -> pd.DataFrame:
    """Load Airbnb listings data from Inside Airbnb."""
    print("Loading Austin Airbnb data...")
    df = pd.read_csv(url)
    print(f"Rows loaded: {len(df):,}")
    return df

# -------------------------------------------------------------------
# Data Cleaning & Preparation
# -------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare Airbnb data for analysis."""
    
    possible_cols = [
        "id", "neighbourhood", "room_type", "price",
        "minimum_nights", "number_of_reviews", "reviews_per_month",
        "availability_365", "calculated_host_listings_count",
        "accommodates", "bedrooms", "bathrooms_text",
        "review_scores_rating", "host_is_superhost"
    ]
    
    df = df[[c for c in possible_cols if c in df.columns]].copy()
    
    # Price cleaning
    df["price"] = (
        df["price"]
        .replace(r"[\$,]", "", regex=True)
        .astype(float)
    )
    
    # Filtering
    df = df[(df["price"] >= 10) & (df["price"] <= 1000)]
    df = df[df["availability_365"] > 0]
    
    # Missing values
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
    if "review_scores_rating" in df.columns:
        df["review_scores_rating"] = df["review_scores_rating"].fillna(
            df["review_scores_rating"].median()
        )
    
    df = df.dropna(subset=["price", "minimum_nights", "availability_365"])
    print(f"After cleaning: {len(df):,} rows")
    
    return df

# -------------------------------------------------------------------
# Feature Engineering
# -------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for regression modeling."""
    
    # Group rare neighborhoods
    if "neighbourhood" in df.columns:
        counts = df["neighbourhood"].value_counts()
        rare = counts[counts < 50].index
        df["neighbourhood"] = df["neighbourhood"].replace(rare, "Other")
    
    # IQR clipping
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    exclude = ["id", "price", "review_scores_rating"]
    
    for col in numeric_cols:
        if col in exclude:
            continue
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df[col] = df[col].clip(lower, upper)
    
    # Log-transform target
    df["log_price"] = np.log1p(df["price"])
    
    # Dummy encoding
    cat_cols = ["room_type"]
    if "neighbourhood" in df.columns:
        cat_cols.append("neighbourhood")
    
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
    
    return df

# -------------------------------------------------------------------
# Exploratory Data Analysis
# -------------------------------------------------------------------

def run_eda(df: pd.DataFrame) -> None:
    """Generate exploratory visualizations."""
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(df["price"], bins=60, kde=True)
    plt.title("Nightly Price Distribution (USD)")
    
    plt.subplot(1, 2, 2)
    sns.histplot(df["log_price"], bins=60, kde=True)
    plt.title("Log-Price Distribution")
    
    plt.tight_layout()
    plt.show()
    
    # Normality check
    stats.probplot(df["log_price"], dist="norm", plot=plt)
    plt.title("Q-Q Plot of log_price")
    plt.show()

# -------------------------------------------------------------------
# Modeling
# -------------------------------------------------------------------

def train_model(df: pd.DataFrame):
    """Train and evaluate a multiple linear regression model."""
    
    X = df.drop(columns=["id", "price", "log_price"], errors="ignore")
    y = df["log_price"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    
    X_train_sm = sm.add_constant(X_train)
    X_test_sm = sm.add_constant(X_test)
    
    model = sm.OLS(y_train, X_train_sm).fit()
    print(model.summary())
    
    # Evaluation
    y_pred = model.predict(X_test_sm)
    
    y_test_orig = np.expm1(y_test)
    y_pred_orig = np.expm1(y_pred)
    
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    mae = mean_absolute_error(y_test_orig, y_pred_orig)
    r2 = r2_score(y_test_orig, y_pred_orig)
    
    print("\nModel Performance (Test Set)")
    print("-" * 40)
    print(f"RMSE: ${rmse:,.2f}")
    print(f"MAE:  ${mae:,.2f}")
    print(f"R²:   {r2:.4f}")
    print(f"Adj R² (train): {model.rsquared_adj:.4f}")
    
    return model

# -------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------

def main():
    df = load_data(DATA_URL)
    df = clean_data(df)
    df = engineer_features(df)
    run_eda(df)
    train_model(df)
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()


