"""
STOCK PRICE PREDICTION - COMPLETE PIPELINE
WITH SAMPLE DATASET SUPPORT
=====================================================
Integrated Script: Data Collection → Cleaning → Validation → EDA → Features → Training → Evaluation

This version is optimized to work with sample_multi_stock_data.csv
which contains multiple stocks (AAPL, TSLA, MSFT, GOOGL, AMZN)

Features:
- Loads data from sample CSV with multiple stocks
- Auto-filters by ticker
- Cleans data (missing values, outliers, duplicates)
- Runs comprehensive EDA (statistics, correlations, analysis)
- Engineers 10 technical indicator features
- Trains Linear & Polynomial Regression models
- Evaluates and saves models as .pkl files

Usage:
    python train_stock_models_COMPLETE.py                           # AAPL (default)
    python train_stock_models_COMPLETE.py --ticker TSLA             # TSLA
    python train_stock_models_COMPLETE.py --ticker TCS.NS           # Any ticker
    python train_stock_models_COMPLETE.py --csv your_data.csv       # Custom CSV
"""

import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import argparse
import warnings
import os
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================================
# STAGE 1: DATA COLLECTION (CSV OR YFINANCE)
# ============================================================================

def load_data_csv(filepath, ticker=None):
    """
    Load stock data from CSV file.
    
    Supports:
    - Single stock CSV (Date, Open, High, Low, Close, Volume)
    - Multi-stock CSV (with Ticker column)
    
    If multi-stock, filters to specified ticker.
    """
    print(f"\n{'='*70}")
    print(f"[1] DATA COLLECTION - Loading from CSV")
    print(f"{'='*70}")
    print(f"File: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Loaded {len(df)} rows from file")
        
        # Check if file has Ticker column (multi-stock)
        ticker_col = None
        for col in df.columns:
            if col.lower() == 'ticker':
                ticker_col = col
                break
        
        if ticker_col:
            # Multi-stock file - filter by ticker
            available_tickers = df[ticker_col].unique()
            print(f"✓ Multi-stock file detected")
            print(f"  Available tickers: {', '.join(available_tickers)}")
            
            if ticker:
                df = df[df[ticker_col].astype(str).str.upper() == ticker.upper()]
                if len(df) == 0:
                    raise ValueError(f"Ticker '{ticker}' not found in CSV. Available: {list(available_tickers)}")
                print(f"✓ Filtered to ticker: {ticker}")
            else:
                raise ValueError(f"Multi-stock CSV requires --ticker argument. Available: {list(available_tickers)}")
        else:
            print(f"✓ Single-stock file detected")
        
        print(f"✓ Final rows for analysis: {len(df)}")
        return df
        
    except FileNotFoundError:
        print(f"✗ File not found: {filepath}")
        print(f"  Make sure {filepath} is in the current directory")
        raise
    except Exception as e:
        print(f"✗ Error loading CSV: {e}")
        raise


def download_data_yfinance(ticker, period='5y'):
    """Download stock data from Yahoo Finance."""
    print(f"\n{'='*70}")
    print(f"[1] DATA COLLECTION - Downloading from yfinance")
    print(f"{'='*70}")
    print(f"Ticker: {ticker}, Period: {period}")
    
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        print(f"✓ Downloaded {len(df)} rows")
        return df
    except Exception as e:
        print(f"✗ Error downloading data: {e}")
        raise


# ============================================================================
# STAGE 2: DATA CLEANING
# ============================================================================

def clean_data(df):
    """
    Clean and validate stock price data.
    
    Operations:
    - Standardize column names
    - Convert Date to datetime
    - Remove duplicate rows
    - Handle missing values
    - Detect and handle outliers (IQR method)
    - Validate OHLC logic
    - Sort by date
    """
    print(f"\n{'='*70}")
    print(f"[2] DATA CLEANING")
    print(f"{'='*70}")
    
    df = df.copy()
    initial_rows = len(df)
    
    # Step 1: Standardize column names (case-insensitive)
    print("\n[Step 1] Standardizing column names...")
    df_lower = {c.lower(): c for c in df.columns}
    rename_map = {}
    required = ['date', 'open', 'high', 'low', 'close', 'volume']
    
    for req in required:
        if req in df_lower:
            rename_map[df_lower[req]] = req.capitalize()
    
    df = df.rename(columns=rename_map)
    print(f"✓ Columns standardized: {list(df.columns[:6])}")
    
    # Step 2: Convert Date to datetime
    print("\n[Step 2] Converting Date column to datetime...")
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    print(f"✓ Date conversion complete")
    
    # Step 3: Remove duplicates
    print("\n[Step 3] Removing duplicates...")
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before - after > 0:
        print(f"✓ Removed {before - after} duplicate rows")
    else:
        print(f"✓ No duplicates found")
    
    # Step 4: Handle missing values
    print("\n[Step 4] Handling missing values...")
    before = len(df)
    df = df.dropna()
    after = len(df)
    if before - after > 0:
        print(f"✓ Dropped {before - after} rows with NaN values")
    else:
        print(f"✓ No missing values found")
    
    # Step 5: Detect and handle outliers (IQR method)
    print("\n[Step 5] Detecting and handling outliers (IQR method, multiplier=1.5)...")
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    outlier_count = 0
    
    for col in numeric_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            n_outliers = len(df[(df[col] < lower) | (df[col] > upper)])
            if n_outliers > 0:
                df[col] = df[col].clip(lower=lower, upper=upper)
                outlier_count += n_outliers
    
    if outlier_count > 0:
        print(f"✓ Capped {outlier_count} outlier values")
    else:
        print(f"✓ No outliers detected")
    
    # Step 6: Validate OHLC logic
    print("\n[Step 6] Validating OHLC logic (High ≥ Low, High ≥ Close, Low ≤ Open)...")
    invalid = df[
        (df['High'] < df['Low']) |
        (df['High'] < df['Close']) |
        (df['Low'] > df['Close'])
    ]
    if len(invalid) > 0:
        print(f"⚠ {len(invalid)} rows violate OHLC logic (flagged but not removed)")
        df['ohlc_valid'] = True
        df.loc[invalid.index, 'ohlc_valid'] = False
    else:
        print(f"✓ All rows pass OHLC validation")
    
    # Step 7: Sort by date
    print("\n[Step 7] Sorting by Date...")
    if 'Date' in df.columns:
        df = df.sort_values('Date').reset_index(drop=True)
    print(f"✓ Data sorted and index reset")
    
    # Summary
    print(f"\n[CLEANING SUMMARY]")
    print(f"Initial rows:  {initial_rows}")
    print(f"Final rows:    {len(df)}")
    print(f"Rows removed:  {initial_rows - len(df)}")
    print(f"Status: ✓ CLEAN")
    
    return df


# ============================================================================
# STAGE 3: DATA VALIDATION & QUALITY REPORT
# ============================================================================

def validate_data(df):
    """
    Post-cleaning validation checkpoint.

    Confirms the cleaned data is actually trustworthy before EDA/modeling
    spends time on it. Produces a pass/fail quality report rather than
    modifying the data further.

    Checks:
    - Minimum row count (enough history to engineer features)
    - No remaining nulls
    - No duplicate dates
    - Date continuity (large gaps flagged, e.g. long trading halts)
    - Required OHLCV columns present with correct dtypes
    - Value sanity (no zero/negative prices or volume)
    """
    print(f"\n{'='*70}")
    print(f"[3] DATA VALIDATION & QUALITY REPORT")
    print(f"{'='*70}")

    checks = []

    # Check 1: Minimum rows (need ~20+ for MA20/RSI to compute at all)
    min_rows_required = 30
    passed = len(df) >= min_rows_required
    checks.append(("Sufficient row count (>= 30)", passed, f"{len(df)} rows"))

    # Check 2: No nulls remain
    null_count = df.isnull().sum().sum()
    checks.append(("No missing values", null_count == 0, f"{null_count} nulls found"))

    # Check 3: No duplicate dates
    dup_dates = df['Date'].duplicated().sum() if 'Date' in df.columns else 0
    checks.append(("No duplicate dates", dup_dates == 0, f"{dup_dates} duplicate dates"))

    # Check 4: Date continuity (flag gaps > 7 calendar days as informational)
    gap_note = "N/A"
    large_gaps = 0
    if 'Date' in df.columns and len(df) > 1:
        gaps = df['Date'].sort_values().diff().dt.days.dropna()
        large_gaps = int((gaps > 7).sum())
        gap_note = f"{large_gaps} gap(s) > 7 days"
    checks.append(("Date continuity reviewed", True, gap_note))  # informational, not a hard fail

    # Check 5: Required columns present
    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [c for c in required_cols if c not in df.columns]
    checks.append(("Required OHLCV columns present", len(missing_cols) == 0,
                    f"missing: {missing_cols}" if missing_cols else "all present"))

    # Check 6: No zero/negative prices or volume
    price_cols = [c for c in ['Open', 'High', 'Low', 'Close'] if c in df.columns]
    bad_prices = int((df[price_cols] <= 0).sum().sum()) if price_cols else 0
    bad_volume = int((df['Volume'] <= 0).sum()) if 'Volume' in df.columns else 0
    checks.append(("Positive prices and volume", bad_prices == 0 and bad_volume == 0,
                    f"{bad_prices} bad price(s), {bad_volume} bad volume(s)"))

    # Print report
    print("\nQuality Checks:")
    print("-" * 70)
    all_passed = True
    for name, passed, detail in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        if not passed:
            all_passed = False
        print(f"  {status}  {name:<35} ({detail})")

    print(f"\n[VALIDATION SUMMARY]")
    if all_passed:
        print(f"Status: ✓ DATA IS VALID — proceeding to EDA")
    else:
        print(f"Status: ⚠ ISSUES DETECTED — review before trusting downstream results")
        print(f"(Pipeline will continue, but treat EDA/model outputs with caution)")

    return df


# ============================================================================
# STAGE 4: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

def run_eda(df):
    """
    Comprehensive exploratory data analysis.
    
    Analyses:
    - Summary statistics
    - Missing values
    - Data types
    - Correlation analysis
    - Return analysis
    - Volume analysis
    - Price ranges
    """
    print(f"\n{'='*70}")
    print(f"[4] EXPLORATORY DATA ANALYSIS (EDA)")
    print(f"{'='*70}")
    
    df = df.copy()
    
    # Compute derived features for analysis
    if 'Close' in df.columns:
        df['Daily_Return'] = df['Close'].pct_change() * 100
        df['Intraday_Volatility'] = ((df['High'] - df['Low']) / df['Close']) * 100
        df['Price_Range'] = df['High'] - df['Low']
    
    # --- SECTION 1: BASIC STATISTICS ---
    print(f"\n[Section 1] SUMMARY STATISTICS")
    print("-" * 70)
    
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if all(col in df.columns for col in numeric_cols):
        print("\nPrice Statistics (OHLCV):")
        print(f"  Trading Days: {len(df):>12,.0f}")
        print(f"  Close Mean:   ${df['Close'].mean():>12.2f}")
        print(f"  Close Std:    ${df['Close'].std():>12.2f}")
        print(f"  Close Min:    ${df['Close'].min():>12.2f}")
        print(f"  Close Max:    ${df['Close'].max():>12.2f}")
        print(f"  Price Range:  ${df['Close'].max() - df['Close'].min():>12.2f}")
        
        print("\nVolume Statistics:")
        print(f"  Mean Volume:  {df['Volume'].mean():>12,.0f}")
        print(f"  Min Volume:   {df['Volume'].min():>12,.0f}")
        print(f"  Max Volume:   {df['Volume'].max():>12,.0f}")
    
    # --- SECTION 2: TIME SERIES INFO ---
    print(f"\n[Section 2] TIME SERIES INFORMATION")
    print("-" * 70)
    
    if 'Date' in df.columns:
        print(f"Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        print(f"Total Trading Days: {(df['Date'].max() - df['Date'].min()).days} calendar days")
    
    # --- SECTION 3: RETURNS ANALYSIS ---
    print(f"\n[Section 3] RETURNS ANALYSIS")
    print("-" * 70)
    
    if 'Daily_Return' in df.columns:
        print(f"Daily Return Stats:")
        print(f"  Mean:     {df['Daily_Return'].mean():>10.4f}%")
        print(f"  Std Dev:  {df['Daily_Return'].std():>10.4f}%")
        print(f"  Min:      {df['Daily_Return'].min():>10.4f}%")
        print(f"  Max:      {df['Daily_Return'].max():>10.4f}%")
        print(f"  Skewness: {df['Daily_Return'].skew():>10.4f}")
        print(f"  Kurtosis: {df['Daily_Return'].kurtosis():>10.4f}")
        
        # Up/Down days
        up_days = (df['Close'] > df['Open']).sum()
        down_days = (df['Close'] < df['Open']).sum()
        up_pct = (up_days / len(df)) * 100
        print(f"\nUp Days:   {up_days:>4} ({up_pct:>5.1f}%)")
        print(f"Down Days: {down_days:>4} ({100-up_pct:>5.1f}%)")
    
    # --- SECTION 4: VOLATILITY ANALYSIS ---
    print(f"\n[Section 4] VOLATILITY ANALYSIS")
    print("-" * 70)
    
    if 'Intraday_Volatility' in df.columns:
        print(f"Intraday Volatility (High-Low % of Close):")
        print(f"  Mean: {df['Intraday_Volatility'].mean():>10.4f}%")
        print(f"  Std:  {df['Intraday_Volatility'].std():>10.4f}%")
        print(f"  Min:  {df['Intraday_Volatility'].min():>10.4f}%")
        print(f"  Max:  {df['Intraday_Volatility'].max():>10.4f}%")
    
    # --- SECTION 5: CORRELATION MATRIX ---
    print(f"\n[Section 5] CORRELATION MATRIX")
    print("-" * 70)
    
    if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
        corr = df[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
        print("\nPearson Correlation:")
        print(corr.to_string())
    
    # --- SECTION 6: DATA QUALITY ---
    print(f"\n[Section 6] DATA QUALITY CHECK")
    print("-" * 70)
    
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✓ No missing values found")
    else:
        print("Missing values per column:")
        print(missing[missing > 0])
    
    print(f"\n[EDA SUMMARY]")
    print(f"✓ Comprehensive analysis complete")
    print(f"Status: ✓ READY FOR MODELING")
    
    return df


# ============================================================================
# STAGE 5: FEATURE ENGINEERING
# ============================================================================

def build_features(df):
    """
    Engineer 10 technical indicator features from OHLCV data.
    
    Features created:
    1-3:   MA5, MA10, MA20 (moving averages)
    4:     Volatility5 (5-day rolling std of returns)
    5:     RSI (14-day Relative Strength Index)
    6-9:   Lag1, Lag2, Lag3, Lag5 (lagged closes)
    10:    Volume
    Target: Next day's closing price
    """
    print(f"\n{'='*70}")
    print(f"[5] FEATURE ENGINEERING")
    print(f"{'='*70}")
    
    d = df.copy()
    
    print("\nCreating features...")
    
    # Moving averages
    d["MA5"] = d["Close"].rolling(5).mean()
    d["MA10"] = d["Close"].rolling(10).mean()
    d["MA20"] = d["Close"].rolling(20).mean()
    print("✓ Moving averages (MA5, MA10, MA20)")
    
    # Volatility
    d["Return"] = d["Close"].pct_change()
    d["Volatility5"] = d["Return"].rolling(5).std()
    print("✓ Volatility (5-day rolling std)")
    
    # RSI (14-day)
    delta = d["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    d["RSI"] = 100 - (100 / (1 + rs))
    print("✓ RSI (14-day Relative Strength Index)")
    
    # Lagged closes
    for lag in [1, 2, 3, 5]:
        d[f"Lag{lag}"] = d["Close"].shift(lag)
    print("✓ Lagged closes (Lag1, Lag2, Lag3, Lag5)")
    
    # Target (next day's close)
    d["Target"] = d["Close"].shift(-1)
    print("✓ Target (next day's closing price)")
    
    # Remove rows with NaN
    before = len(d)
    d = d.dropna()
    after = len(d)
    
    print(f"\nFeature Matrix:")
    print(f"  Total features: 10")
    print(f"  Rows with NaN removed: {before - after}")
    print(f"  Final rows ready for modeling: {after}")
    print(f"Status: ✓ FEATURES READY")
    
    return d


# ============================================================================
# STAGE 6: MODEL TRAINING & EVALUATION
# ============================================================================

def train_and_evaluate(feat):
    """
    Train two regression models and evaluate their performance.
    
    Models:
    1. Linear Regression
    2. Polynomial Regression (degree 2)
    
    Metrics: MAE, RMSE, R²
    """
    print(f"\n{'='*70}")
    print(f"[6] MODEL TRAINING & EVALUATION")
    print(f"{'='*70}")
    
    FEATURE_ORDER = [
        "MA5", "MA10", "MA20", "Volatility5", "RSI",
        "Lag1", "Lag2", "Lag3", "Lag5", "Volume",
    ]
    
    X = feat[FEATURE_ORDER]
    y = feat["Target"]
    
    # Train/test split (80/20 time-ordered split)
    split = int(len(feat) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    print(f"\nTrain/Test Split (Time-Ordered 80/20):")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples:     {len(X_test)}")
    print(f"  Features:         10")
    
    # ====== MODEL 1: LINEAR REGRESSION ======
    print(f"\n[Model 1] LINEAR REGRESSION")
    print("-" * 70)
    
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    linear_preds = linear_model.predict(X_test)
    
    linear_mae = mean_absolute_error(y_test, linear_preds)
    linear_rmse = mean_squared_error(y_test, linear_preds) ** 0.5
    linear_r2 = r2_score(y_test, linear_preds)
    
    print(f"Equation: y = w₁·x₁ + w₂·x₂ + ... + w₁₀·x₁₀ + b")
    print(f"\nResults (on test set):")
    print(f"  MAE (Mean Absolute Error):  ${linear_mae:>8.2f}")
    print(f"  RMSE (Root Mean Sq Error):  ${linear_rmse:>8.2f}")
    print(f"  R² (Coefficient of Determination): {linear_r2:>8.4f}")
    print(f"\n✓ Linear model trained")
    
    # ====== MODEL 2: POLYNOMIAL REGRESSION ======
    print(f"\n[Model 2] POLYNOMIAL REGRESSION (degree=2)")
    print("-" * 70)
    
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    poly_preds = poly_model.predict(X_test_poly)
    
    poly_mae = mean_absolute_error(y_test, poly_preds)
    poly_rmse = mean_squared_error(y_test, poly_preds) ** 0.5
    poly_r2 = r2_score(y_test, poly_preds)
    
    print(f"Equation: y = w₁·x₁ + w₂·x₁² + w₃·x₁·x₂ + ... + b")
    print(f"Features expanded: 10 → 65 polynomial terms")
    print(f"\nResults (on test set):")
    print(f"  MAE (Mean Absolute Error):  ${poly_mae:>8.2f}")
    print(f"  RMSE (Root Mean Sq Error):  ${poly_rmse:>8.2f}")
    print(f"  R² (Coefficient of Determination): {poly_r2:>8.4f}")
    print(f"\n✓ Polynomial model trained")
    
    # ====== COMPARISON ======
    print(f"\n[MODEL COMPARISON]")
    print("-" * 70)
    
    print(f"\n{'Metric':<30} {'Linear':<15} {'Polynomial':<15}")
    print("-" * 70)
    print(f"{'MAE':<30} ${linear_mae:<14.2f} ${poly_mae:<14.2f}")
    print(f"{'RMSE':<30} ${linear_rmse:<14.2f} ${poly_rmse:<14.2f}")
    print(f"{'R²':<30} {linear_r2:<14.4f} {poly_r2:<14.4f}")
    
    if linear_mae < poly_mae:
        print(f"\n✓ LINEAR REGRESSION performs better (lower MAE)")
    else:
        print(f"\n✓ POLYNOMIAL REGRESSION performs better (lower MAE)")
    
    print(f"\nStatus: ✓ TRAINING COMPLETE")
    
    return linear_model, poly_model, poly


# ============================================================================
# STAGE 7: SAVE MODELS
# ============================================================================

def save_models(linear_model, poly_model, poly):
    """Save trained models to disk as .pkl files."""
    print(f"\n{'='*70}")
    print(f"[7] SAVING MODELS")
    print(f"{'='*70}")
    
    try:
        joblib.dump(linear_model, "stock_linear_model.pkl")
        print("✓ Saved: stock_linear_model.pkl")
        
        joblib.dump(poly_model, "stock_poly_model.pkl")
        print("✓ Saved: stock_poly_model.pkl")
        
        joblib.dump(poly, "stock_poly_features.pkl")
        print("✓ Saved: stock_poly_features.pkl")
        
        print(f"\nStatus: ✓ All models saved successfully")
        print(f"\nNext step: streamlit run app.py")
        
    except Exception as e:
        print(f"✗ Error saving models: {e}")
        raise


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(ticker='AAPL', period='5y', csv_path='sample_multi_stock_data.csv'):
    """Run the complete pipeline."""
    
    print("\n" + "="*70)
    print("STOCK PRICE PREDICTION - COMPLETE PIPELINE")
    print("="*70)
    
    try:
        # STAGE 1: Data Collection
        if os.path.exists(csv_path):
            # Use local CSV
            df = load_data_csv(csv_path, ticker=ticker)
        else:
            # Download from yfinance if CSV doesn't exist
            print(f"\nℹ {csv_path} not found, downloading from yfinance instead...")
            df = download_data_yfinance(ticker, period)
        
        # STAGE 2: Data Cleaning
        df = clean_data(df)
        
        # STAGE 3: Data Validation & Quality Report
        df = validate_data(df)
        
        # STAGE 4: EDA
        df = run_eda(df)
        
        # STAGE 5: Feature Engineering
        feat = build_features(df)
        
        # STAGE 6: Training & Evaluation
        linear_model, poly_model, poly = train_and_evaluate(feat)
        
        # STAGE 7: Save Models
        save_models(linear_model, poly_model, poly)
        
        # Final Summary
        print(f"\n{'='*70}")
        print("✓✓✓ PIPELINE COMPLETE ✓✓✓")
        print("="*70)
        print(f"\nAll stages completed successfully!")
        print(f"\nNext steps:")
        print(f"  1. Review the output above")
        print(f"  2. Run the web app:")
        print(f"     streamlit run app.py")
        print(f"  3. Open http://localhost:8501 in your browser")
        print(f"  4. Type your ticker to get predictions!")
        print(f"\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stock Price Prediction - Complete Pipeline with Dataset Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_stock_models_COMPLETE.py                    # AAPL from sample dataset
  python train_stock_models_COMPLETE.py --ticker TSLA      # TSLA from sample dataset
  python train_stock_models_COMPLETE.py --ticker TCS.NS    # TCS (NSE) from sample dataset
  python train_stock_models_COMPLETE.py --csv my_data.csv  # Your own CSV file
        """
    )
    parser.add_argument('--ticker', type=str, default='AAPL', 
                        help='Stock ticker to analyze (default: AAPL)')
    parser.add_argument('--csv', type=str, default='sample_multi_stock_data.csv',
                        help='CSV file path (default: sample_multi_stock_data.csv)')
    
    args = parser.parse_args()
    
    main(ticker=args.ticker, csv_path=args.csv)
