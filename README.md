# Stock Price Prediction

Predicts next-day stock closing price using Linear Regression and Polynomial
Regression, with a Streamlit web app for interactive predictions.

## Files

- `train_stock_models.py` — downloads historical stock data (via `yfinance`),
  engineers technical-indicator features, trains a Linear Regression model
  and a Polynomial Regression model, and saves them as `.pkl` files.
- `app.py` — Streamlit app that loads the saved models and predicts the
  next closing price for any ticker you enter.
- `requirements.txt` — Python dependencies.

## Setup

```bash
pip install -r requirements.txt
```

## 1. Train the models

```bash
python train_stock_models.py
```

Edit `TICKER` at the top of `train_stock_models.py` to train on a different
stock (default: `AAPL`). This creates:

- `stock_linear_model.pkl`
- `stock_poly_model.pkl`
- `stock_poly_features.pkl`

You can also train on your own CSV instead of live data — set
`USE_LOCAL_CSV = True` and `CSV_PATH = "your_file.csv"` in the script. The
CSV needs `Date, Open, High, Low, Close, Volume` columns.

## 2. Run the app

```bash
streamlit run app.py
```

Opens in your browser at `http://localhost:8501`. Enter any ticker symbol,
choose a model (Linear or Polynomial Regression), and click **Fetch &
Predict**.

## How it works

**Features used:** 5/10/20-day moving averages, 5-day volatility, 14-day
RSI, lagged closing prices (1/2/3/5 days back), and volume.

**Models:**
- Linear Regression — straightforward baseline
- Polynomial Regression (degree 2) — `PolynomialFeatures` + `LinearRegression`
  to capture non-linear relationships between features

## ⚠️ Disclaimer

This is an educational project, not a trading strategy. Stock prices are
influenced by many unpredictable real-world factors that these models
cannot see (news, earnings, macro events, sentiment). Backtested accuracy
does not guarantee future performance. Do not use this for real trading or
investment decisions.
