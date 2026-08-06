import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import yfinance as yf

# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Price Prediction System")
st.write("Predict next-day stock closing price using Machine Learning")

# ---------------------------------------------------------------------
# Load models (with error handling in case files are missing)
# ---------------------------------------------------------------------
MODEL_FILES = {
    "linear": "stock_linear_model.pkl",
    "poly_model": "stock_polynomial_model.pkl",
    "poly_features": "stock_polynomial_features.pkl",
}

missing = [name for name, path in MODEL_FILES.items() if not os.path.exists(path)]
if missing:
    st.error(
        f"Missing model file(s): {', '.join(MODEL_FILES[m] for m in missing)}. "
        "Run train_stock_models.py first to generate them."
    )
    st.stop()

linear = joblib.load(MODEL_FILES["linear"])
poly_model = joblib.load(MODEL_FILES["poly_model"])
poly = joblib.load(MODEL_FILES["poly_features"])

# Must match FEATURE_ORDER used in train_stock_models.py
FEATURE_ORDER = [
    "MA5", "MA10", "MA20", "Volatility5", "RSI",
    "Lag1", "Lag2", "Lag3", "Lag5", "Volume",
]

# ---------------------------------------------------------------------
# Sidebar - model choice + ticker
# ---------------------------------------------------------------------
model_choice = st.sidebar.selectbox(
    "Choose Model",
    ("Linear Regression", "Polynomial Regression")
)

ticker = st.sidebar.text_input("Ticker Symbol", "AAPL")
period = st.sidebar.selectbox("History Period", ["6mo", "1y", "2y", "5y"], index=2)


# ---------------------------------------------------------------------
# Feature engineering — must match train_stock_models.py exactly
# ---------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Return"] = d["Close"].pct_change()
    d["MA5"] = d["Close"].rolling(5).mean()
    d["MA10"] = d["Close"].rolling(10).mean()
    d["MA20"] = d["Close"].rolling(20).mean()
    d["Volatility5"] = d["Return"].rolling(5).std()

    delta = d["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    d["RSI"] = 100 - (100 / (1 + rs))

    for lag in [1, 2, 3, 5]:
        d[f"Lag{lag}"] = d["Close"].shift(lag)

    return d.dropna()


@st.cache_data(ttl=3600)
def load_ticker_data(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# ---------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------
if st.sidebar.button("Fetch & Predict") or "raw_data" in st.session_state:
    with st.spinner(f"Fetching data for {ticker}..."):
        try:
            raw = load_ticker_data(ticker, period)
            if raw.empty:
                st.error(f"No data found for ticker '{ticker}'. Check the symbol and try again.")
                st.stop()
            st.session_state["raw_data"] = raw
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            st.stop()

    raw = st.session_state["raw_data"]
    feat = build_features(raw)

    if feat.empty:
        st.error("Not enough historical data to compute features. Try a longer period.")
        st.stop()

    latest = feat[FEATURE_ORDER].iloc[[-1]]
    last_close = float(raw["Close"].iloc[-1])
    last_date = feat.index[-1].strftime("%Y-%m-%d")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"{ticker.upper()} — Recent Price History")
        st.line_chart(raw["Close"].tail(120))

    with col2:
        st.subheader("Latest Data")
        st.metric("Last Close", f"${last_close:,.2f}")
        st.caption(f"As of {last_date}")

        with st.spinner("Predicting..."):
            try:
                if model_choice == "Linear Regression":
                    prediction = linear.predict(latest)
                else:
                    poly_input = poly.transform(latest)
                    prediction = poly_model.predict(poly_input)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        predicted_price = float(prediction[0])
        change = predicted_price - last_close
        change_pct = (change / last_close) * 100

        st.metric(
            "Predicted Next Close",
            f"${predicted_price:,.2f}",
            f"{change:+.2f} ({change_pct:+.2f}%)"
        )
        st.success(f"Model used: {model_choice}")

    st.caption(
        "⚠️ Educational example only — not financial advice. "
        "Predictions are based on limited historical price patterns and "
        "should not be used for real trading decisions."
    )
else:
    st.info("Enter a ticker in the sidebar and click **Fetch & Predict** to get started.")
