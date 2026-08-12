import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# Custom styling — dark trading-terminal theme with amber/emerald accents
# ---------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-deep:      #0B0E14;
        --bg-panel:     #131826;
        --bg-panel-2:   #1A2233;
        --border-soft:  #232B3D;
        --amber:        #F5A623;
        --amber-soft:   rgba(245, 166, 35, 0.14);
        --emerald:      #2ECC91;
        --rose:         #FF5C5C;
        --text-hi:      #F4F6FA;
        --text-mid:     #A9B2C3;
        --text-dim:     #6B7488;
    }

    html, body, [class*="css"]  {
        font-family: 'Space Grotesk', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 0%, rgba(245,166,35,0.07), transparent 40%),
            radial-gradient(circle at 85% 15%, rgba(46,204,145,0.06), transparent 35%),
            var(--bg-deep);
    }

    /* Hide default Streamlit chrome */
    #MainMenu, footer, header {visibility: hidden;}

    /* ---------------- Sidebar ---------------- */
    section[data-testid="stSidebar"] {
        background: var(--bg-panel);
        border-right: 1px solid var(--border-soft);
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* ---------------- Hero header ---------------- */
    .hero-wrap {
        position: relative;
        padding: 2.1rem 2.4rem;
        border-radius: 18px;
        margin-bottom: 1.6rem;
        background: linear-gradient(135deg, rgba(245,166,35,0.10) 0%, rgba(19,24,38,0.4) 60%);
        border: 1px solid var(--border-soft);
        overflow: hidden;
    }
    .hero-eyebrow {
        color: var(--amber);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        letter-spacing: 0.18em;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: var(--text-hi);
        margin: 0;
        line-height: 1.15;
    }
    .hero-sub {
        color: var(--text-mid);
        font-size: 1rem;
        margin-top: 0.5rem;
        max-width: 640px;
    }
    .ticker-tape {
        margin-top: 1.1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: var(--text-dim);
        letter-spacing: 0.04em;
    }

    /* ---------------- Section labels ---------------- */
    .section-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.14em;
        color: var(--amber);
        font-weight: 600;
        margin: 0.2rem 0 0.7rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-label::after {
        content: "";
        flex: 1;
        height: 1px;
        background: var(--border-soft);
    }

    /* ---------------- Metric cards ---------------- */
    .metric-card {
        background: var(--bg-panel);
        border: 1px solid var(--border-soft);
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.9rem;
        transition: border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: var(--amber);
    }
    .metric-label {
        color: var(--text-dim);
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    .metric-value {
        color: var(--text-hi);
        font-size: 1.9rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.1;
    }
    .metric-caption {
        color: var(--text-dim);
        font-size: 0.78rem;
        margin-top: 0.4rem;
    }
    .delta-up {
        color: var(--emerald);
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    .delta-down {
        color: var(--rose);
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ---------------- Prediction spotlight ---------------- */
    .predict-card {
        background: linear-gradient(155deg, rgba(245,166,35,0.16), rgba(19,24,38,0.5));
        border: 1px solid rgba(245,166,35,0.35);
        border-radius: 16px;
        padding: 1.4rem 1.5rem;
        margin-bottom: 1rem;
    }
    .predict-label {
        color: var(--amber);
        font-size: 0.78rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    .predict-value {
        font-size: 2.6rem;
        font-weight: 700;
        color: var(--text-hi);
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.05;
    }

    /* ---------------- Model pill ---------------- */
    .model-pill {
        display: inline-block;
        background: var(--amber-soft);
        color: var(--amber);
        border: 1px solid rgba(245,166,35,0.4);
        border-radius: 999px;
        padding: 0.3rem 0.9rem;
        font-size: 0.8rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 0.6rem;
    }

    /* ---------------- Disclaimer ---------------- */
    .disclaimer {
        margin-top: 1.4rem;
        padding: 0.9rem 1.1rem;
        background: rgba(255,92,92,0.06);
        border: 1px solid rgba(255,92,92,0.25);
        border-radius: 10px;
        color: var(--text-mid);
        font-size: 0.82rem;
        line-height: 1.5;
    }

    /* ---------------- Sidebar labels ---------------- */
    .sb-title {
        color: var(--text-hi);
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 0.2rem;
    }
    .sb-sub {
        color: var(--text-dim);
        font-size: 0.8rem;
        margin-bottom: 1.4rem;
    }

    /* Streamlit input tweaks */
    div[data-baseweb="select"] > div, .stTextInput input {
        background-color: var(--bg-panel-2) !important;
        border: 1px solid var(--border-soft) !important;
        color: var(--text-hi) !important;
        border-radius: 8px !important;
    }
    .stButton button {
        background: linear-gradient(135deg, var(--amber), #E08E00);
        color: #14100A;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        width: 100%;
        letter-spacing: 0.02em;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(245,166,35,0.35);
        color: #14100A;
    }

    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 3.5rem 1rem;
        background: var(--bg-panel);
        border: 1px dashed var(--border-soft);
        border-radius: 16px;
        color: var(--text-mid);
    }
    .empty-state .glyph {
        font-size: 2.6rem;
        margin-bottom: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------
st.markdown("""
<div class="hero-wrap">
    <div class="hero-eyebrow">MACHINE LEARNING · MARKET FORECASTING</div>
    <div class="hero-title">📈 Stock Price Prediction System</div>
    <div class="hero-sub">
        Forecast tomorrow's closing price from today's OHLCV data using
        Linear and Polynomial Regression models trained on historical trends.
    </div>
    <div class="ticker-tape">AAPL · MSFT · GOOGL · AMZN · TSLA &nbsp;|&nbsp; MODEL-DRIVEN &nbsp;|&nbsp; NOT FINANCIAL ADVICE</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Load models (with error handling in case files are missing)
# ---------------------------------------------------------------------
MODEL_FILES = {
    "linear": "stock_linear_model.pkl",
    "poly_model": "stock_poly_model.pkl",
    "poly_features": "stock_poly_features.pkl",
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

# ---------------------------------------------------------------------
# Load local dataset (same data the models were trained on)
# ---------------------------------------------------------------------
DATASET_CANDIDATES = ["sample_multi_stock_data_5yr.csv", "sample_multi_stock_data.csv"]
DATASET_PATH = next((p for p in DATASET_CANDIDATES if os.path.exists(p)), None)

if DATASET_PATH is None:
    st.error(
        f"No dataset found. Expected one of: {', '.join(DATASET_CANDIDATES)} "
        "in the same folder as app.py."
    )
    st.stop()

@st.cache_data
def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

dataset = load_dataset(DATASET_PATH)
AVAILABLE_TICKERS = sorted(dataset["Ticker"].unique().tolist())

# Must match FEATURE_ORDER used in train_stock_models.py
FEATURE_ORDER = [
    "MA5", "MA10", "MA20", "Volatility5", "RSI",
    "Lag1", "Lag2", "Lag3", "Lag5", "Volume",
]

# ---------------------------------------------------------------------
# Sidebar - model choice + ticker
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sb-title">⚙️ Controls</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-sub">Configure your prediction request</div>', unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Choose Model",
        ("Linear Regression", "Polynomial Regression")
    )

    ticker = st.selectbox("Ticker Symbol", AVAILABLE_TICKERS)

    st.markdown("<br>", unsafe_allow_html=True)
    fetch_clicked = st.button("🔮  Predict from Dataset")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">ABOUT</div>', unsafe_allow_html=True)
    st.caption(
        f"Predicting from your own dataset ({DATASET_PATH}) — the same data "
        "the models were trained on. No live internet fetch involved."
    )

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


def get_ticker_history(ticker: str) -> pd.DataFrame:
    """Pull one ticker's OHLCV history from the local dataset, sorted by date."""
    df = dataset[dataset["Ticker"] == ticker].copy()
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.set_index("Date")
    return df[["Open", "High", "Low", "Close", "Volume"]]


# ---------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------
if fetch_clicked or "raw_data" in st.session_state:
    with st.spinner(f"Loading {ticker} history from dataset..."):
        try:
            raw = get_ticker_history(ticker)
            if raw.empty:
                st.error(f"No data found for ticker '{ticker}' in {DATASET_PATH}.")
                st.stop()
            st.session_state["raw_data"] = raw
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
            st.stop()

    raw = st.session_state["raw_data"]
    feat = build_features(raw)

    if feat.empty:
        st.error("Not enough historical data for this ticker to compute features.")
        st.stop()

    latest = feat[FEATURE_ORDER].iloc[[-1]]
    last_close = float(raw["Close"].iloc[-1])
    last_date = feat.index[-1].strftime("%Y-%m-%d")
    day_high = float(raw["High"].iloc[-1])
    day_low = float(raw["Low"].iloc[-1])
    day_volume = float(raw["Volume"].iloc[-1])

    with st.spinner("Running model..."):
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
    is_up = change >= 0

    # ---------------- Layout: chart + stats ----------------
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        show_full_range = st.checkbox("Show full dataset history (instead of last 120 days)", value=False)
        chart_data = raw["Close"] if show_full_range else raw["Close"].tail(120)
        range_label = "FULL HISTORY" if show_full_range else "120D"
        st.markdown(f'<div class="section-label">{ticker} · PRICE HISTORY ({range_label})</div>', unsafe_allow_html=True)
        st.line_chart(chart_data, height=320)

        # Snapshot row
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Day High</div>
                <div class="metric-value" style="font-size:1.4rem;">${day_high:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Day Low</div>
                <div class="metric-value" style="font-size:1.4rem;">${day_low:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Volume</div>
                <div class="metric-value" style="font-size:1.4rem;">{day_volume:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-label">FORECAST</div>', unsafe_allow_html=True)

        delta_class = "delta-up" if is_up else "delta-down"
        arrow = "▲" if is_up else "▼"

        st.markdown(f"""
        <div class="predict-card">
            <div class="predict-label">Predicted Next Close</div>
            <div class="predict-value">${predicted_price:,.2f}</div>
            <div class="{delta_class}" style="margin-top:0.5rem; font-size:0.95rem;">
                {arrow} {change:+.2f} ({change_pct:+.2f}%)
            </div>
            <div class="model-pill">{model_choice}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Last Close · {last_date}</div>
            <div class="metric-value">${last_close:,.2f}</div>
            <div class="metric-caption">Most recent confirmed closing price</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Educational example only — not financial advice.</strong>
        Predictions are based on limited historical price patterns and technical
        indicators, and should not be used for real trading decisions.
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="empty-state">
        <div class="glyph">📊</div>
        <div style="font-size:1.05rem; font-weight:600; color:#F4F6FA; margin-bottom:0.4rem;">
            No forecast yet
        </div>
        <div>Enter a ticker in the sidebar and click <strong>Fetch &amp; Predict</strong> to get started.</div>
    </div>
    """, unsafe_allow_html=True)
