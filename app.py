import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

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
# NOTE: Models are trained LIVE, per prediction request, using only the
# date range selected in the sidebar. A single fixed model can't produce
# predictions that vary by duration — training fresh per request is what
# makes duration actually matter. train_stock_models_WITH_DATASET.py is
# still useful standalone for producing a saved, documented model as a
# project deliverable, but app.py no longer depends on its .pkl output.
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Load local dataset
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

    period = st.selectbox(
        "History Period",
        ["3mo", "6mo", "1y", "2y", "5y"],
        index=4  # defaults to 5y
    )

    st.markdown("<br>", unsafe_allow_html=True)
    fetch_clicked = st.button("🔮  Train & Predict")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">ABOUT</div>', unsafe_allow_html=True)
    st.caption(
        f"A fresh model is trained live using only the selected period's "
        f"data from {DATASET_PATH} — predictions genuinely change with "
        "the history window you pick, not just the chart."
    )

# ---------------------------------------------------------------------
# Feature engineering — ADAPTIVE: the feature set scales to how much
# history is available, so short periods (e.g. 6mo) drop long-window
# features (MA200, MA50) that can't be computed from limited data,
# instead of failing outright. This is also WHY predictions genuinely
# vary by period: shorter windows train on fewer rows AND fewer/shorter
# indicators, longer windows get the full feature set.
# ---------------------------------------------------------------------
# Minimum rows needed after dropna to leave a workable train/test split
MIN_TRAINABLE_ROWS = 40


def build_features_adaptive(df: pd.DataFrame):
    """Builds technical indicators, using only windows that fit the data.

    Returns (feature_dataframe, feature_column_list).
    """
    d = df.copy()
    n = len(d)

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

    feature_cols = [
        "MA5", "MA10", "MA20", "Volatility5", "RSI",
        "Lag1", "Lag2", "Lag3", "Lag5",
    ]

    # Longer lag — needs 20 rows of runway, safe even for short periods
    if n >= 20 + MIN_TRAINABLE_ROWS:
        d["Lag20"] = d["Close"].shift(20)
        feature_cols.append("Lag20")

    # Medium-term trend — needs enough rows before AND after the window
    if n >= 50 + MIN_TRAINABLE_ROWS:
        d["MA50"] = d["Close"].rolling(50).mean()
        feature_cols.append("MA50")

    # Long-term trend — only include with a full year+ of history
    if n >= 200 + MIN_TRAINABLE_ROWS:
        d["MA200"] = d["Close"].rolling(200).mean()
        feature_cols.append("MA200")

    feature_cols.append("Volume")  # always available, no rolling window needed

    d["Target"] = d["Close"].shift(-1)
    d = d.dropna()

    return d, feature_cols


def get_ticker_history(ticker: str) -> pd.DataFrame:
    """Pull one ticker's OHLCV history from the local dataset, sorted by date."""
    df = dataset[dataset["Ticker"] == ticker].copy()
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.set_index("Date")
    return df[["Open", "High", "Low", "Close", "Volume"]]


PERIOD_DAYS = {
    "3mo": 91,
    "6mo": 182,
    "1y": 365,
    "2y": 730,
    "5y": 1825,
}


def trim_by_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Slice a ticker's history down to the selected lookback window."""
    days = PERIOD_DAYS.get(period)
    if days is None or df.empty:
        return df
    cutoff = df.index.max() - pd.Timedelta(days=days)
    return df[df.index >= cutoff]


def train_live(feat: pd.DataFrame, feature_cols: list, model_choice: str):
    """Fit a fresh model on exactly this period's data and predict the
    next close. Returns (prediction, r2_on_holdout, n_features_used)."""
    X = feat[feature_cols]
    y = feat["Target"]

    # Time-ordered 80/20 split (no shuffle — this is time-series data)
    split = max(int(len(X) * 0.8), 1)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    latest_row = X.iloc[[-1]]

    if model_choice == "Linear Regression":
        model = LinearRegression().fit(X_train, y_train)
        prediction = model.predict(latest_row)[0]
        test_pred = model.predict(X_test) if len(X_test) > 0 else None
    else:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        model = LinearRegression().fit(X_train_poly, y_train)
        prediction = model.predict(poly.transform(latest_row))[0]
        test_pred = model.predict(poly.transform(X_test)) if len(X_test) > 0 else None

    r2 = r2_score(y_test, test_pred) if test_pred is not None and len(y_test) > 1 else None
    return float(prediction), r2, len(feature_cols)


# ---------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------
if fetch_clicked or "raw_data" in st.session_state:
    with st.spinner(f"Loading {ticker} history from dataset..."):
        try:
            full_history = get_ticker_history(ticker)
            if full_history.empty:
                st.error(f"No data found for ticker '{ticker}' in {DATASET_PATH}.")
                st.stop()
            raw = trim_by_period(full_history, period)
            if len(raw) < MIN_TRAINABLE_ROWS + 5:
                st.error(
                    f"'{period}' only has {len(raw)} trading day(s) for {ticker} — "
                    f"not enough to train a model (need at least "
                    f"{MIN_TRAINABLE_ROWS + 5}). Try a longer period."
                )
                st.stop()
            st.session_state["raw_data"] = raw
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
            st.stop()

    raw = st.session_state["raw_data"]

    with st.spinner(f"Training a fresh {model_choice} model on {period} of data..."):
        feat, feature_cols = build_features_adaptive(raw)

        if len(feat) < MIN_TRAINABLE_ROWS:
            st.error(
                f"After computing indicators, only {len(feat)} usable rows remain "
                f"for '{period}' — not enough to train reliably. Try a longer period."
            )
            st.stop()

        try:
            predicted_price, r2, n_features = train_live(feat, feature_cols, model_choice)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    last_close = float(raw["Close"].iloc[-1])
    last_date = feat.index[-1].strftime("%Y-%m-%d")
    day_high = float(raw["High"].iloc[-1])
    day_low = float(raw["Low"].iloc[-1])
    day_volume = float(raw["Volume"].iloc[-1])

    change = predicted_price - last_close
    change_pct = (change / last_close) * 100
    is_up = change >= 0

    # ---------------- Layout: chart + stats ----------------
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.markdown(f'<div class="section-label">{ticker} · PRICE HISTORY ({period.upper()})</div>', unsafe_allow_html=True)
        st.line_chart(raw["Close"], height=320)

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

        r2_display = f"{r2:.3f}" if r2 is not None else "n/a (too few rows to hold out)"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">This Run's Model</div>
            <div class="metric-caption">
                Trained on {len(feat)} rows from '{period}' · {n_features} features ·
                Holdout R²: {r2_display}
            </div>
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
