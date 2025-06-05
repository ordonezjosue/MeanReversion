import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Mean Reversion Signal Tracker", layout="wide")
st.title("🔁 Mean Reversion Signal Tracker")

# --- User Inputs ---
ticker = st.text_input("Enter Ticker Symbol (e.g. SPY, AAPL):", "SPY")
date_range = st.slider("Select Lookback Period (Days):", min_value=30, max_value=365, value=90)

if ticker:
    # --- Fetch Data ---
    df = yf.download(ticker, period=f"{date_range}d")
    st.write("Available columns:", df.columns.tolist())
    if df.empty:
        st.error("No data found. Please check the ticker symbol.")
    else:
        if 'Close' in df.columns:
            df['20MA'] = df['Close'].rolling(window=20).mean()
            df['Upper Band'] = df['20MA'] + 2 * df['Close'].rolling(window=20).std()
            df['Lower Band'] = df['20MA'] - 2 * df['Close'].rolling(window=20).std()
        else:
            st.error("❌ 'Close' column not found in the data. Unable to compute signals.")
        df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().add(1).rolling(14).apply(lambda x: (x[x > 1].sum() / x[x <= 1].sum()) if x[x <= 1].sum() != 0 else 0)))
        df.dropna(inplace=True)

        # --- Signal Logic ---
        df['Put Signal'] = (df['Close'] > df['Upper Band']) & (df['RSI'] > 70)
        df['Call Signal'] = (df['Close'] < df['Lower Band']) & (df['RSI'] < 30)

        latest = df.iloc[-1]

        st.subheader("📉 Latest Signal")
        if latest['Put Signal']:
            st.markdown("### 🟥 PUT Debit Spread Suggested")
        elif latest['Call Signal']:
            st.markdown("### 🟩 CALL Debit Spread Suggested")
        else:
            st.info("No clear mean reversion signal at this time.")

        # --- Plot ---
        st.subheader("📊 Price & Bollinger Bands")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df.index, df['Close'], label='Close', color='blue')
        ax.plot(df.index, df['20MA'], label='20MA', color='orange')
        ax.fill_between(df.index, df['Upper Band'], df['Lower Band'], color='gray', alpha=0.3)
        ax.set_title(f"{ticker} Mean Reversion Chart")
        ax.legend()
        st.pyplot(fig)

        st.subheader("🔍 Recent Signals Table")
        st.dataframe(df[['Close', 'RSI', '20MA', 'Upper Band', 'Lower Band', 'Call Signal', 'Put Signal']].tail(15))
