import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openai

# --- Streamlit Config ---
st.set_page_config(page_title="Mean Reversion Signal Tracker", layout="wide")
st.title("🔁 Mean Reversion Signal Tracker")

# --- Load API Key from Streamlit Secrets ---
openai.api_key = st.secrets["OPENAI_API_KEY"]

# --- AI Strategy Function ---
def generate_strategy_recommendation(ticker, rsi, trend_direction, pe_ratio, premium_available):
    prompt = f"""
You are a professional options strategist. A trader is considering selling a put credit spread on {ticker} with:

- Delta: 25
- DTE: 30–45
- Minimum Premium: $50
- Exit Rules: 50% profit or 2x loss

Current Market Info:
- RSI: {rsi}
- Trend Direction: {trend_direction}
- P/E Ratio: {pe_ratio}
- Premium Available: ${premium_available}

Should they sell the put credit spread now? Respond with a clear YES or NO and explain in concise bullet points with pros and cons.
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"⚠️ ChatGPT error: {e}"

# --- User Inputs ---
ticker = st.text_input("Enter Ticker Symbol (e.g. SPY, AAPL):", "SPY")
date_range = st.slider("Select Lookback Period (Days):", min_value=30, max_value=365, value=90)

# --- Fetch Data ---
if ticker:
    df = yf.download(ticker, period=f"{date_range}d")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        st.error("❌ No data found. Please check the ticker symbol.")
    elif 'Close' not in df.columns:
        st.error("❌ 'Close' column not found in the data.")
    else:
        # --- Technical Indicators ---
        df['20MA'] = df['Close'].rolling(window=20).mean()
        df['Upper Band'] = df['20MA'] + 2 * df['Close'].rolling(window=20).std()
        df['Lower Band'] = df['20MA'] - 2 * df['Close'].rolling(window=20).std()

        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        df.dropna(inplace=True)

        # --- Signal Logic ---
        df['Put Signal'] = (df['Close'] > df['Upper Band']) & (df['RSI'] > 70)
        df['Call Signal'] = (df['Close'] < df['Lower Band']) & (df['RSI'] < 30)

        latest = df.iloc[-1]

        # --- Signal Display ---
        st.subheader("📉 Latest Signal")
        if latest['Put Signal']:
            st.markdown("### 🟥 **PUT Debit Spread Suggested**")
        elif latest['Call Signal']:
            st.markdown("### 🟩 **CALL Debit Spread Suggested**")
        else:
            st.info("No clear mean reversion signal at this time.")

        # --- Chart ---
        st.subheader("📊 Price & Bollinger Bands")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df.index, df['Close'], label='Close', color='blue')
        ax.plot(df.index, df['20MA'], label='20MA', color='orange')
        ax.fill_between(df.index, df['Upper Band'], df['Lower Band'], color='gray', alpha=0.3)
        ax.set_title(f"{ticker} Mean Reversion Chart", fontsize=14)
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # --- Signal Table ---
        st.subheader("🔍 Recent Signals Table")
        signal_table = df[['Close', 'RSI', '20MA', 'Upper Band', 'Lower Band', 'Call Signal', 'Put Signal']].tail(15).copy()
        signal_table.index = signal_table.index.strftime("%Y-%m-%d")
        st.dataframe(signal_table.style.format("{:.2f}", subset=['Close', 'RSI', '20MA', 'Upper Band', 'Lower Band']))

        # --- AI Strategy Recommendation ---
        st.subheader("🤖 AI Strategy Recommendation")
        trend_direction = "uptrend" if df['20MA'].iloc[-1] > df['20MA'].iloc[-20] else "downtrend"
        rsi = round(latest['RSI'], 2)
        pe_ratio = round(yf.Ticker(ticker).info.get('trailingPE', 20.0), 2)
        premium_available = 55  # <-- You can replace this with live options API later

        ai_response = generate_strategy_recommendation(ticker, rsi, trend_direction, pe_ratio, premium_available)
        st.markdown(ai_response)
