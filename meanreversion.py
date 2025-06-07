import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from openai import OpenAI

# --- Streamlit Config ---
st.set_page_config(page_title="Mean Reversion Signal Tracker", layout="wide")
st.title("🔁 Mean Reversion Signal Tracker + Strategy Scorer")

# --- Initialize OpenAI Client (v1.x) ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- AI Strategy Function ---
def generate_strategy_recommendation(ticker, rsi, trend_direction, pe_ratio, premium_available, score):
    prompt = f"""
You are a professional options strategist. A trader is evaluating a put credit spread on {ticker}.

Strategy:
- 25 delta put credit spread
- 30–45 DTE
- Minimum premium: $50
- Exit: 50% profit or 2x loss

Market Data:
- RSI: {rsi}
- Trend Direction: {trend_direction}
- P/E Ratio: {pe_ratio}
- Premium Estimate: ${premium_available}
- Custom Signal Score: {score}/100

Do you recommend the trade? Respond in bullet points with reasoning.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # ✅ Updated here
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ ChatGPT error: {e}"

# --- Trade Setup Scoring Function ---
def score_trade_setup(rsi, trend, ivr, premium, earnings_days_away):
    score = 0
    if 35 <= rsi <= 60: score += 20
    if trend == "uptrend": score += 20
    if ivr >= 30: score += 25
    if premium >= 50: score += 20
    if earnings_days_away > 7: score += 15
    return score

# --- Inputs ---
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

        # --- Strategy Inputs ---
        st.subheader("🤖 AI Strategy Recommendation")
        trend_direction = "uptrend" if df['20MA'].iloc[-1] > df['20MA'].iloc[-20] else "downtrend"
        rsi = round(latest['RSI'], 2)
        stock_info = yf.Ticker(ticker).info
        pe_ratio = round(stock_info.get('trailingPE', 20.0), 2)
        earnings_date = stock_info.get('earningsDate', datetime.today() + timedelta(days=30))
        if isinstance(earnings_date, list):  # Sometimes it's a list
            earnings_date = earnings_date[0]
        earnings_days_away = (pd.to_datetime(earnings_date) - pd.to_datetime(datetime.today())).days
        ivr = 35  # Placeholder IVR
        premium_available = 55  # Placeholder premium

        score = score_trade_setup(rsi, trend_direction, ivr, premium_available, earnings_days_away)
        st.markdown(f"**Custom Trade Score:** `{score}/100`")

        ai_response = generate_strategy_recommendation(ticker, rsi, trend_direction, pe_ratio, premium_available, score)
        st.markdown(ai_response)
