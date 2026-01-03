import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="AI Stock Analyzer", layout="wide")

st.title("🤖 AI Multi-Stock Market Analysis & Prediction Dashboard")
st.write("Supports NSE (.NS) & US Stocks — Example: TCS, ITC, SBIN, AAPL")


# ---------- FORMAT NSE SYMBOL ----------
def format_symbol(symbol):
    symbol = symbol.strip().upper()
    if "." not in symbol:
        return symbol + ".NS"
    return symbol


# ---------- MARKET MOOD ----------
def market_mood(df):
    last = df["Close"].iloc[-1]
    prev = df["Close"].iloc[-20]
    change = ((last - prev) / prev) * 100
    if change > 5:
        return "🟢 Bullish"
    elif change < -5:
        return "🔴 Bearish"
    return "⚪ Neutral"


# ---------- RISK ----------
def risk_score(df):
    returns = df["Close"].pct_change()
    vol = returns.std() * 100
    if vol < 1.2:
        return "🟢 Low Risk"
    elif vol < 2.5:
        return "🟡 Medium Risk"
    else:
        return "🔴 High Risk"


# ---------- CRASH WARNING ----------
def crash_warning(df):
    last = df["Close"].iloc[-1]
    week = df["Close"].iloc[-7]
    drop = ((week - last) / week) * 100
    return "⚠ Possible Downtrend" if drop > 6 else "✔ Stable"


# ---------- PRICE PREDICTION ----------
def predict_price(df):
    df["Days"] = np.arange(len(df))
    X = df[["Days"]]
    y = df["Close"]
    model = LinearRegression()
    model.fit(X, y)
    future = np.array([[len(df)+30]])
    return model.predict(future)[0]


# ---------- GROWTH ----------
def performance_score(df):
    start = df["Close"].iloc[0]
    end = df["Close"].iloc[-1]
    return ((end - start) / start) * 100


# ---------- PORTFOLIO ----------
def portfolio_recommendation(results):
    sorted_stocks = sorted(results, key=lambda x: x["growth"], reverse=True)
    return {
        "Low Risk": sorted_stocks[-1]["symbol"],
        "Balanced": sorted_stocks[len(sorted_stocks)//2]["symbol"],
        "High Return": sorted_stocks[0]["symbol"]
    }


# ---------- UI INPUT ----------
symbols = st.text_input(
    "Enter Stock Symbols (comma separated):",
    "TCS, ITC, SBIN, WIPRO"
)

if st.button("Analyze"):

    stocks = [format_symbol(s) for s in symbols.split(",")]
    results = []

    for stock in stocks:
        try:
            df = yf.download(stock, period="1y")

            if df.empty:
                st.error(f"Error fetching {stock}")
                continue

            col1, col2 = st.columns(2)
            with col1:
                st.subheader(stock)
                st.line_chart(df["Close"])

            pred = predict_price(df)
            mood = market_mood(df)
            risk = risk_score(df)
            warn = crash_warning(df)
            growth = performance_score(df)

            results.append({
                "symbol": stock,
                "pred": pred,
                "mood": mood,
                "risk": risk,
                "warn": warn,
                "growth": growth
            })

            with col2:
                st.write(f"**Predicted Price (30 days): ₹{pred:.2f}**")
                st.write(f"Market Mood: {mood}")
                st.write(f"Risk Level: {risk}")
                st.write(f"Crash Signal: {warn}")
                st.write(f"Performance Growth: {growth:.2f}%")

        except Exception as e:
            st.error(f"{stock} failed — {e}")

    if results:

        st.subheader("🏆 Performance Ranking")

        ranked = sorted(results, key=lambda x: x["growth"], reverse=True)

        for i, r in enumerate(ranked, 1):
            st.write(f"{i}. **{r['symbol']}** — {r['growth']:.2f}% growth")

        st.subheader("💼 Suggested Portfolio")

        p = portfolio_recommendation(results)

        st.success(f"✔ Safe Investor → {p['Low Risk']}")
        st.success(f"✔ Balanced Investor → {p['Balanced']}")
        st.success(f"✔ High Return Investor → {p['High Return']}")

        st.subheader("🤖 AI Insights")

        for r in results:
            st.info(
                f"{r['symbol']} → {r['mood']} | {r['risk']} | {r['warn']}"
            )
