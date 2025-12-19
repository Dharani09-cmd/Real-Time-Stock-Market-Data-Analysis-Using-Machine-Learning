import streamlit as st
import matplotlib.pyplot as plt
from stock_prediction import predict_price

st.set_page_config(page_title="Stock Market Predictor", layout="wide")

st.title("📈 Real-Time Stock Market Data Analysis & Prediction")

st.write("This application predicts future stock prices using Machine Learning.")

ticker = st.text_input("Enter Stock Symbol (Example: AAPL, TSLA, MSFT)", "AAPL")

if st.button("Predict"):
    try:
        price, data = predict_price(ticker)

        st.success(f"✅ Predicted Price After 30 Days: ${price}")

        st.subheader("📊 Stock Price Trend")
        plt.figure(figsize=(10,4))
        plt.plot(data['Close'], label="Closing Price")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        st.pyplot(plt)

    except Exception as e:
        st.error("Invalid stock symbol or data not available.")
