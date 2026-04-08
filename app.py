import streamlit as st
import matplotlib.pyplot as plt
from lstm_model import (
    load_data, prepare_data, train_model,
    predict_future, get_signal,
    get_news_sentiment, get_confidence
)
from auth import create_table, signup, login

create_table()

st.set_page_config(page_title="Smart Stock AI", layout="wide")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    st.title("🔐 Smart Stock AI Login")

    menu = ["Login", "Signup"]
    choice = st.sidebar.selectbox("Menu", menu)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if choice == "Signup":
        if st.button("Create Account"):
            if signup(username, password):
                st.success("Account created!")
            else:
                st.error("Username exists")

    elif choice == "Login":
        if st.button("Login"):
            if login(username, password):
                st.session_state.logged_in = True
                st.success("Welcome!")
            else:
                st.error("Invalid credentials")

else:
    st.title("📈 Smart Stock Prediction AI")

    if st.button("Logout"):
        st.session_state.logged_in = False

    stock = st.text_input("Enter Stock Symbol", "AAPL")

    if st.button("🚀 Predict"):

        df = load_data(stock)
        X, y, scaler = prepare_data(df)
        model = train_model(X, y)

        future_prices = predict_future(model, df, scaler)

        current_price = df['Close'].iloc[-1]
        predicted_price = future_prices[0]

        signal = get_signal(current_price, predicted_price)
        sentiment = get_news_sentiment(stock)
        confidence = get_confidence(future_prices)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Current Price", f"{current_price:.2f}")
            st.metric("Prediction", f"{predicted_price:.2f}")

        with col2:
            st.metric("Signal", signal)
            st.metric("Confidence", f"{confidence}%")

        st.subheader(f"📰 Market Sentiment: {sentiment}")

        fig, ax = plt.subplots()
        ax.plot(df['Close'][-100:], label="Past")
        ax.plot(range(len(df['Close'][-100:]),
                     len(df['Close'][-100:]) + 30),
                future_prices, linestyle="dashed", label="Future")
        ax.legend()
        st.pyplot(fig)
