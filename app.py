import streamlit as st
import random
import matplotlib.pyplot as plt

from lstm_model import load_data, prepare_data, train_model, predict_future, get_signal, get_confidence
from auth import create_table, signup, login, get_user

create_table()

# SESSION
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "otp" not in st.session_state:
    st.session_state.otp = None

if "user" not in st.session_state:
    st.session_state.user = None

st.set_page_config(page_title="Smart Stock AI", layout="wide")

# LOGIN / SIGNUP
if not st.session_state.logged_in:

    st.title("🔐 Smart Stock AI")

    menu = ["Login", "Signup"]
    choice = st.selectbox("Select Option", menu)

    # LOGIN
    if choice == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            user = login(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.user = user
                st.success("Login successful")
            else:
                st.error("Invalid credentials")

    # SIGNUP
    elif choice == "Signup":

        username = st.text_input("Username")
        name = st.text_input("Full Name")
        location = st.text_input("Location")
        mobile = st.text_input("Mobile Number")
        email = st.text_input("Email")

        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Re-enter Password", type="password")

        # OTP generation
        if st.button("Send OTP"):
            st.session_state.otp = str(random.randint(1000, 9999))
            st.success(f"OTP sent to {mobile} & {email}: {st.session_state.otp}")

        entered_otp = st.text_input("Enter OTP")

        if st.button("Signup"):

            if password != confirm_password:
                st.error("Passwords do not match")

            elif entered_otp != st.session_state.otp:
                st.error("Invalid OTP")

            else:
                if signup(username, name, location, mobile, email, password):
                    st.success("Account created successfully!")
                else:
                    st.error("Username already exists")

# DASHBOARD
else:
    st.title("📈 Smart Stock Prediction AI")

    menu = ["Dashboard", "Profile"]
    choice = st.sidebar.selectbox("Menu", menu)

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False

    # 📊 DASHBOARD
    if choice == "Dashboard":

        stock = st.text_input("Enter Stock Symbol", "AAPL")

        if st.button("Predict"):

            df = load_data(stock)
            X, y = prepare_data(df)
            model = train_model(X, y)

            future_prices = predict_future(model, df)

            current_price = df['Close'].iloc[-1]
            predicted_price = future_prices[0]

            signal = get_signal(current_price, predicted_price)
            confidence = get_confidence(future_prices)

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Current Price", f"{current_price:.2f}")
                st.metric("Prediction", f"{predicted_price:.2f}")

            with col2:
                st.metric("Signal", signal)
                st.metric("Confidence", f"{confidence}%")

            # Graph
            fig, ax = plt.subplots()
            ax.plot(df['Close'][-100:], label="Past")
            ax.plot(range(len(df['Close'][-100:]),
                         len(df['Close'][-100:]) + 30),
                    future_prices, linestyle="dashed", label="Future")
            ax.legend()
            st.pyplot(fig)

    # 👤 PROFILE PAGE
    elif choice == "Profile":

        user = st.session_state.user

        st.subheader("👤 User Profile")

        st.write(f"**Username:** {user[0]}")
        st.write(f"**Name:** {user[1]}")
        st.write(f"**Location:** {user[2]}")
        st.write(f"**Mobile:** {user[3]}")
        st.write(f"**Email:** {user[4]}")
