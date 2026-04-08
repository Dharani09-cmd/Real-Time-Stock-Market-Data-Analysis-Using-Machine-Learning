import streamlit as st
import random
# ✅ SESSION INIT (VERY IMPORTANT)
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "otp" not in st.session_state:
    st.session_state.otp = None

if "user" not in st.session_state:
    st.session_state.user = None


# 🔐 LOGIN / SIGNUP
if not st.session_state.logged_in:

    st.title("🔐 Smart Stock AI")

    tab1, tab2 = st.tabs(["Login", "Signup"])

    # ---------------- LOGIN ----------------
    with tab1:
        st.subheader("Login")

        username_login = st.text_input("Username", key="login_user")
        password_login = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):

            if not username_login or not password_login:
                st.error("Please enter username and password")

            else:
                user = login(username_login, password_login)

                if user:
                    st.session_state.logged_in = True
                    st.session_state.user = user
                    st.success("Login successful")

                    st.rerun()   # 🔥 IMPORTANT

                else:
                    st.error("Invalid credentials")

    # ---------------- SIGNUP ----------------
    with tab2:
        st.subheader("Create Account")

        username = st.text_input("Username", key="signup_user")
        name = st.text_input("Full Name")
        location = st.text_input("Location", "Hyderabad")
        mobile = st.text_input("Mobile Number")
        email = st.text_input("Email")

        password = st.text_input("Password", type="password", key="pass1")
        confirm_password = st.text_input("Re-enter Password", type="password", key="pass2")

        # OTP button
        if st.button("Send OTP"):
            st.session_state.otp = str(random.randint(1000, 9999))
            st.success(f"OTP sent: {st.session_state.otp}")

        entered_otp = st.text_input("Enter OTP")

        if st.button("Signup"):

            if not username or not password or not name:
                st.error("Please fill all fields")

            elif password != confirm_password:
                st.error("Passwords do not match")

            elif st.session_state.otp is None:
                st.error("Please generate OTP first")

            elif entered_otp != st.session_state.otp:
                st.error("Invalid OTP")

            else:
                if signup(username, name, location, mobile, email, password):
                    st.success("Account created successfully! Please login →")

                    # 🔥 reset fields
                    st.session_state.otp = None

                else:
                    st.error("Username already exists")
