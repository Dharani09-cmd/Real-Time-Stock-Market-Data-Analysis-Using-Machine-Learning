# LOGIN / SIGNUP
if not st.session_state.logged_in:

    st.title("🔐 Smart Stock AI")

    tab1, tab2 = st.tabs(["Login", "Signup"])   # 🔥 Better UI

    # ---------------- LOGIN ----------------
    with tab1:
        st.subheader("Login")

        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
            user = login(username, password)

            if user:
                st.session_state.logged_in = True
                st.session_state.user = user
                st.success("Login successful")

                st.rerun()   # 🔥 VERY IMPORTANT (fixes your issue)

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

        # OTP
        if st.button("Send OTP"):
            st.session_state.otp = str(random.randint(1000, 9999))
            st.success(f"OTP sent: {st.session_state.otp}")  # demo

        entered_otp = st.text_input("Enter OTP")

        if st.button("Signup"):

            if not username or not password:
                st.error("Please fill all fields")

            elif password != confirm_password:
                st.error("Passwords do not match")

            elif entered_otp != st.session_state.otp:
                st.error("Invalid OTP")

            else:
                if signup(username, name, location, mobile, email, password):
                    st.success("Account created successfully! Please login →")

                else:
                    st.error("Username already exists")
