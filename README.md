# 📈 Smart Stock Prediction AI

## 📌 Project Description

Smart Stock Prediction AI is an advanced web-based application that analyzes stock market data and predicts future stock prices using deep learning. The system integrates technical indicators, sentiment analysis, and confidence scoring to provide intelligent trading insights.

---

## 🎯 Objectives

* Analyze historical and real-time stock data
* Predict future stock prices using deep learning
* Provide Buy/Sell recommendations
* Integrate sentiment analysis for better decision-making
* Display results through an interactive dashboard
* Ensure secure user authentication

---

## 🧠 Technologies Used

* Python
* Pandas, NumPy
* TensorFlow / Keras
* Streamlit
* Matplotlib
* SQLite
* Yahoo Finance API

---

## 🤖 Machine Learning Model

* Algorithm: Long Short-Term Memory (LSTM)
* Input: Historical stock prices + technical indicators
* Output: Future stock price prediction (30 days)
* Advantage: Captures time-series patterns and trends

---

## 📊 Features

✔ Real-time stock data fetching
✔ LSTM-based deep learning prediction
✔ 30-day future forecasting graph
✔ Buy / Sell trading signals
✔ Confidence score calculation
✔ News sentiment analysis (NLP-based)
✔ Secure Login / Signup system
✔ Interactive and mobile-friendly UI
✔ Cloud deployment ready

---

## 📂 Project Structure

```
lstm-stock-project/
│
├── app.py              # Streamlit dashboard
├── lstm_model.py       # ML logic
├── auth.py             # Authentication system
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## 🔐 User Authentication

* Users can create an account (Signup)
* Users can log in securely
* Passwords are stored using hashing (SHA-256)
* SQLite database is used for storage

---

## 🚀 How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Deployment (Render)

This project can be deployed using Render

### Build Command:

```bash
pip install -r requirements.txt
```

### Start Command:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

---

## 📈 Output

* Displays current stock price
* Predicts next-day price
* Shows 30-day future trend graph
* Provides Buy/Sell signal
* Displays confidence score
* Shows market sentiment

---

## ⚠️ Limitations

* Predictions are based on historical data only
* External factors (economic events, news) are limited
* Model retrains during execution (can be optimized further)
* Not intended for real financial trading

---

## 🔮 Future Enhancements

* Real-time live stock streaming
* Advanced sentiment analysis with APIs
* Pre-trained model saving (faster predictions)
* Mobile app version
* Portfolio tracking system

---

## 🏁 Conclusion

This project demonstrates the power of deep learning in financial forecasting. By combining LSTM, technical indicators, and sentiment analysis, it provides a comprehensive and intelligent stock prediction system.

---

## 🎤 Viva Explanation

> This project uses an LSTM deep learning model along with technical indicators and sentiment analysis to predict stock prices. It also includes a confidence scoring system and secure user authentication, making it a complete AI-based financial analysis platform.

---
