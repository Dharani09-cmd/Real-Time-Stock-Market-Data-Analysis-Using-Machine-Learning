import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# -------------------------------
# Fetch stock data
# -------------------------------
def get_stock_data(ticker):
    df = yf.download(
        ticker,
        period="5y",
        progress=False
    )

    if df.empty:
        raise ValueError("No data found for this stock symbol")

    df = df[['Close']].copy()

    # Predict price after 30 days
    df['Prediction'] = df['Close'].shift(-30)

    df.dropna(inplace=True)
    return df


# -------------------------------
# Train ML model
# -------------------------------
def train_model(df):
    X = df[['Close']].values
    y = df['Prediction'].values

    model = LinearRegression()
    model.fit(X, y)

    return model


# -------------------------------
# Predict future price
# -------------------------------
def predict_price(ticker):
    df = get_stock_data(ticker)
    model = train_model(df)

    last_close_price = df[['Close']].tail(1).values
    prediction = model.predict(last_close_price)

    return round(float(prediction[0]), 2), df
