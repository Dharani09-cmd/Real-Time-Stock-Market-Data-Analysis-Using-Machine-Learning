import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def get_stock_data(ticker):
    df = yf.download(ticker, period="5y")
    df = df[['Close']]
    df['Prediction'] = df['Close'].shift(-30)
    df.dropna(inplace=True)
    return df

def train_model(df):
    X = np.array(df.drop(['Prediction'], axis=1))
    y = np.array(df['Prediction'])

    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_price(ticker):
    df = get_stock_data(ticker)
    model = train_model(df)

    last_price = df[['Close']].tail(1)
    prediction = model.predict(last_price)

    return round(prediction[0], 2), df
