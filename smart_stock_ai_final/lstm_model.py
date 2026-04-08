import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import ta
from textblob import TextBlob
import requests

def load_data(stock):
    df = yf.download(stock, start="2020-01-01")
    df['MA'] = df['Close'].rolling(10).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df.dropna(inplace=True)
    return df

def prepare_data(df):
    data = df[['Close','MA','RSI']].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
        y.append(scaled[i,0])

    return np.array(X), np.array(y), scaler

def train_model(X,y):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X,y,epochs=2,batch_size=32,verbose=0)
    return model

def predict_future(model, df, scaler, days=30):
    data = df[['Close','MA','RSI']].values
    scaled = scaler.transform(data)
    last = scaled[-60:]
    predictions = []

    for _ in range(days):
        pred = model.predict(last.reshape(1,60,3), verbose=0)[0][0]
        new_row = [pred, last[-1][1], last[-1][2]]
        last = np.vstack([last[1:], new_row])
        temp = [[pred,0,0]]
        price = scaler.inverse_transform(temp)[0][0]
        predictions.append(price)

    return predictions

def get_signal(current, predicted):
    return "BUY 🟢" if predicted>current else "SELL 🔴"

def get_confidence(predictions):
    import numpy as np
    std = np.std(predictions)
    return round(max(0,100-(std*10)),2)

def get_news_sentiment(stock):
    return "Neutral 🟡"  # safe fallback (no API key needed)
