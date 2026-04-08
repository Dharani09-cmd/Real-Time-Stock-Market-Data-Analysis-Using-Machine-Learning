import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
import ta

def load_data(stock):
    df = yf.download(stock, start="2020-01-01")
    df['MA'] = df['Close'].rolling(10).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df.dropna(inplace=True)
    return df

def prepare_data(df):
    data = df[['Close','MA','RSI']]
    
    X, y = [], []
    
    for i in range(10, len(data)):
        X.append(data.iloc[i-10:i].values.flatten())
        y.append(data.iloc[i]['Close'])
    
    return np.array(X), np.array(y)

def train_model(X, y):
    model = RandomForestRegressor(n_estimators=50)
    model.fit(X, y)
    return model

def predict_future(model, df, days=30):
    data = df[['Close','MA','RSI']]
    last = data.iloc[-10:].values.flatten()
    
    preds = []
    
    for _ in range(days):
        pred = model.predict([last])[0]
        preds.append(pred)
        
        last = np.roll(last, -3)
        last[-3] = pred
    
    return preds

def get_signal(current, predicted):
    return "BUY 🟢" if predicted > current else "SELL 🔴"

def get_confidence(preds):
    return round(max(0, 100 - np.std(preds)), 2)

def get_news_sentiment(stock):
    return "Neutral 🟡"
