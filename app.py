from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

app = Flask(__name__)

def predict_stock(stock):
    data = yf.download(stock, start="2020-01-01", end="2024-01-01")
    data = data[['Close']]

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    time_step = 60

    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)))
    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=2, batch_size=32, verbose=0)

    last_60 = scaled_data[-60:]
    last_60 = last_60.reshape(1,60,1)

    prediction = model.predict(last_60)
    prediction = scaler.inverse_transform(prediction)

    return float(prediction[0][0])


@app.route("/")
def home():
    return "LSTM Stock Prediction Running 🚀"


@app.route("/predict")
def predict():
    stock = request.args.get("stock", "AAPL")
    price = predict_stock(stock)

    return jsonify({
        "stock": stock,
        "predicted_price": price
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
