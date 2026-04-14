import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from preprocessing import preprocess_data
from lstm_model import train_lstm


markets = ["nifty50", "sp500", "dax40"]

for market in markets:

    print("\nChecking:", market.upper())

    df, mean, std = preprocess_data(f"data/{market}.csv")

    data = df['Normalized_Return'].values

    window = 12
    X = []
    y = []

    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])

    X = np.array(X)
    y = np.array(y)

    # -------- Linear Regression --------
    linear_model = LinearRegression()
    linear_model.fit(X, y)

    linear_pred = linear_model.predict(X)

    rmse_linear = np.sqrt(mean_squared_error(y, linear_pred))

    # -------- LSTM Model --------
    y_true_lstm, y_pred_lstm = train_lstm(data)

    rmse_lstm = np.sqrt(mean_squared_error(y_true_lstm, y_pred_lstm))

    # -------- Improvement --------
    improvement = ((rmse_linear - rmse_lstm) / rmse_linear) * 100

    print("Linear RMSE :", rmse_linear)
    print("LSTM RMSE   :", rmse_lstm)
    print("Improvement :", improvement, "%")