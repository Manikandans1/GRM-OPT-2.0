import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def train_lstm(data):

    # Use last 150 months for training
    data = data[-150:]

    X = []
    y = []
    window = 12

    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])

    X = np.array(X)
    y = np.array(y)

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # -------- Improved LSTM Model --------
    model = Sequential()

    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(32))
    model.add(Dropout(0.2))

    model.add(Dense(1))

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse'
    )

    # Training
    model.fit(
        X,
        y,
        epochs=20,
        batch_size=8,
        verbose=0
    )

    predictions = model.predict(X)

    return y, predictions.flatten()
