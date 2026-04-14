import pandas as pd
import numpy as np

def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)

    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df.dropna(subset=['Close'], inplace=True)

    # Log Returns
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df.dropna(inplace=True)

    # Z-score normalization
    mean = df['Log_Return'].mean()
    std = df['Log_Return'].std()

    df['Normalized_Return'] = (df['Log_Return'] - mean) / std

    return df, mean, std