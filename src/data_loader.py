import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from typing import Tuple, Dict

def download_ohlcv(ticker: str, start: str = None, end: str = None) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f'No data downloaded for {ticker}')
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.index = pd.to_datetime(df.index)
    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    # Keep OHLCV as-is; add simple engineered features if desired
    df2 = df.copy()
    df2['Return'] = df2['Close'].pct_change().fillna(0)
    df2['HL_PCT'] = (df2['High'] - df2['Low']) / df2['Low']
    # Use forward-fill method via .ffill() to remain compatible with pandas 3.x
    df2 = df2.ffill().fillna(0)
    return df2

def create_windows(df: pd.DataFrame, window: int = 60, horizon: int = 7,
                   feature_cols=None, scaler: MinMaxScaler = None
                  ) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Returns X, y, metadata with scalers and columns.
    X shape: (n_samples, window, n_features)
    y shape: (n_samples, horizon)  -> multi-step predicting Close for next `horizon` days
    """
    if feature_cols is None:
        feature_cols = df.columns.tolist()
    data = df[feature_cols].values.astype('float32')

    # fit scaler if not provided
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(data)

    scaled = scaler.transform(data)
    n_samples = len(scaled) - window - horizon + 1
    if n_samples <= 0:
        raise ValueError('Not enough data for the given window/horizon')

    X = np.zeros((n_samples, window, len(feature_cols)), dtype='float32')
    y = np.zeros((n_samples, horizon), dtype='float32')

    for i in range(n_samples):
        X[i] = scaled[i:i+window]
        # y: next `horizon` days closing price (use original Close column index)
        close_index = feature_cols.index('Close')
        future_close = scaled[i+window:i+window+horizon, close_index]
        y[i] = future_close

    meta = {'scaler': scaler, 'feature_cols': feature_cols}
    return X, y, meta

def train_val_test_split(X, y, train_frac=0.7, val_frac=0.15):
    n = len(X)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return (X[:train_end], y[:train_end],
            X[train_end:val_end], y[train_end:val_end],
            X[val_end:], y[val_end:])
