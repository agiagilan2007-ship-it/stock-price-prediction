import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.data_loader import download_ohlcv, create_features, create_windows

import tensorflow as tf
import joblib


def load_meta(meta_path):
    with open(meta_path, 'r') as f:
        return json.load(f)


def invert_scaling(scaler, data, feature_index):
    # data shape: (n_samples, horizon) scaled for that feature channel in create_windows (Close index)
    # We reconstruct a full feature vector using zeros for other features to inverse_transform
    n, horizon = data.shape
    dummy = np.zeros((n * horizon, scaler.scale_.shape[0]))
    dummy[:, feature_index] = data.reshape(-1)
    inverted = scaler.inverse_transform(dummy)[:, feature_index]
    return inverted.reshape(n, horizon)


def main(args):
    ticker = args.ticker
    window = args.window
    horizon = args.horizon
    model_path = args.model

    if args.data_path:
        import pandas as pd
        df = pd.read_csv(args.data_path, parse_dates=['Date'], index_col='Date')
        df = df[['Open','High','Low','Close','Volume']]
    else:
        df = download_ohlcv(ticker, start=args.start, end=args.end)

    df = create_features(df)
    feature_cols = ['Open','High','Low','Close','Volume']
    X, y, meta = create_windows(df, window=window, horizon=horizon, feature_cols=feature_cols)

    # split: keep last chunk for test
    train_n = int(len(X) * 0.85)
    X_test, y_test = X[train_n:], y[train_n:]

    # load model without compiling to avoid deserialization issues with legacy HDF5 training configs
    model = tf.keras.models.load_model(model_path, compile=False)
    pred = model.predict(X_test)

    # try to load a persisted scaler saved during training, fall back to the scaler returned by create_windows
    model_dir = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)
    scaler_file = os.path.join(model_dir, 'scaler.joblib')
    scaler = None
    if os.path.exists(scaler_file):
        try:
            scaler = joblib.load(scaler_file)
        except Exception as e:
            print('Warning: failed to load scaler.joblib:', e)
            scaler = meta.get('scaler', None)
    else:
        scaler = meta.get('scaler', None)

    if scaler is None:
        print('No scaler found in metadata or saved scaler; results will remain scaled.')
        y_true = y_test
        y_pred = pred
    else:
        close_idx = feature_cols.index('Close')
        y_true = invert_scaling(scaler, y_test, close_idx)
        y_pred = invert_scaling(scaler, pred, close_idx)

    # compute per-horizon metrics
    rmses = np.sqrt(((y_true - y_pred) ** 2).mean(axis=0))
    maes = np.abs(y_true - y_pred).mean(axis=0)
    for i in range(horizon):
        print(f'Horizon +{i+1}: RMSE={rmses[i]:.4f}, MAE={maes[i]:.4f}')

    # plot first sample predictions vs true
    n_plot = min(5, len(y_true))
    for i in range(n_plot):
        plt.figure(figsize=(8,3))
        plt.plot(range(1, horizon+1), y_true[i], label='True')
        plt.plot(range(1, horizon+1), y_pred[i], label='Pred')
        plt.title(f'Sample {i} — horizon predictions')
        plt.xlabel('Days ahead')
        plt.legend()
        plt.tight_layout()
        if args.plot:
            plt.show()
        else:
            out = args.out or 'pred_plots'
            os.makedirs(out, exist_ok=True)
            plt.savefig(os.path.join(out, f'pred_{i}.png'))
            plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, default='AAPL')
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--start', type=str, default=None)
    parser.add_argument('--end', type=str, default=None)
    parser.add_argument('--model', type=str, required=True, help='Path to saved model file or directory')
    parser.add_argument('--window', type=int, default=60)
    parser.add_argument('--horizon', type=int, default=7)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()
    main(args)
