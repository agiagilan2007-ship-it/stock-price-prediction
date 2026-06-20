import argparse
import os
import json
from datetime import datetime
import numpy as np

from src.data_loader import download_ohlcv, create_features, create_windows, train_val_test_split
from src.model import build_lstm_model, build_transformer_model

import tensorflow as tf

def save_metadata(path, meta):
    with open(path, 'w') as f:
        json.dump({'feature_cols': meta['feature_cols']}, f)

def main(args):
    ticker = args.ticker
    window = args.window
    horizon = args.horizon
    model_type = args.model
    epochs = args.epochs
    batch_size = args.batch_size

    if args.data_path:
        import pandas as pd
        df = pd.read_csv(args.data_path, parse_dates=['Date'], index_col='Date')
        df = df[['Open','High','Low','Close','Volume']]
    else:
        df = download_ohlcv(ticker, start=args.start, end=args.end)

    df = create_features(df)
    feature_cols = ['Open','High','Low','Close','Volume']  # keep Close at index used in create_windows
    X, y, meta = create_windows(df, window=window, horizon=horizon, feature_cols=feature_cols)

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)

    input_shape = X_train.shape[1:]  # (window, features)
    if model_type == 'lstm':
        model = build_lstm_model(input_shape, horizon=horizon)
    else:
        model = build_transformer_model(input_shape, horizon=horizon)

    timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    out_dir = args.out_dir or f'models/{model_type}_{ticker}_{timestamp}'
    os.makedirs(out_dir, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(out_dir, 'best.h5'), save_best_only=True, monitor='val_loss'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4)
    ]

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks)

    # save final model and metadata
    model.save(os.path.join(out_dir, 'final.h5'))
    save_metadata(os.path.join(out_dir, 'meta.json'), meta)
    print('Saved model to', out_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, default='AAPL')
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--start', type=str, default=None)
    parser.add_argument('--end', type=str, default=None)
    parser.add_argument('--model', choices=['lstm','transformer'], default='lstm')
    parser.add_argument('--window', type=int, default=60)
    parser.add_argument('--horizon', type=int, default=7)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--out-dir', type=str, default=None)
    args = parser.parse_args()
    main(args)
