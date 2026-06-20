# stock-price-prediction

Starter project for multi-step stock price prediction using TensorFlow.
- Models: LSTM baseline and simple Transformer alternative.
- Data: OHLCV via `yfinance` or local CSVs.
- Default settings: 60-step input window, 7-day prediction horizon, MinMax scaling.

Quick start
1. Create and activate a Python 3.9+ virtual environment.
2. pip install -r requirements.txt
3. Train an LSTM on AAPL (defaults):
   python src/train.py --ticker AAPL --model lstm
4. Evaluate:
   python src/evaluate.py --ticker AAPL --model models/lstm_latest --plot

Files
- notebooks/stock_price_prediction.ipynb — interactive notebook
- src/data_loader.py — data download & preprocessing
- src/model.py — model definitions
- src/train.py — training CLI
- src/evaluate.py — evaluation CLI

License: MIT
