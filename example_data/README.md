If you prefer to provide CSV files instead of using yfinance, place them in ./data/ as CSVs with columns:
Date,Open,High,Low,Close,Adj Close,Volume

Example usage in scripts:
python src/train.py --data-path ./data/AAPL.csv --ticker AAPL --model lstm
