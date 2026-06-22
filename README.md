# 📈 Stock Price Prediction

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green)

> Multi-step stock price prediction using **LSTM** and **Transformer** models built with TensorFlow.  
> Predicts the next **7 days** of closing prices from a **60-step** historical window.

---

## 🗂️ Project Structure

```
stock-price-prediction/
├── example_data/          # Sample CSV datasets
├── notebooks/
│   └── stock_price_prediction.ipynb   # Interactive walkthrough
├── src/
│   ├── data_loader.py     # Data download & preprocessing
│   ├── model.py           # LSTM & Transformer model definitions
│   ├── train.py           # Training CLI
│   └── evaluate.py        # Evaluation & plotting CLI
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Set up your environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the LSTM model on AAPL

```bash
python src/train.py --ticker AAPL --model lstm
```

### 3. Evaluate and plot predictions

```bash
python src/evaluate.py --ticker AAPL --model models/lstm_latest --plot
```

---

## 📊 Sample Output

After running evaluation, you'll see a plot like this:

```
Epoch 1/50 - loss: 0.0412 - val_loss: 0.0389
Epoch 2/50 - loss: 0.0298 - val_loss: 0.0271
...
Epoch 50/50 - loss: 0.0091 - val_loss: 0.0104

Test MAE  : 2.47
Test RMSE : 3.81
```

> 📌 **Tip:** Use `--plot` flag with `evaluate.py` to save a prediction vs. actual chart to `outputs/`.

---

## 🧠 Models

| Model       | Description                                  | Default Config          |
|-------------|----------------------------------------------|-------------------------|
| LSTM        | 2-layer stacked LSTM with dropout            | 128 units, dropout=0.2  |
| Transformer | Single-head attention + feedforward layers   | d_model=64, heads=4     |

Switch between them via the `--model` flag:

```bash
python src/train.py --ticker TSLA --model transformer
```

---

## 📦 Data

- Source: [`yfinance`](https://github.com/ranaroussi/yfinance) or local CSVs in `example_data/`
- Features: **OHLCV** (Open, High, Low, Close, Volume)
- Scaling: **MinMax normalization** per feature
- Input window: **60 time steps**
- Prediction horizon: **7 days**

---

## 🔧 Requirements

```
tensorflow>=2.10
yfinance
numpy
pandas
matplotlib
scikit-learn
```

Install all with:

```bash
pip install -r requirements.txt
```

---

## 📓 Notebook

Open the interactive notebook for a step-by-step walkthrough:

```bash
jupyter notebook notebooks/stock_price_prediction.ipynb
```

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.
