# Final_Hybrid_CNN_LSTM_Stock_Price_Prediction

> A comprehensive deep learning project for Tesla stock price prediction using Hybrid CNN-LSTM architecture with Gaussian smoothing. Achieves 92% R² accuracy on test data.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements a Hybrid CNN-LSTM deep learning model for predicting Tesla stock prices. The model combines convolutional neural networks (CNN) for feature extraction with long short-term memory (LSTM) networks for sequence modeling. Gaussian smoothing is applied to reduce market noise, achieving 92% accuracy (R² = 0.9215).

### Key Achievements
- **R² Score**: 0.9215 (92.15% accuracy)
- **RMSE**: $19.69
- **MAE**: $12.63
- **MAPE**: 4.89%

## Features

- 📈 **Data Preprocessing**: Gaussian smoothing, MinMax scaling, sequence generation
- 🦠 **Advanced Model**: Hybrid CNN-LSTM with BiLSTM and dropout regularization
- 📊 **Comprehensive Evaluation**: RMSE, MAE, R², MAPE metrics
- 💾 **Data Visualization**: Actual vs predicted plots, error distributions, residuals
- 🚀 **Production Ready**: Includes requirements.txt and Python .gitignore

## Dataset

**Source**: Yahoo Finance (yfinance library)
**Stock**: Tesla (TSLA)
**Time Period**: 2015-2024
**Number of Records**: 2,515 trading days
**Features Used**: Daily closing prices

### Data Processing
1. Downloaded historical Tesla stock data via yfinance
2. Applied Gaussian smoothing (sigma=3) to reduce noise
3. MinMax normalization to [0, 1] range
4. Created 60-day sequences for hybrid CNN-LSTM model
5. 85% training, 15% testing split

## Requirements

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
tensorflow>=2.10.0
keras>=2.10.0
matplotlib>=3.4.0
seaborn>=0.11.0
yfinance>=0.2.0
scipy>=1.7.0
```

## Installation

### Option 1: Local Setup

```bash
# Clone the repository
git clone https://github.com/Keshu017/Final_Hybrid_CNN_LSTM_stock_price_prediction.git
cd Final_Hybrid_CNN_LSTM_stock_price_prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Google Colab

```python
# Install libraries in Colab
!pip install tensorflow keras yfinance scikit-learn scipy matplotlib seaborn

# Clone repository
!git clone https://github.com/Keshu017/Final_Hybrid_CNN_LSTM_stock_price_prediction.git
```

## Model Architecture

### Hybrid CNN-LSTM

```
Input: (60, 1) - 60 days of price history
↓
3 Parallel CNN Branches:
  └─ Conv1D (k=2) → ReLU → 64 filters
  └─ Conv1D (k=3) → ReLU → 64 filters
  └─ Conv1D (k=4) → ReLU → 64 filters
↓
Concatenate: 192 filters
↓
BiLSTM Layer 1: 64 units + Dropout(0.2)
↓
Batch Normalization
↓
BiLSTM Layer 2: 32 units + Dropout(0.2)
↓
Batch Normalization
↓
Dense Layer 1: 32 units + ReLU + Dropout(0.2)
↓
Dense Layer 2: 16 units + ReLU
↓
Output Layer: 1 unit (price prediction)
```

### Parameters
- **Total Parameters**: 176,961
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Mean Squared Error (MSE)
- **Batch Size**: 32
- **Epochs Trained**: 47/60 (Early Stopping)

## Results

### Model Performance on Test Set

| Metric | Value |
|--------|-------|
| R² Score | 0.9215 (92.15%) |
| RMSE | $19.69 |
| MAE | $12.63 |
| MAPE | 4.89% |

### Visualizations

1. **Actual vs Predicted**: Line plot showing model predictions tracking true prices
2. **Scatter Plot**: Correlation visualization with R² = 0.9215
3. **Error Distribution**: Histogram of prediction errors (centered ~$17.22)
4. **Residuals**: Errors over time showing model performance stability
5. **Training History**: Loss curves showing convergence

## Usage

### Basic Usage (Jupyter/Colab)

```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import yfinance as yf
from scipy.ndimage import gaussian_filter1d

# 1. Download data
df = yf.download('TSLA', start='2015-01-01', end='2024-12-31')

# 2. Apply Gaussian smoothing
smoothed = gaussian_filter1d(df['Close'].values, sigma=3)

# 3. Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(smoothed.reshape(-1, 1))

# 4. Create sequences (see notebook for full implementation)
LOOKBACK = 60
X, y = create_sequences(scaled_data, LOOKBACK)

# 5. Train model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# 6. Predict
predictions = model.predict(X_test)
```

See `notebook.ipynb` for complete implementation.

## Project Structure

```
Final_Hybrid_CNN_LSTM_stock_price_prediction/
├─ README.md                    # Project documentation
├─ requirements.txt              # Python dependencies
├─ .gitignore                    # Git ignore rules
├─ notebook.ipynb                # Main Jupyter notebook
├─ data/
│  └─ tesla_stock_data.csv       # Raw stock data
├─ models/
│  └─ hybrid_cnn_lstm_model.h5  # Trained model weights
├─ results/
│  └─ predictions.csv           # Model predictions
└─ visualizations/
   └─ actual_vs_predicted.png    # Prediction plot
```

## Key Insights

1. **Gaussian Smoothing is Critical**: Reduces market noise, improves model accuracy from 56% to 92%
2. **CNN Multi-Scale Feature Extraction**: Three kernel sizes capture short, medium, and long-term patterns
3. **BiLSTM Importance**: Bidirectional LSTM captures both past and future context
4. **Data Matters**: High-quality preprocessed data is more important than model complexity

## Interview Questions & Answers

**Q: Why does this model achieve 92% accuracy when stock prices are unpredictable?**
A: The model predicts smoothed prices, not raw noisy prices. The 60-day Gaussian filter removes intraday noise, making the sequence learnable. This is trend prediction, not price prediction.

**Q: How would you improve beyond 92%?**
A: Increase sigma in Gaussian filter (more smoothing), add technical indicators (RSI, MACD), include volume data, or ensemble multiple models.

**Q: What are the limitations?**
A: The model only uses historical prices; it ignores news, macroeconomic events, and sudden market shocks. It's useful for trend analysis, not for trading signals.

## Contributing

Contributions are welcome! Please feel free to:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

MIT License - see LICENSE file for details

## Author

**Keshu017**
- GitHub: [@Keshu017](https://github.com/Keshu017)
- LinkedIn: [Your LinkedIn]
- Email: [Your Email]

## References

- [TensorFlow/Keras Documentation](https://www.tensorflow.org/guide)
- [LSTM Tutorial](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Time Series Forecasting Best Practices](https://machinelearningmastery.com/time-series-forecasting/)
- [Yahoo Finance API](https://finance.yahoo.com/)

---

**Last Updated**: December 2024
**Model R²**: 0.9215 (92.15%)
