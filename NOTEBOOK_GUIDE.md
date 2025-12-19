# Jupyter Notebook Guide - Final Hybrid CNN-LSTM Stock Price Prediction

## Overview
This guide provides instructions for running the Jupyter/Colab notebook for the Hybrid CNN-LSTM stock price prediction project. The notebook contains the complete implementation, training, and evaluation of the deep learning model.

## Notebook Location
- **Colab**: [Link to Colab Notebook](https://colab.research.google.com/drive/1Q7wlHfviSMisa3TrM9u35JuuyxaQeZZ5)
- **GitHub**: Download `Final_CNN_LSTM.ipynb` from this repository

## Prerequisites
Before running the notebook, ensure you have:
- Python 3.7+
- Jupyter Notebook or Google Colab
- All dependencies from `requirements.txt` installed

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

# Launch Jupyter
jupyter notebook
```

### Option 2: Google Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com/)
2. Click "File" → "Open notebook" → "GitHub"
3. Enter: `Keshu017/Final_Hybrid_CNN_LSTM_stock_price_prediction`
4. Select the notebook from the list
5. Click "Open in Colab"

## Notebook Sections

### 1. Library Imports & Setup
- Imports TensorFlow, Keras, scikit-learn, pandas, numpy
- Configures GPU if available in Colab
- Sets random seeds for reproducibility

### 2. Data Download & Exploration
- Downloads Tesla (TSLA) stock data from Yahoo Finance (2015-2024)
- Displays dataset shape and basic statistics
- Total records: 2,515 trading days

### 3. Data Preprocessing
- **Gaussian Smoothing**: Applied with σ=3 to reduce market noise
- **MinMax Scaling**: Scales data to [0, 1] range
- **Sequence Creation**: Creates 60-day lookback windows
- **Train-Test Split**: 85% training (2,086 samples), 15% testing (369 samples)

### 4. Model Architecture
- **Hybrid CNN-LSTM**:
  - 3 Parallel CNN branches (kernel sizes: 2, 3, 4)
  - 2 Bidirectional LSTM layers (64→32 units)
  - Dropout & Batch Normalization for regularization
  - Dense head (32→16→1)
  - Total parameters: 176,961

### 5. Model Training
- **Optimizer**: Adam (LR: 0.001 with decay)
- **Loss Function**: Mean Squared Error (MSE)
- **Callbacks**:
  - Early Stopping (patience: 10, best epoch: 40)
  - Learning Rate Reduction on Plateau
- **Training**: 47/60 epochs

### 6. Model Evaluation
Generates comprehensive metrics:
- **R² Score**: 0.9215 (92.15% accuracy)
- **RMSE**: $19.69
- **MAE**: $12.63
- **MAPE**: 4.89%

### 7. Visualizations
Creates 4 comprehensive plots:
1. **Actual vs Predicted**: Line plot showing model predictions
2. **Scatter Plot**: Correlation visualization with R² = 0.9215
3. **Error Distribution**: Histogram of prediction errors
4. **Residuals Over Time**: Error stability analysis
5. **Training History**: Loss curves showing convergence
6. **Final Summary**: Performance metrics and model quality assessment

## Output Files Generated

The notebook generates the following output files:

```
├── cnn_lstm_prediction_analysis.png
│   └── 4-subplot visualization of model performance
├── training_history_and_summary.png
│   └── Loss curves and model summary
└── final_model_results_summary.png
    └── Comprehensive results visualization
```

## Key Results

### Model Performance
| Metric | Value | Status |
|--------|-------|--------|
| R² Score | 0.9215 | ✓ Excellent (>90%) |
| RMSE | $19.69 | ✓ Very Low |
| MAE | $12.63 | ✓ Acceptable |
| MAPE | 4.89% | ✓ Strong |
| Accuracy | 92.15% | ✓ Excellent |

### Architecture Summary
```
Input (60, 1)
    ↓
3 Parallel CNN (K=2,3,4) × 64 filters
    ↓
Concatenate → (60, 192)
    ↓
BiLSTM (64) + Dropout(0.2)
    ↓
Batch Normalization
    ↓
BiLSTM (32) + Dropout(0.2)
    ↓
Batch Normalization
    ↓
Dense (32) + ReLU + Dropout(0.2)
    ↓
Dense (16) + ReLU
    ↓
Dense (1) → Output Price
```

## Running the Notebook

1. **In Colab**:
   - Click "Runtime" → "Run all" to execute all cells
   - Or click the play button on each cell individually
   - Check GPU status: click "Runtime" → "Change runtime type" → Select GPU

2. **In Jupyter**:
   ```bash
   jupyter notebook Final_CNN_LSTM.ipynb
   ```
   - Press `Shift + Enter` to run each cell
   - Or click Cell → Run All Cells

## Expected Execution Time
- **Google Colab (GPU)**: ~5-10 minutes
- **Local Machine (CPU)**: ~20-30 minutes
- **Local Machine (GPU)**: ~5-10 minutes

## Troubleshooting

### Issue: Memory Error
- **Solution**: Reduce batch size (line with `batch_size=32` → `batch_size=16`)
- **Alternative**: Use Google Colab with GPU support

### Issue: Import Errors
- **Solution**: Ensure all dependencies are installed:
  ```bash
  pip install -r requirements.txt
  ```

### Issue: Slow Execution
- **Solution**: Enable GPU in Colab (Runtime → Change runtime type → GPU)
- Use batch processing for data loading

### Issue: Different Results
- **Solution**: Random seeds are set to 42 for reproducibility
- Small variations may occur due to hardware differences

## Customization Guide

### Change Stock Symbol
Modify the download line:
```python
df = yf.download('AAPL', start='2015-01-01', end='2024-12-31')  # Use AAPL for Apple
```

### Adjust Gaussian Smoothing
Modify sigma value:
```python
smoothed_close = gaussian_filter1d(df['Close'].values, sigma=5)  # Increase for more smoothing
```

### Modify Lookback Window
```python
LOOKBACK = 30  # Change from 60 to 30
```

### Change Train-Test Split
```python
split_idx = int(len(X) * 0.80)  # Change from 0.85 to 0.80
```

## Interview Q&A Based on This Notebook

### Q: Why does this model achieve 92% accuracy when stock prices are unpredictable?
**A**: The model predicts smoothed prices, not raw noisy prices. Gaussian smoothing removes intraday noise, making the smoothed trend learnable. The 92% R² represents trend prediction, not speculative price prediction.

### Q: What role does Gaussian smoothing play?
**A**: Smoothing reduces market noise from daily fluctuations, helping the model learn underlying trends rather than random variations. It improved accuracy from 56% to 92%.

### Q: Why use CNN + LSTM instead of pure LSTM?
**A**: CNN extracts spatial patterns across different time scales (kernels 2,3,4). LSTM captures temporal dependencies. Together, they learn both local patterns and long-term trends.

### Q: How does the 60-day lookback window help?
**A**: It captures approximately 3 months of historical data, sufficient for market trends while remaining computationally efficient. Longer windows may capture irrelevant historical data.

### Q: What would you do to improve beyond 92%?
**A**: Add technical indicators (RSI, MACD), include volume data, ensemble multiple models, or use attention mechanisms for dynamic feature weighting.

## Model Deployment

For production deployment, save the trained model:
```python
# In notebook
model_final.save('stock_price_predictor.h5')

# In production
from tensorflow.keras.models import load_model
model = load_model('stock_price_predictor.h5')
predictions = model.predict(X_new)
```

## License
MIT License - See LICENSE file for details

## Author
Keshu017 | GitHub: [@Keshu017](https://github.com/Keshu017)

## References
- [TensorFlow Keras Documentation](https://www.tensorflow.org/guide)
- [LSTM Tutorial](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Time Series Forecasting](https://machinelearningmastery.com/time-series-forecasting/)
- [Yahoo Finance API](https://finance.yahoo.com/)

---
**Last Updated**: December 2024
**Model Version**: Final Hybrid CNN-LSTM v1.0
**Python Version**: 3.7+
