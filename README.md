# Coffee Futures LSTM Forecasting

This repository contains an LSTM-based time series forecasting model designed to predict **1-day ahead prices** of coffee futures. The model uses historical daily closing prices and is optimized through rolling window cross-validation.

## Overview

- **Objective**: Predict next-day coffee futures prices using LSTM neural networks.
- **Model**: Multi-layer LSTM with dropout regularization.
- **Training**: Includes data normalization, sequence generation, hyperparameter tuning, and early stopping.
- **Evaluation**: Model performance assessed using RMSE, MAE, and MAPE across 5 folds.

## Files

- `coffee_futures_lstm_1day.py`: Full Python script for data loading, preprocessing, model training, cross-validation, evaluation, and visualization.
- `coffee_futures_lstm_1day_model.h5`: Saved Keras model trained on the best-performing hyperparameters (generated after training).
- `Coffee_Futures_WisdomTree.csv`: Historical price data (not included in this repo – place your dataset in the `sample_data/` directory).

## How to Use

1. Place your coffee futures CSV file in the appropriate path (e.g., `/content/sample_data/` if using Google Colab).
2. Run the `coffee_futures_lstm_1day.py` script or adapt it inside a Jupyter Notebook environment.
3. The script will:
   - Preprocess the data
   - Train multiple LSTM models via cross-validation
   - Evaluate and save the final model
   - Output validation metrics and prediction plots

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy, Pandas, scikit-learn, Matplotlib

Install dependencies via pip:

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib
```

## Notebook Version

For a more detailed explanation of each step, **please refer to the Jupyter notebook version of this code**, which includes visual outputs and additional commentary.

## License

This project is open-source and available under the MIT License.

---

Let me know if you'd like a separate section for Colab users or want to include sample outputs (like plots or metrics screenshots).
