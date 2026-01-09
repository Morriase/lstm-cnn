# LSTM-CNN XAUUSD Trading System

A hybrid deep learning system for XAUUSD (Gold) price prediction using LSTM-CNN architecture, designed for MetaTrader 5 integration.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LSTM-CNN Hybrid Model                        │
├─────────────────────────────────────────────────────────────────┤
│  Input: [batch, 30 timesteps, N features]                       │
│                                                                 │
│  ┌─────────────┐         ┌─────────────┐                       │
│  │    LSTM     │         │   Conv1D    │                       │
│  │  (50 units) │         │ (64 filters)│                       │
│  │  + Dropout  │         │ + MaxPool   │                       │
│  └──────┬──────┘         └──────┬──────┘                       │
│         │                       │                               │
│         └───────────┬───────────┘                               │
│                     │                                           │
│              ┌──────▼──────┐                                    │
│              │   Concat    │                                    │
│              │  + Dense    │                                    │
│              │  + Dropout  │                                    │
│              └──────┬──────┘                                    │
│                     │                                           │
│              ┌──────▼──────┐                                    │
│              │   Output    │                                    │
│              │ (1 value)   │                                    │
│              └─────────────┘                                    │
│                                                                 │
│  Output: Next bar close price prediction                        │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
├── python/
│   └── src/
│       ├── train.py           # Training pipeline orchestrator
│       ├── lstm_cnn_model.py  # Model architecture & ONNX export
│       ├── data_cleaner.py    # NaN handling, outlier removal
│       ├── sequence_builder.py # Sliding window sequences
│       └── metrics.py         # RMSE, MAE, R², Directional Accuracy
│
├── Experts/lstm-cnn/
│   ├── LSTM_CNN_EA.mq5        # Main Expert Advisor
│   ├── Include/
│   │   ├── Feature_Engine.mqh    # Technical indicators
│   │   ├── Correlation_Pruner.mqh # Feature selection
│   │   ├── TradeManager.mqh      # Position management
│   │   └── ChartDisplay.mqh      # Visual dashboard
│   ├── Scripts/
│   │   └── Data_Exporter.mq5     # Export training data
│   └── Models/
│       └── lstm_cnn_xauusd.onnx  # Trained model
```

## Data Pipeline

1. **MT5 Data Export** → `Data_Exporter.mq5` exports OHLCV + indicators to CSV
2. **Data Cleaning** → Forward fill NaN, cap outliers (z-score > 3)
3. **Normalization** → Min-max scaling to [0, 1]
4. **Sequencing** → 30-bar sliding windows for LSTM input
5. **Train/Test Split** → 80/20 chronological split (no data leakage)

## Features Used

| Feature | Description |
|---------|-------------|
| Volume | Trading volume |
| BB_Upper | Bollinger Band upper |
| RSI | Relative Strength Index |
| MACD | MACD line |
| MACD_Histogram | MACD histogram |
| OBV | On-Balance Volume |

## Training on Kaggle (2x T4 GPU)

```python
# Cell 1
!git clone https://github.com/Morriase/lstm-cnn.git
!pip install -q loguru

# Cell 2
import sys
sys.path.append('/kaggle/working/lstm-cnn/python/src')
from train import TrainingPipeline

pipeline = TrainingPipeline()
results = pipeline.run_full_pipeline('/kaggle/input/lstm-cnn/lstm_cnn_data/training_data.csv')

# Cell 3
from IPython.display import FileLink
FileLink('/kaggle/working/results/lstm_cnn_xauusd.onnx')
```

## Metrics

- **RMSE** - Root Mean Squared Error
- **MAE** - Mean Absolute Error  
- **R²** - Coefficient of determination
- **Directional Accuracy** - % of correct up/down predictions

## MT5 Integration

1. Copy `lstm_cnn_xauusd.onnx` to `MQL5/Experts/lstm-cnn/Models/`
2. Compile `LSTM_CNN_EA.mq5` in MetaEditor
3. Attach EA to XAUUSD chart
4. Configure risk parameters in EA inputs

## Requirements

- Python 3.8+
- TensorFlow 2.10+
- MetaTrader 5 (build 3000+)
- ONNX Runtime
