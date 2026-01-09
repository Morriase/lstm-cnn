# LSTM-CNN XAUUSD Trading System

A dual-model deep learning system for XAUUSD (Gold) trading, combining price prediction with profitability classification for MetaTrader 5.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Dual-Model Trading System                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────┐    ┌─────────────────────────┐    │
│  │   Price Predictor       │    │  Profitability Classifier│    │
│  │   (LSTM-CNN Hybrid)     │    │  (CNN-only)              │    │
│  ├─────────────────────────┤    ├─────────────────────────┤    │
│  │ LSTM(50) + Conv1D(64)   │    │ Conv1D(32) → MaxPool    │    │
│  │ → Concat → Dense(32)    │    │ Conv1D(16) → GlobalMax  │    │
│  │ → Output(1)             │    │ Dense(16) → Output(2)   │    │
│  └───────────┬─────────────┘    └───────────┬─────────────┘    │
│              │                              │                   │
│              ▼                              ▼                   │
│       Next Close Price              P(long_wins), P(short_wins) │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                      Trading Logic                              │
│  direction = sign(predicted - current)                          │
│  magnitude = |predicted - current| / ATR                        │
│  IF magnitude >= 0.5 AND P_profitable > threshold:              │
│      TRADE in direction                                         │
│  ELSE:                                                          │
│      HOLD                                                       │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
├── python/
│   └── src/
│       ├── train.py              # Training pipeline (both models)
│       ├── lstm_cnn_model.py     # Price predictor architecture
│       ├── profitability_model.py # Profitability classifier
│       ├── data_cleaner.py       # NaN handling, outlier removal
│       ├── sequence_builder.py   # Sliding windows + scaler export
│       └── metrics.py            # RMSE, MAE, R², Directional Accuracy
│
├── Experts/lstm-cnn/
│   ├── LSTM_CNN_EA.mq5           # Main Expert Advisor
│   ├── Include/
│   │   ├── Feature_Engine.mqh    # Technical indicators
│   │   ├── Correlation_Pruner.mqh # Feature selection
│   │   └── TradeManager.mqh      # Position management
│   ├── Scripts/
│   │   └── Data_Exporter.mq5     # Export training data + triple barrier labels
│   └── Models/
│       ├── lstm_cnn_xauusd.onnx  # Price predictor
│       └── profitability_classifier.onnx # Profitability model
```

## Triple Barrier Labeling

Data_Exporter.mq5 generates labels for the profitability classifier:

| Parameter | Default | Description |
|-----------|---------|-------------|
| SL_ATR_Mult | 3.0 | Stop Loss = 3 × ATR (1R) |
| TP_ATR_Mult | 6.0 | Take Profit = 6 × ATR (2R) |
| MaxBarsForward | 48 | Max bars to look for TP/SL hit |

Labels:
- `1` = TP hit first (profitable trade)
- `0` = SL hit first (losing trade)
- `-1` = Neither hit within MaxBarsForward (excluded from training)

## Data Pipeline

1. **MT5 Data Export** → Features + Target + Long_Outcome + Short_Outcome
2. **Data Cleaning** → Forward fill NaN, cap outliers (z-score > 3)
3. **Normalization** → Min-max scaling to [0, 1], export scalers.csv
4. **Sequencing** → 30-bar sliding windows
5. **Train/Test Split** → 80/20 chronological (no data leakage)

## Training on Kaggle (2x T4 GPU)

```python
# Cell 1: Setup
!git clone https://github.com/Morriase/lstm-cnn.git
!pip install -q loguru

# Cell 2: Train both models
import sys
sys.path.append('/kaggle/working/lstm-cnn/python/src')
from train import TrainingPipeline

pipeline = TrainingPipeline()
results = pipeline.run_full_pipeline('/kaggle/input/lstm-cnn/lstm_cnn_data/training_data.csv')

# Cell 3: Download models
from IPython.display import FileLink
display(FileLink('/kaggle/working/results/lstm_cnn_xauusd.onnx'))
display(FileLink('/kaggle/working/results/profitability_classifier.onnx'))
display(FileLink('/kaggle/working/results/scalers.csv'))
```

## Output Files

| File | Description |
|------|-------------|
| `lstm_cnn_xauusd.onnx` | Price predictor model |
| `profitability_classifier.onnx` | Trade profitability model |
| `scalers.csv` | Min/max values for feature normalization |

## MT5 Integration

1. Copy ONNX models to `MQL5/Experts/lstm-cnn/Models/`
2. Copy `scalers.csv` to `MQL5/Experts/lstm-cnn/Models/`
3. Compile `LSTM_CNN_EA.mq5` in MetaEditor
4. Attach EA to XAUUSD H1 chart
5. Configure risk parameters

## Requirements

- Python 3.8+, TensorFlow 2.10+, PyTorch (for ONNX export)
- MetaTrader 5 (build 3000+)
- ONNX Runtime
