# LSTM-CNN XAUUSD Trading Model - Training Guide

## Kaggle Training (2x T4 Tesla GPUs)

### Quick Start

1. Upload your `training_data.csv` (exported from MT5 Data_Exporter) to Kaggle
2. Clone/upload this repo to Kaggle
3. Enable GPU accelerator (Settings → Accelerator → GPU T4 x2)
4. Run training:

```python
import sys
sys.path.append('/kaggle/working/python/src')

from train import TrainingPipeline

# Configure for Kaggle GPUs
config = {
    'epochs': 150,
    'batch_size': 128,          # Will be doubled for 2 GPUs (256 effective)
    'lookback_window': 30,
    'use_mixed_precision': True, # FP16 for T4 Tensor Cores
    'output_dir': '/kaggle/working/results'
}

pipeline = TrainingPipeline(config=config)
results = pipeline.run_full_pipeline('/kaggle/input/your-dataset/training_data.csv')

# Download the ONNX model
from IPython.display import FileLink
FileLink('/kaggle/working/results/lstm_cnn_xauusd.onnx')
```

### GPU Features

- **MirroredStrategy**: Automatically distributes training across both T4 GPUs
- **Mixed Precision (FP16)**: 2-3x faster training using Tensor Cores
- **Memory Growth**: Prevents OOM errors by allocating GPU memory as needed
- **Batch Scaling**: Batch size automatically scales with number of GPUs

### Recommended Settings for Kaggle

| Parameter | Value | Notes |
|-----------|-------|-------|
| batch_size | 128 | Per GPU, 256 effective with 2 GPUs |
| epochs | 150 | Early stopping will kick in earlier |
| learning_rate | 0.001 | Default, auto-reduced on plateau |
| use_mixed_precision | True | Essential for T4 performance |

### CSV Format (from Data_Exporter.mq5)

The training data should have columns matching Feature_Engine output:
- Open, High, Low, Close, Volume
- SMA_10, SMA_50, EMA_10, EMA_50
- BB_Upper, BB_Middle, BB_Lower
- RSI, MACD, MACD_Signal, MACD_Histogram, OBV
- Target (last column - next bar close price)

### Output Files

After training, you'll find in `results/`:
- `lstm_cnn_xauusd.onnx` - Model for MT5 (copy to `Experts/lstm-cnn/Models/`)
- `training_results.json` - Metrics and config
- `predictions.png` - Actual vs predicted plot
- `loss_curves.png` - Training/validation loss

### Copying Model to MT5

1. Download `lstm_cnn_xauusd.onnx` from Kaggle
2. Copy to: `MQL5/Experts/lstm-cnn/Models/lstm_cnn_xauusd.onnx`
3. Recompile `LSTM_CNN_EA.mq5` in MetaEditor


### Full Kaggle Notebook Example

```python
# Cell 1: Setup
!pip install -q tf2onnx onnx onnxruntime

import sys
sys.path.append('/kaggle/working/python/src')

# Cell 2: Check GPU
import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")
print(f"GPUs: {tf.config.list_physical_devices('GPU')}")

# Cell 3: Train
from train import TrainingPipeline

config = {
    'epochs': 150,
    'batch_size': 128,
    'lookback_window': 30,
    'lstm_units': 64,
    'cnn_filters': 64,
    'dense_units': 32,
    'learning_rate': 0.001,
    'early_stopping_patience': 15,
    'use_cross_validation': True,
    'cv_folds': 5,
    'output_dir': '/kaggle/working/results'
}

pipeline = TrainingPipeline(config=config)
results = pipeline.run_full_pipeline('/kaggle/input/xauusd-training/training_data.csv')

# Cell 4: Results
print(f"Test RMSE: {results['evaluation_metrics']['rmse']:.4f}")
print(f"Test MAE: {results['evaluation_metrics']['mae']:.4f}")
print(f"Test R²: {results['evaluation_metrics']['r2']:.4f}")

# Cell 5: Download model
from IPython.display import FileLink
FileLink(results['onnx_path'])
```

### Troubleshooting

**OOM Error**: Reduce batch_size to 64 or 32

**Slow Training**: Ensure GPU accelerator is enabled and mixed_precision is True

**NaN Loss**: Check your CSV for missing values, the DataCleaner should handle them

**Poor Results**: Try increasing epochs, adjusting lookback_window, or collecting more training data

### Requirements

```
tensorflow>=2.10.0
torch>=1.12.0
onnx>=1.12.0
onnxruntime>=1.12.0
pandas>=1.4.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
```
