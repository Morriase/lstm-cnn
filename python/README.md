# LSTM-CNN XAUUSD Trading Model - Kaggle Training

## Copy & Paste into Kaggle Notebook

```python
# Cell 1: Setup
!git clone https://github.com/Morriase/lstm-cnn.git
!pip install -q tf2onnx onnx onnxruntime

# Cell 2: Train
import sys
sys.path.append('/kaggle/working/lstm-cnn/python/src')
from train import TrainingPipeline

pipeline = TrainingPipeline({
    'epochs': 150,
    'batch_size': 128,
    'lookback_window': 30,
    'output_dir': '/kaggle/working/results'
})
results = pipeline.run_full_pipeline('/kaggle/input/lstm-cnn/lstm_cnn_data/training_data.csv')

# Cell 3: Download
from IPython.display import FileLink
FileLink('/kaggle/working/results/lstm_cnn_xauusd.onnx')
```

Enable **GPU T4 x2** in Kaggle Settings before running.
