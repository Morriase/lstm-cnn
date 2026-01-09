# Requirements Document

## Introduction

This system implements a hybrid LSTM-CNN deep learning model for XAUUSD (Gold) price forecasting, based on the architecture described in Fozap (2025). The system is split between MQL5 (data preparation, feature engineering, inference) and Python (model training only). The trained model is exported to ONNX format for real-time inference within MetaTrader 5.

## Glossary

- **Feature_Engine**: MQL5 module responsible for computing technical indicators and features from raw OHLCV data
- **Correlation_Pruner**: MQL5 module that removes redundant features based on correlation analysis
- **Quality_Scorer**: MQL5 module that evaluates and scores feature quality for model input
- **Data_Exporter**: MQL5 script that exports prepared datasets to CSV for Python training
- **LSTM_CNN_Trainer**: Python module that builds, trains, and exports the hybrid model
- **ONNX_Inference_Engine**: MQL5 EA component that loads ONNX model and performs predictions
- **Lookback_Window**: Number of historical bars used as input sequence (default: 30)
- **Technical_Indicator**: Mathematical calculation based on price/volume data (SMA, EMA, RSI, MACD, Bollinger Bands, OBV)

## Requirements

### Requirement 1: Feature Engineering Module

**User Story:** As a trader, I want technical indicators computed consistently in MQL5, so that the same calculations are used during both training data preparation and live inference.

#### Acceptance Criteria

1. THE Feature_Engine SHALL compute Simple Moving Averages (SMA) for periods 10 and 50
2. THE Feature_Engine SHALL compute Exponential Moving Averages (EMA) for periods 10 and 50
3. THE Feature_Engine SHALL compute Bollinger Bands (20-period SMA with 2 standard deviations)
4. THE Feature_Engine SHALL compute Relative Strength Index (RSI) with 14-period default
5. THE Feature_Engine SHALL compute MACD (12, 26, 9 periods) with signal line
6. THE Feature_Engine SHALL compute On-Balance Volume (OBV)
7. THE Feature_Engine SHALL normalize all features to [0, 1] range using Min-Max scaling
8. WHEN raw OHLCV data is provided, THE Feature_Engine SHALL return a structured array of computed features
9. THE Feature_Engine SHALL be implemented as a reusable .mqh include file with OOP design

### Requirement 2: Correlation Pruning Module

**User Story:** As a data scientist, I want redundant features removed based on correlation analysis, so that the model receives only meaningful, non-redundant inputs.

#### Acceptance Criteria

1. THE Correlation_Pruner SHALL compute pairwise Pearson correlation coefficients between all features
2. WHEN two features have correlation above a configurable threshold (default: 0.85), THE Correlation_Pruner SHALL remove the feature with lower variance
3. THE Correlation_Pruner SHALL output a list of retained feature indices
4. THE Correlation_Pruner SHALL be implemented as a reusable .mqh include file with OOP design
5. WHEN pruning is complete, THE Correlation_Pruner SHALL log which features were removed and why

### Requirement 3: Quality Scoring Module

**User Story:** As a data scientist, I want feature quality assessed before training, so that only high-quality features contribute to model predictions.

#### Acceptance Criteria

1. THE Quality_Scorer SHALL evaluate each feature based on variance, missing value ratio, and predictive correlation with target
2. THE Quality_Scorer SHALL assign a quality score between 0 and 1 for each feature
3. WHEN a feature's quality score falls below a configurable threshold (default: 0.3), THE Quality_Scorer SHALL flag it for exclusion
4. THE Quality_Scorer SHALL be implemented as a reusable .mqh include file with OOP design
5. THE Quality_Scorer SHALL output a quality report for all features

### Requirement 4: Data Export for Training

**User Story:** As a developer, I want prepared data exported to CSV format, so that Python can load it for training with minimal cleaning.

#### Acceptance Criteria

1. THE Data_Exporter SHALL export feature-engineered, pruned, and quality-filtered data to CSV
2. THE Data_Exporter SHALL include column headers matching feature names
3. THE Data_Exporter SHALL export target values (next-bar close price or direction) as the final column
4. THE Data_Exporter SHALL support configurable date ranges for training/testing splits
5. WHEN exporting, THE Data_Exporter SHALL preserve the exact feature order used by the Feature_Engine
6. THE Data_Exporter SHALL log the export configuration (date range, feature count, row count)
7. THE Data_Exporter SHALL mark missing/invalid values with a consistent placeholder (e.g., NaN or empty)

### Requirement 4.1: Data Cleaning in Python

**User Story:** As a data scientist, I want historical data cleaned before training, so that the model learns from complete, valid sequences.

#### Acceptance Criteria

1. THE Python_Data_Cleaner SHALL apply forward fill imputation for missing values caused by market closures or data gaps
2. THE Python_Data_Cleaner SHALL detect and handle outliers using configurable thresholds (e.g., 3 standard deviations)
3. THE Python_Data_Cleaner SHALL remove rows where forward fill cannot resolve missing values (e.g., start of dataset)
4. THE Python_Data_Cleaner SHALL log the number of rows affected by each cleaning operation
5. THE Python_Data_Cleaner SHALL validate that no NaN values remain after cleaning
6. THE Python_Data_Cleaner SHALL NOT alter feature calculations - only handle missing/invalid values
7. WHEN cleaning is complete, THE Python_Data_Cleaner SHALL output a clean dataset ready for sequence creation

### Requirement 5: Sequence Preparation

**User Story:** As a model developer, I want data structured into sequences with a lookback window, so that the LSTM can learn temporal patterns.

#### Acceptance Criteria

1. THE Sequence_Builder (Python) SHALL create sequences of configurable lookback length (default: 30 bars) from cleaned data
2. WHEN preparing sequences, THE Sequence_Builder SHALL structure data as [samples, timesteps, features]
3. THE MQL5 Feature_Engine SHALL create identical sequence structures during live inference
4. IF insufficient historical data exists for a complete sequence, THE system SHALL skip that sample
5. THE Sequence_Builder SHALL create target values aligned with each sequence (next-bar prediction)
6. THE Sequence_Builder SHALL ensure no data leakage between sequences (no future data in inputs)

### Requirement 6: LSTM-CNN Hybrid Model Architecture

**User Story:** As a model developer, I want a hybrid architecture combining LSTM and CNN layers, so that both temporal dependencies and spatial patterns are captured.

#### Acceptance Criteria

1. THE LSTM_CNN_Trainer SHALL implement an LSTM layer with configurable units (default: 50) and 20% dropout
2. THE LSTM_CNN_Trainer SHALL implement a CNN layer with configurable filters (default: 64) and kernel size (default: 3)
3. THE LSTM_CNN_Trainer SHALL fuse LSTM and CNN outputs in a fully connected layer
4. THE LSTM_CNN_Trainer SHALL use Adam optimizer with configurable learning rate (default: 0.001)
5. THE LSTM_CNN_Trainer SHALL support configurable epochs (default: 150) and batch size (default: 64)
6. THE LSTM_CNN_Trainer SHALL implement early stopping based on validation loss
7. THE LSTM_CNN_Trainer SHALL NOT perform any data transformation - only receive pre-processed data

### Requirement 7: Model Export to ONNX

**User Story:** As a developer, I want the trained model exported to ONNX format, so that it can be loaded and executed within MetaTrader 5.

#### Acceptance Criteria

1. WHEN training completes, THE LSTM_CNN_Trainer SHALL export the model to ONNX format
2. THE LSTM_CNN_Trainer SHALL specify input shape matching [batch, lookback_window, num_features]
3. THE LSTM_CNN_Trainer SHALL specify output shape matching the prediction target
4. THE LSTM_CNN_Trainer SHALL validate the ONNX model loads correctly before saving
5. THE LSTM_CNN_Trainer SHALL save the ONNX file to a configurable output path

### Requirement 8: ONNX Inference in MQL5

**User Story:** As a trader, I want the trained model to make predictions in real-time within MetaTrader 5, so that I can receive trading signals.

#### Acceptance Criteria

1. THE ONNX_Inference_Engine SHALL load the ONNX model file at EA initialization
2. THE ONNX_Inference_Engine SHALL use the same Feature_Engine module to prepare live input data
3. THE ONNX_Inference_Engine SHALL use the same Correlation_Pruner configuration used during training
4. WHEN a new bar forms, THE ONNX_Inference_Engine SHALL prepare the input sequence and run inference
5. THE ONNX_Inference_Engine SHALL output the prediction (price forecast or direction probability)
6. IF the ONNX model fails to load, THE ONNX_Inference_Engine SHALL log an error and disable trading
7. THE ONNX_Inference_Engine SHALL handle ONNX runtime errors gracefully without crashing

### Requirement 9: Training Pipeline Orchestration

**User Story:** As a developer, I want a clear training workflow, so that I can reproduce model training consistently.

#### Acceptance Criteria

1. THE training pipeline SHALL follow this sequence: MQL5 data preparation → CSV export → Python training → ONNX export
2. THE training pipeline SHALL use an 80/20 train/test split based on date
3. THE training pipeline SHALL implement cross-validation within the training phase
4. THE training pipeline SHALL log all hyperparameters used during training
5. THE training pipeline SHALL save training metrics (RMSE, MAE, MAPE, R²) to a results file

### Requirement 10: Model Evaluation Metrics

**User Story:** As a data scientist, I want comprehensive evaluation metrics, so that I can assess model performance accurately.

#### Acceptance Criteria

1. THE LSTM_CNN_Trainer SHALL compute Root Mean Squared Error (RMSE)
2. THE LSTM_CNN_Trainer SHALL compute Mean Absolute Error (MAE)
3. THE LSTM_CNN_Trainer SHALL compute Mean Absolute Percentage Error (MAPE)
4. THE LSTM_CNN_Trainer SHALL compute R² score
5. THE LSTM_CNN_Trainer SHALL generate loss curves (training vs validation)
6. THE LSTM_CNN_Trainer SHALL generate actual vs predicted price plots
7. THE LSTM_CNN_Trainer SHALL save all metrics and plots to configurable output directory

### Requirement 11: Configuration Management

**User Story:** As a developer, I want configuration handled appropriately for each component, so that the EA is self-contained and training is reproducible.

#### Acceptance Criteria

1. THE MQL5 EA SHALL use `input` parameters for all configurable settings (compiled into .ex5)
2. THE MQL5 EA SHALL NOT read external configuration files at runtime
3. THE Python training scripts SHALL use a separate configuration file (JSON/YAML/Python dict)
4. THE Feature_Engine.mqh SHALL use consistent default parameter values matching training
5. WHEN the EA is compiled, all settings SHALL be embedded - no external dependencies
