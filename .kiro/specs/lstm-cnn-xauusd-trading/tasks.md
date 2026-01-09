# Implementation Plan: LSTM-CNN XAUUSD Trading System

## Overview

This implementation follows a strict order: MQL5 data pipeline modules first, then Python training components, then the inference EA. Each MQL5 module is a reusable `.mqh` include file with OOP design.

## Tasks

- [x] 1. Set up project structure
  - Create MQL5 directory structure under `Experts/lstm-cnn/`
  - Create `Include/` folder for .mqh modules
  - Create `Scripts/` folder for data export script
  - Create `Models/` folder for ONNX output
  - Create Python project structure with `src/` and `tests/`
  - _Requirements: 11.1, 11.2_

- [x] 2. Implement Feature_Engine.mqh
  - [x] 2.1 Create CFeatureEngine class skeleton with member variables
    - Define indicator period members, normalization bounds arrays
    - Implement Init() with default parameters
    - _Requirements: 1.9_
  
  - [x] 2.2 Implement SMA calculation
    - CalculateSMA() using iMA() or manual calculation
    - Support configurable periods (10, 50)
    - _Requirements: 1.1_
  
  - [x] 2.3 Implement EMA calculation
    - CalculateEMA() using iMA() with MODE_EMA or manual recursive formula
    - Support configurable periods (10, 50)
    - _Requirements: 1.2_
  
  - [x] 2.4 Implement Bollinger Bands calculation
    - CalculateBollingerBands() returning upper, middle, lower
    - Use 20-period SMA with 2 standard deviations
    - _Requirements: 1.3_
  
  - [x] 2.5 Implement RSI calculation
    - CalculateRSI() using iRSI() or manual calculation
    - 14-period default, ensure output in [0, 100]
    - _Requirements: 1.4_
  
  - [x] 2.6 Implement MACD calculation
    - CalculateMACD() returning macd, signal, histogram
    - Use periods 12, 26, 9
    - _Requirements: 1.5_
  
  - [x] 2.7 Implement OBV calculation
    - CalculateOBV() using cumulative volume based on price direction
    - _Requirements: 1.6_
  
  - [x] 2.8 Implement normalization
    - Normalize() using Min-Max scaling to [0, 1]
    - SetNormalizationBounds() to store min/max from training
    - _Requirements: 1.7_
  
  - [x] 2.9 Implement ComputeFeatures() and ComputeSequence()
    - ComputeFeatures() returns all indicators for single bar
    - ComputeSequence() returns [lookback, features] matrix
    - _Requirements: 1.8, 5.3_

  - [x] 2.10 Write property tests for indicator calculations
    - **Property 1: Technical Indicator Formula Correctness**
    - **Property 2: Indicator Value Range Invariants**
    - **Validates: Requirements 1.1-1.7**

- [x] 3. Implement Correlation_Pruner.mqh
  - [x] 3.1 Create CCorrelationPruner class skeleton
    - Define threshold, retained indices array
    - Implement Init() with default threshold 0.85
    - _Requirements: 2.4_
  
  - [x] 3.2 Implement Pearson correlation calculation
    - PearsonCorrelation() between two arrays
    - ComputeCorrelationMatrix() for all feature pairs
    - _Requirements: 2.1_
  
  - [x] 3.3 Implement variance calculation
    - GetVariance() for a feature column
    - _Requirements: 2.2_
  
  - [x] 3.4 Implement pruning logic
    - PruneFeatures() removes lower-variance feature when correlation > threshold
    - Store retained indices for later use
    - _Requirements: 2.2, 2.3_
  
  - [x] 3.5 Implement ApplyPruning() for inference
    - Apply stored pruning indices to new data
    - _Requirements: 2.3_

  - [x] 3.6 Write property tests for correlation pruning
    - **Property 3: Correlation Matrix Mathematical Properties**
    - **Property 4: Correlation Pruning Correctness**
    - **Validates: Requirements 2.1-2.3**

- [x] 4. Implement Quality_Scorer.mqh (use fuzzy logic)
  - [x] 4.1 Create CQualityScorer class and FeatureQualityReport struct
    - Define threshold, reports array
    - Implement Init() with default threshold 0.3
    - _Requirements: 3.4_
  
  - [x] 4.2 Implement scoring components
    - ScoreVariance() - higher variance = better
    - ScoreMissingRatio() - fewer missing = better
    - ScoreTargetCorrelation() - higher correlation with target = better
    - _Requirements: 3.1_
  
  - [x] 4.3 Implement overall scoring and filtering
    - ComputeOverallScore() combining components
    - GetQualityIndices() returning features above threshold
    - GenerateReport() for logging
    - _Requirements: 3.2, 3.3, 3.5_

  - [x] 4.4 Write property tests for quality scoring
    - **Property 2: Indicator Value Range Invariants** (quality scores in [0,1])
    - **Validates: Requirements 3.2**

- [x] 5. Implement Data_Exporter.mq5 script
  - [x] 5.1 Create script with input parameters
    - Symbol, timeframe, date range inputs
    - Output path input
    - _Requirements: 4.4_
  
  - [x] 5.2 Implement data collection loop
    - Iterate through historical bars
    - Call Feature_Engine for each bar
    - _Requirements: 4.1_
  
  - [x] 5.3 Apply correlation pruning and quality filtering
    - Run Correlation_Pruner on collected data
    - Run Quality_Scorer and filter features
    - _Requirements: 4.1_
  
  - [x] 5.4 Implement CSV export
    - Write headers matching feature names
    - Write data rows with target as last column
    - Mark missing values consistently
    - _Requirements: 4.2, 4.3, 4.5, 4.7_
  
  - [x] 5.5 Implement logging
    - Log export configuration, feature count, row count
    - _Requirements: 4.6_

  - [x] 5.6 Write property tests for data export
    - **Property 5: Data Export Feature Order Preservation**
    - **Validates: Requirements 4.2, 4.3, 4.5**

- [x] 6. Checkpoint - MQL5 modules complete
  - Ensure all .mqh modules compile without errors
  - Run Data_Exporter script to generate training CSV
  - Verify CSV has expected columns and data
  - Ask the user if questions arise

- [x] 7. Implement Python data_cleaner.py
  - [x] 7.1 Create DataCleaner class
    - __init__ with outlier_threshold parameter
    - _Requirements: 4.1.2_
  
  - [x] 7.2 Implement forward_fill()
    - Apply pandas forward fill for NaN values
    - _Requirements: 4.1.1_
  
  - [x] 7.3 Implement handle_outliers()
    - Detect outliers using z-score threshold
    - Cap or remove outliers
    - _Requirements: 4.1.2_
  
  - [x] 7.4 Implement validation and logging
    - validate_no_nan() raises error if NaN remains
    - Log rows affected by each operation
    - _Requirements: 4.1.3, 4.1.4, 4.1.5_
  
  - [x] 7.5 Implement clean() pipeline
    - Chain forward_fill, handle_outliers, validate
    - Return clean DataFrame
    - _Requirements: 4.1.6, 4.1.7_

  - [x] 7.6 Write property tests for data cleaning
    - **Property 6: Data Cleaning Preserves Valid Values**
    - **Validates: Requirements 4.1.1, 4.1.5, 4.1.6, 4.1.7**

- [x] 8. Implement Python sequence_builder.py
  - [x] 8.1 Create SequenceBuilder class
    - __init__ with lookback parameter (default 30)
    - _Requirements: 5.1_
  
  - [x] 8.2 Implement create_sequences()
    - Create sliding window sequences
    - Output shape [samples, timesteps, features]
    - Align targets with sequences (next-bar prediction)
    - _Requirements: 5.2, 5.5_
  
  - [x] 8.3 Implement data leakage prevention
    - Ensure no future data in input sequences
    - Skip samples with insufficient history
    - _Requirements: 5.4, 5.6_

  - [x] 8.4 Write property tests for sequence building
    - **Property 7: Sequence Structure Correctness**
    - **Validates: Requirements 5.1, 5.2, 5.5, 5.6**

- [x] 9. Implement Python lstm_cnn_model.py
  - [x] 9.1 Create LSTMCNNModel class
    - __init__ with config dict (lookback, num_features, lstm_units, etc.)
    - _Requirements: 6.1-6.5_
  
  - [x] 9.2 Implement build_model()
    - LSTM layer with configurable units and dropout
    - CNN layer with configurable filters and kernel size
    - Fusion in fully connected layer
    - Adam optimizer with configurable learning rate
    - _Requirements: 6.1, 6.2, 6.3, 6.4_
  
  - [x] 9.3 Implement train()
    - Support configurable epochs and batch size
    - Implement early stopping on validation loss
    - Return training history
    - _Requirements: 6.5, 6.6, 6.7_
  
  - [x] 9.4 Implement export_onnx()
    - Export model to ONNX format
    - Specify correct input/output shapes
    - Validate ONNX loads correctly
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [x] 9.5 Write property tests for ONNX export
    - **Property 8: ONNX Shape Consistency**
    - **Validates: Requirements 7.2, 7.3, 7.4**

- [-] 10. Implement Python metrics.py
  - [x] 10.1 Implement evaluation metrics
    - compute_rmse(), compute_mae(), compute_mape(), compute_r2()
    - _Requirements: 10.1, 10.2, 10.3, 10.4_
  
  - [x] 10.2 Implement visualization
    - plot_loss_curves() for training vs validation
    - plot_predictions() for actual vs predicted
    - _Requirements: 10.5, 10.6_
  
  - [x] 10.3 Implement results saving
    - Save metrics to JSON/CSV
    - Save plots to output directory
    - _Requirements: 10.7_

  - [-] 10.4 Write property tests for metrics
    - **Property 11: Evaluation Metric Formula Correctness**
    - **Validates: Requirements 10.1-10.4**

- [x] 11. Implement Python training pipeline
  - [x] 11.1 Create train.py main script
    - Load CSV from MQL5 export
    - Run DataCleaner
    - Run SequenceBuilder
    - Split train/test by date (80/20)
    - _Requirements: 9.1, 9.2_
  
  - [x] 11.2 Implement training loop
    - Build and train model
    - Implement cross-validation
    - Log hyperparameters
    - _Requirements: 9.3, 9.4_
  
  - [x] 11.3 Implement evaluation and export
    - Compute all metrics on test set
    - Generate plots
    - Export ONNX model
    - Save results file
    - _Requirements: 9.5, 10.1-10.7_

  - [x] 11.4 Write property tests for train/test split
    - **Property 10: Train/Test Split Date Boundary**
    - **Validates: Requirements 9.2**

- [x] 12. Checkpoint - Python training complete
  - Run full training pipeline on exported CSV
  - Verify ONNX model is created
  - Review metrics and plots
  - Ask the user if questions arise

- [x] 13. Implement LSTM_CNN_EA.mq5 (for trades management and chart display, use the modules; ChartDisplay.mqh and TradeManager.mqh)
  - [x] 13.1 Create EA skeleton with embedded ONNX
    - #resource directive for ONNX model
    - Input parameters for settings and normalization bounds
    - _Requirements: 8.1, 11.1, 11.2_
  
  - [x] 13.2 Implement OnInit()
    - Load ONNX from buffer
    - Set input/output shapes
    - Initialize Feature_Engine and Correlation_Pruner
    - _Requirements: 8.1, 8.2, 8.3_
  
  - [x] 13.3 Implement OnTick() inference
    - Detect new bar
    - Compute feature sequence using Feature_Engine
    - Apply pruning
    - Run ONNX inference
    - _Requirements: 8.4, 8.5_
  
  - [x] 13.4 Implement error handling
    - Handle ONNX load failure
    - Handle inference errors gracefully
    - _Requirements: 8.6, 8.7_

  - [x] 13.5 Write property tests for inference consistency
    - **Property 9: Inference Feature Consistency**
    - **Validates: Requirements 8.2, 8.3**

- [ ] 14. Final checkpoint - Full system integration
  - Compile EA with embedded ONNX model
  - Test EA in MT5 Strategy Tester
  - Verify predictions are generated on new bars
  - Ensure no runtime errors
  - Ask the user if questions arise

## Notes

- All tasks including property tests are required
- MQL5 modules must be completed before Python training (data dependency)
- ONNX model must be trained before EA can be compiled with embedded model
- Normalization bounds from training must be transferred to EA input parameters
- All .mqh modules use consistent default parameters to ensure training/inference parity
