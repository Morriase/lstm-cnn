"""
Profitability Classifier Model for LSTM-CNN XAUUSD Trading System

This module implements a CNN-only classifier that predicts the probability
of a trade being profitable (hitting TP before SL) for both long and short directions.

Architecture:
- Conv1D(32) → MaxPool → Conv1D(16) → GlobalMaxPool → Dense(16) → Dense(2) → Sigmoid
- Output: [P(long_wins), P(short_wins)]

Used in conjunction with the price predictor model:
- Price predictor determines direction and magnitude
- Profitability classifier filters trades by win probability

Requirements: Dual-model trading system with triple barrier labeling
"""

import os
import sys
import warnings
from typing import Dict, Any, Optional, Tuple, List

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger('tensorflow').setLevel(logging.CRITICAL)

import numpy as np

from loguru import logger


class ProfitabilityModelError(Exception):
    """Custom exception for profitability model errors."""
    pass


DEFAULT_PROFITABILITY_CONFIG = {
    'lookback': 30,
    'num_features': 6,
    'conv1_filters': 32,      # Reduced to prevent overfitting
    'conv2_filters': 16,      # Reduced to prevent overfitting
    'kernel_size': 3,
    'dense_units': 16,        # Reduced to prevent overfitting
    'dropout_rate': 0.5,      # Increased significantly for regularization
    'learning_rate': 0.0005,
    'epochs': 100,            # Reduced - early stopping will handle it
    'batch_size': 256,        # Larger batch for more stable gradients
    'early_stopping_patience': 20,  # Give it more time to find good weights
    'l2_reg': 0.01,           # L2 regularization
}


class ProfitabilityClassifier:
    """
    CNN classifier for predicting trade profitability.
    
    Predicts P(long_wins) and P(short_wins) independently using
    triple barrier labeled data. Only trained on samples where
    either TP or SL was hit (excludes -1 outcomes).
    
    Architecture:
        Input: [batch, lookback, features]
        Conv1D(32, kernel=3) → ReLU → MaxPool(2)
        Conv1D(16, kernel=3) → ReLU → GlobalMaxPool
        Dense(16) → ReLU → Dropout
        Dense(2) → Sigmoid
        Output: [P(long_wins), P(short_wins)]
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize profitability classifier with configuration."""
        self.config = DEFAULT_PROFITABILITY_CONFIG.copy()
        if config is not None:
            self.config.update(config)
        
        self.model = None
        self.history = None
        self._is_built = False
    
    def build_model(self) -> 'tf.keras.Model':
        """
        Build the CNN classifier architecture.
        
        Returns:
            Compiled Keras model
        """
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import (
                Input, Conv1D, Dense, Dropout,
                MaxPooling1D, GlobalMaxPooling1D, BatchNormalization
            )
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.regularizers import l2
        except ImportError as e:
            raise ProfitabilityModelError(f"TensorFlow not installed: {e}")
        
        try:
            lookback = self.config['lookback']
            num_features = self.config['num_features']
            conv1_filters = self.config['conv1_filters']
            conv2_filters = self.config['conv2_filters']
            kernel_size = self.config['kernel_size']
            dense_units = self.config['dense_units']
            dropout_rate = self.config['dropout_rate']
            learning_rate = self.config['learning_rate']
            l2_reg = self.config.get('l2_reg', 0.01)
            
            # Input layer
            inputs = Input(shape=(lookback, num_features), name='input')
            
            # First Conv block with L2 regularization
            x = Conv1D(conv1_filters, kernel_size, activation='relu', 
                      padding='same', name='conv1',
                      kernel_regularizer=l2(l2_reg))(inputs)
            x = BatchNormalization(name='bn1')(x)
            x = MaxPooling1D(pool_size=2, name='maxpool1')(x)
            x = Dropout(dropout_rate / 2, name='dropout1')(x)  # Light dropout after conv
            
            # Second Conv block with L2 regularization
            x = Conv1D(conv2_filters, kernel_size, activation='relu',
                      padding='same', name='conv2',
                      kernel_regularizer=l2(l2_reg))(x)
            x = BatchNormalization(name='bn2')(x)
            x = GlobalMaxPooling1D(name='global_maxpool')(x)
            
            # Dense layers with strong regularization
            x = Dense(dense_units, activation='relu', name='dense1',
                     kernel_regularizer=l2(l2_reg))(x)
            x = Dropout(dropout_rate, name='dropout')(x)
            
            # Two separate output heads for independent binary classification
            long_output = Dense(1, activation='sigmoid', name='long_output', dtype='float32')(x)
            short_output = Dense(1, activation='sigmoid', name='short_output', dtype='float32')(x)
            
            self.model = Model(inputs=inputs, outputs=[long_output, short_output], 
                              name='profitability_classifier')
            
            # Compile with separate losses for each output
            optimizer = Adam(learning_rate=learning_rate)
            self.model.compile(
                optimizer=optimizer,
                loss={'long_output': 'binary_crossentropy', 'short_output': 'binary_crossentropy'},
                metrics={'long_output': 'accuracy', 'short_output': 'accuracy'}
            )
            
            self._is_built = True
            total_params = self.model.count_params()
            logger.info(f"Model built: {total_params:,} parameters")
            
            return self.model
            
        except Exception as e:
            raise ProfitabilityModelError(f"Failed to build model: {e}")


    def filter_valid_samples(
        self,
        X: np.ndarray,
        long_outcomes: np.ndarray,
        short_outcomes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter samples to only include those with valid outcomes.
        
        Excludes samples where outcome == -1 (neither TP nor SL hit).
        For training, we only want samples with clear win/loss outcomes.
        
        Args:
            X: Input sequences [samples, lookback, features]
            long_outcomes: Long trade outcomes (1=win, 0=loss, -1=neither)
            short_outcomes: Short trade outcomes (1=win, 0=loss, -1=neither)
        
        Returns:
            Tuple of (X_filtered, y_long, y_short) where:
            - X_filtered: Sequences with at least one valid outcome
            - y_long: Binary labels for long trades (NaN for invalid)
            - y_short: Binary labels for short trades (NaN for invalid)
        """
        # Create mask for samples with at least one valid outcome
        long_valid = long_outcomes >= 0
        short_valid = short_outcomes >= 0
        any_valid = long_valid | short_valid
        
        n_total = len(X)
        n_valid = np.sum(any_valid)
        n_long_valid = np.sum(long_valid)
        n_short_valid = np.sum(short_valid)
        
        logger.info(f"Filtering samples: {n_valid}/{n_total} have valid outcomes")
        logger.info(f"  Long valid: {n_long_valid}, Short valid: {n_short_valid}")
        
        # Filter X
        X_filtered = X[any_valid]
        
        # Create y arrays (use -1 for invalid, will be masked during training)
        y_long = long_outcomes[any_valid].astype(np.float32)
        y_short = short_outcomes[any_valid].astype(np.float32)
        
        # Replace -1 with NaN for masking
        y_long[y_long < 0] = np.nan
        y_short[y_short < 0] = np.nan
        
        return X_filtered, y_long, y_short
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: int = 1
    ) -> Dict[str, List[float]]:
        """
        Train the profitability classifier.
        
        Args:
            X_train: Training sequences [samples, lookback, features]
            y_train: Training labels [samples, 2] for [long_outcome, short_outcome]
            X_val: Validation sequences (optional)
            y_val: Validation labels (optional)
            verbose: Verbosity mode
        
        Returns:
            Training history dictionary
        """
        if not self._is_built or self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        try:
            import tensorflow as tf
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        except ImportError as e:
            raise ProfitabilityModelError(f"TensorFlow not installed: {e}")
        
        try:
            epochs = self.config['epochs']
            batch_size = self.config['batch_size']
            patience = self.config['early_stopping_patience']
            
            # Ensure correct dtypes
            X_train = X_train.astype(np.float32)
            y_train = y_train.astype(np.float32)
            
            # Filter out samples with any NaN in labels
            # (simpler than sample weights for multi-output)
            valid_mask = ~np.any(np.isnan(y_train), axis=1)
            X_train = X_train[valid_mask]
            y_train = y_train[valid_mask]
            
            # Split y into two separate arrays for two-head model
            y_long = y_train[:, 0:1]  # Keep as 2D [samples, 1]
            y_short = y_train[:, 1:2]
            
            # Compute class weights to handle imbalance
            # This makes the model pay more attention to the minority class (wins)
            long_pos = np.sum(y_long == 1)
            long_neg = np.sum(y_long == 0)
            short_pos = np.sum(y_short == 1)
            short_neg = np.sum(y_short == 0)
            
            # Weight = total / (2 * class_count) - higher weight for minority
            long_weight_0 = len(y_long) / (2 * long_neg) if long_neg > 0 else 1.0
            long_weight_1 = len(y_long) / (2 * long_pos) if long_pos > 0 else 1.0
            short_weight_0 = len(y_short) / (2 * short_neg) if short_neg > 0 else 1.0
            short_weight_1 = len(y_short) / (2 * short_pos) if short_pos > 0 else 1.0
            
            logger.info(f"Training on {len(X_train)} samples (filtered NaN)")
            logger.info(f"Long class balance: {long_pos} wins / {long_neg} losses ({100*long_pos/(long_pos+long_neg):.1f}% win rate)")
            logger.info(f"Short class balance: {short_pos} wins / {short_neg} losses ({100*short_pos/(short_pos+short_neg):.1f}% win rate)")
            
            # Use class weights instead of oversampling for multi-output model
            # This is cleaner and doesn't duplicate data
            # Weight formula: total / (2 * class_count) gives higher weight to minority class
            class_weight_long = {0: long_weight_0, 1: long_weight_1}
            class_weight_short = {0: short_weight_0, 1: short_weight_1}
            
            logger.info(f"Long class weights: loss={long_weight_0:.2f}, win={long_weight_1:.2f}")
            logger.info(f"Short class weights: loss={short_weight_0:.2f}, win={short_weight_1:.2f}")
            
            # Create sample weights that combine both outputs
            # For each sample, use the max weight from either output
            sample_weights = np.ones(len(X_train), dtype=np.float32)
            for i in range(len(X_train)):
                long_w = long_weight_1 if y_long[i, 0] == 1 else long_weight_0
                short_w = short_weight_1 if y_short[i, 0] == 1 else short_weight_0
                sample_weights[i] = max(long_w, short_w)
            
            logger.info(f"Sample weights range: [{sample_weights.min():.2f}, {sample_weights.max():.2f}]")
            
            
            validation_data = None
            if X_val is not None and y_val is not None:
                X_val = X_val.astype(np.float32)
                y_val = y_val.astype(np.float32)
                # Filter validation too
                val_valid_mask = ~np.any(np.isnan(y_val), axis=1)
                X_val = X_val[val_valid_mask]
                y_val = y_val[val_valid_mask]
                y_val_long = y_val[:, 0:1]
                y_val_short = y_val[:, 1:2]
                validation_data = (X_val, {'long_output': y_val_long, 'short_output': y_val_short})
            
            # Custom callback for progress
            class ProgressCallback(tf.keras.callbacks.Callback):
                def __init__(self, print_every=10):
                    super().__init__()
                    self.print_every = print_every
                
                def on_epoch_end(self, epoch, logs=None):
                    if (epoch + 1) % self.print_every == 0 or epoch == 0:
                        loss = logs.get('loss', 0)
                        long_acc = logs.get('long_output_accuracy', 0) * 100
                        short_acc = logs.get('short_output_accuracy', 0) * 100
                        val_loss = logs.get('val_loss', 0)
                        val_long_acc = logs.get('val_long_output_accuracy', 0) * 100
                        val_short_acc = logs.get('val_short_output_accuracy', 0) * 100
                        print(f"  [Profit] Epoch {epoch+1:3d} | loss: {loss:.4f} | long_acc: {long_acc:.1f}% | short_acc: {short_acc:.1f}% | val_loss: {val_loss:.4f}")
            
            callbacks = [
                ProgressCallback(print_every=10),
                EarlyStopping(
                    monitor='val_loss',  # Monitor validation loss to prevent overfitting
                    patience=patience,
                    restore_best_weights=True,
                    verbose=0
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=patience // 2,
                    min_lr=1e-6,
                    verbose=0
                )
            ]
            
            history = self.model.fit(
                X_train, {'long_output': y_long, 'short_output': y_short},
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data,
                sample_weight=sample_weights,  # Apply class balancing via sample weights
                callbacks=callbacks,
                verbose=0
            )
            
            self.history = history.history
            logger.info("Profitability model training complete!")
            
            return self.history
            
        except Exception as e:
            raise ProfitabilityModelError(f"Training failed: {e}")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict profitability probabilities.
        
        Args:
            X: Input sequences [samples, lookback, features]
        
        Returns:
            Tuple of (p_long_wins, p_short_wins) arrays
        """
        if self.model is None:
            raise ValueError("Model not built or trained.")
        
        X = X.astype(np.float32)
        long_pred, short_pred = self.model.predict(X, verbose=0)
        
        return long_pred.flatten(), short_pred.flatten()
    
    def export_onnx(self, output_path: str, validate: bool = True) -> str:
        """
        Export the trained model to ONNX format.
        
        Args:
            output_path: Path to save the ONNX model
            validate: If True, validate the exported model
        
        Returns:
            Path to the saved ONNX model
        """
        if not self._is_built or self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        try:
            import torch
            import torch.nn as nn
        except ImportError as e:
            raise ProfitabilityModelError(f"PyTorch not installed: {e}")
        
        try:
            lookback = self.config['lookback']
            num_features = self.config['num_features']
            
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            logger.info(f"Exporting profitability model to ONNX: {output_path}")
            
            # Create PyTorch equivalent
            pytorch_model = self._create_pytorch_model()
            self._transfer_weights_to_pytorch(pytorch_model)
            pytorch_model.eval()
            
            # Export to ONNX
            dummy_input = torch.randn(1, lookback, num_features)
            
            torch.onnx.export(
                pytorch_model,
                dummy_input,
                output_path,
                input_names=['input'],
                output_names=['output'],
                opset_version=14,
                do_constant_folding=True,
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            logger.info(f"ONNX model saved to: {output_path}")
            
            if validate:
                self._validate_onnx(output_path)
            
            return output_path
            
        except Exception as e:
            raise ProfitabilityModelError(f"ONNX export failed: {e}")
    
    def _create_pytorch_model(self) -> 'torch.nn.Module':
        """Create PyTorch equivalent of the Keras model."""
        import torch
        import torch.nn as nn
        
        lookback = self.config['lookback']
        num_features = self.config['num_features']
        conv1_filters = self.config['conv1_filters']
        conv2_filters = self.config['conv2_filters']
        kernel_size = self.config['kernel_size']
        dense_units = self.config['dense_units']
        dropout_rate = self.config['dropout_rate']
        
        class ProfitabilityCNN(nn.Module):
            def __init__(self):
                super().__init__()
                
                # Conv layers (input: [batch, features, lookback] in PyTorch)
                self.conv1 = nn.Conv1d(num_features, conv1_filters, kernel_size, padding='same')
                self.bn1 = nn.BatchNorm1d(conv1_filters)
                self.maxpool1 = nn.MaxPool1d(2)
                self.conv2 = nn.Conv1d(conv1_filters, conv2_filters, kernel_size, padding='same')
                self.bn2 = nn.BatchNorm1d(conv2_filters)
                self.global_maxpool = nn.AdaptiveMaxPool1d(1)
                
                # Dense layers
                self.dense1 = nn.Linear(conv2_filters, dense_units)
                self.dropout = nn.Dropout(dropout_rate)
                
                # Two output heads
                self.long_output = nn.Linear(dense_units, 1)
                self.short_output = nn.Linear(dense_units, 1)
                
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                # x: [batch, lookback, features] -> [batch, features, lookback]
                x = x.permute(0, 2, 1)
                
                x = self.relu(self.bn1(self.conv1(x)))
                x = self.maxpool1(x)
                x = self.relu(self.bn2(self.conv2(x)))
                x = self.global_maxpool(x).squeeze(-1)
                
                x = self.relu(self.dense1(x))
                # Note: dropout is only applied during training, not inference
                # x = self.dropout(x)  # Skip dropout for inference
                
                long_out = self.sigmoid(self.long_output(x))
                short_out = self.sigmoid(self.short_output(x))
                
                # Concatenate for single output tensor [batch, 2]
                return torch.cat([long_out, short_out], dim=1)
        
        return ProfitabilityCNN()
    
    def _transfer_weights_to_pytorch(self, pytorch_model: 'torch.nn.Module') -> None:
        """Transfer weights from Keras to PyTorch model."""
        import torch
        
        keras_weights = {layer.name: layer.get_weights() for layer in self.model.layers}
        
        # Conv1
        if 'conv1' in keras_weights and len(keras_weights['conv1']) >= 2:
            kernel, bias = keras_weights['conv1'][:2]
            pytorch_model.conv1.weight.data = torch.tensor(
                kernel.transpose(2, 1, 0), dtype=torch.float32
            )
            pytorch_model.conv1.bias.data = torch.tensor(bias, dtype=torch.float32)
        
        # BatchNorm1
        if 'bn1' in keras_weights and len(keras_weights['bn1']) >= 4:
            gamma, beta, moving_mean, moving_var = keras_weights['bn1'][:4]
            pytorch_model.bn1.weight.data = torch.tensor(gamma, dtype=torch.float32)
            pytorch_model.bn1.bias.data = torch.tensor(beta, dtype=torch.float32)
            pytorch_model.bn1.running_mean.data = torch.tensor(moving_mean, dtype=torch.float32)
            pytorch_model.bn1.running_var.data = torch.tensor(moving_var, dtype=torch.float32)
        
        # Conv2
        if 'conv2' in keras_weights and len(keras_weights['conv2']) >= 2:
            kernel, bias = keras_weights['conv2'][:2]
            pytorch_model.conv2.weight.data = torch.tensor(
                kernel.transpose(2, 1, 0), dtype=torch.float32
            )
            pytorch_model.conv2.bias.data = torch.tensor(bias, dtype=torch.float32)
        
        # BatchNorm2
        if 'bn2' in keras_weights and len(keras_weights['bn2']) >= 4:
            gamma, beta, moving_mean, moving_var = keras_weights['bn2'][:4]
            pytorch_model.bn2.weight.data = torch.tensor(gamma, dtype=torch.float32)
            pytorch_model.bn2.bias.data = torch.tensor(beta, dtype=torch.float32)
            pytorch_model.bn2.running_mean.data = torch.tensor(moving_mean, dtype=torch.float32)
            pytorch_model.bn2.running_var.data = torch.tensor(moving_var, dtype=torch.float32)
        
        # Dense1
        if 'dense1' in keras_weights and len(keras_weights['dense1']) >= 2:
            kernel, bias = keras_weights['dense1'][:2]
            pytorch_model.dense1.weight.data = torch.tensor(kernel.T, dtype=torch.float32)
            pytorch_model.dense1.bias.data = torch.tensor(bias, dtype=torch.float32)
        
        # Long output
        if 'long_output' in keras_weights and len(keras_weights['long_output']) >= 2:
            kernel, bias = keras_weights['long_output'][:2]
            pytorch_model.long_output.weight.data = torch.tensor(kernel.T, dtype=torch.float32)
            pytorch_model.long_output.bias.data = torch.tensor(bias, dtype=torch.float32)
        
        # Short output
        if 'short_output' in keras_weights and len(keras_weights['short_output']) >= 2:
            kernel, bias = keras_weights['short_output'][:2]
            pytorch_model.short_output.weight.data = torch.tensor(kernel.T, dtype=torch.float32)
            pytorch_model.short_output.bias.data = torch.tensor(bias, dtype=torch.float32)
    
    def _validate_onnx(self, onnx_path: str) -> bool:
        """Validate the exported ONNX model."""
        try:
            import onnx
            import onnxruntime as ort
        except ImportError as e:
            raise ProfitabilityModelError(f"onnx/onnxruntime not installed: {e}")
        
        try:
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model structure validation passed")
            
            # Test inference
            session = ort.InferenceSession(onnx_path)
            input_name = session.get_inputs()[0].name
            
            sample_input = np.random.randn(
                1, self.config['lookback'], self.config['num_features']
            ).astype(np.float32)
            
            result = session.run(None, {input_name: sample_input})
            logger.info(f"ONNX inference test passed, output shape: {result[0].shape}")
            
            return True
            
        except Exception as e:
            raise ProfitabilityModelError(f"ONNX validation failed: {e}")
