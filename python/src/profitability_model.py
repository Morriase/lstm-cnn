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
    'conv1_filters': 32,
    'conv2_filters': 16,
    'kernel_size': 3,
    'dense_units': 16,
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'epochs': 100,
    'batch_size': 64,
    'early_stopping_patience': 10,
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
                MaxPooling1D, GlobalMaxPooling1D
            )
            from tensorflow.keras.optimizers import Adam
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
            
            logger.info(f"Building profitability model: lookback={lookback}, features={num_features}")
            
            # Input layer
            inputs = Input(shape=(lookback, num_features), name='input')
            
            # First Conv block
            x = Conv1D(conv1_filters, kernel_size, activation='relu', 
                      padding='same', name='conv1')(inputs)
            x = MaxPooling1D(pool_size=2, name='maxpool1')(x)
            
            # Second Conv block
            x = Conv1D(conv2_filters, kernel_size, activation='relu',
                      padding='same', name='conv2')(x)
            x = GlobalMaxPooling1D(name='global_maxpool')(x)
            
            # Dense layers
            x = Dense(dense_units, activation='relu', name='dense1')(x)
            x = Dropout(dropout_rate, name='dropout')(x)
            
            # Output: 2 independent probabilities (not softmax - they're independent)
            outputs = Dense(2, activation='sigmoid', name='output', dtype='float32')(x)
            
            self.model = Model(inputs=inputs, outputs=outputs, name='profitability_classifier')
            
            # Compile with binary crossentropy per output
            optimizer = Adam(learning_rate=learning_rate)
            self.model.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction='sum_over_batch_size'),
                metrics=['accuracy']
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
            
            logger.info(f"Training on {len(X_train)} samples (filtered NaN)")
            
            validation_data = None
            if X_val is not None and y_val is not None:
                X_val = X_val.astype(np.float32)
                y_val = y_val.astype(np.float32)
                # Filter validation too
                val_valid_mask = ~np.any(np.isnan(y_val), axis=1)
                X_val = X_val[val_valid_mask]
                y_val = y_val[val_valid_mask]
                validation_data = (X_val, y_val)
            
            # Custom callback for progress
            class ProgressCallback(tf.keras.callbacks.Callback):
                def __init__(self, print_every=10):
                    super().__init__()
                    self.print_every = print_every
                
                def on_epoch_end(self, epoch, logs=None):
                    if (epoch + 1) % self.print_every == 0 or epoch == 0:
                        loss = logs.get('loss', 0)
                        acc = logs.get('accuracy', 0) * 100
                        val_loss = logs.get('val_loss', 0)
                        val_acc = logs.get('val_accuracy', 0) * 100
                        print(f"  [Profit] Epoch {epoch+1:3d} | BCE: {loss:.4f} | val_BCE: {val_loss:.4f} | acc: {acc:.1f}% | val_acc: {val_acc:.1f}%")
            
            callbacks = [
                ProgressCallback(print_every=10),
                EarlyStopping(
                    monitor='val_loss' if validation_data else 'loss',
                    patience=patience,
                    restore_best_weights=True,
                    verbose=0
                ),
                ReduceLROnPlateau(
                    monitor='val_loss' if validation_data else 'loss',
                    factor=0.5,
                    patience=patience // 2,
                    min_lr=1e-6,
                    verbose=0
                )
            ]
            
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data,
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
        predictions = self.model.predict(X, verbose=0)
        
        p_long_wins = predictions[:, 0]
        p_short_wins = predictions[:, 1]
        
        return p_long_wins, p_short_wins
    
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
                self.maxpool1 = nn.MaxPool1d(2)
                self.conv2 = nn.Conv1d(conv1_filters, conv2_filters, kernel_size, padding='same')
                self.global_maxpool = nn.AdaptiveMaxPool1d(1)
                
                # Dense layers
                self.dense1 = nn.Linear(conv2_filters, dense_units)
                self.dropout = nn.Dropout(dropout_rate)
                self.output = nn.Linear(dense_units, 2)
                
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                # x: [batch, lookback, features] -> [batch, features, lookback]
                x = x.permute(0, 2, 1)
                
                x = self.relu(self.conv1(x))
                x = self.maxpool1(x)
                x = self.relu(self.conv2(x))
                x = self.global_maxpool(x).squeeze(-1)
                
                x = self.relu(self.dense1(x))
                x = self.dropout(x)
                x = self.sigmoid(self.output(x))
                
                return x
        
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
        
        # Conv2
        if 'conv2' in keras_weights and len(keras_weights['conv2']) >= 2:
            kernel, bias = keras_weights['conv2'][:2]
            pytorch_model.conv2.weight.data = torch.tensor(
                kernel.transpose(2, 1, 0), dtype=torch.float32
            )
            pytorch_model.conv2.bias.data = torch.tensor(bias, dtype=torch.float32)
        
        # Dense1
        if 'dense1' in keras_weights and len(keras_weights['dense1']) >= 2:
            kernel, bias = keras_weights['dense1'][:2]
            pytorch_model.dense1.weight.data = torch.tensor(kernel.T, dtype=torch.float32)
            pytorch_model.dense1.bias.data = torch.tensor(bias, dtype=torch.float32)
        
        # Output
        if 'output' in keras_weights and len(keras_weights['output']) >= 2:
            kernel, bias = keras_weights['output'][:2]
            pytorch_model.output.weight.data = torch.tensor(kernel.T, dtype=torch.float32)
            pytorch_model.output.bias.data = torch.tensor(bias, dtype=torch.float32)
        
        logger.info("Weights transferred to PyTorch model")
    
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
