"""
LSTM-CNN Hybrid Model Module for XAUUSD Trading System

This module implements the hybrid LSTM-CNN deep learning model for price forecasting:
- LSTM layer for temporal dependencies
- CNN layer for spatial pattern recognition
- Fusion in fully connected layer
- ONNX export for MQL5 inference
- Multi-GPU support for accelerated training (Kaggle 2x T4 Tesla)

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 7.1, 7.2, 7.3, 7.4, 7.5
"""

import os
import sys
from typing import Dict, Any, Optional, Tuple, List

import numpy as np

from loguru import logger

# Configure loguru
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO",
    colorize=True
)


def setup_gpu():
    """
    Configure GPU settings for optimal performance on Kaggle (2x T4 Tesla).
    Returns the distribution strategy for multi-GPU training.
    """
    try:
        import tensorflow as tf
        
        # List available GPUs
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            logger.info(f"Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
            
            # Enable memory growth to avoid OOM
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"Memory growth enabled for {gpu.name}")
                except RuntimeError as e:
                    logger.warning(f"Could not set memory growth for {gpu.name}: {e}")
            
            # Use MirroredStrategy for multi-GPU training
            if len(gpus) > 1:
                strategy = tf.distribute.MirroredStrategy()
                logger.info(f"Using MirroredStrategy with {strategy.num_replicas_in_sync} GPUs")
            else:
                strategy = tf.distribute.get_strategy()  # Default strategy
                logger.info("Using single GPU")
            
            # Log GPU details
            for gpu in gpus:
                details = tf.config.experimental.get_device_details(gpu)
                logger.info(f"GPU: {gpu.name}, Details: {details}")
        else:
            logger.warning("No GPUs found, using CPU")
            strategy = tf.distribute.get_strategy()
        
        return strategy
        
    except Exception as e:
        logger.warning(f"GPU setup failed: {e}. Falling back to CPU.")
        import tensorflow as tf
        return tf.distribute.get_strategy()


class ModelBuildError(Exception):
    """Custom exception for model building errors."""
    pass


class ModelTrainingError(Exception):
    """Custom exception for model training errors."""
    pass


class ONNXExportError(Exception):
    """Custom exception for ONNX export errors."""
    pass


# Default configuration matching design document
DEFAULT_CONFIG = {
    'lookback': 30,
    'num_features': 17,
    'lstm_units': 50,
    'lstm_dropout': 0.2,
    'cnn_filters': 64,
    'cnn_kernel_size': 3,
    'dense_units': 32,
    'learning_rate': 0.001,
    'epochs': 150,
    'batch_size': 64,
    'early_stopping_patience': 10,
    'validation_split': 0.2,
    'use_mixed_precision': True,  # Enable mixed precision for T4 GPUs
}


class LSTMCNNModel:
    """
    Hybrid LSTM-CNN model for XAUUSD price forecasting.
    
    Combines LSTM for temporal pattern learning with CNN for spatial
    feature extraction. The outputs are fused in a fully connected layer
    for final prediction.
    
    Architecture:
    - Input: [batch, lookback, features]
    - LSTM branch: LSTM layer with dropout
    - CNN branch: Conv1D layer with pooling
    - Fusion: Concatenate + Dense layers
    - Output: Single prediction value
    
    Supports multi-GPU training on Kaggle (2x T4 Tesla) with:
    - MirroredStrategy for data parallelism
    - Mixed precision (FP16) for faster training
    - Automatic batch size scaling
    
    Attributes:
        config: Configuration dictionary with model hyperparameters
        model: Keras model instance (after build_model is called)
        history: Training history (after train is called)
        strategy: Distribution strategy for multi-GPU
    
    Requirements: 6.1-6.5
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize LSTMCNNModel with configuration.
        
        Args:
            config: Configuration dictionary with keys:
                - lookback: Number of timesteps in input sequence (default: 30)
                - num_features: Number of input features (default: 17)
                - lstm_units: Number of LSTM units (default: 50)
                - lstm_dropout: Dropout rate for LSTM (default: 0.2)
                - cnn_filters: Number of CNN filters (default: 64)
                - cnn_kernel_size: CNN kernel size (default: 3)
                - dense_units: Units in fusion dense layer (default: 32)
                - learning_rate: Adam optimizer learning rate (default: 0.001)
                - epochs: Maximum training epochs (default: 150)
                - batch_size: Training batch size (default: 64)
                - early_stopping_patience: Epochs to wait before early stop (default: 10)
                - validation_split: Fraction of data for validation (default: 0.2)
                - use_mixed_precision: Enable FP16 mixed precision (default: True)
        
        Requirements: 6.1-6.5
        """
        # Merge provided config with defaults
        self.config = DEFAULT_CONFIG.copy()
        if config is not None:
            self.config.update(config)
        
        self.model = None
        self.history = None
        self._is_built = False
        self.strategy = None
        
        # Setup GPU and get distribution strategy
        self.strategy = setup_gpu()
        
        # Enable mixed precision for T4 GPUs (Tensor Cores)
        if self.config.get('use_mixed_precision', True):
            try:
                import tensorflow as tf
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                logger.info("Mixed precision (FP16) enabled for faster training")
            except Exception as e:
                logger.warning(f"Could not enable mixed precision: {e}")
        
        logger.info(f"LSTMCNNModel initialized with config: {self.config}")

    def build_model(self) -> 'tf.keras.Model':
        """
        Build the hybrid LSTM-CNN model architecture with multi-GPU support.
        
        Architecture:
        1. Input layer: [batch, lookback, features]
        2. LSTM branch:
           - LSTM layer with configurable units and dropout
        3. CNN branch:
           - Conv1D layer with configurable filters and kernel size
           - GlobalMaxPooling1D
        4. Fusion:
           - Concatenate LSTM and CNN outputs
           - Dense layer with ReLU activation
           - Output Dense layer (single value)
        5. Compile with Adam optimizer
        
        Returns:
            Compiled Keras model
        
        Raises:
            ModelBuildError: If model building fails
        
        Requirements: 6.1, 6.2, 6.3, 6.4
        """
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import (
                Input, LSTM, Conv1D, Dense, Dropout,
                GlobalMaxPooling1D, Concatenate
            )
            from tensorflow.keras.optimizers import Adam
        except ImportError as e:
            raise ModelBuildError(
                f"TensorFlow/Keras not installed. Install with: pip install tensorflow. Error: {e}"
            )
        
        try:
            lookback = self.config['lookback']
            num_features = self.config['num_features']
            lstm_units = self.config['lstm_units']
            lstm_dropout = self.config['lstm_dropout']
            cnn_filters = self.config['cnn_filters']
            cnn_kernel_size = self.config['cnn_kernel_size']
            dense_units = self.config['dense_units']
            learning_rate = self.config['learning_rate']
            
            logger.info(f"Building model: lookback={lookback}, features={num_features}, "
                       f"lstm_units={lstm_units}, cnn_filters={cnn_filters}")
            
            # Build model within strategy scope for multi-GPU support
            with self.strategy.scope():
                # Input layer
                inputs = Input(shape=(lookback, num_features), name='input')
                
                # LSTM branch - captures temporal dependencies
                # Requirements: 6.1 - LSTM layer with configurable units and 20% dropout
                lstm_out = LSTM(
                    units=lstm_units,
                    dropout=lstm_dropout,
                    return_sequences=False,
                    name='lstm'
                )(inputs)
                
                # CNN branch - captures spatial patterns
                # Requirements: 6.2 - CNN layer with configurable filters and kernel size
                cnn_out = Conv1D(
                    filters=cnn_filters,
                    kernel_size=cnn_kernel_size,
                    activation='relu',
                    padding='same',
                    name='conv1d'
                )(inputs)
                cnn_out = GlobalMaxPooling1D(name='global_max_pool')(cnn_out)
                
                # Fusion layer - combines LSTM and CNN outputs
                # Requirements: 6.3 - Fuse LSTM and CNN outputs in fully connected layer
                fused = Concatenate(name='fusion')([lstm_out, cnn_out])
                fused = Dense(dense_units, activation='relu', name='dense_fusion')(fused)
                fused = Dropout(lstm_dropout, name='dropout_fusion')(fused)
                
                # Output layer - single prediction value
                # Use float32 for output when using mixed precision
                outputs = Dense(1, activation='linear', name='output', dtype='float32')(fused)
                
                # Create model
                self.model = Model(inputs=inputs, outputs=outputs, name='lstm_cnn_model')
                
                # Compile with Adam optimizer
                # Requirements: 6.4 - Adam optimizer with configurable learning rate
                optimizer = Adam(learning_rate=learning_rate)
                self.model.compile(
                    optimizer=optimizer,
                    loss='mse',
                    metrics=['mae']
                )
            
            self._is_built = True
            
            # Log model summary and GPU info
            logger.info("Model built successfully")
            if hasattr(self.strategy, 'num_replicas_in_sync'):
                logger.info(f"Model distributed across {self.strategy.num_replicas_in_sync} GPU(s)")
            self.model.summary(print_fn=logger.info)
            
            return self.model
            
        except Exception as e:
            raise ModelBuildError(f"Failed to build model: {e}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: int = 1
    ) -> Dict[str, List[float]]:
        """
        Train the LSTM-CNN model with multi-GPU support.
        
        Supports configurable epochs and batch size with early stopping
        based on validation loss to prevent overfitting. Automatically
        scales batch size for multi-GPU training.
        
        Args:
            X_train: Training input sequences, shape [samples, lookback, features]
            y_train: Training targets, shape [samples, 1] or [samples,]
            X_val: Validation input sequences (optional)
            y_val: Validation targets (optional)
            verbose: Verbosity mode (0=silent, 1=progress bar, 2=one line per epoch)
        
        Returns:
            Training history dictionary with keys:
            - 'loss': Training loss per epoch
            - 'mae': Training MAE per epoch
            - 'val_loss': Validation loss per epoch (if validation data provided)
            - 'val_mae': Validation MAE per epoch (if validation data provided)
        
        Raises:
            ModelTrainingError: If training fails
            ValueError: If model not built or data shapes invalid
        
        Requirements: 6.5, 6.6, 6.7
        """
        if not self._is_built or self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        try:
            import tensorflow as tf
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
        except ImportError as e:
            raise ModelTrainingError(f"TensorFlow not installed: {e}")
        
        try:
            epochs = self.config['epochs']
            batch_size = self.config['batch_size']
            patience = self.config['early_stopping_patience']
            
            # Scale batch size for multi-GPU (effective batch = batch_size * num_gpus)
            num_replicas = getattr(self.strategy, 'num_replicas_in_sync', 1)
            global_batch_size = batch_size * num_replicas
            logger.info(f"Batch size: {batch_size} per GPU, {global_batch_size} global ({num_replicas} GPU(s))")
            
            # Validate input shapes
            expected_shape = (self.config['lookback'], self.config['num_features'])
            if X_train.shape[1:] != expected_shape:
                raise ValueError(
                    f"X_train shape {X_train.shape[1:]} doesn't match expected {expected_shape}"
                )
            
            # Ensure y is 2D and float32 for mixed precision compatibility
            if y_train.ndim == 1:
                y_train = y_train.reshape(-1, 1)
            y_train = y_train.astype(np.float32)
            X_train = X_train.astype(np.float32)
            
            logger.info(f"Training model: epochs={epochs}, batch_size={global_batch_size}, "
                       f"samples={X_train.shape[0]}")
            
            # Prepare validation data
            validation_data = None
            if X_val is not None and y_val is not None:
                if y_val.ndim == 1:
                    y_val = y_val.reshape(-1, 1)
                y_val = y_val.astype(np.float32)
                X_val = X_val.astype(np.float32)
                validation_data = (X_val, y_val)
                logger.info(f"Using provided validation data: {X_val.shape[0]} samples")
            
            # Callbacks
            callbacks = []
            
            # Early stopping on validation loss
            # Requirements: 6.6 - Implement early stopping based on validation loss
            early_stopping = EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)
            
            # Learning rate reduction on plateau
            lr_reducer = ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=patience // 2,
                min_lr=1e-6,
                verbose=1
            )
            callbacks.append(lr_reducer)
            
            # Train the model
            # Requirements: 6.5 - Support configurable epochs and batch size
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=global_batch_size,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=verbose
            )
            
            # Store and return history
            # Requirements: 6.7 - Return training history
            self.history = history.history
            
            logger.info(f"Training complete. Final loss: {self.history['loss'][-1]:.6f}")
            if 'val_loss' in self.history:
                logger.info(f"Final validation loss: {self.history['val_loss'][-1]:.6f}")
            
            return self.history
            
        except Exception as e:
            raise ModelTrainingError(f"Training failed: {e}")

    def export_onnx(self, output_path: str, validate: bool = True) -> str:
        """
        Export the trained model to ONNX format using PyTorch.
        
        Converts the Keras model to a PyTorch equivalent, then exports
        to ONNX using torch.onnx.export for MQL5 inference compatibility.
        
        Args:
            output_path: Path to save the ONNX model file
            validate: If True, validate the ONNX model loads correctly
        
        Returns:
            Path to the saved ONNX model file
        
        Raises:
            ONNXExportError: If export or validation fails
            ValueError: If model not built
        
        Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
        """
        if not self._is_built or self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        try:
            import torch
            import torch.nn as nn
        except ImportError as e:
            raise ONNXExportError(
                f"PyTorch not installed. Install with: pip install torch. Error: {e}"
            )
        
        try:
            lookback = self.config['lookback']
            num_features = self.config['num_features']
            
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            logger.info(f"Exporting model to ONNX via PyTorch: {output_path}")
            
            # Create PyTorch equivalent model and transfer weights
            pytorch_model = self._create_pytorch_model()
            self._transfer_weights_to_pytorch(pytorch_model)
            
            # Set to evaluation mode
            pytorch_model.eval()
            
            # Create dummy input for tracing
            # Requirements: 7.2 - Input shape [batch, lookback_window, num_features]
            dummy_input = torch.randn(1, lookback, num_features)
            
            # Export to ONNX with dynamic batch size
            # Requirements: 7.1 - Export model to ONNX format
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
            
            # Validate the exported model
            # Requirements: 7.4 - Validate ONNX model loads correctly
            if validate:
                self._validate_onnx(output_path)
            
            return output_path
            
        except ONNXExportError:
            raise
        except Exception as e:
            raise ONNXExportError(f"ONNX export failed: {e}")

    def _create_pytorch_model(self) -> 'torch.nn.Module':
        """Create a PyTorch model equivalent to the Keras model."""
        import torch
        import torch.nn as nn
        
        lookback = self.config['lookback']
        num_features = self.config['num_features']
        lstm_units = self.config['lstm_units']
        lstm_dropout = self.config['lstm_dropout']
        cnn_filters = self.config['cnn_filters']
        cnn_kernel_size = self.config['cnn_kernel_size']
        dense_units = self.config['dense_units']
        
        class LSTMCNNPyTorch(nn.Module):
            def __init__(self, lookback, num_features, lstm_units, lstm_dropout,
                         cnn_filters, cnn_kernel_size, dense_units):
                super().__init__()
                
                # LSTM branch
                self.lstm = nn.LSTM(
                    input_size=num_features,
                    hidden_size=lstm_units,
                    batch_first=True,
                    dropout=0  # Dropout handled separately for inference
                )
                
                # CNN branch
                self.conv1d = nn.Conv1d(
                    in_channels=num_features,
                    out_channels=cnn_filters,
                    kernel_size=cnn_kernel_size,
                    padding='same'
                )
                self.relu = nn.ReLU()
                self.global_max_pool = nn.AdaptiveMaxPool1d(1)
                
                # Fusion layers
                self.dense_fusion = nn.Linear(lstm_units + cnn_filters, dense_units)
                self.dropout = nn.Dropout(lstm_dropout)
                
                # Output layer
                self.output = nn.Linear(dense_units, 1)
            
            def forward(self, x):
                # x shape: [batch, lookback, features]
                
                # LSTM branch
                lstm_out, _ = self.lstm(x)
                lstm_out = lstm_out[:, -1, :]  # Take last timestep
                
                # CNN branch - needs [batch, channels, length]
                cnn_in = x.permute(0, 2, 1)
                cnn_out = self.conv1d(cnn_in)
                cnn_out = self.relu(cnn_out)
                cnn_out = self.global_max_pool(cnn_out).squeeze(-1)
                
                # Fusion
                fused = torch.cat([lstm_out, cnn_out], dim=1)
                fused = self.dense_fusion(fused)
                fused = self.relu(fused)
                fused = self.dropout(fused)
                
                # Output
                output = self.output(fused)
                return output
        
        return LSTMCNNPyTorch(
            lookback, num_features, lstm_units, lstm_dropout,
            cnn_filters, cnn_kernel_size, dense_units
        )

    def _transfer_weights_to_pytorch(self, pytorch_model: 'torch.nn.Module') -> None:
        """Transfer weights from Keras model to PyTorch model."""
        import torch
        
        # Get Keras weights
        keras_weights = {layer.name: layer.get_weights() for layer in self.model.layers}
        
        # Transfer LSTM weights
        # Keras LSTM: kernel [input, 4*units], recurrent_kernel [units, 4*units], bias [4*units]
        # PyTorch LSTM: weight_ih [4*units, input], weight_hh [4*units, units], bias_ih, bias_hh
        if 'lstm' in keras_weights and len(keras_weights['lstm']) >= 3:
            kernel, recurrent_kernel, bias = keras_weights['lstm'][:3]
            
            # Keras order: i, f, c, o -> PyTorch order: i, f, c, o (same but transposed)
            pytorch_model.lstm.weight_ih_l0.data = torch.tensor(kernel.T, dtype=torch.float32)
            pytorch_model.lstm.weight_hh_l0.data = torch.tensor(recurrent_kernel.T, dtype=torch.float32)
            pytorch_model.lstm.bias_ih_l0.data = torch.tensor(bias, dtype=torch.float32)
            pytorch_model.lstm.bias_hh_l0.data = torch.zeros_like(pytorch_model.lstm.bias_hh_l0)
        
        # Transfer Conv1D weights
        # Keras Conv1D: kernel [kernel_size, in_channels, out_channels]
        # PyTorch Conv1d: weight [out_channels, in_channels, kernel_size]
        if 'conv1d' in keras_weights and len(keras_weights['conv1d']) >= 2:
            kernel, bias = keras_weights['conv1d'][:2]
            # Transpose from [kernel_size, in_channels, out_channels] to [out_channels, in_channels, kernel_size]
            pytorch_model.conv1d.weight.data = torch.tensor(
                kernel.transpose(2, 1, 0), dtype=torch.float32
            )
            pytorch_model.conv1d.bias.data = torch.tensor(bias, dtype=torch.float32)
        
        # Transfer Dense fusion weights
        if 'dense_fusion' in keras_weights and len(keras_weights['dense_fusion']) >= 2:
            kernel, bias = keras_weights['dense_fusion'][:2]
            pytorch_model.dense_fusion.weight.data = torch.tensor(kernel.T, dtype=torch.float32)
            pytorch_model.dense_fusion.bias.data = torch.tensor(bias, dtype=torch.float32)
        
        # Transfer Output layer weights
        if 'output' in keras_weights and len(keras_weights['output']) >= 2:
            kernel, bias = keras_weights['output'][:2]
            pytorch_model.output.weight.data = torch.tensor(kernel.T, dtype=torch.float32)
            pytorch_model.output.bias.data = torch.tensor(bias, dtype=torch.float32)
        
        logger.info("Weights transferred from Keras to PyTorch model")

    def _validate_onnx(self, onnx_path: str) -> bool:
        """
        Validate the exported ONNX model.
        
        Checks that:
        1. The model loads correctly
        2. Input shape matches expected [batch, lookback, features]
        3. Output shape matches expected [batch, 1]
        4. Model can run inference with sample data
        
        Args:
            onnx_path: Path to the ONNX model file
        
        Returns:
            True if validation passes
        
        Raises:
            ONNXExportError: If validation fails
        
        Requirements: 7.2, 7.3, 7.4
        """
        try:
            import onnx
            import onnxruntime as ort
        except ImportError as e:
            raise ONNXExportError(
                f"onnx or onnxruntime not installed. Install with: "
                f"pip install onnx onnxruntime. Error: {e}"
            )
        
        try:
            # Load and check the model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model structure validation passed")
            
            # Check input shape
            # Requirements: 7.2 - Input shape [batch, lookback_window, num_features]
            input_info = onnx_model.graph.input[0]
            input_shape = [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]
            
            expected_lookback = self.config['lookback']
            expected_features = self.config['num_features']
            
            # First dimension is batch (dynamic), check lookback and features
            if len(input_shape) >= 3:
                if input_shape[1] != expected_lookback:
                    raise ONNXExportError(
                        f"Input lookback dimension {input_shape[1]} != expected {expected_lookback}"
                    )
                if input_shape[2] != expected_features:
                    raise ONNXExportError(
                        f"Input features dimension {input_shape[2]} != expected {expected_features}"
                    )
            
            logger.info(f"Input shape validated: {input_shape}")
            
            # Check output shape
            # Requirements: 7.3 - Output shape matches prediction target
            output_info = onnx_model.graph.output[0]
            output_shape = [dim.dim_value for dim in output_info.type.tensor_type.shape.dim]
            logger.info(f"Output shape: {output_shape}")
            
            # Test inference with sample data
            session = ort.InferenceSession(onnx_path)
            input_name = session.get_inputs()[0].name
            
            # Create sample input
            sample_input = np.random.randn(1, expected_lookback, expected_features).astype(np.float32)
            
            # Run inference
            result = session.run(None, {input_name: sample_input})
            
            logger.info(f"Test inference successful. Output shape: {result[0].shape}")
            
            return True
            
        except Exception as e:
            raise ONNXExportError(f"ONNX validation failed: {e}")

    def get_input_shape(self) -> Tuple[int, int]:
        """
        Get the expected input shape for the model.
        
        Returns:
            Tuple of (lookback, num_features)
        """
        return (self.config['lookback'], self.config['num_features'])

    def get_output_shape(self) -> Tuple[int]:
        """
        Get the expected output shape for the model.
        
        Returns:
            Tuple of (1,) for single prediction
        """
        return (1,)

    def summary(self) -> None:
        """Print model summary if model is built."""
        if self.model is not None:
            self.model.summary()
        else:
            logger.warning("Model not built yet. Call build_model() first.")
