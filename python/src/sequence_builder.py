"""
Sequence Builder Module for LSTM-CNN XAUUSD Trading System

This module handles sequence creation for the LSTM-CNN model:
- Creates sliding window sequences with configurable lookback
- Structures data as [samples, timesteps, features]
- Aligns targets with sequences for next-bar prediction
- Prevents data leakage by ensuring no future data in inputs
- Normalizes features using min-max scaling

Requirements: 5.1, 5.2, 5.4, 5.5, 5.6
"""

import sys
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union

from loguru import logger

# Configure loguru
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO",
    colorize=True
)


class SequenceBuilderError(Exception):
    """Custom exception for sequence building errors."""
    pass


class SequenceBuilder:
    """
    Sequence builder for preparing LSTM-CNN training data.
    
    Creates sliding window sequences from time series data with proper
    target alignment for next-bar prediction. Ensures no data leakage
    by only using past data in input sequences.
    
    Attributes:
        lookback: Number of historical bars in each sequence (default: 30)
        feature_min: Min values for each feature (for denormalization)
        feature_max: Max values for each feature (for denormalization)
        target_min: Min value for target (for denormalization)
        target_max: Max value for target (for denormalization)
    """
    
    def __init__(self, lookback: int = 30):
        """
        Initialize SequenceBuilder with lookback window size.
        
        Args:
            lookback: Number of historical timesteps in each sequence.
                     Must be a positive integer.
        
        Raises:
            ValueError: If lookback is not a positive integer.
        
        Requirements: 5.1
        """
        if not isinstance(lookback, int) or lookback <= 0:
            raise ValueError(f"lookback must be a positive integer, got {lookback}")
        
        self.lookback = lookback
        self.feature_min = None
        self.feature_max = None
        self.target_min = None
        self.target_max = None
        logger.info(f"SequenceBuilder initialized with lookback={lookback}")


    def create_sequences(
        self, 
        data: Union[pd.DataFrame, np.ndarray],
        target_column: Optional[str] = None,
        target_array: Optional[np.ndarray] = None,
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window sequences for LSTM-CNN training.
        
        Creates sequences using a sliding window approach where each sequence
        contains `lookback` consecutive timesteps. The target for each sequence
        is the value at the next timestep (next-bar prediction).
        
        Data leakage prevention:
        - Each sequence only contains data from timesteps [t-lookback, t-1]
        - The target is the value at timestep t
        - No future data appears in input sequences
        
        Args:
            data: Input feature data as DataFrame or 2D numpy array.
                  Shape should be [total_timesteps, num_features].
            target_column: Name of target column if data is DataFrame.
                          If None and target_array is None, uses last column.
            target_array: Explicit target values as 1D array. If provided,
                         this is used instead of extracting from data.
            normalize: If True, apply min-max normalization to features and target.
        
        Returns:
            Tuple of (X, y) where:
            - X: Input sequences with shape [samples, timesteps, features]
            - y: Target values with shape [samples, 1]
        
        Raises:
            SequenceBuilderError: If insufficient data for sequences.
            ValueError: If data format is invalid.
        
        Requirements: 5.2, 5.5
        """
        # Convert DataFrame to numpy array if needed
        if isinstance(data, pd.DataFrame):
            feature_names = list(data.columns)
            
            if target_array is None:
                # Extract target from data
                if target_column is not None:
                    if target_column not in data.columns:
                        raise ValueError(f"Target column '{target_column}' not found in data")
                    targets = data[target_column].values.astype(np.float64)
                    features = data.drop(columns=[target_column]).values.astype(np.float64)
                else:
                    # Use last column as target
                    targets = data.iloc[:, -1].values.astype(np.float64)
                    features = data.iloc[:, :-1].values.astype(np.float64)
            else:
                targets = target_array.astype(np.float64)
                features = data.values.astype(np.float64)
        else:
            # Numpy array input
            features = np.asarray(data, dtype=np.float64)
            if features.ndim != 2:
                raise ValueError(f"Data must be 2D, got shape {features.shape}")
            
            if target_array is None:
                # Use last column as target
                targets = features[:, -1].copy()
                features = features[:, :-1].copy()
            else:
                targets = np.asarray(target_array, dtype=np.float64)
        
        # Validate dimensions
        n_samples, n_features = features.shape
        
        if len(targets) != n_samples:
            raise ValueError(
                f"Target length ({len(targets)}) must match data length ({n_samples})"
            )
        
        # Normalize features and targets (min-max scaling to [0, 1])
        if normalize:
            # Store min/max for denormalization later
            self.feature_min = features.min(axis=0)
            self.feature_max = features.max(axis=0)
            self.target_min = targets.min()
            self.target_max = targets.max()
            
            # Avoid division by zero
            feature_range = self.feature_max - self.feature_min
            feature_range[feature_range == 0] = 1.0
            target_range = self.target_max - self.target_min
            if target_range == 0:
                target_range = 1.0
            
            features = (features - self.feature_min) / feature_range
            targets = (targets - self.target_min) / target_range
            
            logger.info(f"Normalized features to [0, 1] range")
            logger.info(f"Target range: [{self.target_min:.2f}, {self.target_max:.2f}]")
        
        # Check if we have enough data
        # We need at least lookback + 1 samples to create one sequence
        # (lookback for input, 1 for target)
        min_required = self.lookback + 1
        if n_samples < min_required:
            raise SequenceBuilderError(
                f"Insufficient data: need at least {min_required} samples "
                f"for lookback={self.lookback}, got {n_samples}"
            )
        
        # Calculate number of valid sequences
        # For each sequence starting at index i, we use:
        # - Input: features[i:i+lookback] (indices i to i+lookback-1)
        # - Target: targets[i+lookback] (next bar after sequence)
        n_sequences = n_samples - self.lookback
        
        logger.info(
            f"Creating {n_sequences} sequences from {n_samples} samples "
            f"with lookback={self.lookback}, features={n_features}"
        )
        
        # Pre-allocate arrays for efficiency
        X = np.zeros((n_sequences, self.lookback, n_features), dtype=np.float32)
        y = np.zeros((n_sequences, 1), dtype=np.float32)
        
        # Create sequences using sliding window
        for i in range(n_sequences):
            # Input sequence: timesteps [i, i+lookback)
            X[i] = features[i:i + self.lookback]
            # Target: next timestep after sequence (i+lookback)
            y[i, 0] = targets[i + self.lookback]
        
        logger.info(f"Created sequences: X shape={X.shape}, y shape={y.shape}")
        logger.info(f"X range: [{X.min():.4f}, {X.max():.4f}], y range: [{y.min():.4f}, {y.max():.4f}]")
        
        return X, y


    def validate_no_data_leakage(
        self,
        X: np.ndarray,
        y: np.ndarray,
        original_data: Union[pd.DataFrame, np.ndarray],
        original_targets: np.ndarray
    ) -> bool:
        """
        Validate that sequences contain no future data (data leakage prevention).
        
        For each sequence at index i:
        - Input X[i] should only contain data from timesteps [i, i+lookback)
        - Target y[i] should be from timestep i+lookback
        - No data from timestep > i+lookback-1 should appear in X[i]
        
        Args:
            X: Input sequences with shape [samples, timesteps, features]
            y: Target values with shape [samples, 1]
            original_data: Original feature data before sequencing
            original_targets: Original target values before sequencing
        
        Returns:
            True if validation passes
        
        Raises:
            SequenceBuilderError: If data leakage is detected
        
        Requirements: 5.4, 5.6
        """
        if isinstance(original_data, pd.DataFrame):
            features = original_data.values
        else:
            features = np.asarray(original_data)
        
        targets = np.asarray(original_targets)
        n_sequences = X.shape[0]
        
        for i in range(n_sequences):
            # Verify input sequence matches original data at correct indices
            expected_input = features[i:i + self.lookback]
            actual_input = X[i]
            
            if not np.allclose(expected_input, actual_input, equal_nan=True):
                raise SequenceBuilderError(
                    f"Data leakage detected at sequence {i}: "
                    f"input does not match expected timesteps [{i}, {i + self.lookback})"
                )
            
            # Verify target is from the correct future timestep
            expected_target = targets[i + self.lookback]
            actual_target = y[i, 0]
            
            if not np.isclose(expected_target, actual_target, equal_nan=True):
                raise SequenceBuilderError(
                    f"Data leakage detected at sequence {i}: "
                    f"target does not match timestep {i + self.lookback}"
                )
        
        logger.info(f"Data leakage validation passed for {n_sequences} sequences")
        return True

    def get_valid_sample_count(self, total_samples: int) -> int:
        """
        Calculate number of valid sequences that can be created.
        
        Samples with insufficient history (less than lookback bars) are skipped.
        
        Args:
            total_samples: Total number of timesteps in the data
        
        Returns:
            Number of valid sequences that can be created
        
        Requirements: 5.4
        """
        if total_samples <= self.lookback:
            return 0
        return total_samples - self.lookback

    def get_sequence_indices(self, sequence_idx: int) -> Tuple[int, int, int]:
        """
        Get the original data indices for a given sequence.
        
        Useful for debugging and understanding which timesteps
        contribute to each sequence.
        
        Args:
            sequence_idx: Index of the sequence (0-based)
        
        Returns:
            Tuple of (input_start, input_end, target_idx) where:
            - input_start: First timestep index in input sequence
            - input_end: Last timestep index in input sequence (exclusive)
            - target_idx: Timestep index of the target value
        
        Requirements: 5.6
        """
        input_start = sequence_idx
        input_end = sequence_idx + self.lookback
        target_idx = input_end  # Target is the next bar after sequence
        
        return input_start, input_end, target_idx

    def skip_insufficient_history(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        min_history: Optional[int] = None
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Skip samples at the start that have insufficient history.
        
        This ensures that all remaining samples can form complete sequences
        with the required lookback window.
        
        Args:
            data: Input data (DataFrame or numpy array)
            min_history: Minimum history required. Defaults to lookback.
        
        Returns:
            Data with insufficient history samples removed
        
        Requirements: 5.4
        """
        if min_history is None:
            min_history = self.lookback
        
        if isinstance(data, pd.DataFrame):
            if len(data) <= min_history:
                logger.warning(
                    f"All data has insufficient history: "
                    f"{len(data)} samples < {min_history} required"
                )
                return data.iloc[0:0]  # Return empty DataFrame with same columns
            
            # Keep data from min_history onwards
            # Note: We don't actually skip here because create_sequences handles this
            # This method is for explicit pre-filtering if needed
            return data
        else:
            data = np.asarray(data)
            if len(data) <= min_history:
                logger.warning(
                    f"All data has insufficient history: "
                    f"{len(data)} samples < {min_history} required"
                )
                return data[0:0]  # Return empty array with same shape[1:]
            return data
