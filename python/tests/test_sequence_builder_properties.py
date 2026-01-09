"""
Property-Based Tests for SequenceBuilder Module

Feature: lstm-cnn-xauusd-trading, Property 7: Sequence Structure Correctness
Validates: Requirements 5.1, 5.2, 5.5, 5.6

Tests verify:
- Shape SHALL be [samples, L, F] for lookback L and F features
- Each sequence SHALL contain exactly L consecutive timesteps
- Target SHALL correspond to the bar immediately after the sequence
- No future data SHALL appear in input sequences (no data leakage)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings, assume, HealthCheck

from sequence_builder import SequenceBuilder, SequenceBuilderError


# Strategies for generating test data
@st.composite
def valid_time_series_data(draw, min_rows=35, max_rows=100, min_cols=2, max_cols=8):
    """Generate valid time series data for sequence building."""
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    n_features = draw(st.integers(min_value=min_cols, max_value=max_cols))
    
    # Generate feature data
    data = draw(st.lists(
        st.lists(
            st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
            min_size=n_features,
            max_size=n_features
        ),
        min_size=n_rows,
        max_size=n_rows
    ))
    
    return np.array(data, dtype=np.float64)


@st.composite
def lookback_and_data(draw, min_lookback=5, max_lookback=30):
    """Generate a lookback value and compatible data."""
    lookback = draw(st.integers(min_value=min_lookback, max_value=max_lookback))
    # Ensure we have enough data for at least a few sequences
    min_rows = lookback + 5
    max_rows = lookback + 50
    
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    n_features = draw(st.integers(min_value=2, max_value=6))
    
    data = draw(st.lists(
        st.lists(
            st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
            min_size=n_features,
            max_size=n_features
        ),
        min_size=n_rows,
        max_size=n_rows
    ))
    
    return lookback, np.array(data, dtype=np.float64)


class TestSequenceBuilderProperties:
    """
    Property-based tests for SequenceBuilder.
    
    Feature: lstm-cnn-xauusd-trading, Property 7: Sequence Structure Correctness
    Validates: Requirements 5.1, 5.2, 5.5, 5.6
    """


    @given(lookback_data=lookback_and_data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_output_shape_correctness(self, lookback_data):
        """
        Property: Shape SHALL be [samples, L, F] for lookback L and F features.
        
        For any sequence created with lookback L and F features:
        - X shape must be [n_samples, L, F]
        - y shape must be [n_samples, 1]
        - n_samples = total_rows - L
        
        Feature: lstm-cnn-xauusd-trading, Property 7: Sequence Structure Correctness
        Validates: Requirements 5.1, 5.2
        """
        lookback, data = lookback_data
        n_rows, n_cols = data.shape
        n_features = n_cols - 1  # Last column is target
        
        builder = SequenceBuilder(lookback=lookback)
        X, y = builder.create_sequences(data)
        
        expected_samples = n_rows - lookback
        
        # Verify X shape
        assert X.shape == (expected_samples, lookback, n_features), \
            f"X shape {X.shape} != expected ({expected_samples}, {lookback}, {n_features})"
        
        # Verify y shape
        assert y.shape == (expected_samples, 1), \
            f"y shape {y.shape} != expected ({expected_samples}, 1)"

    @given(lookback_data=lookback_and_data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_sequence_contains_consecutive_timesteps(self, lookback_data):
        """
        Property: Each sequence SHALL contain exactly L consecutive timesteps.
        
        For any sequence X[i], it must contain data from timesteps
        [i, i+1, ..., i+L-1] in that exact order.
        
        Feature: lstm-cnn-xauusd-trading, Property 7: Sequence Structure Correctness
        Validates: Requirements 5.2
        """
        lookback, data = lookback_data
        features = data[:, :-1]  # All columns except last (target)
        
        builder = SequenceBuilder(lookback=lookback)
        X, y = builder.create_sequences(data)
        
        n_sequences = X.shape[0]
        
        # Check each sequence contains correct consecutive timesteps
        for i in range(n_sequences):
            expected_sequence = features[i:i + lookback]
            actual_sequence = X[i]
            
            assert np.allclose(expected_sequence, actual_sequence), \
                f"Sequence {i} does not contain consecutive timesteps [{i}, {i + lookback})"

    @given(lookback_data=lookback_and_data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_target_is_next_bar(self, lookback_data):
        """
        Property: Target SHALL correspond to the bar immediately after the sequence.
        
        For any sequence X[i] ending at timestep i+L-1, the target y[i]
        must be the value at timestep i+L (next bar prediction).
        
        Feature: lstm-cnn-xauusd-trading, Property 7: Sequence Structure Correctness
        Validates: Requirements 5.5
        """
        lookback, data = lookback_data
        targets = data[:, -1]  # Last column is target
        
        builder = SequenceBuilder(lookback=lookback)
        X, y = builder.create_sequences(data)
        
        n_sequences = X.shape[0]
        
        # Check each target corresponds to the next bar
        for i in range(n_sequences):
            expected_target = targets[i + lookback]
            actual_target = y[i, 0]
            
            assert np.isclose(expected_target, actual_target), \
                f"Target at sequence {i}: expected {expected_target}, got {actual_target}"

    @given(lookback_data=lookback_and_data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_no_future_data_in_sequences(self, lookback_data):
        """
        Property: No future data SHALL appear in input sequences (no data leakage).
        
        For any sequence X[i], no data from timestep >= i+L should appear.
        The sequence should only contain data from timesteps [i, i+L).
        
        Feature: lstm-cnn-xauusd-trading, Property 7: Sequence Structure Correctness
        Validates: Requirements 5.6
        """
        lookback, data = lookback_data
        features = data[:, :-1]
        targets = data[:, -1]
        
        builder = SequenceBuilder(lookback=lookback)
        X, y = builder.create_sequences(data)
        
        n_sequences = X.shape[0]
        
        # For each sequence, verify no future data is present
        for i in range(n_sequences):
            sequence_end_idx = i + lookback - 1  # Last valid index in sequence
            
            # The sequence should only contain data up to sequence_end_idx
            # Check that the last row of the sequence matches the data at sequence_end_idx
            last_row_in_sequence = X[i, -1, :]
            expected_last_row = features[sequence_end_idx]
            
            assert np.allclose(last_row_in_sequence, expected_last_row), \
                f"Sequence {i} contains data beyond timestep {sequence_end_idx}"
            
            # Verify target is strictly after the sequence
            target_idx = i + lookback
            assert target_idx > sequence_end_idx, \
                f"Target index {target_idx} is not after sequence end {sequence_end_idx}"

    @given(lookback_data=lookback_and_data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_insufficient_history_samples_skipped(self, lookback_data):
        """
        Property: Samples with insufficient history SHALL be skipped.
        
        The first L-1 timesteps cannot form complete sequences and must
        be excluded from the output.
        
        Feature: lstm-cnn-xauusd-trading, Property 7: Sequence Structure Correctness
        Validates: Requirements 5.4, 5.6
        """
        lookback, data = lookback_data
        n_rows = data.shape[0]
        
        builder = SequenceBuilder(lookback=lookback)
        X, y = builder.create_sequences(data)
        
        # Number of sequences should be total_rows - lookback
        # This means the first lookback rows are "used up" to form the first sequence
        expected_sequences = n_rows - lookback
        actual_sequences = X.shape[0]
        
        assert actual_sequences == expected_sequences, \
            f"Expected {expected_sequences} sequences, got {actual_sequences}"
        
        # Verify the valid sample count method
        valid_count = builder.get_valid_sample_count(n_rows)
        assert valid_count == expected_sequences, \
            f"get_valid_sample_count returned {valid_count}, expected {expected_sequences}"

    @given(st.integers(min_value=1, max_value=100))
    @settings(max_examples=100)
    def test_lookback_initialization(self, lookback):
        """
        Property: SequenceBuilder SHALL accept any positive integer lookback.
        
        Feature: lstm-cnn-xauusd-trading, Property 7: Sequence Structure Correctness
        Validates: Requirements 5.1
        """
        builder = SequenceBuilder(lookback=lookback)
        assert builder.lookback == lookback

    @given(st.integers(min_value=-100, max_value=0))
    @settings(max_examples=100)
    def test_invalid_lookback_rejected(self, lookback):
        """
        Property: SequenceBuilder SHALL reject non-positive lookback values.
        
        Feature: lstm-cnn-xauusd-trading, Property 7: Sequence Structure Correctness
        Validates: Requirements 5.1
        """
        with pytest.raises(ValueError):
            SequenceBuilder(lookback=lookback)

    @given(lookback_data=lookback_and_data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_sequence_indices_correctness(self, lookback_data):
        """
        Property: get_sequence_indices SHALL return correct index boundaries.
        
        Feature: lstm-cnn-xauusd-trading, Property 7: Sequence Structure Correctness
        Validates: Requirements 5.6
        """
        lookback, data = lookback_data
        
        builder = SequenceBuilder(lookback=lookback)
        X, y = builder.create_sequences(data)
        
        n_sequences = X.shape[0]
        
        for i in range(n_sequences):
            input_start, input_end, target_idx = builder.get_sequence_indices(i)
            
            # Verify index calculations
            assert input_start == i, f"input_start should be {i}, got {input_start}"
            assert input_end == i + lookback, f"input_end should be {i + lookback}, got {input_end}"
            assert target_idx == i + lookback, f"target_idx should be {i + lookback}, got {target_idx}"
            
            # Verify input_end - input_start == lookback
            assert input_end - input_start == lookback, \
                f"Sequence length should be {lookback}, got {input_end - input_start}"
