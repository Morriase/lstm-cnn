"""
Property-Based Tests for Train/Test Split

Feature: lstm-cnn-xauusd-trading, Property 10: Train/Test Split Date Boundary
Validates: Requirements 9.2

Tests verify:
- For any 80/20 split, all training samples SHALL have timestamps before all test samples
- No overlap between training and test sets
- Split ratio is correctly applied
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings, assume, HealthCheck

from train import TrainingPipeline


# Strategies for generating test data
@st.composite
def time_series_dataframe(draw, min_rows=50, max_rows=200, min_cols=3, max_cols=8):
    """
    Generate a DataFrame representing time series data in chronological order.
    Includes a timestamp-like index to verify date ordering.
    """
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    n_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))
    
    # Create feature columns
    col_names = [f"feature_{i}" for i in range(n_cols - 1)]
    col_names.append("target")  # Last column is target
    
    data = {}
    for col in col_names:
        values = draw(st.lists(
            st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
            min_size=n_rows,
            max_size=n_rows
        ))
        data[col] = values
    
    # Add a timestamp index to track chronological order
    df = pd.DataFrame(data)
    df['_original_index'] = np.arange(n_rows)
    
    return df


@st.composite
def train_split_ratio(draw):
    """Generate valid train split ratios between 0.5 and 0.95."""
    return draw(st.floats(min_value=0.5, max_value=0.95))


class TestTrainTestSplitProperties:
    """
    Property-based tests for train/test split.
    
    Feature: lstm-cnn-xauusd-trading, Property 10: Train/Test Split Date Boundary
    Validates: Requirements 9.2
    """
    
    @given(df=time_series_dataframe(), ratio=train_split_ratio())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_train_samples_before_test_samples(self, df, ratio):
        """
        Property: For any split, all training samples SHALL have indices before all test samples.
        
        Feature: lstm-cnn-xauusd-trading, Property 10: Train/Test Split Date Boundary
        Validates: Requirements 9.2
        """
        pipeline = TrainingPipeline({'train_split': ratio})
        pipeline.clean_data = df.drop(columns=['_original_index'])
        
        train_df, test_df, train_indices, test_indices = pipeline.split_train_test_by_date()
        
        # All training indices should be less than all test indices
        if len(train_indices) > 0 and len(test_indices) > 0:
            max_train_idx = train_indices.max()
            min_test_idx = test_indices.min()
            
            assert max_train_idx < min_test_idx, \
                f"Training indices ({max_train_idx}) should be before test indices ({min_test_idx})"
    
    @given(df=time_series_dataframe(), ratio=train_split_ratio())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_no_overlap_between_train_and_test(self, df, ratio):
        """
        Property: For any split, there SHALL be no overlap between training and test sets.
        
        Feature: lstm-cnn-xauusd-trading, Property 10: Train/Test Split Date Boundary
        Validates: Requirements 9.2
        """
        pipeline = TrainingPipeline({'train_split': ratio})
        pipeline.clean_data = df.drop(columns=['_original_index'])
        
        train_df, test_df, train_indices, test_indices = pipeline.split_train_test_by_date()
        
        # Check no overlap in indices
        train_set = set(train_indices)
        test_set = set(test_indices)
        overlap = train_set.intersection(test_set)
        
        assert len(overlap) == 0, f"Found overlapping indices: {overlap}"
    
    @given(df=time_series_dataframe(), ratio=train_split_ratio())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_split_ratio_correctly_applied(self, df, ratio):
        """
        Property: For any split ratio, the actual split SHALL match the requested ratio.
        
        Feature: lstm-cnn-xauusd-trading, Property 10: Train/Test Split Date Boundary
        Validates: Requirements 9.2
        """
        pipeline = TrainingPipeline({'train_split': ratio})
        pipeline.clean_data = df.drop(columns=['_original_index'])
        
        train_df, test_df, train_indices, test_indices = pipeline.split_train_test_by_date()
        
        n_total = len(df)
        n_train = len(train_df)
        
        # Expected split index
        expected_train_size = int(n_total * ratio)
        
        # Allow for integer rounding
        assert n_train == expected_train_size, \
            f"Train size {n_train} doesn't match expected {expected_train_size}"
    
    @given(df=time_series_dataframe(), ratio=train_split_ratio())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_all_data_accounted_for(self, df, ratio):
        """
        Property: For any split, train + test SHALL equal total samples.
        
        Feature: lstm-cnn-xauusd-trading, Property 10: Train/Test Split Date Boundary
        Validates: Requirements 9.2
        """
        pipeline = TrainingPipeline({'train_split': ratio})
        pipeline.clean_data = df.drop(columns=['_original_index'])
        
        train_df, test_df, train_indices, test_indices = pipeline.split_train_test_by_date()
        
        n_total = len(df)
        n_train = len(train_df)
        n_test = len(test_df)
        
        assert n_train + n_test == n_total, \
            f"Train ({n_train}) + Test ({n_test}) != Total ({n_total})"
    
    @given(df=time_series_dataframe())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_default_80_20_split(self, df):
        """
        Property: Default split SHALL be 80/20.
        
        Feature: lstm-cnn-xauusd-trading, Property 10: Train/Test Split Date Boundary
        Validates: Requirements 9.2
        """
        pipeline = TrainingPipeline()  # Uses default config
        pipeline.clean_data = df.drop(columns=['_original_index'])
        
        train_df, test_df, train_indices, test_indices = pipeline.split_train_test_by_date()
        
        n_total = len(df)
        n_train = len(train_df)
        
        expected_train_size = int(n_total * 0.8)
        
        assert n_train == expected_train_size, \
            f"Default split should be 80%, got {n_train}/{n_total}"
    
    @given(df=time_series_dataframe(), ratio=train_split_ratio())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_chronological_order_preserved(self, df, ratio):
        """
        Property: For any split, chronological order SHALL be preserved within each set.
        
        Feature: lstm-cnn-xauusd-trading, Property 10: Train/Test Split Date Boundary
        Validates: Requirements 9.2
        """
        pipeline = TrainingPipeline({'train_split': ratio})
        
        # Keep original index for verification
        df_with_idx = df.copy()
        pipeline.clean_data = df_with_idx.drop(columns=['_original_index'])
        
        train_df, test_df, train_indices, test_indices = pipeline.split_train_test_by_date()
        
        # Training indices should be monotonically increasing
        if len(train_indices) > 1:
            assert np.all(np.diff(train_indices) > 0), \
                "Training indices are not in chronological order"
        
        # Test indices should be monotonically increasing
        if len(test_indices) > 1:
            assert np.all(np.diff(test_indices) > 0), \
                "Test indices are not in chronological order"
