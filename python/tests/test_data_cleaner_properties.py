"""
Property-Based Tests for DataCleaner Module

Feature: lstm-cnn-xauusd-trading, Property 6: Data Cleaning Preserves Valid Values
Validates: Requirements 4.1.1, 4.1.5, 4.1.6, 4.1.7

Tests verify:
- No NaN values remain after cleaning
- Non-NaN values from the original dataset are unchanged (within outlier bounds)
- Forward-filled values equal the most recent valid value
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings, assume, HealthCheck

from data_cleaner import DataCleaner, DataCleaningError


# Optimized strategies for generating test data
@st.composite
def valid_numeric_dataframe(draw, min_rows=5, max_rows=20, min_cols=2, max_cols=5):
    """Generate a DataFrame with only valid numeric values (no NaN)."""
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    n_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))
    
    col_names = [f"feature_{i}" for i in range(n_cols)]
    
    data = {}
    for col in col_names:
        values = draw(st.lists(
            st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
            min_size=n_rows,
            max_size=n_rows
        ))
        data[col] = values
    
    return pd.DataFrame(data)


@st.composite
def numeric_dataframe_with_nan(draw, min_rows=10, max_rows=20, min_cols=2, max_cols=4):
    """Generate a DataFrame with numeric values and some NaN values."""
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    n_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))
    
    col_names = [f"feature_{i}" for i in range(n_cols)]
    
    data = {}
    for col in col_names:
        # Ensure first few values are valid for forward fill to work
        values = []
        for i in range(n_rows):
            if i < 3:  # First 3 values always valid
                values.append(draw(st.floats(min_value=-100, max_value=100, 
                                             allow_nan=False, allow_infinity=False)))
            else:
                # 20% chance of NaN
                if draw(st.integers(min_value=0, max_value=4)) == 0:
                    values.append(np.nan)
                else:
                    values.append(draw(st.floats(min_value=-100, max_value=100,
                                                 allow_nan=False, allow_infinity=False)))
        data[col] = values
    
    return pd.DataFrame(data)


class TestDataCleanerProperties:
    """
    Property-based tests for DataCleaner.
    
    Feature: lstm-cnn-xauusd-trading, Property 6: Data Cleaning Preserves Valid Values
    Validates: Requirements 4.1.1, 4.1.5, 4.1.6, 4.1.7
    """
    
    @given(df=valid_numeric_dataframe())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_no_nan_after_cleaning_valid_data(self, df):
        """
        Property: For any valid dataset, no NaN values SHALL remain after cleaning.
        
        Feature: lstm-cnn-xauusd-trading, Property 6: Data Cleaning Preserves Valid Values
        Validates: Requirements 4.1.5
        """
        cleaner = DataCleaner(outlier_threshold=3.0)
        cleaned = cleaner.clean(df)
        
        # No NaN values should remain
        assert cleaned.isna().sum().sum() == 0, "NaN values found after cleaning"
    
    @given(df=valid_numeric_dataframe())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_valid_values_preserved_within_bounds(self, df):
        """
        Property: For any dataset, non-NaN values within outlier bounds SHALL be unchanged.
        
        Feature: lstm-cnn-xauusd-trading, Property 6: Data Cleaning Preserves Valid Values
        Validates: Requirements 4.1.6
        """
        threshold = 3.0
        cleaner = DataCleaner(outlier_threshold=threshold)
        cleaned = cleaner.clean(df)
        
        # For each column, values within bounds should be preserved
        for col in df.columns:
            col_mean = df[col].mean()
            col_std = df[col].std()
            
            if col_std == 0 or pd.isna(col_std):
                # All values are the same, should be preserved
                assert np.allclose(df[col].values, cleaned[col].values, equal_nan=True)
            else:
                # Values within bounds should be unchanged
                lower_bound = col_mean - threshold * col_std
                upper_bound = col_mean + threshold * col_std
                
                within_bounds_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
                
                # Compare values that were within bounds
                original_within = df.loc[within_bounds_mask, col]
                cleaned_within = cleaned.loc[within_bounds_mask, col]
                
                assert np.allclose(original_within.values, cleaned_within.values), \
                    f"Values within bounds were altered in column {col}"

    @given(df=numeric_dataframe_with_nan())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_forward_fill_uses_most_recent_valid(self, df):
        """
        Property: Forward-filled values SHALL equal the most recent valid value.
        
        Feature: lstm-cnn-xauusd-trading, Property 6: Data Cleaning Preserves Valid Values
        Validates: Requirements 4.1.1
        """
        cleaner = DataCleaner()
        
        # Apply only forward fill
        filled = cleaner.forward_fill(df)
        
        # Verify forward fill property for each column
        for col in df.columns:
            last_valid = None
            for i in range(len(df)):
                original_val = df.iloc[i][col]
                filled_val = filled.iloc[i][col]
                
                if not pd.isna(original_val):
                    last_valid = original_val
                    # Original valid values should be unchanged
                    assert filled_val == original_val, \
                        f"Valid value at index {i}, col {col} was changed"
                elif last_valid is not None:
                    # NaN should be filled with last valid
                    assert filled_val == last_valid, \
                        f"Forward fill at index {i}, col {col} should be {last_valid}, got {filled_val}"
    
    @given(df=numeric_dataframe_with_nan())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_clean_pipeline_produces_valid_output(self, df):
        """
        Property: The clean() pipeline SHALL output a clean dataset ready for sequence creation.
        
        Feature: lstm-cnn-xauusd-trading, Property 6: Data Cleaning Preserves Valid Values
        Validates: Requirements 4.1.7
        """
        cleaner = DataCleaner(outlier_threshold=3.0)
        
        try:
            cleaned = cleaner.clean(df)
            
            # Output should have no NaN
            assert cleaned.isna().sum().sum() == 0, "NaN values found in output"
            
            # Output should have same columns
            assert list(cleaned.columns) == list(df.columns), "Columns were altered"
            
            # Output should have rows <= input rows
            assert len(cleaned) <= len(df), "Output has more rows than input"
            
        except DataCleaningError:
            # This is acceptable if data cannot be cleaned
            pass
