"""
Property-Based Tests for Metrics Module

Feature: lstm-cnn-xauusd-trading, Property 8: Metrics Computation Correctness
Validates: Requirements 10.1, 10.2, 10.3, 10.4

Tests verify:
- RMSE SHALL be non-negative and zero only for perfect predictions
- MAE SHALL be non-negative and zero only for perfect predictions
- MAPE SHALL be non-negative and zero only for perfect predictions
- R² SHALL be 1.0 for perfect predictions
- All metrics SHALL handle edge cases correctly
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume, HealthCheck

from metrics import (
    compute_rmse, compute_mae, compute_mape, compute_r2,
    compute_all_metrics, MetricsError
)


@st.composite
def matching_arrays(draw, min_size=5, max_size=100):
    """Generate two arrays of the same length with valid numeric values."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    y_true = draw(st.lists(
        st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
        min_size=size, max_size=size
    ))
    y_pred = draw(st.lists(
        st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
        min_size=size, max_size=size
    ))
    return np.array(y_true), np.array(y_pred)


@st.composite
def non_zero_arrays(draw, min_size=5, max_size=100):
    """Generate arrays where y_true has no zero values (for MAPE)."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    y_true = draw(st.lists(
        st.one_of(
            st.floats(min_value=0.1, max_value=1000, allow_nan=False, allow_infinity=False),
            st.floats(min_value=-1000, max_value=-0.1, allow_nan=False, allow_infinity=False)
        ),
        min_size=size, max_size=size
    ))
    y_pred = draw(st.lists(
        st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
        min_size=size, max_size=size
    ))
    return np.array(y_true), np.array(y_pred)


@st.composite
def varying_arrays(draw, min_size=5, max_size=100):
    """Generate arrays where y_true has non-zero variance (for R²)."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    base_values = draw(st.lists(
        st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
        min_size=size, max_size=size
    ))
    y_true = np.array(base_values)
    if np.std(y_true) == 0 and size > 1:
        y_true[0] = y_true[0] + 1.0
    y_pred = draw(st.lists(
        st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
        min_size=size, max_size=size
    ))
    return y_true, np.array(y_pred)



class TestRMSEProperties:
    """
    Property-based tests for RMSE computation.
    
    Feature: lstm-cnn-xauusd-trading, Property 8: Metrics Computation Correctness
    Validates: Requirements 10.1
    """
    
    @given(arrays=matching_arrays())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_rmse_non_negative(self, arrays):
        """Property: RMSE SHALL always be non-negative."""
        y_true, y_pred = arrays
        rmse = compute_rmse(y_true, y_pred)
        assert rmse >= 0, f"RMSE should be non-negative, got {rmse}"
    
    @given(st.lists(
        st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
        min_size=5, max_size=100
    ))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_rmse_zero_for_perfect_predictions(self, values):
        """Property: RMSE SHALL be zero when predictions equal actual values."""
        y = np.array(values)
        rmse = compute_rmse(y, y)
        assert np.isclose(rmse, 0.0), f"RMSE should be 0 for perfect predictions, got {rmse}"
    
    @given(arrays=matching_arrays())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_rmse_symmetry(self, arrays):
        """Property: RMSE SHALL be symmetric: RMSE(y, y_hat) == RMSE(y_hat, y)."""
        y_true, y_pred = arrays
        rmse_forward = compute_rmse(y_true, y_pred)
        rmse_reverse = compute_rmse(y_pred, y_true)
        assert np.isclose(rmse_forward, rmse_reverse), \
            f"RMSE should be symmetric: {rmse_forward} != {rmse_reverse}"
    
    @given(arrays=matching_arrays())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_rmse_greater_equal_mae(self, arrays):
        """Property: RMSE SHALL be greater than or equal to MAE."""
        y_true, y_pred = arrays
        rmse = compute_rmse(y_true, y_pred)
        mae = compute_mae(y_true, y_pred)
        assert rmse >= mae - 1e-10, f"RMSE ({rmse}) should be >= MAE ({mae})"



class TestMAEProperties:
    """
    Property-based tests for MAE computation.
    
    Feature: lstm-cnn-xauusd-trading, Property 8: Metrics Computation Correctness
    Validates: Requirements 10.2
    """
    
    @given(arrays=matching_arrays())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_mae_non_negative(self, arrays):
        """Property: MAE SHALL always be non-negative."""
        y_true, y_pred = arrays
        mae = compute_mae(y_true, y_pred)
        assert mae >= 0, f"MAE should be non-negative, got {mae}"
    
    @given(st.lists(
        st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
        min_size=5, max_size=100
    ))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_mae_zero_for_perfect_predictions(self, values):
        """Property: MAE SHALL be zero when predictions equal actual values."""
        y = np.array(values)
        mae = compute_mae(y, y)
        assert np.isclose(mae, 0.0), f"MAE should be 0 for perfect predictions, got {mae}"
    
    @given(arrays=matching_arrays())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_mae_symmetry(self, arrays):
        """Property: MAE SHALL be symmetric: MAE(y, y_hat) == MAE(y_hat, y)."""
        y_true, y_pred = arrays
        mae_forward = compute_mae(y_true, y_pred)
        mae_reverse = compute_mae(y_pred, y_true)
        assert np.isclose(mae_forward, mae_reverse), \
            f"MAE should be symmetric: {mae_forward} != {mae_reverse}"
    
    @given(
        st.lists(st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False), min_size=5, max_size=50),
        st.lists(st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False), min_size=5, max_size=50),
        st.lists(st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False), min_size=5, max_size=50)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_mae_triangle_inequality(self, a_list, b_list, c_list):
        """Property: MAE SHALL satisfy triangle inequality: MAE(a,c) <= MAE(a,b) + MAE(b,c)."""
        min_len = min(len(a_list), len(b_list), len(c_list))
        assume(min_len >= 5)
        a = np.array(a_list[:min_len])
        b = np.array(b_list[:min_len])
        c = np.array(c_list[:min_len])
        mae_ac = compute_mae(a, c)
        mae_ab = compute_mae(a, b)
        mae_bc = compute_mae(b, c)
        assert mae_ac <= mae_ab + mae_bc + 1e-10, \
            f"Triangle inequality violated: {mae_ac} > {mae_ab} + {mae_bc}"



class TestMAPEProperties:
    """
    Property-based tests for MAPE computation.
    
    Feature: lstm-cnn-xauusd-trading, Property 8: Metrics Computation Correctness
    Validates: Requirements 10.3
    """
    
    @given(arrays=non_zero_arrays())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_mape_non_negative(self, arrays):
        """Property: MAPE SHALL always be non-negative."""
        y_true, y_pred = arrays
        mape = compute_mape(y_true, y_pred)
        assert mape >= 0, f"MAPE should be non-negative, got {mape}"
    
    @given(st.lists(
        st.one_of(
            st.floats(min_value=0.1, max_value=1000, allow_nan=False, allow_infinity=False),
            st.floats(min_value=-1000, max_value=-0.1, allow_nan=False, allow_infinity=False)
        ),
        min_size=5, max_size=100
    ))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_mape_zero_for_perfect_predictions(self, values):
        """Property: MAPE SHALL be zero when predictions equal actual values."""
        y = np.array(values)
        mape = compute_mape(y, y)
        assert np.isclose(mape, 0.0), f"MAPE should be 0 for perfect predictions, got {mape}"
    
    @given(st.lists(
        st.floats(min_value=0.0, max_value=0.0, allow_nan=False, allow_infinity=False),
        min_size=5, max_size=20
    ))
    @settings(max_examples=50)
    def test_mape_raises_for_all_zeros(self, values):
        """Property: MAPE SHALL raise MetricsError when all y_true values are zero."""
        y_true = np.array(values)
        y_pred = np.ones_like(y_true)
        with pytest.raises(MetricsError):
            compute_mape(y_true, y_pred)



class TestR2Properties:
    """
    Property-based tests for R-squared computation.
    
    Feature: lstm-cnn-xauusd-trading, Property 8: Metrics Computation Correctness
    Validates: Requirements 10.4
    """
    
    @given(st.lists(
        st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
        min_size=5, max_size=100
    ))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_r2_one_for_perfect_predictions(self, values):
        """Property: R-squared SHALL be 1.0 when predictions equal actual values."""
        y = np.array(values)
        if np.std(y) == 0:
            return  # Skip constant arrays
        r2 = compute_r2(y, y)
        assert np.isclose(r2, 1.0), f"R2 should be 1.0 for perfect predictions, got {r2}"
    
    @given(arrays=varying_arrays())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_r2_zero_for_mean_predictor(self, arrays):
        """Property: R-squared SHALL be 0.0 when predicting the mean of y_true."""
        y_true, _ = arrays
        assume(np.std(y_true) > 0)
        y_pred_mean = np.full_like(y_true, np.mean(y_true))
        r2 = compute_r2(y_true, y_pred_mean)
        assert np.isclose(r2, 0.0, atol=1e-10), \
            f"R2 should be 0.0 for mean predictor, got {r2}"
    
    @given(st.lists(
        st.floats(min_value=1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=5, max_size=20
    ))
    @settings(max_examples=50)
    def test_r2_handles_zero_variance(self, values):
        """Property: R-squared SHALL handle zero variance in y_true appropriately."""
        y_true = np.array(values)
        y_pred = np.array(values) + 1.0  # Different from y_true
        with pytest.raises(MetricsError):
            compute_r2(y_true, y_pred)
    
    @given(st.lists(
        st.floats(min_value=1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=5, max_size=20
    ))
    @settings(max_examples=50)
    def test_r2_perfect_with_zero_variance(self, values):
        """Property: R-squared SHALL be 1.0 for perfect predictions even with zero variance."""
        y = np.array(values)
        r2 = compute_r2(y, y)
        assert r2 == 1.0, f"R2 should be 1.0 for perfect predictions, got {r2}"



class TestMetricsEdgeCases:
    """
    Property-based tests for edge cases and error handling.
    
    Feature: lstm-cnn-xauusd-trading, Property 8: Metrics Computation Correctness
    Validates: Requirements 10.1, 10.2, 10.3, 10.4
    """
    
    @given(
        st.lists(st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False), min_size=5, max_size=50),
        st.lists(st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False), min_size=5, max_size=50)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_length_mismatch_raises_error(self, y_true_list, y_pred_list):
        """Property: All metrics SHALL raise MetricsError for mismatched array lengths."""
        assume(len(y_true_list) != len(y_pred_list))
        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)
        with pytest.raises(MetricsError):
            compute_rmse(y_true, y_pred)
        with pytest.raises(MetricsError):
            compute_mae(y_true, y_pred)
        with pytest.raises(MetricsError):
            compute_mape(y_true, y_pred)
        with pytest.raises(MetricsError):
            compute_r2(y_true, y_pred)
    
    def test_empty_arrays_raise_error(self):
        """Property: All metrics SHALL raise MetricsError for empty arrays."""
        y_true = np.array([])
        y_pred = np.array([])
        with pytest.raises(MetricsError):
            compute_rmse(y_true, y_pred)
        with pytest.raises(MetricsError):
            compute_mae(y_true, y_pred)
        with pytest.raises(MetricsError):
            compute_mape(y_true, y_pred)
        with pytest.raises(MetricsError):
            compute_r2(y_true, y_pred)
    
    @given(arrays=matching_arrays())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_compute_all_metrics_consistency(self, arrays):
        """Property: compute_all_metrics SHALL return consistent values with individual functions."""
        y_true, y_pred = arrays
        assume(np.std(y_true) > 0)  # Ensure R2 can be computed
        all_metrics = compute_all_metrics(y_true, y_pred)
        assert np.isclose(all_metrics['rmse'], compute_rmse(y_true, y_pred))
        assert np.isclose(all_metrics['mae'], compute_mae(y_true, y_pred))
        assert np.isclose(all_metrics['r2'], compute_r2(y_true, y_pred))
    
    @given(
        st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_rmse_scales_with_error_magnitude(self, base, scale):
        """Property: RMSE SHALL scale linearly with error magnitude."""
        y_true = np.array([base, base + 1, base + 2, base + 3, base + 4])
        error = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        y_pred_1 = y_true + error
        y_pred_scaled = y_true + error * scale
        rmse_1 = compute_rmse(y_true, y_pred_1)
        rmse_scaled = compute_rmse(y_true, y_pred_scaled)
        assert np.isclose(rmse_scaled, rmse_1 * scale, rtol=1e-10), \
            f"RMSE should scale linearly: {rmse_scaled} != {rmse_1} * {scale}"
