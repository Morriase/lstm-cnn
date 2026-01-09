"""
Property-Based Tests for LSTMCNNModel Module

Feature: lstm-cnn-xauusd-trading, Property 8: ONNX Shape Consistency
Validates: Requirements 7.2, 7.3, 7.4

Tests verify:
- Input shape SHALL match [batch, lookback_window, num_features]
- Output shape SHALL match the prediction target dimensions
- The model SHALL load successfully via ONNX runtime
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
import tempfile
import shutil
from hypothesis import given, strategies as st, settings, HealthCheck, assume

# Check TensorFlow availability
tf_available = True
skip_reason = None
try:
    import tensorflow as tf
    # Suppress TF warnings for cleaner test output
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
except ImportError as e:
    tf_available = False
    skip_reason = f"TensorFlow not installed: {e}"

# Check ONNX runtime availability
onnx_available = True
try:
    import onnx
    import onnxruntime as ort
except (ImportError, AttributeError) as e:
    onnx_available = False
    if skip_reason is None:
        skip_reason = f"ONNX dependencies not available: {e}"

# Check PyTorch availability (used for ONNX export)
torch_available = True
torch_error = None
try:
    import torch
except ImportError as e:
    torch_available = False
    torch_error = str(e)

if tf_available:
    from lstm_cnn_model import LSTMCNNModel, ONNXExportError, ModelBuildError


# Strategies for generating test configurations
@st.composite
def valid_model_config(draw):
    """Generate valid model configurations for testing."""
    lookback = draw(st.integers(min_value=5, max_value=30))
    num_features = draw(st.integers(min_value=3, max_value=10))
    lstm_units = draw(st.integers(min_value=8, max_value=32))
    cnn_filters = draw(st.integers(min_value=8, max_value=32))
    cnn_kernel_size = draw(st.integers(min_value=2, max_value=min(5, lookback)))
    
    return {
        'lookback': lookback,
        'num_features': num_features,
        'lstm_units': lstm_units,
        'lstm_dropout': 0.1,
        'cnn_filters': cnn_filters,
        'cnn_kernel_size': cnn_kernel_size,
        'dense_units': 16,
        'learning_rate': 0.001,
        'epochs': 1,
        'batch_size': 8,
        'early_stopping_patience': 5,
    }


@pytest.mark.skipif(not tf_available, reason="TensorFlow not installed")
class TestLSTMCNNModelBuildProperties:
    """
    Property-based tests for model building functionality.
    
    These tests verify the model builds correctly with various configurations.
    """

    @given(config=valid_model_config())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_model_builds_with_valid_config(self, config):
        """
        Property: Model SHALL build successfully with any valid configuration.
        
        For any valid configuration, the model should build without errors
        and have the expected input/output shapes.
        """
        model = LSTMCNNModel(config)
        keras_model = model.build_model()
        
        # Verify model is built
        assert model._is_built
        assert keras_model is not None
        
        # Verify input shape
        input_shape = keras_model.input_shape
        assert input_shape[1] == config['lookback']
        assert input_shape[2] == config['num_features']
        
        # Verify output shape
        output_shape = keras_model.output_shape
        assert output_shape[1] == 1  # Single prediction

    @given(config=valid_model_config())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_model_prediction_shape(self, config):
        """
        Property: Model predictions SHALL have shape [batch, 1].
        
        For any valid input, the model should produce predictions
        with the correct output shape.
        """
        model = LSTMCNNModel(config)
        model.build_model()
        
        # Create sample input
        batch_size = 5
        sample_input = np.random.randn(
            batch_size, 
            config['lookback'], 
            config['num_features']
        ).astype(np.float32)
        
        # Get prediction
        prediction = model.model.predict(sample_input, verbose=0)
        
        # Verify output shape
        assert prediction.shape == (batch_size, 1)


@pytest.mark.skipif(
    not tf_available or not onnx_available or not torch_available, 
    reason=skip_reason or torch_error or "PyTorch not available for ONNX export"
)
class TestLSTMCNNModelONNXProperties:
    """
    Property-based tests for ONNX export functionality.
    
    Feature: lstm-cnn-xauusd-trading, Property 8: ONNX Shape Consistency
    Validates: Requirements 7.2, 7.3, 7.4
    """

    @given(config=valid_model_config())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_onnx_input_shape_matches_config(self, config):
        """
        Property: Input shape SHALL match [batch, lookback_window, num_features].
        
        For any valid model configuration, the exported ONNX model's input
        shape must match the configured lookback and num_features dimensions.
        
        Feature: lstm-cnn-xauusd-trading, Property 8: ONNX Shape Consistency
        Validates: Requirements 7.2
        """
        temp_dir = tempfile.mkdtemp()
        try:
            # Build model
            model = LSTMCNNModel(config)
            model.build_model()
            
            # Export to ONNX
            onnx_path = os.path.join(temp_dir, 'test_model.onnx')
            model.export_onnx(onnx_path, validate=False)
            
            # Load and check input shape
            onnx_model = onnx.load(onnx_path)
            input_info = onnx_model.graph.input[0]
            input_shape = [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]
            
            # Verify shape dimensions (batch is dynamic, so check lookback and features)
            assert len(input_shape) == 3, f"Expected 3D input, got {len(input_shape)}D"
            assert input_shape[1] == config['lookback'], \
                f"Lookback {input_shape[1]} != config {config['lookback']}"
            assert input_shape[2] == config['num_features'], \
                f"Features {input_shape[2]} != config {config['num_features']}"
                
        finally:
            shutil.rmtree(temp_dir)

    @given(config=valid_model_config())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_onnx_output_shape_is_single_prediction(self, config):
        """
        Property: Output shape SHALL match the prediction target dimensions.
        
        For any valid model configuration, the exported ONNX model's output
        shape must be [batch, 1] for single value prediction.
        
        Feature: lstm-cnn-xauusd-trading, Property 8: ONNX Shape Consistency
        Validates: Requirements 7.3
        """
        temp_dir = tempfile.mkdtemp()
        try:
            # Build model
            model = LSTMCNNModel(config)
            model.build_model()
            
            # Export to ONNX
            onnx_path = os.path.join(temp_dir, 'test_model.onnx')
            model.export_onnx(onnx_path, validate=False)
            
            # Load and check output shape
            onnx_model = onnx.load(onnx_path)
            output_info = onnx_model.graph.output[0]
            output_shape = [dim.dim_value for dim in output_info.type.tensor_type.shape.dim]
            
            # Output should be [batch, 1] - batch is dynamic (0), prediction is 1
            assert len(output_shape) == 2, f"Expected 2D output, got {len(output_shape)}D"
            assert output_shape[1] == 1, f"Output dim {output_shape[1]} != 1"
                
        finally:
            shutil.rmtree(temp_dir)

    @given(config=valid_model_config())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_onnx_loads_successfully(self, config):
        """
        Property: The model SHALL load successfully via ONNX runtime.
        
        For any valid model configuration, the exported ONNX model must
        load without errors using both onnx.load() and onnxruntime.
        
        Feature: lstm-cnn-xauusd-trading, Property 8: ONNX Shape Consistency
        Validates: Requirements 7.4
        """
        temp_dir = tempfile.mkdtemp()
        try:
            # Build model
            model = LSTMCNNModel(config)
            model.build_model()
            
            # Export to ONNX
            onnx_path = os.path.join(temp_dir, 'test_model.onnx')
            model.export_onnx(onnx_path, validate=False)
            
            # Test loading with onnx
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            # Test loading with onnxruntime
            session = ort.InferenceSession(onnx_path)
            assert session is not None
            
            # Verify we can get input/output info
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            assert len(inputs) == 1
            assert len(outputs) == 1
                
        finally:
            shutil.rmtree(temp_dir)

    @given(config=valid_model_config(), batch_size=st.integers(min_value=1, max_value=5))
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_onnx_inference_produces_correct_output_shape(self, config, batch_size):
        """
        Property: ONNX inference SHALL produce output matching input batch size.
        
        For any valid input batch, the ONNX model inference must produce
        output with shape [batch_size, 1].
        
        Feature: lstm-cnn-xauusd-trading, Property 8: ONNX Shape Consistency
        Validates: Requirements 7.2, 7.3, 7.4
        """
        temp_dir = tempfile.mkdtemp()
        try:
            # Build model
            model = LSTMCNNModel(config)
            model.build_model()
            
            # Export to ONNX
            onnx_path = os.path.join(temp_dir, 'test_model.onnx')
            model.export_onnx(onnx_path, validate=False)
            
            # Create inference session
            session = ort.InferenceSession(onnx_path)
            input_name = session.get_inputs()[0].name
            
            # Create sample input with specified batch size
            sample_input = np.random.randn(
                batch_size, 
                config['lookback'], 
                config['num_features']
            ).astype(np.float32)
            
            # Run inference
            result = session.run(None, {input_name: sample_input})
            
            # Verify output shape
            assert result[0].shape == (batch_size, 1), \
                f"Output shape {result[0].shape} != expected ({batch_size}, 1)"
                
        finally:
            shutil.rmtree(temp_dir)

    @given(config=valid_model_config())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_keras_and_onnx_produce_similar_outputs(self, config):
        """
        Property: Keras and ONNX models SHALL produce similar predictions.
        
        For any valid input, the Keras model and exported ONNX model
        should produce numerically similar outputs (within tolerance).
        
        Feature: lstm-cnn-xauusd-trading, Property 8: ONNX Shape Consistency
        Validates: Requirements 7.2, 7.3, 7.4
        """
        temp_dir = tempfile.mkdtemp()
        try:
            # Build model
            model = LSTMCNNModel(config)
            model.build_model()
            
            # Create sample input
            sample_input = np.random.randn(
                1, 
                config['lookback'], 
                config['num_features']
            ).astype(np.float32)
            
            # Get Keras prediction
            keras_pred = model.model.predict(sample_input, verbose=0)
            
            # Export to ONNX
            onnx_path = os.path.join(temp_dir, 'test_model.onnx')
            model.export_onnx(onnx_path, validate=False)
            
            # Get ONNX prediction
            session = ort.InferenceSession(onnx_path)
            input_name = session.get_inputs()[0].name
            onnx_pred = session.run(None, {input_name: sample_input})[0]
            
            # Compare predictions (allow small numerical differences)
            np.testing.assert_allclose(
                keras_pred, onnx_pred, 
                rtol=1e-4, atol=1e-5,
                err_msg="Keras and ONNX predictions differ significantly"
            )
                
        finally:
            shutil.rmtree(temp_dir)
