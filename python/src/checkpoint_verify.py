"""
Checkpoint Verification Script for LSTM-CNN XAUUSD Trading System

This script verifies the complete Python training pipeline works correctly
by generating synthetic XAUUSD-like data and running the full pipeline.

Used for Task 12: Checkpoint - Python training complete
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_xauusd_data(
    n_samples: int = 2000,
    start_price: float = 1800.0,
    volatility: float = 0.002,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic XAUUSD-like data for pipeline verification.
    
    Creates realistic price data with OHLCV and technical indicator columns
    matching the expected MQL5 Data_Exporter output format.
    
    Args:
        n_samples: Number of bars to generate
        start_price: Starting price for XAUUSD
        volatility: Daily volatility factor
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with columns matching MQL5 export format
    """
    np.random.seed(seed)
    
    logger.info(f"Generating {n_samples} synthetic XAUUSD bars...")
    
    # Generate price series using geometric Brownian motion
    returns = np.random.normal(0, volatility, n_samples)
    prices = start_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    data = {
        'Open': prices * (1 + np.random.uniform(-0.001, 0.001, n_samples)),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.002, n_samples))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.002, n_samples))),
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, n_samples).astype(float),
    }
    
    # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
    data['High'] = np.maximum(data['High'], np.maximum(data['Open'], data['Close']))
    data['Low'] = np.minimum(data['Low'], np.minimum(data['Open'], data['Close']))
    
    df = pd.DataFrame(data)
    
    # Add technical indicators (simplified versions)
    # SMA
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # EMA
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # Bollinger Bands
    bb_sma = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = bb_sma + 2 * bb_std
    df['BB_Middle'] = bb_sma
    df['BB_Lower'] = bb_sma - 2 * bb_std
    
    # RSI (simplified)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # OBV (simplified)
    obv = np.zeros(n_samples)
    for i in range(1, n_samples):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv[i] = obv[i-1] + df['Volume'].iloc[i]
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv[i] = obv[i-1] - df['Volume'].iloc[i]
        else:
            obv[i] = obv[i-1]
    df['OBV'] = obv
    
    # Target column (next bar close - for prediction)
    df['Target'] = df['Close'].shift(-1)
    
    # Drop rows with NaN (from rolling calculations and target shift)
    df = df.dropna().reset_index(drop=True)
    
    logger.info(f"Generated synthetic data: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"Columns: {list(df.columns)}")
    
    return df


def verify_training_pipeline(output_dir: str = 'results') -> dict:
    """
    Verify the complete training pipeline works correctly.
    
    Steps:
    1. Generate synthetic data
    2. Save as CSV (simulating MQL5 export)
    3. Run full training pipeline
    4. Verify ONNX model is created
    5. Verify metrics and plots are generated
    
    Returns:
        Dictionary with verification results
    """
    from train import TrainingPipeline
    
    results = {
        'success': False,
        'data_generated': False,
        'pipeline_ran': False,
        'onnx_created': False,
        'metrics_saved': False,
        'plots_saved': False,
        'errors': []
    }
    
    try:
        # Step 1: Generate synthetic data
        logger.info("=" * 60)
        logger.info("Step 1: Generating synthetic XAUUSD data")
        logger.info("=" * 60)
        
        df = generate_synthetic_xauusd_data(n_samples=2000)
        results['data_generated'] = True
        
        # Step 2: Save as CSV
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        csv_path = os.path.join(output_dir, 'synthetic_xauusd_data.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved synthetic data to: {csv_path}")
        
        # Step 3: Configure and run training pipeline
        logger.info("=" * 60)
        logger.info("Step 2: Running training pipeline")
        logger.info("=" * 60)
        
        config = {
            'output_dir': output_dir,
            'lookback_window': 30,
            'epochs': 10,  # Reduced for verification
            'batch_size': 32,
            'use_cross_validation': False,  # Skip CV for faster verification
            'early_stopping_patience': 3,
            'target_column': 'Target',
        }
        
        pipeline = TrainingPipeline(config=config)
        pipeline_results = pipeline.run_full_pipeline(csv_path, verbose=1)
        results['pipeline_ran'] = True
        
        # Step 4: Verify ONNX model
        logger.info("=" * 60)
        logger.info("Step 3: Verifying ONNX model")
        logger.info("=" * 60)
        
        onnx_path = pipeline_results.get('onnx_path')
        if onnx_path and os.path.exists(onnx_path):
            results['onnx_created'] = True
            logger.info(f"ONNX model verified at: {onnx_path}")
            
            # Verify ONNX can be loaded
            import onnxruntime as ort
            session = ort.InferenceSession(onnx_path)
            input_shape = session.get_inputs()[0].shape
            output_shape = session.get_outputs()[0].shape
            logger.info(f"ONNX input shape: {input_shape}")
            logger.info(f"ONNX output shape: {output_shape}")
        else:
            results['errors'].append(f"ONNX model not found at expected path: {onnx_path}")
        
        # Step 5: Verify metrics
        logger.info("=" * 60)
        logger.info("Step 4: Verifying metrics and results")
        logger.info("=" * 60)
        
        saved_files = pipeline_results.get('saved_files', {})
        if saved_files.get('json') and os.path.exists(saved_files['json']):
            results['metrics_saved'] = True
            logger.info(f"Metrics JSON verified at: {saved_files['json']}")
            
            # Print metrics
            import json
            with open(saved_files['json'], 'r') as f:
                metrics_data = json.load(f)
            
            if 'metrics' in metrics_data:
                logger.info("Evaluation Metrics:")
                for key, value in metrics_data['metrics'].items():
                    if value is not None:
                        logger.info(f"  {key}: {value:.6f}")
        else:
            results['errors'].append("Metrics JSON not found")
        
        # Step 6: Verify plots
        logger.info("=" * 60)
        logger.info("Step 5: Verifying plots")
        logger.info("=" * 60)
        
        plots = pipeline_results.get('plots', {})
        plots_found = []
        for plot_name, plot_path in plots.items():
            if plot_path and os.path.exists(plot_path):
                plots_found.append(plot_name)
                logger.info(f"Plot verified: {plot_name} at {plot_path}")
        
        if plots_found:
            results['plots_saved'] = True
        else:
            results['errors'].append("No plots found")
        
        # Overall success
        results['success'] = (
            results['data_generated'] and
            results['pipeline_ran'] and
            results['onnx_created'] and
            results['metrics_saved']
        )
        
        # Summary
        logger.info("=" * 60)
        logger.info("VERIFICATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Data Generated: {'✓' if results['data_generated'] else '✗'}")
        logger.info(f"Pipeline Ran: {'✓' if results['pipeline_ran'] else '✗'}")
        logger.info(f"ONNX Created: {'✓' if results['onnx_created'] else '✗'}")
        logger.info(f"Metrics Saved: {'✓' if results['metrics_saved'] else '✗'}")
        logger.info(f"Plots Saved: {'✓' if results['plots_saved'] else '✗'}")
        logger.info(f"Overall Success: {'✓' if results['success'] else '✗'}")
        
        if results['errors']:
            logger.warning("Errors encountered:")
            for error in results['errors']:
                logger.warning(f"  - {error}")
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline verification failed: {e}")
        import traceback
        traceback.print_exc()
        results['errors'].append(str(e))
        return results


def main():
    """Main entry point for checkpoint verification."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify LSTM-CNN training pipeline')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    
    args = parser.parse_args()
    
    results = verify_training_pipeline(args.output_dir)
    
    if results['success']:
        print("\n" + "=" * 60)
        print("CHECKPOINT VERIFICATION PASSED")
        print("=" * 60)
        print("The Python training pipeline is working correctly.")
        print(f"Results saved to: {args.output_dir}/")
        return 0
    else:
        print("\n" + "=" * 60)
        print("CHECKPOINT VERIFICATION FAILED")
        print("=" * 60)
        print("Errors:")
        for error in results['errors']:
            print(f"  - {error}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
