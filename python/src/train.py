"""
Training Pipeline for LSTM-CNN XAUUSD Trading System

This module orchestrates the complete training workflow:
1. Load CSV data from MQL5 export
2. Clean data using DataCleaner
3. Build sequences using SequenceBuilder
4. Split train/test by date (80/20)
5. Train LSTM-CNN model with cross-validation
6. Evaluate and export ONNX model

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 10.1-10.7
"""

import os
import sys
import json
import warnings
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

# Silence warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger('tensorflow').setLevel(logging.CRITICAL)
logging.getLogger('absl').setLevel(logging.CRITICAL)

import numpy as np
import pandas as pd

from loguru import logger

# Configure loguru - clean format
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO",
    colorize=True
)

from data_cleaner import DataCleaner, DataCleaningError
from sequence_builder import SequenceBuilder, SequenceBuilderError
from lstm_cnn_model import LSTMCNNModel, ModelBuildError, ModelTrainingError, ONNXExportError
from profitability_model import ProfitabilityClassifier, ProfitabilityModelError
from metrics import (
    MetricsReporter, compute_all_metrics,
    plot_loss_curves, plot_predictions,
    save_metrics_json, save_metrics_csv
)


# Kaggle dataset path (hardcoded)
KAGGLE_DATA_PATH = '/kaggle/input/lstm-cnn/lstm_cnn_data/training_data.csv'

# Default training configuration
DEFAULT_TRAINING_CONFIG = {
    'lookback_window': 30,
    'target_column': None,
    'train_split': 0.8,
    'lstm_units': 50,
    'lstm_dropout': 0.2,
    'cnn_filters': 64,
    'cnn_kernel_size': 3,
    'dense_units': 32,
    'learning_rate': 0.001,
    'epochs': 150,
    'batch_size': 128,
    'early_stopping_patience': 10,
    'cv_folds': 5,
    'use_cross_validation': True,
    'outlier_threshold': 3.0,
    'output_dir': '/kaggle/working/results',
    'model_filename': 'lstm_cnn_xauusd.onnx',
    # Dual-model settings
    'train_profitability_model': True,
    'profitability_model_filename': 'profitability_classifier.onnx',
    'scalers_filename': 'scalers.csv',
    'long_outcome_column': 'Long_Outcome',
    'short_outcome_column': 'Short_Outcome',
}



class TrainingPipeline:
    """
    Complete training pipeline for LSTM-CNN XAUUSD model.
    
    Orchestrates data loading, cleaning, sequence building, model training,
    evaluation, and ONNX export.
    
    Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize training pipeline with configuration."""
        self.config = DEFAULT_TRAINING_CONFIG.copy()
        if config is not None:
            self.config.update(config)
        
        self.data_cleaner = DataCleaner(
            outlier_threshold=self.config['outlier_threshold']
        )
        self.sequence_builder = SequenceBuilder(
            lookback=self.config['lookback_window']
        )
        self.model = None
        self.profitability_model = None
        self.metrics_reporter = None
        
        self.raw_data = None
        self.clean_data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.train_indices = None
        self.test_indices = None
        
        # Triple barrier outcome data
        self.long_outcomes_train = None
        self.short_outcomes_train = None
        self.long_outcomes_test = None
        self.short_outcomes_test = None
        
        self.training_history = None
        self.profitability_history = None
        self.evaluation_metrics = None
        self.cv_results = None
        
        logger.info(f"TrainingPipeline initialized with config: {self.config}")
    
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Load CSV data exported from MQL5. Requirements: 9.1"""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        logger.info(f"Loading data from: {csv_path}")
        self.raw_data = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(self.raw_data)} rows, {len(self.raw_data.columns)} columns")
        return self.raw_data
    
    def clean_data_step(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Clean data using DataCleaner. Requirements: 9.1"""
        if df is None:
            df = self.raw_data
        if df is None:
            raise ValueError("No data to clean. Call load_data() first.")
        
        logger.info("Running data cleaning pipeline...")
        self.clean_data = self.data_cleaner.clean(df)
        logger.info(f"Cleaned data shape: {self.clean_data.shape}")
        return self.clean_data

    
    def split_train_test_by_date(
        self,
        df: Optional[pd.DataFrame] = None,
        train_ratio: Optional[float] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Split data into train/test sets by date (chronological order).
        Uses 80/20 split ensuring all training samples have timestamps
        before all test samples.
        
        Requirements: 9.2
        """
        if df is None:
            df = self.clean_data
        if df is None:
            raise ValueError("No data to split. Call clean_data_step() first.")
        if train_ratio is None:
            train_ratio = self.config['train_split']
        
        n_samples = len(df)
        split_idx = int(n_samples * train_ratio)
        
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        self.train_indices = np.arange(0, split_idx)
        self.test_indices = np.arange(split_idx, n_samples)
        
        logger.info(f"Train/test split: {len(train_df)} train, {len(test_df)} test "
                   f"({train_ratio*100:.0f}%/{(1-train_ratio)*100:.0f}%)")
        return train_df, test_df, self.train_indices, self.test_indices
    
    def build_sequences(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build sequences for training and testing. Requirements: 9.1"""
        if target_column is None:
            target_column = self.config['target_column']
        
        logger.info("Building sequences...")
        
        # Check for triple barrier outcome columns
        long_col = self.config.get('long_outcome_column', 'Long_Outcome')
        short_col = self.config.get('short_outcome_column', 'Short_Outcome')
        has_outcomes = long_col in train_df.columns and short_col in train_df.columns
        
        if has_outcomes:
            logger.info("Found triple barrier outcome columns, extracting...")
            # Extract outcomes before dropping them for sequence building
            train_long = train_df[long_col].values
            train_short = train_df[short_col].values
            test_long = test_df[long_col].values
            test_short = test_df[short_col].values
            
            # Drop outcome columns for feature sequences
            train_features = train_df.drop(columns=[long_col, short_col])
            test_features = test_df.drop(columns=[long_col, short_col])
        else:
            train_features = train_df
            test_features = test_df
            train_long = train_short = test_long = test_short = None
        
        self.X_train, self.y_train = self.sequence_builder.create_sequences(
            train_features, target_column=target_column
        )
        self.X_test, self.y_test = self.sequence_builder.create_sequences(
            test_features, target_column=target_column
        )
        
        # Align outcomes with sequences (offset by lookback)
        if has_outcomes:
            lookback = self.config['lookback_window']
            self.long_outcomes_train = train_long[lookback:]
            self.short_outcomes_train = train_short[lookback:]
            self.long_outcomes_test = test_long[lookback:]
            self.short_outcomes_test = test_short[lookback:]
            
            logger.info(f"Triple barrier outcomes aligned: train={len(self.long_outcomes_train)}, "
                       f"test={len(self.long_outcomes_test)}")
        
        logger.info(f"Training sequences: X={self.X_train.shape}, y={self.y_train.shape}")
        logger.info(f"Test sequences: X={self.X_test.shape}, y={self.y_test.shape}")
        return self.X_train, self.y_train, self.X_test, self.y_test

    
    def build_model(self) -> LSTMCNNModel:
        """Build the LSTM-CNN model. Requirements: 9.3"""
        if self.X_train is None:
            raise ValueError("No training data. Call build_sequences() first.")
        
        num_features = self.X_train.shape[2]
        model_config = {
            'lookback': self.config['lookback_window'],
            'num_features': num_features,
            'lstm_units': self.config['lstm_units'],
            'lstm_dropout': self.config['lstm_dropout'],
            'cnn_filters': self.config['cnn_filters'],
            'cnn_kernel_size': self.config['cnn_kernel_size'],
            'dense_units': self.config['dense_units'],
            'learning_rate': self.config['learning_rate'],
            'epochs': self.config['epochs'],
            'batch_size': self.config['batch_size'],
            'early_stopping_patience': self.config['early_stopping_patience'],
        }
        
        self.model = LSTMCNNModel(config=model_config)
        self.model.build_model()
        return self.model
    
    def train_model(
        self,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: int = 1
    ) -> Dict[str, List[float]]:
        """Train the model. Requirements: 9.3"""
        if X_train is None:
            X_train = self.X_train
        if y_train is None:
            y_train = self.y_train
        if self.model is None:
            self.build_model()
        
        logger.info("Training model...")
        self.training_history = self.model.train(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            verbose=verbose
        )
        return self.training_history
    
    def _create_fresh_model(self) -> LSTMCNNModel:
        """Create a new model instance with current config."""
        num_features = self.X_train.shape[2]
        model_config = {
            'lookback': self.config['lookback_window'],
            'num_features': num_features,
            'lstm_units': self.config['lstm_units'],
            'lstm_dropout': self.config['lstm_dropout'],
            'cnn_filters': self.config['cnn_filters'],
            'cnn_kernel_size': self.config['cnn_kernel_size'],
            'dense_units': self.config['dense_units'],
            'learning_rate': self.config['learning_rate'],
            'epochs': self.config['epochs'],
            'batch_size': self.config['batch_size'],
            'early_stopping_patience': self.config['early_stopping_patience'],
        }
        return LSTMCNNModel(config=model_config)

    
    def cross_validate(
        self,
        n_folds: Optional[int] = None,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Perform time-series cross-validation using expanding window approach.
        Requirements: 9.3
        """
        if n_folds is None:
            n_folds = self.config['cv_folds']
        if self.X_train is None or self.y_train is None:
            raise ValueError("No training data. Call build_sequences() first.")
        
        logger.info(f"Starting {n_folds}-fold time-series cross-validation...")
        
        n_samples = len(self.X_train)
        fold_size = n_samples // (n_folds + 1)
        
        cv_metrics = []
        
        for fold in range(n_folds):
            train_end = (fold + 1) * fold_size
            val_start = train_end
            val_end = min(val_start + fold_size, n_samples)
            
            if val_end <= val_start:
                logger.warning(f"Skipping fold {fold + 1}: insufficient validation data")
                continue
            
            X_fold_train = self.X_train[:train_end]
            y_fold_train = self.y_train[:train_end]
            X_fold_val = self.X_train[val_start:val_end]
            y_fold_val = self.y_train[val_start:val_end]
            
            logger.info(f"Fold {fold + 1}/{n_folds}: train={len(X_fold_train)}, val={len(X_fold_val)}")
            
            fold_model = self._create_fresh_model()
            fold_model.build_model()
            fold_model.train(X_fold_train, y_fold_train, X_val=X_fold_val, y_val=y_fold_val, verbose=verbose)
            
            y_pred = fold_model.model.predict(X_fold_val, verbose=0)
            fold_metrics = compute_all_metrics(y_fold_val, y_pred)
            fold_metrics['fold'] = fold + 1
            cv_metrics.append(fold_metrics)
            
            logger.info(f"Fold {fold + 1} metrics: RMSE={fold_metrics['rmse']:.4f}, "
                       f"MAE={fold_metrics['mae']:.4f}, R²={fold_metrics['r2']:.4f}")
        
        self.cv_results = {
            'n_folds': len(cv_metrics),
            'fold_metrics': cv_metrics,
            'mean_rmse': np.mean([m['rmse'] for m in cv_metrics]),
            'std_rmse': np.std([m['rmse'] for m in cv_metrics]),
            'mean_mae': np.mean([m['mae'] for m in cv_metrics]),
            'std_mae': np.std([m['mae'] for m in cv_metrics]),
            'mean_r2': np.mean([m['r2'] for m in cv_metrics]),
            'std_r2': np.std([m['r2'] for m in cv_metrics]),
        }
        
        logger.info(f"CV Results: RMSE={self.cv_results['mean_rmse']:.4f}±{self.cv_results['std_rmse']:.4f}")
        return self.cv_results

    
    def evaluate(
        self,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Evaluate model on test set. Requirements: 9.5, 10.1-10.4"""
        if X_test is None:
            X_test = self.X_test
        if y_test is None:
            y_test = self.y_test
        if self.model is None:
            raise ValueError("No model to evaluate. Call train_model() first.")
        
        logger.info("Evaluating model on test set...")
        y_pred = self.model.model.predict(X_test, verbose=0)
        self.evaluation_metrics = compute_all_metrics(y_test, y_pred)
        
        logger.info("=" * 50)
        logger.info("TEST RESULTS")
        logger.info("=" * 50)
        logger.info(f"RMSE: {self.evaluation_metrics['rmse']:.4f}")
        logger.info(f"MAE: {self.evaluation_metrics['mae']:.4f}")
        logger.info(f"R²: {self.evaluation_metrics['r2']:.4f}")
        if self.evaluation_metrics.get('mape'):
            logger.info(f"MAPE: {self.evaluation_metrics['mape']:.2f}%")
        if self.evaluation_metrics.get('directional_accuracy'):
            logger.info(f"Directional Accuracy: {self.evaluation_metrics['directional_accuracy']:.1f}%")
        logger.info("=" * 50)
        
        return self.evaluation_metrics
    
    def export_onnx(self, output_path: Optional[str] = None) -> str:
        """Export model to ONNX format. Requirements: 9.5"""
        if self.model is None:
            raise ValueError("No model to export. Call train_model() first.")
        
        if output_path is None:
            output_dir = self.config['output_dir']
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path = os.path.join(output_dir, self.config['model_filename'])
        
        logger.info(f"Exporting model to ONNX: {output_path}")
        return self.model.export_onnx(output_path)
    
    def train_profitability_model(
        self,
        verbose: int = 1
    ) -> Optional[Dict[str, List[float]]]:
        """
        Train the profitability classifier on triple barrier labeled data.
        
        Only trains on samples where either TP or SL was hit (excludes -1 outcomes).
        
        Returns:
            Training history or None if no outcome data available
        """
        if self.long_outcomes_train is None or self.short_outcomes_train is None:
            logger.warning("No triple barrier outcomes available, skipping profitability model")
            return None
        
        logger.info("=" * 50)
        logger.info("Training Profitability Classifier")
        logger.info("=" * 50)
        
        # Initialize profitability model
        num_features = self.X_train.shape[2]
        prof_config = {
            'lookback': self.config['lookback_window'],
            'num_features': num_features,
            'epochs': self.config['epochs'],
            'batch_size': self.config['batch_size'],
            'early_stopping_patience': self.config['early_stopping_patience'],
        }
        
        self.profitability_model = ProfitabilityClassifier(config=prof_config)
        self.profitability_model.build_model()
        
        # Filter to valid samples and prepare labels
        X_prof, y_long, y_short = self.profitability_model.filter_valid_samples(
            self.X_train,
            self.long_outcomes_train,
            self.short_outcomes_train
        )
        
        # Stack labels: [samples, 2] for [long_outcome, short_outcome]
        y_prof = np.stack([y_long, y_short], axis=1)
        
        # Prepare validation data
        X_val_prof = None
        y_val_prof = None
        if self.long_outcomes_test is not None:
            X_val_prof, y_long_val, y_short_val = self.profitability_model.filter_valid_samples(
                self.X_test,
                self.long_outcomes_test,
                self.short_outcomes_test
            )
            y_val_prof = np.stack([y_long_val, y_short_val], axis=1)
        
        # Train
        self.profitability_history = self.profitability_model.train(
            X_prof, y_prof,
            X_val=X_val_prof, y_val=y_val_prof,
            verbose=verbose
        )
        
        return self.profitability_history
    
    def export_profitability_onnx(self, output_path: Optional[str] = None) -> Optional[str]:
        """Export profitability model to ONNX format."""
        if self.profitability_model is None:
            logger.warning("No profitability model to export")
            return None
        
        if output_path is None:
            output_dir = self.config['output_dir']
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path = os.path.join(output_dir, self.config['profitability_model_filename'])
        
        logger.info(f"Exporting profitability model to ONNX: {output_path}")
        return self.profitability_model.export_onnx(output_path)
    
    def export_scalers(self, output_path: Optional[str] = None) -> str:
        """Export normalization scalers to CSV for MQL5."""
        if output_path is None:
            output_dir = self.config['output_dir']
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path = os.path.join(output_dir, self.config['scalers_filename'])
        
        # Get feature names from clean data (excluding target and outcome columns)
        feature_names = None
        if self.clean_data is not None:
            cols = list(self.clean_data.columns)
            # Remove target and outcome columns
            for col in ['Target', 'Long_Outcome', 'Short_Outcome']:
                if col in cols:
                    cols.remove(col)
            feature_names = cols
        
        return self.sequence_builder.export_scalers_csv(output_path, feature_names)
    
    def generate_plots(
        self,
        y_test: Optional[np.ndarray] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, str]:
        """Generate evaluation plots. Requirements: 10.5, 10.6"""
        if y_test is None:
            y_test = self.y_test
        if output_dir is None:
            output_dir = self.config['output_dir']
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        saved_plots = {}
        
        y_pred = self.model.model.predict(self.X_test, verbose=0)
        
        predictions_path = os.path.join(output_dir, 'predictions.png')
        plot_predictions(y_test, y_pred, output_path=predictions_path)
        saved_plots['predictions'] = predictions_path
        
        if self.training_history is not None:
            loss_path = os.path.join(output_dir, 'loss_curves.png')
            plot_loss_curves(self.training_history, output_path=loss_path)
            saved_plots['loss_curves'] = loss_path
        
        return saved_plots

    
    def save_results(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """Save metrics and results to files. Requirements: 10.7"""
        if output_dir is None:
            output_dir = self.config['output_dir']
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        saved_files = {}
        
        results = {
            'config': self.config,
            'evaluation_metrics': self.evaluation_metrics,
            'cv_results': self.cv_results,
            'timestamp': datetime.now().isoformat(),
        }
        
        json_path = os.path.join(output_dir, 'training_results.json')
        save_metrics_json(self.evaluation_metrics or {}, json_path, additional_info=results)
        saved_files['json'] = json_path
        
        csv_path = os.path.join(output_dir, 'training_metrics.csv')
        save_metrics_csv(self.evaluation_metrics or {}, csv_path)
        saved_files['csv'] = csv_path
        
        return saved_files
    
    def log_hyperparameters(self) -> Dict[str, Any]:
        """Log all hyperparameters used during training. Requirements: 9.4"""
        hyperparams = {
            'lookback_window': self.config['lookback_window'],
            'train_split': self.config['train_split'],
            'lstm_units': self.config['lstm_units'],
            'lstm_dropout': self.config['lstm_dropout'],
            'cnn_filters': self.config['cnn_filters'],
            'cnn_kernel_size': self.config['cnn_kernel_size'],
            'dense_units': self.config['dense_units'],
            'learning_rate': self.config['learning_rate'],
            'epochs': self.config['epochs'],
            'batch_size': self.config['batch_size'],
            'early_stopping_patience': self.config['early_stopping_patience'],
            'cv_folds': self.config['cv_folds'],
            'outlier_threshold': self.config['outlier_threshold'],
        }
        
        logger.info("Hyperparameters:")
        for key, value in hyperparams.items():
            logger.info(f"  {key}: {value}")
        
        return hyperparams

    
    def run_full_pipeline(
        self,
        csv_path: str,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Sequence: Load → Clean → Split → Sequences → CV → Train → Evaluate → Export
        
        For dual-model system:
        - Price predictor trained on all samples
        - Profitability classifier trained on samples with valid outcomes
        
        Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 10.1-10.7
        """
        print("\n" + "=" * 60)
        print("LSTM-CNN XAUUSD DUAL-MODEL TRAINING PIPELINE")
        print("=" * 60)
        
        # Step 1: Load data
        print("\n[1/8] Loading data...")
        self.load_data(csv_path)
        
        # Step 2: Clean data
        print("[2/8] Cleaning data...")
        self.clean_data_step()
        
        # Step 3: Split train/test by date
        print("[3/8] Splitting train/test...")
        train_df, test_df, _, _ = self.split_train_test_by_date()
        
        # Step 4: Build sequences
        print("[4/8] Building sequences...")
        self.build_sequences(train_df, test_df)
        
        # Step 5: Cross-validation (optional)
        if self.config['use_cross_validation']:
            print("[5/8] Cross-validation...")
            self.cross_validate(verbose=0)
        
        # Step 6: Train price predictor
        print("\n" + "-" * 60)
        print("MODEL 1: PRICE PREDICTOR (LSTM-CNN)")
        print("-" * 60)
        print(f"Training on {len(self.X_train)} samples...")
        self.build_model()
        self.train_model(X_val=self.X_test, y_val=self.y_test, verbose=0)
        
        # Step 7: Train profitability classifier
        profitability_onnx_path = None
        if self.config.get('train_profitability_model', True) and self.long_outcomes_train is not None:
            print("\n" + "-" * 60)
            print("MODEL 2: PROFITABILITY CLASSIFIER (CNN)")
            print("-" * 60)
            
            # Count valid samples
            long_valid = np.sum(self.long_outcomes_train >= 0)
            short_valid = np.sum(self.short_outcomes_train >= 0)
            print(f"Training on {long_valid} long + {short_valid} short labeled samples...")
            
            self.train_profitability_model(verbose=0)
        
        # Step 8: Evaluate and export
        print("\n" + "-" * 60)
        print("EVALUATION & EXPORT")
        print("-" * 60)
        
        self.evaluate()
        plots = self.generate_plots()
        onnx_path = self.export_onnx()
        
        if self.profitability_model is not None:
            profitability_onnx_path = self.export_profitability_onnx()
        
        scalers_path = self.export_scalers()
        saved_files = self.save_results()
        
        results = {
            'evaluation_metrics': self.evaluation_metrics,
            'cv_results': self.cv_results,
            'onnx_path': onnx_path,
            'profitability_onnx_path': profitability_onnx_path,
            'scalers_path': scalers_path,
            'plots': plots,
            'saved_files': saved_files,
        }
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Price predictor:    {onnx_path}")
        if profitability_onnx_path:
            print(f"Profitability model: {profitability_onnx_path}")
        print(f"Scalers:            {scalers_path}")
        print("=" * 60 + "\n")
        
        return results


def main():
    """Main entry point for training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train LSTM-CNN XAUUSD model')
    parser.add_argument('--csv-path', default=KAGGLE_DATA_PATH, help='Path to training CSV file')
    parser.add_argument('--output-dir', default='/kaggle/working/results', help='Output directory')
    parser.add_argument('--lookback', type=int, default=30, help='Lookback window')
    parser.add_argument('--epochs', type=int, default=150, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--no-cv', action='store_true', help='Skip cross-validation')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')
    
    args = parser.parse_args()
    
    config = {
        'output_dir': args.output_dir,
        'lookback_window': args.lookback,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'use_cross_validation': not args.no_cv,
    }
    
    pipeline = TrainingPipeline(config=config)
    results = pipeline.run_full_pipeline(args.csv_path, verbose=args.verbose)
    
    print("\nTraining complete!")
    print(f"Test RMSE: {results['evaluation_metrics']['rmse']:.4f}")
    print(f"Test MAE: {results['evaluation_metrics']['mae']:.4f}")
    print(f"Test R²: {results['evaluation_metrics']['r2']:.4f}")


if __name__ == '__main__':
    main()
