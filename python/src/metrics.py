"""
Metrics Module for LSTM-CNN XAUUSD Trading System

This module implements evaluation metrics and visualization for model performance:
- RMSE, MAE, MAPE, R² metrics
- Loss curves and prediction plots
- Results saving to JSON/CSV

Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7
"""

import logging
import json
import os
from typing import Dict, Any, Optional, List, Union

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MetricsError(Exception):
    """Custom exception for metrics computation errors."""
    pass


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error.
    
    RMSE = sqrt(mean((y - ŷ)²))
    
    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
    
    Returns:
        RMSE value
    
    Raises:
        MetricsError: If arrays have different lengths or are empty
    
    Requirements: 10.1
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise MetricsError(f"Array length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    if len(y_true) == 0:
        raise MetricsError("Cannot compute RMSE on empty arrays")
    
    squared_errors = (y_true - y_pred) ** 2
    mse = np.mean(squared_errors)
    rmse = np.sqrt(mse)
    
    return float(rmse)


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.
    
    MAE = mean(|y - ŷ|)
    
    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
    
    Returns:
        MAE value
    
    Raises:
        MetricsError: If arrays have different lengths or are empty
    
    Requirements: 10.2
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise MetricsError(f"Array length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    if len(y_true) == 0:
        raise MetricsError("Cannot compute MAE on empty arrays")
    
    absolute_errors = np.abs(y_true - y_pred)
    mae = np.mean(absolute_errors)
    
    return float(mae)


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Percentage Error.
    
    MAPE = mean(|y - ŷ| / |y|) * 100
    
    Note: Values where y_true is zero are excluded from calculation
    to avoid division by zero.
    
    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
    
    Returns:
        MAPE value as percentage
    
    Raises:
        MetricsError: If arrays have different lengths, are empty,
                     or all y_true values are zero
    
    Requirements: 10.3
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise MetricsError(f"Array length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    if len(y_true) == 0:
        raise MetricsError("Cannot compute MAPE on empty arrays")
    
    # Exclude zero values to avoid division by zero
    non_zero_mask = y_true != 0
    
    if not np.any(non_zero_mask):
        raise MetricsError("Cannot compute MAPE: all y_true values are zero")
    
    y_true_nz = y_true[non_zero_mask]
    y_pred_nz = y_pred[non_zero_mask]
    
    percentage_errors = np.abs((y_true_nz - y_pred_nz) / np.abs(y_true_nz))
    mape = np.mean(percentage_errors) * 100
    
    return float(mape)


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute R² (coefficient of determination) score.
    
    R² = 1 - (SS_res / SS_tot)
    where:
        SS_res = Σ(y - ŷ)² (residual sum of squares)
        SS_tot = Σ(y - ȳ)² (total sum of squares)
    
    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
    
    Returns:
        R² score (can be negative for poor models)
    
    Raises:
        MetricsError: If arrays have different lengths, are empty,
                     or y_true has zero variance
    
    Requirements: 10.4
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise MetricsError(f"Array length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    if len(y_true) == 0:
        raise MetricsError("Cannot compute R² on empty arrays")
    
    # Calculate mean of y_true
    y_mean = np.mean(y_true)
    
    # Calculate SS_res (residual sum of squares)
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    # Calculate SS_tot (total sum of squares)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    
    if ss_tot == 0:
        # All y_true values are the same (zero variance)
        # If predictions are perfect, R² = 1, otherwise undefined
        if ss_res == 0:
            return 1.0
        raise MetricsError("Cannot compute R²: y_true has zero variance")
    
    r2 = 1 - (ss_res / ss_tot)
    
    return float(r2)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
    
    Returns:
        Dictionary with keys: 'rmse', 'mae', 'mape', 'r2'
    
    Requirements: 10.1, 10.2, 10.3, 10.4
    """
    metrics = {
        'rmse': compute_rmse(y_true, y_pred),
        'mae': compute_mae(y_true, y_pred),
        'r2': compute_r2(y_true, y_pred)
    }
    
    # MAPE may fail if all y_true are zero
    try:
        metrics['mape'] = compute_mape(y_true, y_pred)
    except MetricsError as e:
        logger.warning(f"Could not compute MAPE: {e}")
        metrics['mape'] = None
    
    return metrics



def plot_loss_curves(
    history: Dict[str, List[float]],
    output_path: Optional[str] = None,
    figsize: tuple = (10, 6)
) -> Optional[Any]:
    """
    Plot training vs validation loss curves.
    
    Args:
        history: Training history dictionary with keys 'loss' and optionally 'val_loss'
        output_path: Path to save the plot (if None, returns figure)
        figsize: Figure size as (width, height)
    
    Returns:
        Matplotlib figure if output_path is None, else None
    
    Requirements: 10.5
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise MetricsError("matplotlib not installed. Install with: pip install matplotlib")
    
    if 'loss' not in history:
        raise MetricsError("History must contain 'loss' key")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Plot training loss
    ax.plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
    
    # Plot validation loss if available
    if 'val_loss' in history:
        ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Training vs Validation Loss', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    if output_path:
        # Ensure directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Loss curves saved to: {output_path}")
        return None
    
    return fig


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Optional[str] = None,
    figsize: tuple = (12, 8),
    max_points: int = 500,
    title: str = 'Actual vs Predicted Prices'
) -> Optional[Any]:
    """
    Plot actual vs predicted values.
    
    Creates two subplots:
    1. Time series comparison of actual vs predicted
    2. Scatter plot with ideal prediction line
    
    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
        output_path: Path to save the plot (if None, returns figure)
        figsize: Figure size as (width, height)
        max_points: Maximum points to plot (for performance)
        title: Plot title
    
    Returns:
        Matplotlib figure if output_path is None, else None
    
    Requirements: 10.6
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise MetricsError("matplotlib not installed. Install with: pip install matplotlib")
    
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise MetricsError(f"Array length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    # Subsample if too many points
    if len(y_true) > max_points:
        indices = np.linspace(0, len(y_true) - 1, max_points, dtype=int)
        y_true_plot = y_true[indices]
        y_pred_plot = y_pred[indices]
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Subplot 1: Time series comparison
    ax1 = axes[0]
    x_axis = range(len(y_true_plot))
    ax1.plot(x_axis, y_true_plot, 'b-', label='Actual', alpha=0.7, linewidth=1)
    ax1.plot(x_axis, y_pred_plot, 'r-', label='Predicted', alpha=0.7, linewidth=1)
    ax1.set_xlabel('Sample Index', fontsize=11)
    ax1.set_ylabel('Price', fontsize=11)
    ax1.set_title('Time Series: Actual vs Predicted', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Scatter plot
    ax2 = axes[1]
    ax2.scatter(y_true_plot, y_pred_plot, alpha=0.5, s=10, c='blue')
    
    # Add ideal prediction line (y = x)
    min_val = min(y_true_plot.min(), y_pred_plot.min())
    max_val = max(y_true_plot.max(), y_pred_plot.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal')
    
    ax2.set_xlabel('Actual Price', fontsize=11)
    ax2.set_ylabel('Predicted Price', fontsize=11)
    ax2.set_title('Scatter: Actual vs Predicted', fontsize=12)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add metrics annotation
    try:
        metrics = compute_all_metrics(y_true, y_pred)
        metrics_text = f"RMSE: {metrics['rmse']:.4f}\nMAE: {metrics['mae']:.4f}\nR²: {metrics['r2']:.4f}"
        if metrics['mape'] is not None:
            metrics_text += f"\nMAPE: {metrics['mape']:.2f}%"
        ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    except MetricsError:
        pass
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        # Ensure directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Predictions plot saved to: {output_path}")
        return None
    
    return fig



def save_metrics_json(
    metrics: Dict[str, Any],
    output_path: str,
    additional_info: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Dictionary of metric values
        output_path: Path to save the JSON file
        additional_info: Optional additional information to include
    
    Returns:
        Path to the saved file
    
    Requirements: 10.7
    """
    # Ensure directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Combine metrics with additional info
    output_data = {'metrics': metrics}
    if additional_info:
        output_data.update(additional_info)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    output_data = convert_to_serializable(output_data)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Metrics saved to JSON: {output_path}")
    return output_path


def save_metrics_csv(
    metrics: Dict[str, Any],
    output_path: str,
    additional_info: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save metrics to CSV file.
    
    Args:
        metrics: Dictionary of metric values
        output_path: Path to save the CSV file
        additional_info: Optional additional information to include
    
    Returns:
        Path to the saved file
    
    Requirements: 10.7
    """
    # Ensure directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Flatten metrics and additional info into single row
    row_data = {}
    
    for key, value in metrics.items():
        if isinstance(value, (int, float, str, type(None))):
            row_data[f"metric_{key}"] = value
    
    if additional_info:
        for key, value in additional_info.items():
            if isinstance(value, (int, float, str, type(None))):
                row_data[key] = value
    
    # Create DataFrame and save
    df = pd.DataFrame([row_data])
    df.to_csv(output_path, index=False)
    
    logger.info(f"Metrics saved to CSV: {output_path}")
    return output_path


class MetricsReporter:
    """
    Comprehensive metrics reporter for model evaluation.
    
    Combines metric computation, visualization, and saving into
    a single interface for easy evaluation workflow.
    
    Requirements: 10.1-10.7
    """
    
    def __init__(self, output_dir: str = 'results'):
        """
        Initialize MetricsReporter.
        
        Args:
            output_dir: Directory to save all outputs
        """
        self.output_dir = output_dir
        self.metrics = {}
        self.history = None
        
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        logger.info(f"MetricsReporter initialized with output_dir: {output_dir}")
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        history: Optional[Dict[str, List[float]]] = None
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Args:
            y_true: Array of actual values
            y_pred: Array of predicted values
            history: Optional training history for loss curves
        
        Returns:
            Dictionary of computed metrics
        
        Requirements: 10.1-10.4
        """
        self.metrics = compute_all_metrics(y_true, y_pred)
        self.history = history
        
        logger.info(f"Evaluation complete: RMSE={self.metrics['rmse']:.4f}, "
                   f"MAE={self.metrics['mae']:.4f}, R²={self.metrics['r2']:.4f}")
        
        return self.metrics
    
    def generate_plots(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str = 'model'
    ) -> Dict[str, str]:
        """
        Generate and save all visualization plots.
        
        Args:
            y_true: Array of actual values
            y_pred: Array of predicted values
            prefix: Filename prefix for saved plots
        
        Returns:
            Dictionary mapping plot names to file paths
        
        Requirements: 10.5, 10.6
        """
        saved_plots = {}
        
        # Generate predictions plot
        predictions_path = os.path.join(self.output_dir, f'{prefix}_predictions.png')
        plot_predictions(y_true, y_pred, output_path=predictions_path)
        saved_plots['predictions'] = predictions_path
        
        # Generate loss curves if history available
        if self.history is not None:
            loss_path = os.path.join(self.output_dir, f'{prefix}_loss_curves.png')
            plot_loss_curves(self.history, output_path=loss_path)
            saved_plots['loss_curves'] = loss_path
        
        return saved_plots
    
    def save_results(
        self,
        prefix: str = 'model',
        additional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Save metrics to JSON and CSV files.
        
        Args:
            prefix: Filename prefix for saved files
            additional_info: Optional additional information to include
        
        Returns:
            Dictionary mapping format names to file paths
        
        Requirements: 10.7
        """
        if not self.metrics:
            raise MetricsError("No metrics to save. Call evaluate() first.")
        
        saved_files = {}
        
        # Save to JSON
        json_path = os.path.join(self.output_dir, f'{prefix}_metrics.json')
        save_metrics_json(self.metrics, json_path, additional_info)
        saved_files['json'] = json_path
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, f'{prefix}_metrics.csv')
        save_metrics_csv(self.metrics, csv_path, additional_info)
        saved_files['csv'] = csv_path
        
        return saved_files
    
    def full_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        history: Optional[Dict[str, List[float]]] = None,
        prefix: str = 'model',
        additional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate complete evaluation report with metrics, plots, and saved files.
        
        Args:
            y_true: Array of actual values
            y_pred: Array of predicted values
            history: Optional training history for loss curves
            prefix: Filename prefix for all saved files
            additional_info: Optional additional information to include
        
        Returns:
            Dictionary containing:
            - 'metrics': Computed metrics
            - 'plots': Paths to saved plots
            - 'files': Paths to saved metric files
        
        Requirements: 10.1-10.7
        """
        # Compute metrics
        metrics = self.evaluate(y_true, y_pred, history)
        
        # Generate plots
        plots = self.generate_plots(y_true, y_pred, prefix)
        
        # Save results
        files = self.save_results(prefix, additional_info)
        
        report = {
            'metrics': metrics,
            'plots': plots,
            'files': files
        }
        
        logger.info(f"Full report generated in: {self.output_dir}")
        
        return report
