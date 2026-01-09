"""
Data Cleaner Module for LSTM-CNN XAUUSD Trading System

This module handles data cleaning operations for the training pipeline:
- Forward fill imputation for missing values
- Outlier detection and handling using z-score
- Validation to ensure no NaN values remain

Requirements: 4.1.1, 4.1.2, 4.1.3, 4.1.4, 4.1.5, 4.1.6, 4.1.7
"""

import sys
import pandas as pd
import numpy as np
from typing import Optional

from loguru import logger

# Configure loguru
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO",
    colorize=True
)


class DataCleaningError(Exception):
    """Custom exception for data cleaning errors."""
    pass


class DataCleaner:
    """
    Data cleaner for preparing training data.
    
    Handles missing values via forward fill and outliers via z-score capping.
    Does NOT alter feature calculations - only handles missing/invalid values.
    
    Attributes:
        outlier_threshold: Z-score threshold for outlier detection (default: 3.0)
    """
    
    def __init__(self, outlier_threshold: float = 3.0):
        """
        Initialize DataCleaner with outlier threshold.
        
        Args:
            outlier_threshold: Z-score threshold for outlier detection.
                              Values beyond this threshold are capped.
        
        Requirements: 4.1.2
        """
        self.outlier_threshold = outlier_threshold
        logger.info(f"DataCleaner initialized with outlier_threshold={outlier_threshold}")

    def forward_fill(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply forward fill imputation for NaN values.
        
        Forward fill propagates the last valid observation forward to fill gaps.
        This is appropriate for time series data where missing values are caused
        by market closures or data gaps.
        
        Args:
            df: Input DataFrame with potential NaN values
            
        Returns:
            DataFrame with NaN values forward-filled
            
        Requirements: 4.1.1
        """
        df_copy = df.copy()
        
        # Count NaN values before
        nan_before = df_copy.isna().sum().sum()
        rows_with_nan_before = df_copy.isna().any(axis=1).sum()
        
        # Apply forward fill
        df_copy = df_copy.ffill()
        
        # Count NaN values after
        nan_after = df_copy.isna().sum().sum()
        rows_affected = rows_with_nan_before - df_copy.isna().any(axis=1).sum()
        
        logger.info(f"Forward fill: {nan_before - nan_after} NaN values filled, "
                   f"{rows_affected} rows affected")
        
        return df_copy
    
    def handle_outliers(self, df: pd.DataFrame, method: str = 'cap') -> pd.DataFrame:
        """
        Detect and handle outliers using z-score threshold.
        
        Outliers are detected using z-score (number of standard deviations from mean).
        Values beyond the threshold are either capped to the threshold boundary
        or the rows are removed.
        
        Args:
            df: Input DataFrame
            method: 'cap' to cap outliers at threshold, 'remove' to remove rows
            
        Returns:
            DataFrame with outliers handled
            
        Requirements: 4.1.2
        """
        df_copy = df.copy()
        total_outliers = 0
        rows_affected = set()
        
        # Only process numeric columns
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Calculate z-scores (handle case where std is 0)
            col_mean = df_copy[col].mean()
            col_std = df_copy[col].std()
            
            if col_std == 0 or pd.isna(col_std):
                continue
                
            z_scores = (df_copy[col] - col_mean) / col_std
            
            # Find outliers
            outlier_mask = np.abs(z_scores) > self.outlier_threshold
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                total_outliers += outlier_count
                rows_affected.update(df_copy.index[outlier_mask].tolist())
                
                if method == 'cap':
                    # Cap outliers at threshold boundaries
                    upper_bound = col_mean + self.outlier_threshold * col_std
                    lower_bound = col_mean - self.outlier_threshold * col_std
                    df_copy[col] = df_copy[col].clip(lower=lower_bound, upper=upper_bound)
                elif method == 'remove':
                    # Mark rows for removal (handled after loop)
                    pass
        
        if method == 'remove' and rows_affected:
            df_copy = df_copy.drop(index=list(rows_affected))
            logger.info(f"Outlier handling: {total_outliers} outliers found, "
                       f"{len(rows_affected)} rows removed")
        else:
            logger.info(f"Outlier handling: {total_outliers} outliers capped, "
                       f"{len(rows_affected)} rows affected")
        
        return df_copy

    def validate_no_nan(self, df: pd.DataFrame) -> bool:
        """
        Validate that no NaN values remain in the DataFrame.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if validation passes
            
        Raises:
            DataCleaningError: If NaN values are found
            
        Requirements: 4.1.3, 4.1.5
        """
        nan_counts = df.isna().sum()
        total_nan = nan_counts.sum()
        
        if total_nan > 0:
            # Find columns with NaN
            cols_with_nan = nan_counts[nan_counts > 0]
            error_msg = f"Validation failed: {total_nan} NaN values remain. " \
                       f"Columns affected: {cols_with_nan.to_dict()}"
            logger.error(error_msg)
            raise DataCleaningError(error_msg)
        
        logger.info("Validation passed: No NaN values remain")
        return True
    
    def clean(self, df: pd.DataFrame, drop_initial_nan: bool = True) -> pd.DataFrame:
        """
        Run the complete cleaning pipeline.
        
        Pipeline sequence:
        1. Replace inf values with NaN
        2. Optionally drop rows where forward fill cannot resolve NaN (start of dataset)
        3. Forward fill remaining NaN values
        4. Backward fill any remaining NaN at the start
        5. Handle outliers using z-score capping
        6. Drop any remaining rows with NaN
        7. Validate no NaN values remain
        
        Args:
            df: Input DataFrame to clean
            drop_initial_nan: If True, drop rows at start where forward fill 
                            cannot resolve NaN values
            
        Returns:
            Cleaned DataFrame ready for sequence creation
            
        Raises:
            DataCleaningError: If NaN values remain after cleaning
            
        Requirements: 4.1.6, 4.1.7
        """
        logger.info(f"Starting data cleaning pipeline. Input shape: {df.shape}")
        initial_rows = len(df)
        
        df_clean = df.copy()
        
        # Step 0: Replace inf/-inf with NaN
        inf_count = np.isinf(df_clean.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            logger.info(f"Replacing {inf_count} inf values with NaN")
            df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        
        # Step 1: Handle initial NaN values that cannot be forward-filled
        if drop_initial_nan:
            # Find first valid index for each column
            first_valid_indices = df_clean.apply(lambda col: col.first_valid_index())
            
            if first_valid_indices.notna().any():
                # Get the maximum first valid index (latest start across all columns)
                max_first_valid = first_valid_indices.max()
                
                if max_first_valid is not None and max_first_valid > df_clean.index[0]:
                    # Find the position of max_first_valid in the index
                    start_pos = df_clean.index.get_loc(max_first_valid)
                    rows_dropped = start_pos
                    df_clean = df_clean.iloc[start_pos:]
                    logger.info(f"Dropped {rows_dropped} initial rows where forward fill "
                               "cannot resolve NaN values")
        
        # Step 2: Forward fill
        df_clean = self.forward_fill(df_clean)
        
        # Step 3: Backward fill any remaining NaN at the start
        nan_before_bfill = df_clean.isna().sum().sum()
        if nan_before_bfill > 0:
            df_clean = df_clean.bfill()
            nan_after_bfill = df_clean.isna().sum().sum()
            logger.info(f"Backward fill: {nan_before_bfill - nan_after_bfill} NaN values filled")
        
        # Step 4: Handle outliers
        df_clean = self.handle_outliers(df_clean)
        
        # Step 5: Drop any remaining rows with NaN (last resort)
        nan_rows = df_clean.isna().any(axis=1).sum()
        if nan_rows > 0:
            logger.warning(f"Dropping {nan_rows} rows with remaining NaN values")
            df_clean = df_clean.dropna()
        
        # Step 6: Validate
        self.validate_no_nan(df_clean)
        
        final_rows = len(df_clean)
        logger.info(f"Cleaning complete. Output shape: {df_clean.shape}. "
                   f"Rows removed: {initial_rows - final_rows}")
        
        return df_clean
