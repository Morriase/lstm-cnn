"""
Multi-Pair XGBoost Model Training for Forex Price Prediction
Trains separate models for each currency pair with per-pair scaler params

Supported Pairs: EURUSD, XAUUSD, GBPUSD, AUDUSD, USDJPY, USDCAD, USDCHF, NZDUSD
Binary Classification: Buy(0) vs Sell(1)

Usage:
  python train_multi_pair.py                    # Train all pairs
  python train_multi_pair.py --pair EURUSD     # Train single pair
  python train_multi_pair.py --pair all        # Train all pairs
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import os
import time
import argparse

# =============================================================================
# CONFIGURATION
# =============================================================================
SUPPORTED_PAIRS = ['EURUSD', 'XAUUSD', 'GBPUSD', 'AUDUSD', 'USDJPY', 'USDCAD', 'USDCHF', 'NZDUSD']

TRAIN_RATIO = 0.8
NUM_CLASSES = 2  # Binary: Buy(0), Sell(1)

# XGBoost params (GPU accelerated)
XGB_PARAMS = {
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'device': 'cuda',
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'early_stopping_rounds': 50,
    'eval_metric': 'logloss',
    'random_state': 42,
    'scale_pos_weight': 1.0,
}

FEATURE_COLUMNS = None
NUM_FEATURES = None


# =============================================================================
# DATA LOADING
# =============================================================================
def find_data_file(pair):
    """Find training data file for a specific pair"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    # Try pair-specific files first, then generic
    filenames = [
        f'training_data_{pair}.csv',
        'training_data.csv' if pair == 'EURUSD' else None,
    ]
    
    paths = []
    for fname in filenames:
        if fname is None:
            continue
        paths.extend([
            f'/kaggle/input/propsai/Prop_firms_AI/{fname}',  # Kaggle dataset path
            f'/kaggle/input/propsai/{fname}',                 # Alternate Kaggle path
            os.path.join(script_dir, fname),
            os.path.join(project_dir, fname),
            os.path.join(project_dir, 'Training_data', fname),
        ])
    
    for p in paths:
        if os.path.exists(p):
            return p
    
    return None


def load_and_preprocess_data(filepath, pair):
    global FEATURE_COLUMNS, NUM_FEATURES
    print(f"Loading {pair} data from {filepath}...")
    df = pd.read_csv(filepath)
    FEATURE_COLUMNS = [col for col in df.columns if col != 'Target']
    NUM_FEATURES = len(FEATURE_COLUMNS)
    print(f"Detected {NUM_FEATURES} features")
    features = df[FEATURE_COLUMNS].values
    targets = df['Target'].values
    print(f"Total samples: {len(df):,}")
    unique, counts = np.unique(targets, return_counts=True)
    print("Class distribution:")
    class_names = ['Buy', 'Sell']
    for cls, count in zip(unique, counts):
        print(f"  {class_names[int(cls)]}: {count:,} ({100*count/len(targets):.1f}%)")
    return features, targets


def chronological_split(features, targets):
    """Simple chronological split - train on older data, test on newer"""
    n_samples = len(features)
    split_idx = int(n_samples * TRAIN_RATIO)
    
    X_train = features[:split_idx]
    y_train = targets[:split_idx]
    X_test = features[split_idx:]
    y_test = targets[split_idx:]
    
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    
    print(f"\nCHRONOLOGICAL SPLIT")
    print(f"Training: {len(X_train):,} samples")
    print(f"Testing:  {len(X_test):,} samples")
    
    return X_train, y_train.astype(np.int32), X_test, y_test.astype(np.int32), scaler


# =============================================================================
# TRAINING
# =============================================================================
def train_model(X_train, y_train, X_test, y_test, pair):
    print(f"\n{'='*60}")
    print(f"TRAINING {pair} MODEL (XGBoost Binary, GPU)")
    print(f"{'='*60}")
    
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    params = XGB_PARAMS.copy()
    params['scale_pos_weight'] = scale_pos_weight
    print(f"Scale pos weight: {scale_pos_weight:.3f}")
    
    model = xgb.XGBClassifier(**params)
    
    print(f"\nStarting training...")
    start_time = time.time()
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=50
    )
    
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.1f}s")
    print(f"Best iteration: {model.best_iteration}")
    
    return model


# =============================================================================
# EVALUATION
# =============================================================================
def evaluate_model(model, X_test, y_test, pair):
    print(f"\n{'='*60}")
    print(f"{pair} TEST EVALUATION")
    print(f"{'='*60}")
    
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    
    class_names = ['Buy', 'Sell']
    cm = confusion_matrix(y_test, preds)
    print("\nConfusion Matrix:")
    print(f"{'':>12} {'Pred Buy':>10} {'Pred Sell':>10}")
    for i, row in enumerate(cm):
        print(f"{'True '+class_names[i]:>12} {row[0]:>10} {row[1]:>10}")
    
    print("\n" + classification_report(y_test, preds, target_names=class_names, digits=3))
    accuracy = accuracy_score(y_test, preds)
    print(f"Overall Accuracy: {100*accuracy:.2f}%")
    
    return accuracy, cm


# =============================================================================
# EXPORT
# =============================================================================
def export_to_onnx(model, output_path, num_features, pair):
    """Export XGBoost model to ONNX via Hummingbird with dynamic input shape"""
    try:
        import torch
        from hummingbird.ml import convert
        
        print(f"Converting {pair} model to ONNX...")
        pytorch_model = convert(model, 'torch')
        dummy_input = torch.randn(1, num_features)
        
        # Export with dynamic batch size for flexibility
        # Input shape will be [batch, num_features] where batch is dynamic
        torch.onnx.export(
            pytorch_model.model,
            dummy_input,
            output_path,
            input_names=['input'],
            output_names=['output', 'probabilities'],
            opset_version=14,
            do_constant_folding=True,
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'},
                'probabilities': {0: 'batch_size'}
            }
        )
        
        print(f"Model exported to {output_path} (features={num_features})")
        verify_onnx_export(output_path, num_features)
        return True
        
    except ImportError as e:
        print(f"Hummingbird not installed: {e}")
        print("Install with: pip install hummingbird-ml")
    except Exception as e:
        print(f"ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Fallback: save native format
    json_path = output_path.replace('.onnx', '.json')
    model.save_model(json_path)
    print(f"Saved native XGBoost model to {json_path}")
    return False


def verify_onnx_export(onnx_path, num_features):
    """Verify ONNX model works"""
    try:
        import onnxruntime as rt
        sess = rt.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name
        dummy_input = np.random.randn(1, num_features).astype(np.float32)
        outputs = sess.run(None, {input_name: dummy_input})
        print(f"ONNX verification passed!")
    except Exception as e:
        print(f"ONNX verification failed: {e}")


def save_scaler_params(scaler, output_path, pair):
    """Save scaler parameters for inference"""
    pd.DataFrame({
        'feature': FEATURE_COLUMNS,
        'min': scaler.data_min_,
        'max': scaler.data_max_,
        'scale': scaler.scale_
    }).to_csv(output_path, index=False)
    print(f"Scaler saved to {output_path}")


# =============================================================================
# SINGLE PAIR TRAINING
# =============================================================================
def train_pair(pair, output_dir):
    """Train model for a single pair"""
    print("\n" + "="*70)
    print(f"  TRAINING {pair}")
    print("="*70)
    
    # Find data file
    data_path = find_data_file(pair)
    if not data_path:
        print(f"ERROR: Data file not found for {pair}")
        print(f"Expected: training_data_{pair}.csv")
        return None
    
    # Load and split data
    features, targets = load_and_preprocess_data(data_path, pair)
    X_train, y_train, X_test, y_test, scaler = chronological_split(features, targets)
    
    # Train
    model = train_model(X_train, y_train, X_test, y_test, pair)
    
    # Evaluate
    accuracy, cm = evaluate_model(model, X_test, y_test, pair)
    
    # Export
    model_path = os.path.join(output_dir, f'ForexXGB_{pair}.onnx')
    scaler_path = os.path.join(output_dir, f'scaler_params_{pair}.csv')
    
    export_to_onnx(model, model_path, NUM_FEATURES, pair)
    save_scaler_params(scaler, scaler_path, pair)
    
    return {
        'pair': pair,
        'accuracy': accuracy,
        'samples': len(features),
        'features': NUM_FEATURES,
        'model_path': model_path,
        'scaler_path': scaler_path
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Train XGBoost models for forex pairs')
    parser.add_argument('--pair', type=str, default='all', 
                        help='Pair to train (e.g., EURUSD) or "all" for all pairs')
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    output_dir = '/kaggle/working' if os.path.exists('/kaggle/working') else os.path.join(project_dir, 'models')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("  MULTI-PAIR FOREX XGBoost TRAINING")
    print("="*70)
    print(f"Output directory: {output_dir}")
    
    # Determine which pairs to train
    if args.pair.lower() == 'all':
        pairs_to_train = SUPPORTED_PAIRS
    else:
        pair = args.pair.upper()
        if pair not in SUPPORTED_PAIRS:
            print(f"ERROR: Unknown pair {pair}")
            print(f"Supported pairs: {', '.join(SUPPORTED_PAIRS)}")
            return
        pairs_to_train = [pair]
    
    print(f"Pairs to train: {', '.join(pairs_to_train)}")
    
    # Train each pair
    results = []
    for pair in pairs_to_train:
        result = train_pair(pair, output_dir)
        if result:
            results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("  TRAINING SUMMARY")
    print("="*70)
    
    if results:
        print(f"\n{'Pair':<10} {'Accuracy':>10} {'Samples':>12} {'Features':>10}")
        print("-" * 45)
        for r in results:
            print(f"{r['pair']:<10} {r['accuracy']*100:>9.2f}% {r['samples']:>12,} {r['features']:>10}")
        
        print(f"\nModels saved to: {output_dir}")
        print("\nGenerated files:")
        for r in results:
            print(f"  - ForexXGB_{r['pair']}.onnx")
            print(f"  - scaler_params_{r['pair']}.csv")
    else:
        print("No models were trained successfully.")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
