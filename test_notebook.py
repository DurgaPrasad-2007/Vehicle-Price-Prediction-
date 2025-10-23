#!/usr/bin/env python3
"""
Test script to validate the Vehicle Price Prediction evaluation notebook functionality
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def test_imports():
    """Test that all required imports work"""
    try:
        from src.data.preprocessing import get_preprocessor
        from src.models.ensemble import get_model
        from src.utils.config import get_config
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_dataset_loading():
    """Test dataset loading"""
    try:
        df = pd.read_csv('dataset.csv')
        print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return None

def test_preprocessing(df):
    """Test preprocessing pipeline"""
    try:
        from src.data.preprocessing import get_preprocessor
        preprocessor = get_preprocessor()
        df_clean = preprocessor.clean_data(df)
        df_features = preprocessor.engineer_features(df_clean)
        X, y = preprocessor.prepare_training_data(df_features)
        print(f"✅ Preprocessing successful: {X.shape[1]} features, {len(y)} samples")
        return X, y
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return None, None

def test_model_initialization():
    """Test model initialization"""
    try:
        from src.models.ensemble import get_model
        model = get_model()
        print("✅ Model initialization successful")
        return model
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return None

def test_config():
    """Test configuration loading"""
    try:
        from src.utils.config import get_config
        config = get_config()
        print(f"✅ Config loaded: API port {config.api.port}")
        return config
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return None

def main():
    print("🚗 Vehicle Price Prediction - Notebook Validation Test")
    print("=" * 60)

    # Test imports
    if not test_imports():
        return False

    # Test config
    if not test_config():
        return False

    # Test dataset
    df = test_dataset_loading()
    if df is None:
        return False

    # Test preprocessing
    X, y = test_preprocessing(df)
    if X is None or y is None:
        return False

    # Test model
    if not test_model_initialization():
        return False

    print("\\n🎉 All tests passed! The evaluation notebook should work correctly.")
    print("\\n📝 To run the full evaluation:")
    print("1. Open vehicle_price_prediction_evaluation.ipynb in Jupyter")
    print("2. Run all cells in order")
    print("3. The notebook will perform comprehensive evaluation and testing")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
