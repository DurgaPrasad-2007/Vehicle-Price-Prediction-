"""
Ensemble ML models for Vehicle Price Prediction
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
# import tensorflow as tf  # Temporarily disabled due to compatibility issues
from loguru import logger

from src.utils.config import get_config


class VehicleEnsembleModel:
    """Ensemble model for vehicle price prediction"""
    
    def __init__(self):
        self.config = get_config()
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.ensemble_weights = self.config.models.ensemble_weights
        self.is_trained = False
        
    def _create_xgboost_model(self) -> xgb.XGBRegressor:
        """Create XGBoost model"""
        return xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.config.models.random_state,
            n_jobs=-1
        )
    
    def _create_lightgbm_model(self) -> lgb.LGBMRegressor:
        """Create LightGBM model"""
        return lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.config.models.random_state,
            n_jobs=-1,
            verbose=-1
        )
    
    def _create_catboost_model(self) -> CatBoostRegressor:
        """Create CatBoost model"""
        return CatBoostRegressor(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            random_seed=self.config.models.random_state,
            verbose=False
        )
    
    def _create_neural_network_model(self, input_dim: int):
        """Create neural network model using MLPRegressor"""
        from sklearn.neural_network import MLPRegressor
        
        model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=500,
            random_state=self.config.models.random_state
        )
        
        return model
    
    def _create_random_forest_model(self) -> RandomForestRegressor:
        """Create Random Forest model"""
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=self.config.models.random_state,
            n_jobs=-1
        )
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, Any], pd.DataFrame, pd.Series]:
        """Train all models"""
        logger.info("Training ensemble models...")
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.models.test_size,
            random_state=self.config.models.random_state
        )
        
        # Further split training data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=self.config.models.validation_size,
            random_state=self.config.models.random_state
        )
        
        # Create scalers
        self.scalers['standard'] = StandardScaler()
        self.scalers['robust'] = RobustScaler()
        
        # Scale features
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_val_scaled = self.scalers['standard'].transform(X_val)
        X_test_scaled = self.scalers['standard'].transform(X_test)
        
        # Train XGBoost
        logger.info("Training XGBoost...")
        xgb_model = self._create_xgboost_model()
        xgb_model.fit(X_train, y_train)
        self.models['xgboost'] = xgb_model
        
        # Train LightGBM
        logger.info("Training LightGBM...")
        lgb_model = self._create_lightgbm_model()
        lgb_model.fit(X_train, y_train)
        self.models['lightgbm'] = lgb_model
        
        # Train CatBoost
        logger.info("Training CatBoost...")
        cat_model = self._create_catboost_model()
        cat_model.fit(X_train, y_train)
        self.models['catboost'] = cat_model
        
        # Train Random Forest
        logger.info("Training Random Forest...")
        rf_model = self._create_random_forest_model()
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model
        
        # Train Neural Network
        logger.info("Training Neural Network...")
        nn_model = self._create_neural_network_model(X_train_scaled.shape[1])
        nn_model.fit(X_train_scaled, y_train)
        self.models['neural_network'] = nn_model
        
        # Set trained flag before evaluation
        self.is_trained = True
        
        # Save preprocessors (use the same preprocessor instance from training)
        # The preprocessors are already saved in the main training function
        
        # Evaluate models
        results = self._evaluate_models(X_test, X_test_scaled, y_test)
        
        logger.info("All models trained successfully")
        
        return results, X_test, y_test
    
    def _evaluate_models(self, X_test: pd.DataFrame, X_test_scaled: np.ndarray, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate all models"""
        logger.info("Evaluating models...")
        
        results = {}
        
        for name, model in self.models.items():
            if name == 'neural_network':
                y_pred = model.predict(X_test_scaled).flatten()
            else:
                y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            results[name] = {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': rmse
            }
            
            logger.info(f"{name.upper()} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
        
        # Evaluate ensemble
        ensemble_pred = self.predict_ensemble(X_test)
        ensemble_mse = mean_squared_error(y_test, ensemble_pred)
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        ensemble_rmse = np.sqrt(ensemble_mse)
        
        results['ensemble'] = {
            'mse': ensemble_mse,
            'mae': ensemble_mae,
            'r2': ensemble_r2,
            'rmse': ensemble_rmse
        }
        
        logger.info(f"ENSEMBLE - RMSE: {ensemble_rmse:.2f}, MAE: {ensemble_mae:.2f}, R²: {ensemble_r2:.4f}")
        
        return results
    
    def predict_ensemble(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble prediction"""
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        predictions = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            if name == 'neural_network':
                X_scaled = self.scalers['standard'].transform(X)
                pred = model.predict(X_scaled)
            else:
                pred = model.predict(X)
            
            predictions[name] = pred
        
        # Weighted ensemble prediction
        ensemble_pred = np.zeros(len(X))
        for name, weight in self.ensemble_weights.items():
            if name in predictions:
                ensemble_pred += weight * predictions[name]
        
        return ensemble_pred
    
    def predict_single(self, vehicle_data: Dict[str, Any]) -> float:
        """Predict price for a single vehicle"""
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        # Convert to DataFrame
        df = pd.DataFrame([vehicle_data])
        
        # Ensure all required features are present
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0  # Default value for missing features
        
        # Reorder columns to match training data
        df = df[self.feature_columns]
        
        # Simple encoding for categorical features (since encoders aren't saved)
        categorical_columns = ['make', 'model', 'fuel', 'transmission', 'trim', 'body', 
                              'exterior_color', 'interior_color', 'drivetrain']
        
        for col in categorical_columns:
            if col in df.columns:
                # Simple hash-based encoding
                df[col] = df[col].astype(str).apply(lambda x: hash(x) % 1000)
        
        # Make prediction
        prediction = self.predict_ensemble(df)
        
        return float(prediction[0])
    
    def save_models(self) -> None:
        """Save all models"""
        models_path = Path(self.config.data.models_path)
        models_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual models
        for name, model in self.models.items():
            joblib.dump(model, models_path / f"{name}_model.pkl")
        
        # Save scalers
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, models_path / f"{name}_scaler.pkl")
        
        # Save ensemble weights
        joblib.dump(self.ensemble_weights, models_path / "ensemble_weights.pkl")
        
        # Save feature columns
        joblib.dump(self.feature_columns, models_path / "feature_columns.pkl")
        
        # Save training status
        joblib.dump(self.is_trained, models_path / "is_trained.pkl")
        
        logger.info("All models saved successfully")
    
    def load_models(self) -> None:
        """Load all models"""
        models_path = Path(self.config.data.models_path)
        
        if not models_path.exists():
            logger.warning("Models directory not found")
            return
        
        # Load individual models
        for name in ['xgboost', 'lightgbm', 'catboost', 'random_forest', 'neural_network']:
            model_path = models_path / f"{name}_model.pkl"
            if model_path.exists():
                self.models[name] = joblib.load(model_path)
        
        # Load scalers
        for name in ['standard', 'robust']:
            scaler_path = models_path / f"{name}_scaler.pkl"
            if scaler_path.exists():
                self.scalers[name] = joblib.load(scaler_path)
        
        # Load ensemble weights
        weights_path = models_path / "ensemble_weights.pkl"
        if weights_path.exists():
            self.ensemble_weights = joblib.load(weights_path)
        
        # Load feature columns
        features_path = models_path / "feature_columns.pkl"
        if features_path.exists():
            self.feature_columns = joblib.load(features_path)
        
        # Load training status
        status_path = models_path / "is_trained.pkl"
        if status_path.exists():
            self.is_trained = joblib.load(status_path)
        
        logger.info("All models loaded successfully")


# Global model instance
_model: Optional[VehicleEnsembleModel] = None


def get_model() -> VehicleEnsembleModel:
    """Get the global model instance"""
    global _model
    if _model is None:
        _model = VehicleEnsembleModel()
    return _model
