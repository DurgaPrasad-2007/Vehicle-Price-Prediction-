"""
MLOps components for Vehicle Price Prediction System
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import joblib
from pathlib import Path
import shap
import lime
import lime.lime_tabular
from loguru import logger

from src.models.ensemble import get_model
from src.data.preprocessing import get_preprocessor


class ModelExplainer:
    """Model explainability using SHAP and LIME"""
    
    def __init__(self):
        self.model = get_model()
        self.preprocessor = get_preprocessor()
        self.shap_explainer = None
        self.lime_explainer = None
        self.is_initialized = False
        
    def initialize_explainers(self, X_sample: pd.DataFrame) -> None:
        """Initialize SHAP and LIME explainers"""
        try:
            logger.info("Initializing model explainers...")
            
            # Initialize SHAP explainer
            if self.model.is_trained:
                # Use a sample of the data for SHAP
                sample_size = min(100, len(X_sample))
                X_sample_shap = X_sample.sample(n=sample_size, random_state=42)
                
                # Create SHAP explainer for ensemble (using XGBoost as proxy)
                if 'xgboost' in self.model.models:
                    self.shap_explainer = shap.TreeExplainer(self.model.models['xgboost'])
                    logger.info("SHAP explainer initialized")
            
            # Initialize LIME explainer
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                X_sample.values,
                feature_names=X_sample.columns.tolist(),
                mode='regression',
                random_state=42
            )
            logger.info("LIME explainer initialized")
            
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize explainers: {e}")
            self.is_initialized = False
    
    def explain_prediction_shap(self, vehicle_data: Dict[str, Any]) -> Dict[str, Any]:
        """Explain prediction using SHAP"""
        if not self.is_initialized or self.shap_explainer is None:
            return {"error": "SHAP explainer not initialized"}
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame([vehicle_data])
            
            # Ensure all required features are present
            for col in self.model.feature_columns:
                if col not in df.columns:
                    df[col] = 0
            
            # Reorder columns
            df = df[self.model.feature_columns]
            
            # Get SHAP values
            shap_values = self.shap_explainer.shap_values(df)
            
            # Create explanation
            explanation = {
                "feature_importance": {},
                "base_value": float(self.shap_explainer.expected_value),
                "prediction": float(np.sum(shap_values[0]) + self.shap_explainer.expected_value)
            }
            
            # Map feature importance
            for i, feature in enumerate(self.model.feature_columns):
                explanation["feature_importance"][feature] = float(shap_values[0][i])
            
            return explanation
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return {"error": f"SHAP explanation failed: {str(e)}"}
    
    def explain_prediction_lime(self, vehicle_data: Dict[str, Any]) -> Dict[str, Any]:
        """Explain prediction using LIME"""
        if not self.is_initialized or self.lime_explainer is None:
            return {"error": "LIME explainer not initialized"}
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame([vehicle_data])
            
            # Ensure all required features are present
            for col in self.model.feature_columns:
                if col not in df.columns:
                    df[col] = 0
            
            # Reorder columns
            df = df[self.model.feature_columns]
            
            # Define prediction function for LIME
            def predict_fn(X):
                predictions = []
                for row in X:
                    row_dict = dict(zip(self.model.feature_columns, row))
                    pred = self.model.predict_single(row_dict)
                    predictions.append(pred)
                return np.array(predictions)
            
            # Get LIME explanation
            explanation = self.lime_explainer.explain_instance(
                df.values[0],
                predict_fn,
                num_features=min(10, len(self.model.feature_columns))
            )
            
            # Convert to dictionary
            lime_explanation = {
                "feature_importance": {},
                "prediction": float(explanation.prediction),
                "intercept": float(explanation.intercept)
            }
            
            # Map feature importance
            for feature_idx, importance in explanation.as_list():
                lime_explanation["feature_importance"][feature_idx] = importance
            
            return lime_explanation
            
        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            return {"error": f"LIME explanation failed: {str(e)}"}
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get overall feature importance from ensemble models"""
        if not self.model.is_trained:
            return {}
        
        try:
            feature_importance = {}
            
            # Get feature importance from tree-based models
            for name, model in self.model.models.items():
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    for i, feature in enumerate(self.model.feature_columns):
                        if feature not in feature_importance:
                            feature_importance[feature] = 0
                        feature_importance[feature] += importances[i] * self.model.ensemble_weights.get(name, 0)
            
            # Normalize importance scores
            total_importance = sum(feature_importance.values())
            if total_importance > 0:
                feature_importance = {k: v/total_importance for k, v in feature_importance.items()}
            
            return feature_importance
            
        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            return {}
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        if not self.model.is_trained:
            return {"error": "Models not trained"}
        
        try:
            # This would typically come from model evaluation
            # For now, return placeholder data
            return {
                "ensemble_rmse": 0.0,
                "ensemble_r2": 0.0,
                "individual_model_performance": {
                    "xgboost": {"rmse": 0.0, "r2": 0.0},
                    "lightgbm": {"rmse": 0.0, "r2": 0.0},
                    "catboost": {"rmse": 0.0, "r2": 0.0},
                    "neural_network": {"rmse": 0.0, "r2": 0.0},
                    "random_forest": {"rmse": 0.0, "r2": 0.0}
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get model performance: {e}")
            return {"error": f"Failed to get model performance: {str(e)}"}


# Global explainer instance
_explainer: Optional[ModelExplainer] = None


def get_explainer() -> ModelExplainer:
    """Get the global explainer instance"""
    global _explainer
    if _explainer is None:
        _explainer = ModelExplainer()
    return _explainer
