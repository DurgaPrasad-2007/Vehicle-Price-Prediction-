"""
Data preprocessing and feature engineering for Vehicle Price Prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import joblib
from loguru import logger

from src.utils.config import get_config


class VehiclePreprocessor:
    """Vehicle data preprocessing and feature engineering"""
    
    def __init__(self):
        self.config = get_config()
        self.label_encoders = {}
        self.scalers = {}
        self.feature_columns = []
        
    def load_vehicle_dataset(self) -> pd.DataFrame:
        """Load the vehicle dataset from CSV"""
        try:
            # Try to load from the Vehicle-Price-Prediction directory
            dataset_path = Path("dataset.csv")
            if dataset_path.exists():
                df = pd.read_csv(dataset_path)
                logger.info(f"Loaded vehicle dataset with shape: {df.shape}")
                return df
            else:
                raise FileNotFoundError("Vehicle dataset not found")
        except Exception as e:
            logger.error(f"Failed to load vehicle dataset: {e}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the vehicle dataset"""
        logger.info("Cleaning vehicle dataset...")
        
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Handle missing values
        # For numerical columns, fill with median
        numerical_cols = ['price', 'year', 'cylinders', 'mileage', 'doors']
        for col in numerical_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # For categorical columns, fill with mode
        categorical_cols = ['make', 'model', 'fuel', 'transmission', 'trim', 
                          'body', 'exterior_color', 'interior_color', 'drivetrain']
        for col in categorical_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown')
        
        # Handle engine column - extract engine size if possible
        if 'engine' in df_clean.columns:
            df_clean['engine'] = df_clean['engine'].fillna('Unknown')
            # Extract engine displacement from engine description
            df_clean['engine_displacement'] = df_clean['engine'].str.extract(r'(\d+\.?\d*)\s*L')
            df_clean['engine_displacement'] = pd.to_numeric(df_clean['engine_displacement'], errors='coerce')
            df_clean['engine_displacement'] = df_clean['engine_displacement'].fillna(df_clean['engine_displacement'].median())
        
        # Remove rows with missing price (target variable)
        df_clean = df_clean.dropna(subset=['price'])
        
        # Remove outliers in price (keep prices between 1000 and 500000)
        df_clean = df_clean[(df_clean['price'] >= 1000) & (df_clean['price'] <= 500000)]
        
        # Remove outliers in year (keep years between 1990 and 2025)
        df_clean = df_clean[(df_clean['year'] >= 1990) & (df_clean['year'] <= 2025)]
        
        # Remove outliers in mileage (keep mileage between 0 and 500000)
        df_clean = df_clean[(df_clean['mileage'] >= 0) & (df_clean['mileage'] <= 500000)]
        
        logger.info(f"Cleaned dataset shape: {df_clean.shape}")
        return df_clean
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for vehicle price prediction"""
        logger.info("Engineering features...")
        
        df_features = df.copy()
        
        # Age feature
        df_features['age'] = 2024 - df_features['year']
        
        # Mileage per year
        df_features['mileage_per_year'] = df_features['mileage'] / (df_features['age'] + 1)
        # Handle infinity values
        df_features['mileage_per_year'] = df_features['mileage_per_year'].replace([np.inf, -np.inf], np.nan)
        df_features['mileage_per_year'] = df_features['mileage_per_year'].fillna(df_features['mileage_per_year'].median())
        
        # Engine power features
        if 'cylinders' in df_features.columns:
            df_features['cylinders'] = df_features['cylinders'].fillna(df_features['cylinders'].median())
            df_features['cylinders_category'] = pd.cut(df_features['cylinders'], 
                                                      bins=[0, 4, 6, 8, 12], 
                                                      labels=['Small', 'Medium', 'Large', 'Very Large'])
        
        # Price categories for analysis
        df_features['price_category'] = pd.cut(df_features['price'], 
                                              bins=[0, 15000, 30000, 50000, 100000, float('inf')], 
                                              labels=['Budget', 'Economy', 'Mid-range', 'Luxury', 'Premium'])
        
        # Brand categories (luxury vs regular)
        luxury_brands = ['BMW', 'Mercedes-Benz', 'Audi', 'Lexus', 'Porsche', 'Jaguar', 'Land Rover', 'Infiniti', 'Acura', 'Cadillac', 'Lincoln']
        df_features['is_luxury_brand'] = df_features['make'].isin(luxury_brands).astype(int)
        
        # Body type categories
        if 'body' in df_features.columns:
            df_features['body_category'] = df_features['body'].map({
                'Sedan': 'Car',
                'Coupe': 'Car', 
                'Convertible': 'Car',
                'Hatchback': 'Car',
                'SUV': 'SUV',
                'Crossover': 'SUV',
                'Pickup Truck': 'Truck',
                'Van': 'Van',
                'Wagon': 'Car'
            }).fillna('Other')
        
        # Transmission type
        if 'transmission' in df_features.columns:
            df_features['is_automatic'] = df_features['transmission'].str.contains('Automatic', case=False, na=False).astype(int)
            df_features['is_manual'] = df_features['transmission'].str.contains('Manual', case=False, na=False).astype(int)
            df_features['is_cvt'] = df_features['transmission'].str.contains('CVT', case=False, na=False).astype(int)
        
        # Fuel efficiency categories
        if 'fuel' in df_features.columns:
            df_features['is_electric'] = df_features['fuel'].str.contains('Electric', case=False, na=False).astype(int)
            df_features['is_hybrid'] = df_features['fuel'].str.contains('Hybrid', case=False, na=False).astype(int)
            df_features['is_diesel'] = df_features['fuel'].str.contains('Diesel', case=False, na=False).astype(int)
        
        # Drivetrain features
        if 'drivetrain' in df_features.columns:
            df_features['is_awd'] = df_features['drivetrain'].str.contains('All-wheel|Four-wheel', case=False, na=False).astype(int)
            df_features['is_fwd'] = df_features['drivetrain'].str.contains('Front-wheel', case=False, na=False).astype(int)
            df_features['is_rwd'] = df_features['drivetrain'].str.contains('Rear-wheel', case=False, na=False).astype(int)
        
        # Door count categories
        if 'doors' in df_features.columns:
            df_features['doors'] = df_features['doors'].fillna(df_features['doors'].median())
            df_features['door_category'] = pd.cut(df_features['doors'], 
                                                 bins=[0, 2, 4, 6], 
                                                 labels=['2-door', '4-door', '6+ door'])
        
        # Color popularity (simplified)
        if 'exterior_color' in df_features.columns:
            common_colors = ['White', 'Black', 'Silver', 'Gray', 'Red', 'Blue']
            df_features['is_common_color'] = df_features['exterior_color'].isin(common_colors).astype(int)
        
        # Handle any remaining infinity or NaN values
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        # Only fill numeric columns with median
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        df_features[numeric_cols] = df_features[numeric_cols].fillna(df_features[numeric_cols].median())
        # Fill remaining categorical columns with mode
        categorical_cols = df_features.select_dtypes(exclude=[np.number]).columns
        for col in categorical_cols:
            df_features[col] = df_features[col].fillna(df_features[col].mode()[0] if not df_features[col].mode().empty else 'Unknown')
        
        logger.info(f"Feature engineering completed. Shape: {df_features.shape}")
        return df_features
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features"""
        logger.info("Encoding categorical features...")
        
        df_encoded = df.copy()
        
        categorical_columns = [
            'make', 'model', 'fuel', 'transmission', 'trim', 'body', 
            'exterior_color', 'interior_color', 'drivetrain', 'cylinders_category',
            'price_category', 'body_category', 'door_category'
        ]
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                if fit:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        df_encoded[col] = df_encoded[col].astype(str)
                        le = self.label_encoders[col]
                        # Map unseen categories to a default value
                        df_encoded[col] = df_encoded[col].map(
                            lambda x: x if x in le.classes_ else le.classes_[0]
                        )
                        df_encoded[col] = le.transform(df_encoded[col])
        
        return df_encoded
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data"""
        logger.info("Preparing training data...")
        
        # Clean and engineer features
        df_processed = self.clean_data(df)
        df_features = self.engineer_features(df_processed)
        df_encoded = self.encode_categorical_features(df_features, fit=True)
        
        # Select features for training
        feature_columns = [
            'year', 'age', 'mileage', 'mileage_per_year', 'cylinders', 
            'doors', 'is_luxury_brand', 'is_automatic', 'is_manual', 'is_cvt',
            'is_electric', 'is_hybrid', 'is_diesel', 'is_awd', 'is_fwd', 'is_rwd',
            'is_common_color', 'engine_displacement'
        ]
        
        # Add encoded categorical features
        categorical_features = [
            'make', 'model', 'fuel', 'transmission', 'trim', 'body',
            'exterior_color', 'interior_color', 'drivetrain'
        ]
        
        for col in categorical_features:
            if col in df_encoded.columns:
                feature_columns.append(col)
        
        # Filter available features
        available_features = [col for col in feature_columns if col in df_encoded.columns]
        self.feature_columns = available_features
        
        X = df_encoded[available_features]
        y = df_encoded['price']
        
        logger.info(f"Training data prepared. Features: {len(available_features)}, Samples: {len(X)}")
        return X, y
    
    def save_dataset(self, df: pd.DataFrame) -> None:
        """Save processed dataset"""
        output_path = Path(self.config.data.processed_path) / self.config.data.dataset_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Dataset saved to {output_path}")
    
    def load_dataset(self) -> pd.DataFrame:
        """Load processed dataset"""
        input_path = Path(self.config.data.processed_path) / self.config.data.dataset_file
        if not input_path.exists():
            raise FileNotFoundError(f"Processed dataset not found at {input_path}")
        
        df = pd.read_csv(input_path)
        logger.info(f"Loaded processed dataset with shape: {df.shape}")
        return df
    
    def save_preprocessors(self) -> None:
        """Save preprocessors"""
        models_path = Path(self.config.data.models_path)
        models_path.mkdir(parents=True, exist_ok=True)
        
        # Save label encoders
        for name, encoder in self.label_encoders.items():
            joblib.dump(encoder, models_path / f"{name}_encoder.pkl")
        
        # Save scalers
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, models_path / f"{name}_scaler.pkl")
        
        # Save feature columns
        joblib.dump(self.feature_columns, models_path / "feature_columns.pkl")
        
        logger.info("Preprocessors saved successfully")
    
    def load_preprocessors(self) -> None:
        """Load preprocessors"""
        models_path = Path(self.config.data.models_path)
        
        if not models_path.exists():
            logger.warning("Models directory not found")
            return
        
        # Load label encoders
        for file_path in models_path.glob("*_encoder.pkl"):
            name = file_path.stem.replace("_encoder", "")
            self.label_encoders[name] = joblib.load(file_path)
        
        # Load scalers
        for file_path in models_path.glob("*_scaler.pkl"):
            name = file_path.stem.replace("_scaler", "")
            self.scalers[name] = joblib.load(file_path)
        
        # Load feature columns
        feature_columns_path = models_path / "feature_columns.pkl"
        if feature_columns_path.exists():
            self.feature_columns = joblib.load(feature_columns_path)
        
        logger.info("Preprocessors loaded successfully")


# Global preprocessor instance
_preprocessor: Optional[VehiclePreprocessor] = None


def get_preprocessor() -> VehiclePreprocessor:
    """Get the global preprocessor instance"""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = VehiclePreprocessor()
    return _preprocessor
