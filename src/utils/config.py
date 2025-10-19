"""
Configuration management for Vehicle Price Prediction System
"""

import yaml
from pathlib import Path
from typing import Any, Dict
from pydantic import BaseModel, Field


class APIConfig(BaseModel):
    """API configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    title: str = "Vehicle Price Prediction API"
    description: str = "Advanced vehicle price prediction system with ML models"
    version: str = "1.0.0"


class DataConfig(BaseModel):
    """Data configuration"""
    raw_path: str = "data/raw"
    processed_path: str = "data/processed"
    models_path: str = "data/models"
    dataset_file: str = "vehicle_dataset.csv"


class ModelConfig(BaseModel):
    """Model configuration"""
    ensemble_weights: Dict[str, float] = Field(default_factory=lambda: {
        "xgboost": 0.30,
        "lightgbm": 0.30,
        "catboost": 0.20,
        "neural_network": 0.15,
        "random_forest": 0.05
    })
    random_state: int = 42
    test_size: float = 0.2
    validation_size: float = 0.2


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = "INFO"
    format: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    file_path: str = "logs/vehicle_price_prediction.log"


class MonitoringConfig(BaseModel):
    """Monitoring configuration"""
    metrics_enabled: bool = True
    prometheus_port: int = 9090


class Config(BaseModel):
    """Main configuration"""
    api: APIConfig = Field(default_factory=APIConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)


def load_config() -> Config:
    """Load configuration from YAML file"""
    config_path = Path("config/config.yaml")
    
    if config_path.exists():
        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file)
        return Config(**config_data)
    else:
        # Return default configuration if file doesn't exist
        return Config()


# Global config instance
_config: Config = None


def get_config() -> Config:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = load_config()
    return _config
