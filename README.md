# Vehicle Price Prediction System

A comprehensive vehicle price prediction system using machine learning to estimate vehicle prices based on specifications, make, model, and other features.

## Overview

This project implements a vehicle price prediction system that analyzes vehicle features to predict accurate prices. It uses ensemble machine learning models to make predictions and provides explanations for its decisions.

## Features

- **Accurate Price Prediction**: Predicts vehicle prices using multiple ML models
- **Multiple ML Models**: XGBoost, LightGBM, CatBoost, Neural Networks, Random Forest
- **Model Explainability**: SHAP and LIME explanations for predictions
- **REST API**: Complete API for integration
- **Modern Web Frontend**: Clean, responsive UI for testing
- **Real-time Predictions**: Fast prediction with confidence scores
- **Interactive Forms**: Easy-to-use specification input forms
- **Live Statistics**: Monitoring dashboard with metrics
- **Comprehensive Logging**: Detailed logging and monitoring

## Architecture

```
Data → Features → Models → API → Response
```

## Technology Stack

- **Python 3.11**
- **FastAPI** for the API
- **XGBoost, LightGBM, CatBoost** for ML models
- **TensorFlow** for neural networks
- **SHAP & LIME** for explanations
- **Docker** for deployment
- **Prometheus & Grafana** for monitoring

## Dataset Features

The system analyzes the following vehicle features:

- **name**: Full name of the vehicle
- **make**: Manufacturer (e.g., Ford, Toyota, BMW)
- **model**: Model name
- **year**: Manufacturing year
- **price**: Target variable (price in USD)
- **engine**: Engine specifications
- **cylinders**: Number of cylinders
- **fuel**: Fuel type (Gasoline, Diesel, Electric, Hybrid)
- **mileage**: Vehicle mileage
- **transmission**: Transmission type (Automatic, Manual, CVT)
- **trim**: Trim level
- **body**: Body style (Sedan, SUV, Coupe, etc.)
- **doors**: Number of doors
- **exterior_color**: Exterior color
- **interior_color**: Interior color
- **drivetrain**: Drivetrain type (FWD, RWD, AWD, 4WD)

## Getting Started

### Prerequisites
- Python 3.11+
- Poetry (for dependency management)

### Installation
```bash
# Install Poetry (if not installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Run the system
poetry run devrun
```

### Quick Start with Frontend
```bash
# Start the complete system (backend + frontend on port 8000)
poetry run devrun

# Or start in serve mode
poetry run python main.py --mode serve
```

### Access the Application
- **Frontend UI**: http://localhost:8000 - Modern web interface for testing
- **Backend API**: http://localhost:8000/api - REST API endpoints
- **API Documentation**: http://localhost:8000/docs - Interactive API docs
- **Health Check**: http://localhost:8000/health - System status
- **Metrics**: http://localhost:8000/metrics - Prometheus metrics

## Data Processing Pipeline

### Feature Engineering
Creates features like:
- Vehicle age calculation
- Mileage per year
- Engine power categories
- Brand luxury classification
- Body type categories
- Transmission type flags
- Fuel efficiency categories
- Drivetrain features
- Color popularity indicators

### Model Training
Trains multiple models and combines them:
- **XGBoost** (30%)
- **LightGBM** (30%) 
- **CatBoost** (20%)
- **Neural Network** (15%)
- **Random Forest** (5%)

## API Usage

### Single Vehicle Price Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "make": "Toyota",
    "model": "Camry",
    "year": 2020,
    "mileage": 25000,
    "cylinders": 4,
    "fuel": "Gasoline",
    "transmission": "Automatic",
    "trim": "LE",
    "body": "Sedan",
    "doors": 4,
    "exterior_color": "White",
    "interior_color": "Black",
    "drivetrain": "Front-wheel Drive"
  }'
```

### Health Check
```bash
curl http://localhost:8000/health
```

### Get Statistics
```bash
curl http://localhost:8000/stats
```

## Testing

Run the test suite:
```bash
poetry run pytest tests/ -v
```

Run a simple system test:
```bash
python test_system.py
```

## Development

### Code Quality
```bash
# Format code
poetry run black src/
poetry run isort src/

# Lint code
poetry run flake8 src/

# Type checking
poetry run mypy src/

# Security scan
poetry run safety check
poetry run bandit -r src/
```

### Pre-commit Hooks
```bash
poetry run pre-commit install
```

## Docker Deployment

### Single Container
```bash
docker build -t vehicle-price-prediction .
docker run -p 8000:8000 vehicle-price-prediction
```

### Multi-Service with Monitoring
```bash
docker-compose up -d
```

This will start:
- **Vehicle Price Prediction API** on port 8000
- **Prometheus** on port 9090
- **Grafana** on port 3000

## Project Structure

```
vehicle-price-prediction/
├── src/                          # Source code
│   ├── data/                     # Data processing
│   ├── models/                   # ML models
│   ├── api/                      # API endpoints
│   ├── monitoring/               # Monitoring
│   ├── mlops/                    # MLOps components
│   └── utils/                    # Utilities
├── config/                       # Configuration files
├── data/                         # Data storage
│   ├── raw/                      # Raw data
│   ├── processed/                # Processed data
│   └── models/                   # Trained models
├── logs/                         # Log files
├── static/                       # Static files
├── tests/                        # Test suite
├── pyproject.toml               # Poetry configuration
├── Dockerfile                   # Container config
├── docker-compose.yml           # Multi-service setup
├── prometheus.yml               # Prometheus config
└── main.py                      # Application entry point
```

## Model Performance

The ensemble model achieves:
- **RMSE**: Low root mean square error
- **MAE**: Low mean absolute error
- **R²**: High coefficient of determination
- **Fast Inference**: Sub-second prediction times

## Monitoring

The system includes comprehensive monitoring:
- **Prometheus Metrics**: Performance and usage metrics
- **Grafana Dashboards**: Visual monitoring
- **Health Checks**: System status monitoring
- **Logging**: Detailed application logs

## Known Issues

- Models need to be trained before first use
- No authentication on API endpoints (TODO)
- Limited input validation (TODO)
- No rate limiting implemented (TODO)
- Hardcoded configuration values (TODO)

## TODO

- [ ] Add proper authentication
- [ ] Implement rate limiting
- [ ] Add more comprehensive tests
- [ ] Improve error handling
- [ ] Add model versioning
- [ ] Implement proper monitoring
- [ ] Add database integration
- [ ] Implement caching
- [ ] Add batch prediction endpoint
- [ ] Implement model retraining pipeline

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support, please open an issue in the repository.
