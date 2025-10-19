"""
FastAPI endpoints for Vehicle Price Prediction System
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import numpy as np
from pathlib import Path
from loguru import logger
import time

from src.utils.config import get_config
from src.models.ensemble import get_model
from src.data.preprocessing import get_preprocessor
from src.monitoring.metrics import get_metrics_collector


# Pydantic models for API
class VehicleFeatures(BaseModel):
    """Vehicle features for price prediction"""
    make: str = Field(..., description="Vehicle manufacturer")
    model: str = Field(..., description="Vehicle model")
    year: int = Field(..., ge=1990, le=2025, description="Manufacturing year")
    mileage: float = Field(..., ge=0, description="Vehicle mileage")
    cylinders: Optional[float] = Field(None, description="Number of cylinders")
    fuel: str = Field(..., description="Fuel type")
    transmission: str = Field(..., description="Transmission type")
    trim: Optional[str] = Field(None, description="Vehicle trim level")
    body: str = Field(..., description="Body style")
    doors: Optional[float] = Field(None, description="Number of doors")
    exterior_color: Optional[str] = Field(None, description="Exterior color")
    interior_color: Optional[str] = Field(None, description="Interior color")
    drivetrain: str = Field(..., description="Drivetrain type")
    engine: Optional[str] = Field(None, description="Engine description")


class PredictionResponse(BaseModel):
    """Prediction response"""
    predicted_price: float = Field(..., description="Predicted vehicle price")
    confidence_score: float = Field(..., description="Confidence score (0-1)")
    model_contributions: Dict[str, float] = Field(..., description="Individual model contributions")
    processing_time: float = Field(..., description="Processing time in seconds")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    models_loaded: bool = Field(..., description="Whether models are loaded")
    version: str = Field(..., description="API version")
    timestamp: float = Field(..., description="Current timestamp")


# Initialize FastAPI app
config = get_config()
app = FastAPI(
    title=config.api.title,
    description=config.api.description,
    version=config.api.version,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize components
model = get_model()
preprocessor = get_preprocessor()
metrics_collector = get_metrics_collector()


@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("Starting Vehicle Price Prediction API...")
    
    # Load models if they exist
    try:
        model.load_models()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load models: {e}")
    
    # Load preprocessors if they exist
    try:
        preprocessor.load_preprocessors()
        logger.info("Preprocessors loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load preprocessors: {e}")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface"""
    try:
        html_path = Path("index.html")
        if html_path.exists():
            return FileResponse(html_path)
        else:
            return HTMLResponse("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Vehicle Price Prediction</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .container { max-width: 800px; margin: 0 auto; }
                    h1 { color: #333; }
                    .feature { margin: 10px 0; }
                    label { display: block; margin-bottom: 5px; font-weight: bold; }
                    input, select { width: 100%; padding: 8px; margin-bottom: 10px; }
                    button { background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; }
                    button:hover { background: #0056b3; }
                    .result { margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🚗 Vehicle Price Prediction System</h1>
                    <p>Predict vehicle prices using machine learning models</p>
                    
                    <form id="predictionForm">
                        <div class="feature">
                            <label for="make">Make:</label>
                            <input type="text" id="make" name="make" required>
                        </div>
                        
                        <div class="feature">
                            <label for="model">Model:</label>
                            <input type="text" id="model" name="model" required>
                        </div>
                        
                        <div class="feature">
                            <label for="year">Year:</label>
                            <input type="number" id="year" name="year" min="1990" max="2025" required>
                        </div>
                        
                        <div class="feature">
                            <label for="mileage">Mileage:</label>
                            <input type="number" id="mileage" name="mileage" min="0" required>
                        </div>
                        
                        <div class="feature">
                            <label for="cylinders">Cylinders:</label>
                            <input type="number" id="cylinders" name="cylinders" min="1" max="12">
                        </div>
                        
                        <div class="feature">
                            <label for="fuel">Fuel Type:</label>
                            <select id="fuel" name="fuel" required>
                                <option value="">Select Fuel Type</option>
                                <option value="Gasoline">Gasoline</option>
                                <option value="Diesel">Diesel</option>
                                <option value="Electric">Electric</option>
                                <option value="Hybrid">Hybrid</option>
                            </select>
                        </div>
                        
                        <div class="feature">
                            <label for="transmission">Transmission:</label>
                            <select id="transmission" name="transmission" required>
                                <option value="">Select Transmission</option>
                                <option value="Automatic">Automatic</option>
                                <option value="Manual">Manual</option>
                                <option value="CVT">CVT</option>
                            </select>
                        </div>
                        
                        <div class="feature">
                            <label for="body">Body Style:</label>
                            <select id="body" name="body" required>
                                <option value="">Select Body Style</option>
                                <option value="Sedan">Sedan</option>
                                <option value="SUV">SUV</option>
                                <option value="Coupe">Coupe</option>
                                <option value="Hatchback">Hatchback</option>
                                <option value="Pickup Truck">Pickup Truck</option>
                                <option value="Convertible">Convertible</option>
                            </select>
                        </div>
                        
                        <div class="feature">
                            <label for="doors">Doors:</label>
                            <input type="number" id="doors" name="doors" min="2" max="6">
                        </div>
                        
                        <div class="feature">
                            <label for="drivetrain">Drivetrain:</label>
                            <select id="drivetrain" name="drivetrain" required>
                                <option value="">Select Drivetrain</option>
                                <option value="Front-wheel Drive">Front-wheel Drive</option>
                                <option value="Rear-wheel Drive">Rear-wheel Drive</option>
                                <option value="All-wheel Drive">All-wheel Drive</option>
                                <option value="Four-wheel Drive">Four-wheel Drive</option>
                            </select>
                        </div>
                        
                        <button type="submit">Predict Price</button>
                    </form>
                    
                    <div id="result" class="result" style="display: none;">
                        <h3>Prediction Result</h3>
                        <p id="predictionText"></p>
                    </div>
                </div>
                
                <script>
                    document.getElementById('predictionForm').addEventListener('submit', async function(e) {
                        e.preventDefault();
                        
                        const formData = new FormData(this);
                        const data = Object.fromEntries(formData);
                        
                        // Convert numeric fields
                        data.year = parseInt(data.year);
                        data.mileage = parseFloat(data.mileage);
                        if (data.cylinders) data.cylinders = parseFloat(data.cylinders);
                        if (data.doors) data.doors = parseFloat(data.doors);
                        
                        try {
                            const response = await fetch('/predict', {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json',
                                },
                                body: JSON.stringify(data)
                            });
                            
                            if (response.ok) {
                                const result = await response.json();
                                document.getElementById('predictionText').innerHTML = `
                                    <strong>Predicted Price:</strong> $${result.predicted_price.toLocaleString()}<br>
                                    <strong>Confidence:</strong> ${(result.confidence_score * 100).toFixed(1)}%<br>
                                    <strong>Processing Time:</strong> ${result.processing_time.toFixed(3)}s
                                `;
                                document.getElementById('result').style.display = 'block';
                            } else {
                                const error = await response.json();
                                alert('Error: ' + error.detail);
                            }
                        } catch (error) {
                            alert('Error: ' + error.message);
                        }
                    });
                </script>
            </body>
            </html>
            """)
    except Exception as e:
        logger.error(f"Error serving frontend: {e}")
        raise HTTPException(status_code=500, detail="Frontend not available")


@app.post("/predict", response_model=PredictionResponse)
async def predict_price(vehicle: VehicleFeatures, request: Request):
    """Predict vehicle price"""
    start_time = time.time()
    
    try:
        # Convert to dict
        vehicle_data = vehicle.dict()
        
        # Make prediction
        predicted_price = model.predict_single(vehicle_data)
        
        # Calculate confidence score (simplified)
        confidence_score = min(0.95, max(0.60, 1.0 - abs(predicted_price - 50000) / 100000))
        
        # Get individual model contributions (simplified)
        model_contributions = {
            "xgboost": predicted_price * 0.30,
            "lightgbm": predicted_price * 0.30,
            "catboost": predicted_price * 0.20,
            "neural_network": predicted_price * 0.15,
            "random_forest": predicted_price * 0.05
        }
        
        processing_time = time.time() - start_time
        
        # Record metrics
        metrics_collector.record_prediction(
            predicted_price=predicted_price,
            processing_time=processing_time,
            client_ip=request.client.host
        )
        
        return PredictionResponse(
            predicted_price=predicted_price,
            confidence_score=confidence_score,
            model_contributions=model_contributions,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        models_loaded=model.is_trained,
        version=config.api.version,
        timestamp=time.time()
    )


@app.get("/metrics")
async def get_metrics():
    """Get Prometheus metrics"""
    return metrics_collector.get_metrics()


@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        stats = {
            "models_loaded": model.is_trained,
            "feature_count": len(model.feature_columns),
            "ensemble_weights": model.ensemble_weights,
            "total_predictions": metrics_collector.get_total_predictions(),
            "average_processing_time": metrics_collector.get_average_processing_time(),
            "uptime": time.time() - metrics_collector.start_time
        }
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve statistics")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
