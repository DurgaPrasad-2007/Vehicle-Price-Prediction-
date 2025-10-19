"""
Monitoring and metrics collection for Vehicle Price Prediction System
"""

import time
from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from loguru import logger


class MetricsCollector:
    """Metrics collector for the vehicle price prediction system"""
    
    def __init__(self):
        self.start_time = time.time()
        
        # Prometheus metrics
        self.predictions_total = Counter(
            'vehicle_predictions_total',
            'Total number of vehicle price predictions',
            ['client_ip']
        )
        
        self.prediction_duration = Histogram(
            'vehicle_prediction_duration_seconds',
            'Time spent on vehicle price predictions',
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        self.predicted_price_histogram = Histogram(
            'vehicle_predicted_price',
            'Distribution of predicted vehicle prices',
            buckets=[10000, 25000, 50000, 75000, 100000, 150000, 200000, 300000, 500000]
        )
        
        self.active_connections = Gauge(
            'vehicle_prediction_active_connections',
            'Number of active connections'
        )
        
        self.models_loaded = Gauge(
            'vehicle_prediction_models_loaded',
            'Whether models are loaded (1=loaded, 0=not loaded)'
        )
        
        # Internal counters
        self._total_predictions = 0
        self._total_processing_time = 0.0
        self._prediction_history = []
        
        logger.info("Metrics collector initialized")
    
    def record_prediction(self, predicted_price: float, processing_time: float, client_ip: str = "unknown") -> None:
        """Record a prediction event"""
        self.predictions_total.labels(client_ip=client_ip).inc()
        self.prediction_duration.observe(processing_time)
        self.predicted_price_histogram.observe(predicted_price)
        
        # Update internal counters
        self._total_predictions += 1
        self._total_processing_time += processing_time
        self._prediction_history.append({
            'timestamp': time.time(),
            'predicted_price': predicted_price,
            'processing_time': processing_time,
            'client_ip': client_ip
        })
        
        # Keep only last 1000 predictions in memory
        if len(self._prediction_history) > 1000:
            self._prediction_history = self._prediction_history[-1000:]
        
        logger.debug(f"Recorded prediction: ${predicted_price:.2f}, {processing_time:.3f}s")
    
    def set_models_loaded(self, loaded: bool) -> None:
        """Set the models loaded status"""
        self.models_loaded.set(1 if loaded else 0)
    
    def get_total_predictions(self) -> int:
        """Get total number of predictions"""
        return self._total_predictions
    
    def get_average_processing_time(self) -> float:
        """Get average processing time"""
        if self._total_predictions == 0:
            return 0.0
        return self._total_processing_time / self._total_predictions
    
    def get_prediction_history(self, limit: int = 100) -> list:
        """Get recent prediction history"""
        return self._prediction_history[-limit:] if self._prediction_history else []
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        return generate_latest()
    
    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get metrics as dictionary"""
        return {
            'total_predictions': self._total_predictions,
            'average_processing_time': self.get_average_processing_time(),
            'uptime_seconds': time.time() - self.start_time,
            'recent_predictions': self.get_prediction_history(10)
        }


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector
