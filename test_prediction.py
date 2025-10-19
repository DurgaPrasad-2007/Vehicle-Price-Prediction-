"""
Test script to verify the prediction fix
"""

import requests
import json

def test_prediction():
    """Test the prediction API"""
    url = "http://localhost:8000/predict"
    
    # Test data
    test_data = {
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
    }
    
    try:
        print("Testing prediction API...")
        response = requests.post(url, json=test_data, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS!")
            print(f"Predicted Price: ${result['predicted_price']:,.2f}")
            print(f"Confidence: {result['confidence_score']:.2%}")
            print(f"Processing Time: {result['processing_time']:.3f}s")
        else:
            print("❌ ERROR!")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_prediction()
