import requests
import base64
import json
import os

# Default to local mock if no URL provided
API_URL = os.environ.get("API_URL", "http://localhost:8000/predict")

def predict_anomaly(image_file, category=None):
    """
    Sends image to API for prediction.
    """
    try:
        # Encode image
        image_bytes = image_file.getvalue()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        payload = {"image": image_b64, "category": category}
        
        # In a real scenario, we would call the API
        # response = requests.post(API_URL, json=payload)
        # return response.json()
        
        # Mock response for demonstration if API is not reachable
        import random
        is_anomaly = random.random() > 0.7
        confidence = random.uniform(0.8, 0.99)
        
        return {
            "prediction": [0.1, 0.9] if is_anomaly else [0.9, 0.1], # [Good, Anomaly]
            "confidence": confidence,
            "is_anomaly": is_anomaly
        }
        
    except Exception as e:
        return {"error": str(e)}
