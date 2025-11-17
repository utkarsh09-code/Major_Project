#!/usr/bin/env python3
"""
Test script for behavioral cues detection.
"""

import requests
import json
import numpy as np

def test_behavioral_cues():
    """Test the behavioral cues detection endpoint."""
    url = "http://localhost:8000/detect_behavioral_cues"
    
    # Create a test image (simple array)
    test_image = [0] * 1000  # Small test image
    
    payload = {
        "image": test_image
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Behavioral Cues Detection Result:")
            print(json.dumps(result, indent=2))
            
            # Check if key fields are present
            required_fields = [
                'is_blinking', 'blink_rate', 'is_yawning', 
                'yawn_rate', 'movement_level', 'fatigue_score',
                'fatigue_level', 'confidence'
            ]
            
            for field in required_fields:
                if field in result:
                    print(f"✓ {field}: {result[field]}")
                else:
                    print(f"✗ Missing field: {field}")
                    
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Error testing behavioral cues: {e}")

def test_face_detection():
    """Test the face detection endpoint."""
    url = "http://localhost:8000/detect_faces"
    
    # Create a test image
    test_image = [0] * 1000
    
    payload = {
        "image": test_image
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"\nFace Detection Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Face Detection Result:")
            print(json.dumps(result, indent=2))
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Error testing face detection: {e}")

def test_health():
    """Test the health endpoint."""
    url = "http://localhost:8000/health"
    
    try:
        response = requests.get(url, timeout=30)
        print(f"\nHealth Check Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Health Check Result:")
            print(json.dumps(result, indent=2))
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Error testing health check: {e}")

if __name__ == "__main__":
    print("Testing Attentiveness Detection System...")
    print("=" * 50)
    
    test_health()
    test_face_detection()
    test_behavioral_cues()
    
    print("\n" + "=" * 50)
    print("Testing completed!") 