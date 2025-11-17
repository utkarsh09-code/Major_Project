#!/usr/bin/env python3
"""Test MediaPipe face detection with workaround."""

import sys
import os

# Suppress TensorFlow import errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    # Try to import MediaPipe solutions directly
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.venv', 'Lib', 'site-packages'))
    
    # Import only what we need
    from mediapipe.python.solutions import face_detection_pb2
    from mediapipe.python.solutions import drawing_utils
    import mediapipe.python.solutions.face_detection as mp_face_detection
    
    print("✓ MediaPipe modules imported successfully")
    
    # Create face detector
    detector = mp_face_detection.FaceDetection()
    print("✓ Face detector created successfully")
    print("\n✅ SUCCESS: MediaPipe works!")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

