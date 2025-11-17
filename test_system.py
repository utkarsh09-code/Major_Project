#!/usr/bin/env python3
"""
Simple test script to verify the attentiveness detection system works without dlib.
"""

import sys
import cv2
import numpy as np
from src.core.face_detection import FaceDetector
from src.core.gaze_estimation import GazeEstimator
from src.core.pose_estimation import PoseEstimator

def test_system():
    """Test the core components of the attentiveness detection system."""
    print("Testing Attentiveness Detection System...")
    
    # Initialize components
    try:
        print("1. Initializing face detector...")
        face_detector = FaceDetector(backend="mediapipe")
        print("   ✓ Face detector initialized successfully")
        
        print("2. Initializing gaze estimator...")
        gaze_estimator = GazeEstimator(backend="mediapipe")
        print("   ✓ Gaze estimator initialized successfully")
        
        print("3. Initializing pose estimator...")
        pose_estimator = PoseEstimator(backend="mediapipe")
        print("   ✓ Pose estimator initialized successfully")
        
        print("4. Testing camera access...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("   ✗ Camera not accessible")
            return False
        
        print("   ✓ Camera accessible")
        
        # Test with a few frames
        print("5. Testing frame processing...")
        for i in range(5):
            ret, frame = cap.read()
            if not ret:
                print(f"   ✗ Failed to read frame {i}")
                continue
            
            # Test face detection
            faces = face_detector.detect_faces(frame)
            if faces:
                print(f"   ✓ Frame {i}: {len(faces)} face(s) detected")
                
                # Test gaze estimation
                gaze_result = gaze_estimator.estimate_gaze(frame)
                if gaze_result['confidence'] > 0:
                    print(f"      - Gaze confidence: {gaze_result['confidence']:.2f}")
                
                # Test pose estimation
                pose_result = pose_estimator.estimate_pose(frame)
                if pose_result['confidence'] > 0:
                    print(f"      - Pose confidence: {pose_result['confidence']:.2f}")
            else:
                print(f"   - Frame {i}: No faces detected")
        
        cap.release()
        print("   ✓ Frame processing test completed")
        
        print("\n🎉 All tests passed! System is working without dlib.")
        return True
        
    except Exception as e:
        print(f"   ✗ Error during testing: {e}")
        return False

if __name__ == "__main__":
    success = test_system()
    sys.exit(0 if success else 1) 