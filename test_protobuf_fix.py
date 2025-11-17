#!/usr/bin/env python3
"""Test protobuf compatibility with MediaPipe."""

import sys

try:
    print("Testing MediaPipe with protobuf...")
    import mediapipe as mp
    print("✓ MediaPipe imported")
    
    mp_face = mp.solutions.face_detection
    print("✓ Face detection module loaded")
    
    detector = mp_face.FaceDetection()
    print("✓ Face detector created successfully")
    print("\n✅ SUCCESS: No protobuf errors!")
    sys.exit(0)
    
except AttributeError as e:
    if "GetPrototype" in str(e):
        print(f"\n❌ ERROR: Protobuf compatibility issue: {e}")
        print("This means protobuf version is incompatible with MediaPipe.")
        sys.exit(1)
    else:
        raise
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    sys.exit(1)



