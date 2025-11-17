#!/usr/bin/env python3
"""
Quick test script to verify face detection is working.
"""

import cv2
import sys
from src.core.face_detection import FaceDetector

def test_face_detection():
    """Test face detection with camera."""
    print("Testing face detection...")
    print("=" * 60)
    
    # Initialize face detector
    print("Initializing face detector (MediaPipe backend)...")
    face_detector = FaceDetector(backend="mediapipe")
    print("✓ Face detector initialized")
    
    # Initialize camera
    print("\nInitializing camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Error: Could not open camera")
        return False
    
    print("✓ Camera opened")
    print("\nPress 'q' to quit, 's' to save screenshot")
    print("-" * 60)
    
    frame_count = 0
    faces_detected = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Detect faces
            faces = face_detector.detect_faces(frame)
            
            if faces:
                faces_detected += 1
                if frame_count % 30 == 0:  # Print every 30 frames
                    print(f"Frame {frame_count}: Detected {len(faces)} face(s)")
                
                # Draw faces
                frame = face_detector.draw_faces(frame, faces)
            else:
                if frame_count % 30 == 0:
                    print(f"Frame {frame_count}: No faces detected")
            
            # Display frame
            cv2.imshow("Face Detection Test", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"face_detection_test_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Total frames processed: {frame_count}")
        print(f"Frames with faces detected: {faces_detected}")
        if frame_count > 0:
            print(f"Detection rate: {faces_detected/frame_count*100:.1f}%")
        print("=" * 60)
        
        return faces_detected > 0

if __name__ == "__main__":
    success = test_face_detection()
    sys.exit(0 if success else 1)

