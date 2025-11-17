#!/usr/bin/env python3
"""
Example: Robust Attentiveness Detection

This script demonstrates the robust attentiveness calculator with:
- Real-time webcam feed
- Overlay of metrics on video
- JSON output every second
- Calibration phase handling
"""

import cv2
import numpy as np
import time
import json
from typing import Dict, Any

from src.core.face_detection import FaceDetector
from src.core.gaze_estimation import GazeEstimator
from src.core.pose_estimation import PoseEstimator
from src.core.behavioral_cues import BehavioralCuesDetector
from src.core.robust_attentiveness import RobustAttentivenessCalculator


def draw_overlay(frame: np.ndarray, result: Dict[str, Any], fps: float) -> np.ndarray:
    """Draw attentiveness metrics overlay on frame."""
    overlay = frame.copy()
    h, w = frame.shape[:2]
    
    # Semi-transparent background for text
    overlay_alpha = 0.7
    cv2.rectangle(overlay, (10, 10), (400, 280), (0, 0, 0), -1)
    cv2.addWeighted(overlay, overlay_alpha, frame, 1 - overlay_alpha, 0, frame)
    
    # Text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    color_white = (255, 255, 255)
    color_green = (0, 255, 0)
    color_red = (0, 0, 255)
    color_yellow = (0, 255, 255)
    
    y_offset = 35
    line_height = 25
    
    # Attentiveness score (large, prominent)
    score = result['attentiveness_score']
    attentive = result['attentive']
    score_color = color_green if attentive else color_red
    
    cv2.putText(frame, f"Attentiveness: {score:.1f}/100", (20, y_offset),
                font, 0.8, score_color, thickness)
    y_offset += line_height
    
    status_text = "ATTENTIVE" if attentive else "NOT ATTENTIVE"
    status_color = color_green if attentive else color_red
    cv2.putText(frame, status_text, (20, y_offset),
                font, 0.7, status_color, thickness)
    y_offset += line_height + 5
    
    # Face presence
    face_status = "Face: DETECTED" if result['face_present'] else "Face: NOT DETECTED"
    face_color = color_green if result['face_present'] else color_red
    cv2.putText(frame, face_status, (20, y_offset),
                font, font_scale, face_color, thickness)
    y_offset += line_height
    
    # Pose angles
    cv2.putText(frame, f"Yaw: {result['yaw']:.1f}°", (20, y_offset),
                font, font_scale, color_white, 1)
    y_offset += line_height
    cv2.putText(frame, f"Pitch: {result['pitch']:.1f}°", (20, y_offset),
                font, font_scale, color_white, 1)
    y_offset += line_height
    cv2.putText(frame, f"Roll: {result['roll']:.1f}°", (20, y_offset),
                font, font_scale, color_white, 1)
    y_offset += line_height
    
    # Gaze
    cv2.putText(frame, f"Gaze off-axis: {result['gaze_off_axis']:.1f}°", (20, y_offset),
                font, font_scale, color_white, 1)
    y_offset += line_height
    
    # Movement
    cv2.putText(frame, f"Movement: {result['movement_px_per_sec']:.1f} px/s", (20, y_offset),
                font, font_scale, color_white, 1)
    y_offset += line_height
    
    # Signal quality
    quality = result['signal_quality']
    quality_color = color_green if quality > 0.7 else (color_yellow if quality > 0.4 else color_red)
    cv2.putText(frame, f"Signal Quality: {quality:.2f}", (20, y_offset),
                font, font_scale, quality_color, 1)
    y_offset += line_height
    
    # Calibration status
    if result['is_calibrating']:
        progress = result['calibration_progress'] * 100
        cv2.putText(frame, f"Calibrating: {progress:.0f}%", (20, y_offset),
                    font, font_scale, color_yellow, 1)
    else:
        cv2.putText(frame, "Calibrated", (20, y_offset),
                    font, font_scale, color_green, 1)
    y_offset += line_height
    
    # FPS and latency
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_offset),
                font, font_scale, color_white, 1)
    y_offset += line_height
    cv2.putText(frame, f"Latency: {result['latency_ms']:.1f}ms", (20, y_offset),
                font, font_scale, color_white, 1)
    
    return frame


def main():
    """Main function to run robust attentiveness detection."""
    print("=" * 60)
    print("Robust Attentiveness Detection System")
    print("=" * 60)
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - Press 'r' to reset calibration")
    print("  - JSON output printed every second")
    print("\nStarting...\n")
    
    # Initialize components
    face_detector = FaceDetector(backend="mediapipe")
    gaze_estimator = GazeEstimator(backend="mediapipe")
    pose_estimator = PoseEstimator(backend="mediapipe")
    behavioral_detector = BehavioralCuesDetector()
    attentiveness_calc = RobustAttentivenessCalculator(calibration_duration_sec=8.0)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Get actual FPS
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    attentiveness_calc.fps = fps
    
    frame_count = 0
    last_json_time = time.time()
    start_time = time.time()
    
    print("Calibration phase: Please look at the screen naturally for 8 seconds...")
    print("After calibration, the system will use your baseline movement patterns.\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Detect face
            faces = face_detector.detect_faces(frame)
            face_detected = len(faces) > 0
            
            # Initialize result dictionaries
            pose_result = None
            gaze_result = None
            behavioral_result = None
            
            if face_detected:
                # Get largest face
                largest_face = face_detector.get_largest_face(faces)
                if largest_face:
                    bbox = largest_face['bbox']
                    
                    # Estimate pose
                    pose_result = pose_estimator.estimate_pose(frame, bbox)
                    
                    # Estimate gaze
                    gaze_result = gaze_estimator.estimate_gaze(frame, bbox)
                    
                    # Detect behavioral cues
                    behavioral_result = behavioral_detector.detect_cues(frame, {
                        'landmarks': largest_face.get('landmarks', [])
                    })
            
            # Calculate attentiveness
            result = attentiveness_calc.calculate_attentiveness(
                frame=frame,
                face_detected=face_detected,
                pose_result=pose_result,
                gaze_result=gaze_result,
                behavioral_result=behavioral_result
            )
            
            # Draw overlay
            display_frame = draw_overlay(frame, result, fps)
            
            # Draw face bounding box if detected
            if face_detected and largest_face:
                bbox = largest_face['bbox']
                cv2.rectangle(display_frame, (bbox[0], bbox[1]),
                            (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                            (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Robust Attentiveness Detection', display_frame)
            
            # Print JSON every second
            if current_time - last_json_time >= 1.0:
                json_output = {
                    'timestamp': current_time,
                    'frame_count': frame_count,
                    'attentiveness_score': result['attentiveness_score'],
                    'attentive': result['attentive'],
                    'face_present': result['face_present'],
                    'yaw': result['yaw'],
                    'pitch': result['pitch'],
                    'roll': result['roll'],
                    'gaze_off_axis': result['gaze_off_axis'],
                    'movement_px_per_sec': result['movement_px_per_sec'],
                    'blink_rate': result['blink_rate'],
                    'signal_quality': result['signal_quality'],
                    'is_calibrating': result['is_calibrating'],
                    'latency_ms': result['latency_ms']
                }
                print(json.dumps(json_output, indent=2))
                print("-" * 60)
                last_json_time = current_time
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("\nResetting calibration...")
                attentiveness_calc.reset(new_calibration=True)
                print("Calibration phase: Please look at the screen naturally for 8 seconds...\n")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final summary
        elapsed = time.time() - start_time
        print(f"\n{'=' * 60}")
        print("Session Summary")
        print(f"{'=' * 60}")
        print(f"Total frames: {frame_count}")
        print(f"Total time: {elapsed:.1f}s")
        print(f"Average FPS: {frame_count / elapsed:.1f}")
        print(f"\nState Summary:")
        state_summary = attentiveness_calc.get_state_summary()
        for key, value in state_summary.items():
            print(f"  {key}: {value}")
        print("=" * 60)


if __name__ == "__main__":
    main()

