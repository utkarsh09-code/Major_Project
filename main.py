#!/usr/bin/env python3
"""
Main entry point for the attentiveness detection system.
Provides command-line interface for real-time monitoring.
"""

import os
import warnings
import logging

# Suppress warnings unless verbose mode
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import time
import argparse
import sys
from typing import Optional

from src.core.face_detection import FaceDetector
from src.core.gaze_estimation import GazeEstimator
from src.core.pose_estimation import PoseEstimator
from src.core.behavioral_cues import BehavioralCuesDetector
from src.core.multimodal_fusion import MultimodalFusion
from src.core.attentiveness_scoring import AttentivenessScorer

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Real-time User Attentiveness Detection System")
    
    parser.add_argument("--camera", "-c", type=int, default=0,
                       help="Camera device index (default: 0)")
    parser.add_argument("--width", "-w", type=int, default=640,
                       help="Frame width (default: 640)")
    parser.add_argument("--height", type=int, default=480,
                       help="Frame height (default: 480)")
    parser.add_argument("--fps", "-f", type=int, default=30,
                       help="Target FPS (default: 30)")
    parser.add_argument("--backend", "-b", type=str, default="mediapipe",
                       choices=["mediapipe", "opencv"],
                       help="Face detection backend (default: mediapipe)")
    parser.add_argument("--fusion", type=str, default="weighted_sum",
                       choices=["weighted_sum", "attention", "ensemble"],
                       help="Fusion strategy (default: weighted_sum)")
    parser.add_argument("--output", "-o", type=str, default="",
                       help="Output video file path (optional)")
    parser.add_argument("--session-id", type=str, default="",
                       help="Session ID for tracking (optional)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--no-display", action="store_true",
                       help="Disable video display (headless mode)")
    parser.add_argument("--gui", "-g", action="store_true",
                       help="Launch GUI interface instead of CLI")
    
    return parser.parse_args()

def initialize_components(backend: str, fusion_strategy: str, session_id: Optional[str] = None):
    """Initialize all system components."""
    print("Initializing attentiveness detection system...")
    
    try:
        # Initialize core components
        face_detector = FaceDetector(backend=backend)
        gaze_estimator = GazeEstimator(backend=backend)
        pose_estimator = PoseEstimator(backend=backend)
        behavioral_detector = BehavioralCuesDetector()
        fusion = MultimodalFusion(fusion_strategy=fusion_strategy)
        scorer = AttentivenessScorer(session_id=session_id)

        print("✓ All components initialized successfully")
        return face_detector, gaze_estimator, pose_estimator, behavioral_detector, fusion, scorer

    except Exception as e:
        # Return explicit tuple of Nones so the caller can detect failure without raising here
        print(f"✗ Error initializing components: {e}")
        return None, None, None, None, None, None

def initialize_camera(camera_index: int, width: int, height: int, fps: int):
    """Initialize camera capture."""
    print(f"Initializing camera (device: {camera_index})...")
    import platform

    # On Windows prefer DirectShow backend which is often more reliable
    if platform.system() == 'Windows':
        try:
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        except Exception:
            cap = cv2.VideoCapture(camera_index)
    else:
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"✗ Error: Could not open camera {camera_index}")
        return None
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    # Verify settings
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"✓ Camera initialized: {actual_width:.0f}x{actual_height:.0f} @ {actual_fps:.1f} FPS")
    return cap

def process_frame(frame: np.ndarray, components: tuple) -> tuple:
    """Process a single frame through all components."""
    face_detector, gaze_estimator, pose_estimator, behavioral_detector, fusion, scorer = components
    
    # Detect faces
    faces = face_detector.detect_faces(frame)
    
    if not faces:
        return frame, faces, None, None, None, None, None
    
    # Log face detection (only occasionally to avoid spam)
    import random
    if random.random() < 0.01:  # 1% of frames
        print(f"✓ Detected {len(faces)} face(s)")
    
    # Get the largest face
    largest_face = face_detector.get_largest_face(faces)
    if not largest_face:
        return frame, faces, None, None, None, None, None
    
    bbox = largest_face['bbox']
    
    # Estimate gaze
    gaze_result = gaze_estimator.estimate_gaze(frame, bbox)
    
    # Estimate pose
    pose_result = pose_estimator.estimate_pose(frame, bbox)
    
    # Detect behavioral cues
    behavioral_result = behavioral_detector.detect_cues(frame, {
        'landmarks': largest_face.get('landmarks', [])
    })
    
    # Fuse features
    fusion_result = fusion.fuse_features(gaze_result, pose_result, behavioral_result)
    
    # Calculate final attention score
    scoring_result = scorer.calculate_attention_score(gaze_result, pose_result, behavioral_result)
    
    return frame, faces, gaze_result, pose_result, behavioral_result, fusion_result, scoring_result

def draw_results(frame: np.ndarray, face_detector, faces: list, gaze_estimator, gaze_result: dict, 
                pose_estimator, pose_result: dict, scoring_result: dict, fps: float = 0.0) -> np.ndarray:
    """Draw detection results on frame with organized, non-overlapping layout."""
    output_frame = frame.copy()
    h, w = frame.shape[:2]
    
    # Draw face detections
    if faces and face_detector:
        output_frame = face_detector.draw_faces(output_frame, faces)
    
    # Draw gaze visualization (only arrows, no text)
    if gaze_result and gaze_estimator:
        output_frame = gaze_estimator.draw_gaze_visualization(output_frame, gaze_result)
    
    # Draw pose axes only (no text overlay from pose estimator to avoid overlap)
    if pose_result and pose_estimator:
        # Draw only the pose axes, not the text
        image_points = pose_result.get('image_points', [])
        if len(image_points) >= 6:
            nose_point = tuple(map(int, image_points[0]))
            axis_length = 50
            confidence = pose_result.get('confidence', 0.0)
            
            # X-axis (red), Y-axis (green), Z-axis (blue)
            cv2.arrowedLine(output_frame, nose_point, 
                           (nose_point[0] + axis_length, nose_point[1]), (0, 0, 255), 2)
            cv2.arrowedLine(output_frame, nose_point, 
                           (nose_point[0], nose_point[1] - axis_length), (0, 255, 0), 2)
            cv2.arrowedLine(output_frame, nose_point, 
                           (nose_point[0], nose_point[1] + axis_length), (255, 0, 0), 2)
    
    # Draw organized information panels (non-overlapping)
    draw_comprehensive_overlay(output_frame, scoring_result, pose_result, gaze_result, fps, w, h, faces)
    
    return output_frame

def draw_comprehensive_overlay(frame: np.ndarray, scoring_result: dict, pose_result: dict, 
                               gaze_result: dict, fps: float, width: int, height: int, faces: list = None):
    """Draw all information in organized, non-overlapping panels."""
    
    # Panel dimensions and spacing
    font_scale = 0.6
    font_thickness = 2
    
    # === TOP-LEFT: Attention Score Panel ===
    att_x = 15
    att_y = 15
    att_panel_w = 280
    att_panel_h = 140
    
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (att_x, att_y), (att_x + att_panel_w, att_y + att_panel_h), 
                 (20, 20, 30), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    # Draw border
    cv2.rectangle(frame, (att_x, att_y), (att_x + att_panel_w, att_y + att_panel_h), 
                 (100, 150, 255), 2)
    
    # Title
    cv2.putText(frame, "ATTENTION STATUS", (att_x + 10, att_y + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 200, 255), 2)
    
    if scoring_result:
        attention_score = scoring_result.get('attention_score', 0.0)
        attention_level = scoring_result.get('attention_level', "Very Low")
        confidence = scoring_result.get('confidence', 0.0)
        
        # Choose color based on attention level
        if attention_level == "High":
            color = (0, 255, 0)  # Green
        elif attention_level == "Medium":
            color = (0, 255, 255)  # Yellow
        elif attention_level == "Low":
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 0, 255)  # Red
        
        # Level
        cv2.putText(frame, f"Level: {attention_level}", (att_x + 10, att_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)
        
        # Score
        cv2.putText(frame, f"Score: {attention_score:.2f}", (att_x + 10, att_y + 75),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        
        # Confidence
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (att_x + 10, att_y + 100),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), 1)
        
        # Progress bar
        bar_x = att_x + 10
        bar_y = att_y + 110
        bar_w = 260
        bar_h = 18
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        # Fill
        fill_w = int(bar_w * attention_score)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
        # Border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)
    else:
        # No data available
        cv2.putText(frame, "Waiting for data...", (att_x + 10, att_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (150, 150, 150), 1)
    
    # === TOP-RIGHT: Head Pose Panel ===
    pose_x = width - 280
    pose_y = 15
    pose_panel_w = 265
    pose_panel_h = 140
    
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (pose_x, pose_y), (pose_x + pose_panel_w, pose_y + pose_panel_h), 
                 (20, 20, 30), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    # Draw border
    cv2.rectangle(frame, (pose_x, pose_y), (pose_x + pose_panel_w, pose_y + pose_panel_h), 
                 (255, 150, 100), 2)
    
    # Title
    cv2.putText(frame, "HEAD POSE", (pose_x + 10, pose_y + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 150), 2)
    
    if pose_result:
        pitch = pose_result.get('pitch', 0.0)
        yaw = pose_result.get('yaw', 0.0)
        roll = pose_result.get('roll', 0.0)
        pose_conf = pose_result.get('confidence', 0.0)
        
        # Color coding for angles
        def get_angle_color(angle):
            abs_angle = abs(angle)
            if abs_angle < 15:
                return (0, 255, 0)  # Green - good
            elif abs_angle < 30:
                return (0, 255, 255)  # Yellow - acceptable
            else:
                return (0, 0, 255)  # Red - poor
        
        y_offset = pose_y + 50
        cv2.putText(frame, f"Pitch: {pitch:+.1f}", (pose_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, get_angle_color(pitch), font_thickness)
        cv2.putText(frame, f"Yaw: {yaw:+.1f}", (pose_x + 10, y_offset + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, get_angle_color(yaw), font_thickness)
        cv2.putText(frame, f"Roll: {roll:+.1f}", (pose_x + 10, y_offset + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, get_angle_color(roll), font_thickness)
        cv2.putText(frame, f"Confidence: {pose_conf:.2f}", (pose_x + 10, y_offset + 75),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), 1)
    else:
        # No pose data
        cv2.putText(frame, "No pose data", (pose_x + 10, pose_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (150, 150, 150), 1)
    
    # === BOTTOM-LEFT: Gaze & Behavior Panel ===
    gaze_x = 15
    gaze_y = height - 140
    gaze_panel_w = 280
    gaze_panel_h = 125
    
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (gaze_x, gaze_y), (gaze_x + gaze_panel_w, gaze_y + gaze_panel_h), 
                 (20, 20, 30), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    # Draw border
    cv2.rectangle(frame, (gaze_x, gaze_y), (gaze_x + gaze_panel_w, gaze_y + gaze_panel_h), 
                 (150, 100, 255), 2)
    
    # Title
    cv2.putText(frame, "GAZE & BEHAVIOR", (gaze_x + 10, gaze_y + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 150, 255), 2)
    
    y_pos = gaze_y + 50
    has_any_data = False
    
    if gaze_result:
        gaze_direction = gaze_result.get('gaze_direction', (0.0, 0.0))
        gaze_conf = gaze_result.get('confidence', 0.0)
        
        if gaze_conf > 0:
            has_any_data = True
            # Calculate gaze off-axis if not present
            if 'gaze_off_axis' in gaze_result:
                gaze_off_axis = gaze_result.get('gaze_off_axis', 0.0)
            else:
                # Calculate from gaze_direction magnitude
                gaze_magnitude = np.sqrt(gaze_direction[0]**2 + gaze_direction[1]**2)
                gaze_off_axis = gaze_magnitude * 30.0  # Convert to degrees
            
            # Gaze off-axis with color coding
            gaze_color = (0, 255, 0) if gaze_off_axis < 10 else (0, 255, 255) if gaze_off_axis < 20 else (0, 0, 255)
            cv2.putText(frame, f"Off-axis: {gaze_off_axis:.1f}", (gaze_x + 10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, gaze_color, font_thickness)
            y_pos += 22
            
            cv2.putText(frame, f"Confidence: {gaze_conf:.2f}", (gaze_x + 10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_pos += 22
    
    if scoring_result:
        gaze_score = scoring_result.get('gaze_score', 0.0)
        behavioral_score = scoring_result.get('behavioral_score', 0.0)
        
        # Gaze score (show if available)
        if gaze_score > 0:
            has_any_data = True
            gaze_score_color = (0, 255, 0) if gaze_score > 0.7 else (0, 255, 255) if gaze_score > 0.4 else (0, 0, 255)
            cv2.putText(frame, f"Gaze Score: {gaze_score:.2f}", (gaze_x + 10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, gaze_score_color, 1)
            y_pos += 18
        
        # Behavioral score
        if behavioral_score > 0:
            has_any_data = True
            behavior_color = (0, 255, 0) if behavioral_score > 0.7 else (0, 255, 255) if behavioral_score > 0.4 else (0, 0, 255)
            cv2.putText(frame, f"Behavior: {behavioral_score:.2f}", (gaze_x + 10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, behavior_color, 1)
    
    if not has_any_data:
        cv2.putText(frame, "No gaze data", (gaze_x + 10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (150, 150, 150), 1)
    
    # === BOTTOM-RIGHT: System Info Panel ===
    info_x = width - 280
    info_y = height - 100
    info_panel_w = 265
    info_panel_h = 85
    
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (info_x, info_y), (info_x + info_panel_w, info_y + info_panel_h), 
                 (20, 20, 30), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    # Draw border
    cv2.rectangle(frame, (info_x, info_y), (info_x + info_panel_w, info_y + info_panel_h), 
                 (100, 255, 100), 2)
    
    # Title
    cv2.putText(frame, "SYSTEM INFO", (info_x + 10, info_y + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 255, 150), 2)
    
    # FPS
    fps_color = (0, 255, 0) if fps > 20 else (0, 255, 255) if fps > 10 else (0, 0, 255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (info_x + 10, info_y + 50),
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, fps_color, font_thickness)
    
    # Face detection status
    has_face = faces is not None and len(faces) > 0
    face_status = "Face: DETECTED" if has_face else "Face: NOT DETECTED"
    face_color = (0, 255, 0) if has_face else (0, 0, 255)
    cv2.putText(frame, face_status, (info_x + 10, info_y + 75),
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, face_color, font_thickness)

def print_status(scoring_result: dict, frame_count: int, fps: float, verbose: bool = False):
    """Print status information."""
    if not scoring_result:
        return
    
    attention_level = scoring_result.get('attention_level', "Very Low")
    attention_score = scoring_result.get('attention_score', 0.0)
    
    if verbose:
        gaze_score = scoring_result.get('gaze_score', 0.0)
        pose_score = scoring_result.get('pose_score', 0.0)
        behavioral_score = scoring_result.get('behavioral_score', 0.0)
        confidence = scoring_result.get('confidence', 0.0)
        
        print(f"Frame {frame_count:4d} | FPS: {fps:5.1f} | "
              f"Attention: {attention_level} ({attention_score:.2f}) | "
              f"Gaze: {gaze_score:.2f} | Pose: {pose_score:.2f} | "
              f"Behavior: {behavioral_score:.2f} | Conf: {confidence:.2f}")
    else:
        if frame_count % 30 == 0:  # Print every 30 frames (1 second at 30 FPS)
            print(f"Attention: {attention_level} ({attention_score:.2f}) | FPS: {fps:.1f}")

def main():
    """Main function."""
    args = parse_arguments()
    
    # Enable verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.ERROR)
    
    # Launch GUI if requested
    if args.gui:
        try:
            from src.gui.main_window import run_gui
            run_gui()
            return
        except Exception as e:
            print(f"Error launching GUI: {e}")
            print("Falling back to CLI mode...")
    
    print("=" * 60)
    print("Real-time User Attentiveness Detection System")
    print("=" * 60)
    print(f"Camera: {args.camera}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Target FPS: {args.fps}")
    print(f"Backend: {args.backend}")
    print(f"Fusion Strategy: {args.fusion}")
    print(f"Session ID: {args.session_id or 'auto-generated'}")
    print("=" * 60)
    
    # Initialize components
    components = initialize_components(args.backend, args.fusion, args.session_id)
    if None in components:
        print("Failed to initialize components. Exiting.")
        sys.exit(1)
    
    # Initialize camera
    cap = initialize_camera(args.camera, args.width, args.height, args.fps)
    if cap is None:
        print("Failed to initialize camera. Exiting.")
        sys.exit(1)
    
    # Initialize video writer if output specified
    video_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.output, fourcc, args.fps, (args.width, args.height))
        if not video_writer.isOpened():
            print(f"⚠️ Warning: could not open video writer for {args.output}. Continuing without recording.")
            video_writer = None
        else:
            print(f"Recording to: {args.output}")
    
    # Processing loop
    frame_count = 0
    start_time = time.time()
    
    print("\n" + "=" * 60)
    print("Starting monitoring...")
    print("=" * 60)
    print("Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save screenshot")
    print("=" * 60)
    print()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame
            processed_frame, faces, gaze_result, pose_result, behavioral_result, fusion_result, scoring_result = \
                process_frame(frame, components)
            
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # Draw results (pass fps for display)
            face_detector, gaze_estimator, pose_estimator, behavioral_detector, fusion, scorer = components
            if any([faces, gaze_result, pose_result, scoring_result]):
                processed_frame = draw_results(processed_frame, 
                                           face_detector, faces,
                                           gaze_estimator, gaze_result, 
                                           pose_estimator, pose_result, 
                                           scoring_result, fps)
            
            # Print status
            print_status(scoring_result, frame_count, fps, args.verbose)
            
            # Write to video file
            if video_writer:
                video_writer.write(processed_frame)
            
            # Display frame
            if not args.no_display:
                cv2.imshow("AI Attentiveness Detection System", processed_frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"attentiveness_screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"Screenshot saved: {filename}")
            else:
                # Headless mode: avoid busy-looping, throttle to target FPS
                if args.fps and args.fps > 0:
                    time.sleep(max(0, 1.0 / args.fps))
    
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user")
    finally:
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # Save session data
        try:
            scorer = components[5] if components and len(components) > 5 else None
            if scorer:
                session_file = None
                try:
                    session_file = scorer.save_session_data()
                except Exception as e:
                    print(f"Warning: failed to save session data: {e}")

                if session_file:
                    print(f"Session data saved: {session_file}")

                try:
                    session_summary = scorer.get_session_summary()
                    print("\n" + "=" * 60)
                    print("SESSION SUMMARY")
                    print("=" * 60)
                    print(f"Session ID: {session_summary.get('session_id', 'N/A')}")
                    print(f"Duration: {session_summary.get('session_duration', 0.0):.1f} seconds")
                    print(f"Total Frames: {session_summary.get('total_frames', 0)}")
                    print(f"Average Attention Score: {session_summary.get('avg_attention_score', 0.0):.2f}")
                    print(f"Focus Percentage: {session_summary.get('focus_percentage', 0.0):.1f}%")
                    print(f"Attention Distribution:")
                    distribution = session_summary.get('attention_distribution', {})
                    total_frames = session_summary.get('total_frames', 0)
                    for level, count in distribution.items():
                        percentage = (count / total_frames * 100) if total_frames > 0 else 0
                        print(f"  {level.title()}: {count} frames ({percentage:.1f}%)")
                    print("=" * 60)
                except Exception as e:
                    print(f"Warning: failed to generate session summary: {e}")
        except Exception:
            # Defensive: don't let cleanup raise
            pass
        
        print("Monitoring ended")

if __name__ == "__main__":
    main() 