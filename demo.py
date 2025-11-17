#!/usr/bin/env python3
"""
Enhanced demo of the attentiveness detection system.
Shows real-time face detection, gaze estimation, pose estimation, and behavioral analysis
with more realistic attention thresholds for natural human movement.
"""

import cv2
import numpy as np
import time
from src.core.face_detection import FaceDetector
from src.core.gaze_estimation import GazeEstimator
from src.core.pose_estimation import PoseEstimator
from src.core.behavioral_cues import BehavioralCuesDetector
from src.core.multimodal_fusion import MultimodalFusion
from src.core.attentiveness_scoring import AttentivenessScorer

def main():
    """Run the enhanced attentiveness detection demo."""
    print("Starting Enhanced Attentiveness Detection Demo...")
    print("Press 'q' to quit, 's' to save screenshot, 'r' to reset session")
    
    # Initialize all components
    face_detector = FaceDetector(backend="mediapipe")
    gaze_estimator = GazeEstimator(backend="mediapipe")
    pose_estimator = PoseEstimator(backend="mediapipe")
    behavioral_detector = BehavioralCuesDetector()
    fusion = MultimodalFusion(fusion_strategy="weighted_sum")
    scorer = AttentivenessScorer()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame
            processed_frame = frame.copy()
            
            # Detect faces
            faces = face_detector.detect_faces(frame)
            
            if faces:
                # Get the largest face
                largest_face = face_detector.get_largest_face(faces)
                if largest_face:
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
                    
                    # Draw visualizations
                    processed_frame = face_detector.draw_faces(processed_frame, faces)
                    processed_frame = gaze_estimator.draw_gaze_visualization(processed_frame, gaze_result)
                    processed_frame = pose_estimator.draw_pose_visualization(processed_frame, pose_result)
                    
                    # Draw comprehensive attention metrics
                    draw_comprehensive_metrics(processed_frame, gaze_result, pose_result, 
                                           behavioral_result, fusion_result, scoring_result)
                    
                    # Print console info
                    if frame_count % 30 == 0:  # Every 30 frames (1 second at 30 FPS)
                        print_comprehensive_info(gaze_result, pose_result, behavioral_result, 
                                              fusion_result, scoring_result)
            else:
                # No faces detected
                cv2.putText(processed_frame, "No face detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw FPS
            fps = frame_count / (time.time() - start_time) if time.time() > start_time else 0
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, processed_frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow("Enhanced Attentiveness Detection Demo", processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"attentiveness_demo_{timestamp}.jpg"
                cv2.imwrite(filename, processed_frame)
                print(f"Screenshot saved: {filename}")
            elif key == ord('r'):
                # Reset session
                scorer.reset_session()
                print("Session reset")
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Save session data
        session_file = scorer.save_session_data()
        if session_file:
            print(f"Session data saved: {session_file}")
        
        print("Demo ended")

def draw_comprehensive_metrics(frame, gaze_result, pose_result, behavioral_result, fusion_result, scoring_result):
    """Draw comprehensive attention metrics on the frame."""
    y_offset = 150
    
    # Attention score (main metric)
    attention_score = scoring_result.get('attention_score', 0.0)
    attention_level = scoring_result.get('attention_level', "Very Low")
    attention_color = get_attention_color(attention_score)
    
    cv2.putText(frame, f"Attention: {attention_level} ({attention_score:.2f})", 
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, attention_color, 2)
    
    # Individual component scores
    gaze_score = scoring_result.get('gaze_score', 0.0)
    pose_score = scoring_result.get('pose_score', 0.0)
    behavioral_score = scoring_result.get('behavioral_score', 0.0)
    
    cv2.putText(frame, f"Gaze: {gaze_score:.2f}", (10, y_offset + 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Pose: {pose_score:.2f}", (10, y_offset + 45), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Behavior: {behavioral_score:.2f}", (10, y_offset + 65), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Behavioral indicators
    blink_rate = behavioral_result.get('blink_rate', 0.0)
    yawn_rate = behavioral_result.get('yawn_rate', 0.0)
    fatigue_score = behavioral_result.get('fatigue_score', 0.0)
    
    cv2.putText(frame, f"Blink Rate: {blink_rate:.1f}/min", (10, y_offset + 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, f"Yawn Rate: {yawn_rate:.1f}/hour", (10, y_offset + 110), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, f"Fatigue: {fatigue_score:.2f}", (10, y_offset + 130), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Confidence indicators
    gaze_conf = gaze_result.get('confidence', 0.0)
    pose_conf = pose_result.get('confidence', 0.0)
    overall_conf = scoring_result.get('confidence', 0.0)
    
    cv2.putText(frame, f"Confidence: {overall_conf:.2f}", (10, y_offset + 155), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Draw attention bar
    draw_attention_bar(frame, attention_score, y_offset + 180)

def get_attention_color(attention_score):
    """Get color based on attention score."""
    if attention_score >= 0.6:
        return (0, 255, 0)  # Green for high attention
    elif attention_score >= 0.35:
        return (0, 255, 255)  # Yellow for medium attention
    elif attention_score >= 0.15:
        return (0, 165, 255)  # Orange for low attention
    else:
        return (0, 0, 255)  # Red for very low attention

def draw_attention_bar(frame, attention_score, y_offset):
    """Draw attention score bar."""
    bar_width = 200
    bar_height = 15
    bar_x = 10
    bar_y = y_offset
    
    # Background bar
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
    
    # Attention bar
    attention_width = int(bar_width * attention_score)
    bar_color = get_attention_color(attention_score)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + attention_width, bar_y + bar_height), bar_color, -1)
    
    # Border
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 1)

def print_comprehensive_info(gaze_result, pose_result, behavioral_result, fusion_result, scoring_result):
    """Print comprehensive attention information to console."""
    attention_level = scoring_result.get('attention_level', "Very Low")
    attention_score = scoring_result.get('attention_score', 0.0)
    gaze_score = scoring_result.get('gaze_score', 0.0)
    pose_score = scoring_result.get('pose_score', 0.0)
    behavioral_score = scoring_result.get('behavioral_score', 0.0)
    
    blink_rate = behavioral_result.get('blink_rate', 0.0)
    yawn_rate = behavioral_result.get('yawn_rate', 0.0)
    fatigue_score = behavioral_result.get('fatigue_score', 0.0)
    
    print(f"Attention: {attention_level} ({attention_score:.2f}) | "
          f"Gaze: {gaze_score:.2f} | Pose: {pose_score:.2f} | "
          f"Behavior: {behavioral_score:.2f} | "
          f"Blink: {blink_rate:.1f}/min | Yawn: {yawn_rate:.1f}/hour | "
          f"Fatigue: {fatigue_score:.2f}")

def calculate_attention_level(gaze_result, pose_result, behavioral_result):
    """Calculate overall attention level with realistic thresholds."""
    gaze_conf = gaze_result.get('confidence', 0.0)
    pose_conf = pose_result.get('confidence', 0.0)
    
    # Check if person is looking at screen (more lenient)
    gaze_direction = gaze_result.get('gaze_direction', (0.0, 0.0))
    gaze_magnitude = np.sqrt(gaze_direction[0]**2 + gaze_direction[1]**2)
    
    # Check if head is relatively still (more lenient)
    pose_angles = [abs(pose_result.get('pitch', 0)), abs(pose_result.get('yaw', 0)), abs(pose_result.get('roll', 0))]
    max_angle = max(pose_angles)
    
    # More realistic thresholds for natural human movement
    if gaze_conf > 0.5 and pose_conf > 0.5 and gaze_magnitude < 0.4 and max_angle < 20:
        return "High"
    elif gaze_conf > 0.3 and pose_conf > 0.3 and gaze_magnitude < 0.6 and max_angle < 30:
        return "Medium"
    else:
        return "Low"

if __name__ == "__main__":
    main() 