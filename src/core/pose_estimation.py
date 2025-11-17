"""
Head Pose Estimation Module

This module estimates 3D head pose (pitch, yaw, roll) using facial landmarks.
It supports multiple backends including MediaPipe and optional Dlib.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import logging
import warnings
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# Import MediaPipe (may show protobuf warning but should still work)
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception as e:
    MP_AVAILABLE = False
    # Suppress warning
    if False:  # Disable warnings
        logging.warning(f"MediaPipe not available: {e}. Pose estimation may not work.")
    mp = None

# Optional imports
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    # Suppress warning
    pass

from ..utils.logger import get_logger

logger = get_logger(__name__)


class PoseEstimator:
    """Estimates 3D head pose using facial landmarks."""
    
    def __init__(self, backend: str = "mediapipe"):
        """
        Initialize pose estimator.
        
        Args:
            backend: Estimation backend ("mediapipe", "dlib")
        """
        self.backend = backend
        self.pose_history = []  # Store recent pose estimates for smoothing
        
        # Initialize MediaPipe face mesh
        if MP_AVAILABLE and mp is not None and backend == "mediapipe":
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            self.mp_face_mesh = None
            self.face_mesh = None
        
        # 3D model points for pose estimation
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0),     # Right mouth corner
        ], dtype=np.float64)
        
        # Camera matrix (assumed values, should be calibrated)
        self.camera_matrix = np.array([
            [1000.0, 0.0, 320.0],
            [0.0, 1000.0, 240.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        
        # Distortion coefficients
        self.dist_coeffs = np.zeros((4, 1))
        
        logger.info(f"Pose estimator initialized with backend: {backend}")
    
    def estimate_pose(self, frame: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """
        Estimate head pose for the detected face.
        
        Args:
            frame: Input frame
            face_bbox: Face bounding box (x, y, w, h)
            
        Returns:
            Dictionary containing pose angles, confidence, and movement metrics
        """
        if self.backend == "mediapipe":
            return self._estimate_pose_mediapipe(frame, face_bbox)
        elif self.backend == "dlib" and DLIB_AVAILABLE:
            return self._estimate_pose_dlib(frame, face_bbox)
        else:
            # Fallback to MediaPipe
            logger.warning(f"Backend {self.backend} not available, using MediaPipe")
            return self._estimate_pose_mediapipe(frame, face_bbox)
    
    def _estimate_pose_mediapipe(self, frame: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """Estimate pose using MediaPipe face mesh."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return self._get_empty_pose_result()
        
        # Use the first detected face
        landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        
        # Extract key facial landmarks for pose estimation
        image_points = self._extract_pose_landmarks(landmarks, w, h)
        
        if len(image_points) < 6:
            return self._get_empty_pose_result()
        
        # Solve PnP to get rotation and translation vectors
        success, rotation_vec, translation_vec = cv2.solvePnP(
            self.model_points, image_points, self.camera_matrix, self.dist_coeffs
        )
        
        if not success:
            return self._get_empty_pose_result()
        
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
        
        # Extract Euler angles (pitch, yaw, roll)
        pitch, yaw, roll = self._rotation_matrix_to_euler_angles(rotation_matrix)
        
        # Convert to degrees
        pitch_deg = np.degrees(pitch)
        yaw_deg = np.degrees(yaw)
        roll_deg = np.degrees(roll)
        
        # CRITICAL: Normalize angles to reasonable ranges for head pose
        # Euler angles can be -180 to +180, but head pose should be -90 to +90
        # Normalize by converting angles > 90° or < -90° to equivalent angles in [-90, 90]
        def normalize_angle(angle):
            """Normalize angle to [-90, 90] range."""
            while angle > 90.0:
                angle -= 180.0
            while angle < -90.0:
                angle += 180.0
            return angle
        
        pitch_deg = normalize_angle(pitch_deg)
        yaw_deg = normalize_angle(yaw_deg)
        roll_deg = normalize_angle(roll_deg)
        
        # Calculate confidence based on landmark visibility and pose stability
        confidence = self._calculate_pose_confidence(image_points, rotation_matrix)
        
        # Detect significant head movement
        is_moving = self._detect_head_movement(pitch_deg, yaw_deg, roll_deg)
        
        # Update pose history for smoothing
        current_pose = (pitch_deg, yaw_deg, roll_deg)
        self.pose_history.append(current_pose)
        if len(self.pose_history) > 10:
            self.pose_history.pop(0)
        
        # Smooth pose angles
        smoothed_pose = self._smooth_pose_angles()
        
        return {
            'pitch': smoothed_pose[0],
            'yaw': smoothed_pose[1],
            'roll': smoothed_pose[2],
            'confidence': confidence,
            'is_moving': is_moving,
            'rotation_matrix': rotation_matrix,
            'translation_vector': translation_vec,
            'image_points': image_points,
            'backend': 'mediapipe'
        }
    
    def _estimate_pose_dlib(self, frame: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """Estimate pose using Dlib (if available)."""
        if not DLIB_AVAILABLE:
            return self._get_empty_pose_result()
        
        try:
            # This would require dlib facial landmarks
            # For now, return empty result
            return self._get_empty_pose_result()
        except Exception as e:
            logger.error(f"Dlib pose estimation failed: {e}")
            return self._get_empty_pose_result()
    
    def _extract_pose_landmarks(self, landmarks, w: int, h: int) -> np.ndarray:
        """Extract key facial landmarks for pose estimation."""
        # MediaPipe face mesh indices for key facial points
        landmark_indices = [
            1,    # Nose tip
            152,  # Chin
            226,  # Left eye left corner
            446,  # Right eye right corner
            57,   # Left mouth corner
            287,  # Right mouth corner
        ]
        
        image_points = []
        for idx in landmark_indices:
            if idx < len(landmarks.landmark):
                landmark = landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                image_points.append([x, y])
        
        return np.array(image_points, dtype=np.float64)
    
    def _rotation_matrix_to_euler_angles(self, R: np.ndarray) -> Tuple[float, float, float]:
        """Convert rotation matrix to Euler angles (pitch, yaw, roll)."""
        # Extract rotation angles from rotation matrix
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            pitch = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(-R[2, 0], sy)
            roll = np.arctan2(R[1, 0], R[0, 0])
        else:
            pitch = np.arctan2(-R[1, 2], R[1, 1])
            yaw = np.arctan2(-R[2, 0], sy)
            roll = 0
        
        return pitch, yaw, roll
    
    def _calculate_pose_confidence(self, image_points: np.ndarray, rotation_matrix: np.ndarray) -> float:
        """Calculate confidence in pose estimation."""
        if len(image_points) < 6:
            return 0.0
        
        # Base confidence
        confidence = 0.5
        
        # Higher confidence if landmarks are well distributed
        if len(image_points) >= 6:
            confidence += 0.3
        
        # Higher confidence if rotation matrix is valid
        if np.linalg.det(rotation_matrix) > 0.9:  # Valid rotation matrix
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _detect_head_movement(self, pitch: float, yaw: float, roll: float) -> bool:
        """Detect significant head movement."""
        if len(self.pose_history) < 2:
            return False
        
        # Calculate movement magnitude
        last_pose = self.pose_history[-1]
        movement = np.sqrt(
            (pitch - last_pose[0])**2 + 
            (yaw - last_pose[1])**2 + 
            (roll - last_pose[2])**2
        )
        
        # Threshold for significant movement (degrees)
        movement_threshold = 5.0
        
        return movement > movement_threshold
    
    def _smooth_pose_angles(self) -> Tuple[float, float, float]:
        """Smooth pose angles using moving average."""
        if not self.pose_history:
            return (0.0, 0.0, 0.0)
        
        # Calculate moving average
        avg_pitch = np.mean([pose[0] for pose in self.pose_history])
        avg_yaw = np.mean([pose[1] for pose in self.pose_history])
        avg_roll = np.mean([pose[2] for pose in self.pose_history])
        
        return (avg_pitch, avg_yaw, avg_roll)
    
    def _get_empty_pose_result(self) -> Dict[str, Any]:
        """Return empty pose estimation result."""
        return {
            'pitch': 0.0,
            'yaw': 0.0,
            'roll': 0.0,
            'confidence': 0.0,
            'is_moving': False,
            'rotation_matrix': np.eye(3),
            'translation_vector': np.zeros(3),
            'image_points': [],
            'backend': self.backend
        }
    
    def get_pose_direction(self, pose_result: Dict[str, Any]) -> str:
        """Get human-readable pose direction."""
        yaw = pose_result.get('yaw', 0.0)
        
        if yaw < -15:
            return "Looking Left"
        elif yaw > 15:
            return "Looking Right"
        else:
            return "Looking Forward"
    
    def get_pose_level(self, pose_result: Dict[str, Any]) -> str:
        """Get pose level based on angles."""
        pitch = abs(pose_result.get('pitch', 0.0))
        yaw = abs(pose_result.get('yaw', 0.0))
        roll = abs(pose_result.get('roll', 0.0))
        
        max_angle = max(pitch, yaw, roll)
        
        if max_angle < 10:
            return "Neutral"
        elif max_angle < 25:
            return "Slight"
        elif max_angle < 45:
            return "Moderate"
        else:
            return "Extreme"
    
    def draw_pose_visualization(self, frame: np.ndarray, pose_result: Dict[str, Any]) -> np.ndarray:
        """Draw pose visualization on the frame."""
        result_frame = frame.copy()
        
        pitch = pose_result.get('pitch', 0.0)
        yaw = pose_result.get('yaw', 0.0)
        roll = pose_result.get('roll', 0.0)
        confidence = pose_result.get('confidence', 0.0)
        is_moving = pose_result.get('is_moving', False)
        
        h, w = frame.shape[:2]
        
        # Draw pose axes
        image_points = pose_result.get('image_points', [])
        if len(image_points) >= 6:
            # Draw coordinate axes
            nose_point = tuple(map(int, image_points[0]))
            
            # Scale for visualization
            axis_length = 50
            
            # Draw axes (simplified)
            color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.4 else (0, 0, 255)
            
            # X-axis (red)
            cv2.arrowedLine(result_frame, nose_point, 
                           (nose_point[0] + axis_length, nose_point[1]), (0, 0, 255), 2)
            
            # Y-axis (green)
            cv2.arrowedLine(result_frame, nose_point, 
                           (nose_point[0], nose_point[1] - axis_length), (0, 255, 0), 2)
            
            # Z-axis (blue)
            cv2.arrowedLine(result_frame, nose_point, 
                           (nose_point[0], nose_point[1] + axis_length), (255, 0, 0), 2)
        
        # Draw pose information
        y_offset = 30
        cv2.putText(result_frame, f"Pitch: {pitch:.1f}°", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result_frame, f"Yaw: {yaw:.1f}°", (10, y_offset + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result_frame, f"Roll: {roll:.1f}°", (10, y_offset + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result_frame, f"Confidence: {confidence:.2f}", (10, y_offset + 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw movement indicator
        if is_moving:
            cv2.putText(result_frame, "MOVING", (w - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return result_frame 