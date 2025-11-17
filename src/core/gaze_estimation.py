"""
Gaze Estimation Module

This module estimates eye gaze direction using facial landmarks.
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
        logging.warning(f"MediaPipe not available: {e}. Gaze estimation may not work.")
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


class GazeEstimator:
    """Estimates eye gaze direction using facial landmarks."""
    
    def __init__(self, backend: str = "mediapipe"):
        """
        Initialize gaze estimator.
        
        Args:
            backend: Estimation backend ("mediapipe", "dlib")
        """
        self.backend = backend
        self.gaze_history = []  # Store recent gaze estimates for smoothing
        
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
        
        # Eye aspect ratio thresholds
        self.ear_threshold = 0.2
        self.ear_history = []
        
        logger.info(f"Gaze estimator initialized with backend: {backend}")
    
    def estimate_gaze(self, frame: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """
        Estimate gaze direction for the detected face.
        
        Args:
            frame: Input frame
            face_bbox: Face bounding box (x, y, w, h)
            
        Returns:
            Dictionary containing gaze direction, confidence, and eye metrics
        """
        if self.backend == "mediapipe":
            return self._estimate_gaze_mediapipe(frame, face_bbox)
        elif self.backend == "dlib" and DLIB_AVAILABLE:
            return self._estimate_gaze_dlib(frame, face_bbox)
        else:
            # Fallback to MediaPipe
            logger.warning(f"Backend {self.backend} not available, using MediaPipe")
            return self._estimate_gaze_mediapipe(frame, face_bbox)
    
    def _estimate_gaze_mediapipe(self, frame: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """Estimate gaze using MediaPipe face mesh."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return self._get_empty_gaze_result()
        
        # Use the first detected face
        landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        
        # Extract eye landmarks
        left_eye_landmarks = self._extract_eye_landmarks(landmarks, 'left', w, h)
        right_eye_landmarks = self._extract_eye_landmarks(landmarks, 'right', w, h)
        
        if not left_eye_landmarks or not right_eye_landmarks:
            return self._get_empty_gaze_result()
        
        # Calculate eye aspect ratios
        left_ear = self._calculate_eye_aspect_ratio(left_eye_landmarks)
        right_ear = self._calculate_eye_aspect_ratio(right_eye_landmarks)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Update EAR history
        self.ear_history.append(avg_ear)
        if len(self.ear_history) > 10:
            self.ear_history.pop(0)
        
        # Detect blinking
        is_blinking = avg_ear < self.ear_threshold
        
        # Estimate gaze direction (simplified)
        gaze_direction = self._estimate_gaze_direction(left_eye_landmarks, right_eye_landmarks, w, h)
        
        # Calculate confidence based on eye visibility and stability
        confidence = self._calculate_gaze_confidence(left_eye_landmarks, right_eye_landmarks, avg_ear)
        
        # Smooth gaze direction using history
        self.gaze_history.append(gaze_direction)
        if len(self.gaze_history) > 5:
            self.gaze_history.pop(0)
        
        smoothed_gaze = self._smooth_gaze_direction()
        
        return {
            'gaze_direction': smoothed_gaze,
            'confidence': confidence,
            'left_ear': left_ear,
            'right_ear': right_ear,
            'avg_ear': avg_ear,
            'is_blinking': is_blinking,
            'left_eye_landmarks': left_eye_landmarks,
            'right_eye_landmarks': right_eye_landmarks,
            'backend': 'mediapipe'
        }
    
    def _estimate_gaze_dlib(self, frame: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """Estimate gaze using Dlib (if available)."""
        if not DLIB_AVAILABLE:
            return self._get_empty_gaze_result()
        
        try:
            # This would require dlib facial landmarks
            # For now, return empty result
            return self._get_empty_gaze_result()
        except Exception as e:
            logger.error(f"Dlib gaze estimation failed: {e}")
            return self._get_empty_gaze_result()
    
    def _extract_eye_landmarks(self, landmarks, eye_side: str, w: int, h: int) -> List[Tuple[int, int]]:
        """Extract eye landmarks from MediaPipe face mesh."""
        if eye_side == 'left':
            # Left eye landmarks (MediaPipe face mesh indices)
            eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        else:  # right eye
            eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        eye_landmarks = []
        for idx in eye_indices:
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            eye_landmarks.append((x, y))
        
        return eye_landmarks
    
    def _calculate_eye_aspect_ratio(self, eye_landmarks: List[Tuple[int, int]]) -> float:
        """Calculate Eye Aspect Ratio (EAR) for blink detection."""
        if len(eye_landmarks) < 6:
            return 0.0
        
        # Vertical distances
        A = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
        B = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
        
        # Horizontal distance
        C = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
        
        # EAR = (A + B) / (2.0 * C)
        ear = (A + B) / (2.0 * C) if C > 0 else 0.0
        
        return ear
    
    def _estimate_gaze_direction(self, left_eye: List[Tuple[int, int]], 
                                right_eye: List[Tuple[int, int]], w: int, h: int) -> Tuple[float, float]:
        """Estimate gaze direction based on eye center positions."""
        if not left_eye or not right_eye:
            return (0.0, 0.0)
        
        # Calculate eye centers
        left_center = np.mean(left_eye, axis=0)
        right_center = np.mean(right_eye, axis=0)
        
        # Calculate face center
        face_center_x = (left_center[0] + right_center[0]) / 2
        face_center_y = (left_center[1] + right_center[1]) / 2
        
        # Calculate gaze direction relative to frame center
        frame_center_x = w / 2
        frame_center_y = h / 2
        
        # Normalize gaze direction to [-1, 1] range
        gaze_x = (face_center_x - frame_center_x) / (w / 2)
        gaze_y = (face_center_y - frame_center_y) / (h / 2)
        
        # Clamp to [-1, 1]
        gaze_x = np.clip(gaze_x, -1.0, 1.0)
        gaze_y = np.clip(gaze_y, -1.0, 1.0)
        
        return (gaze_x, gaze_y)
    
    def _calculate_gaze_confidence(self, left_eye: List[Tuple[int, int]], 
                                  right_eye: List[Tuple[int, int]], avg_ear: float) -> float:
        """Calculate confidence in gaze estimation."""
        if not left_eye or not right_eye:
            return 0.0
        
        # Base confidence on eye visibility and EAR stability
        confidence = 0.5
        
        # Higher confidence if eyes are open
        if avg_ear > self.ear_threshold:
            confidence += 0.3
        
        # Higher confidence if EAR is stable (not blinking)
        if len(self.ear_history) > 1:
            ear_variance = np.var(self.ear_history)
            if ear_variance < 0.01:  # Low variance indicates stable gaze
                confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _smooth_gaze_direction(self) -> Tuple[float, float]:
        """Smooth gaze direction using moving average."""
        if not self.gaze_history:
            return (0.0, 0.0)
        
        # Calculate moving average
        avg_x = np.mean([gaze[0] for gaze in self.gaze_history])
        avg_y = np.mean([gaze[1] for gaze in self.gaze_history])
        
        return (avg_x, avg_y)
    
    def _get_empty_gaze_result(self) -> Dict[str, Any]:
        """Return empty gaze estimation result."""
        return {
            'gaze_direction': (0.0, 0.0),
            'confidence': 0.0,
            'left_ear': 0.0,
            'right_ear': 0.0,
            'avg_ear': 0.0,
            'is_blinking': False,
            'left_eye_landmarks': [],
            'right_eye_landmarks': [],
            'backend': self.backend
        }
    
    def detect_blink(self, ear: float) -> bool:
        """Detect if the person is blinking based on EAR."""
        return ear < self.ear_threshold
    
    def get_blink_rate(self) -> float:
        """Calculate blink rate from recent EAR history."""
        if len(self.ear_history) < 2:
            return 0.0
        
        blink_count = 0
        for i in range(1, len(self.ear_history)):
            if self.ear_history[i] < self.ear_threshold and self.ear_history[i-1] >= self.ear_threshold:
                blink_count += 1
        
        # Assuming 30 FPS, calculate blinks per minute
        time_window = len(self.ear_history) / 30.0  # seconds
        blinks_per_minute = (blink_count / time_window) * 60.0
        
        return blinks_per_minute
    
    def draw_gaze_visualization(self, frame: np.ndarray, gaze_result: Dict[str, Any]) -> np.ndarray:
        """Draw gaze direction visualization on the frame."""
        result_frame = frame.copy()
        
        gaze_direction = gaze_result.get('gaze_direction', (0.0, 0.0))
        confidence = gaze_result.get('confidence', 0.0)
        
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Draw gaze direction arrow
        arrow_length = 50
        arrow_x = int(center_x + gaze_direction[0] * arrow_length)
        arrow_y = int(center_y + gaze_direction[1] * arrow_length)
        
        # Color based on confidence
        color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.4 else (0, 0, 255)
        
        # Draw arrow
        cv2.arrowedLine(result_frame, (center_x, center_y), (arrow_x, arrow_y), color, 2)
        
        # Draw confidence text
        cv2.putText(result_frame, f"Gaze: ({gaze_direction[0]:.2f}, {gaze_direction[1]:.2f})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(result_frame, f"Confidence: {confidence:.2f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw eye landmarks if available
        left_eye = gaze_result.get('left_eye_landmarks', [])
        right_eye = gaze_result.get('right_eye_landmarks', [])
        
        for landmark in left_eye + right_eye:
            cv2.circle(result_frame, landmark, 2, (255, 0, 0), -1)
        
        return result_frame 