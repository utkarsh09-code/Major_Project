"""
Behavioral Cues Detection Module

This module detects behavioral cues such as blinking, yawning, and other
indicators that can be used to assess attentiveness and fatigue.
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Optional
import time

from ..utils.logger import get_logger

logger = get_logger(__name__)


class BehavioralCuesDetector:
    """Detects behavioral cues like blinking, yawning, and movement patterns."""
    
    def __init__(self):
        """Initialize the behavioral cues detector."""
        # Eye aspect ratio thresholds (adjusted for MediaPipe)
        self.ear_threshold = 0.21  # Adjusted for MediaPipe landmarks
        self.ear_history = []
        self.max_ear_history = 30
        
        # Mouth aspect ratio thresholds
        self.mar_threshold = 0.5  # Adjusted for MediaPipe
        self.mar_history = []
        self.max_mar_history = 30
        
        # Blink detection
        self.blink_counter = 0
        self.last_blink_time = time.time()
        self.blink_rate_history = []
        
        # Yawn detection
        self.yawn_counter = 0
        self.last_yawn_time = time.time()
        self.yawn_rate_history = []
        
        # Movement detection
        self.movement_threshold = 0.05  # Reduced for more sensitive detection
        self.movement_history = []
        self.max_movement_history = 10
        
        # Fatigue detection
        self.fatigue_threshold = 0.6
        self.fatigue_history = []
        self.max_fatigue_history = 60
        
        logger.info("Behavioral cues detector initialized")
    
    def detect_cues(self, frame: np.ndarray, face_landmarks: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detect behavioral cues from the frame and face landmarks.
        
        Args:
            frame: Input frame
            face_landmarks: Face landmarks from MediaPipe or other detector
            
        Returns:
            Dictionary containing detected behavioral cues
        """
        # If no face landmarks provided, try to detect faces first
        if face_landmarks is None:
            # Create a simple test result for when no face is detected
            return self._get_empty_cues_result()
        
        # Extract eye and mouth landmarks from MediaPipe format
        eye_landmarks = self._extract_eye_landmarks_mediapipe(face_landmarks)
        mouth_landmarks = self._extract_mouth_landmarks_mediapipe(face_landmarks)
        
        if not eye_landmarks or not mouth_landmarks:
            return self._get_empty_cues_result()
        
        # Calculate aspect ratios
        left_ear = self._calculate_eye_aspect_ratio_mediapipe(eye_landmarks['left'])
        right_ear = self._calculate_eye_aspect_ratio_mediapipe(eye_landmarks['right'])
        avg_ear = (left_ear + right_ear) / 2.0
        
        mar = self._calculate_mouth_aspect_ratio_mediapipe(mouth_landmarks)
        
        # Update histories
        self._update_ear_history(avg_ear)
        self._update_mar_history(mar)
        
        # Detect blinks
        is_blinking = self._detect_blink(avg_ear)
        blink_rate = self._calculate_blink_rate()
        
        # Detect yawns
        is_yawning = self._detect_yawn(mar)
        yawn_rate = self._calculate_yawn_rate()
        
        # Detect movement
        movement_level = self._detect_movement_mediapipe(face_landmarks)
        
        # Calculate fatigue score
        fatigue_score = self._calculate_fatigue_score(blink_rate, yawn_rate, movement_level)
        
        # Detect fatigue level
        fatigue_level = self._detect_fatigue_level(fatigue_score)
        
        return {
            'left_ear': left_ear,
            'right_ear': right_ear,
            'avg_ear': avg_ear,
            'mar': mar,
            'is_blinking': is_blinking,
            'blink_rate': blink_rate,
            'is_yawning': is_yawning,
            'yawn_rate': yawn_rate,
            'movement_level': movement_level,
            'fatigue_score': fatigue_score,
            'fatigue_level': fatigue_level,
            'eye_landmarks': eye_landmarks,
            'mouth_landmarks': mouth_landmarks,
            'confidence': self._calculate_confidence(avg_ear, mar, movement_level)
        }
    
    def _extract_eye_landmarks_mediapipe(self, face_landmarks: Dict[str, Any]) -> Optional[Dict[str, List[tuple]]]:
        """Extract eye landmarks from MediaPipe face mesh."""
        try:
            landmarks = face_landmarks.get('landmarks', [])
            if len(landmarks) < 468:  # MediaPipe face mesh has 468 landmarks
                return None
            
            # MediaPipe eye landmark indices
            left_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            right_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            
            left_eye = [landmarks[i] for i in left_eye_indices if i < len(landmarks)]
            right_eye = [landmarks[i] for i in right_eye_indices if i < len(landmarks)]
            
            if len(left_eye) < 8 or len(right_eye) < 8:
                return None
            
            return {
                'left': left_eye,
                'right': right_eye
            }
            
        except Exception as e:
            logger.error(f"Error extracting eye landmarks: {e}")
            return None
    
    def _extract_mouth_landmarks_mediapipe(self, face_landmarks: Dict[str, Any]) -> Optional[List[tuple]]:
        """Extract mouth landmarks from MediaPipe face mesh."""
        try:
            landmarks = face_landmarks.get('landmarks', [])
            if len(landmarks) < 468:
                return None
            
            # MediaPipe mouth landmark indices
            mouth_indices = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
            
            mouth_landmarks = [landmarks[i] for i in mouth_indices if i < len(landmarks)]
            
            if len(mouth_landmarks) < 8:
                return None
            
            return mouth_landmarks
            
        except Exception as e:
            logger.error(f"Error extracting mouth landmarks: {e}")
            return None
    
    def _calculate_eye_aspect_ratio_mediapipe(self, eye_landmarks: List[tuple]) -> float:
        """Calculate eye aspect ratio for MediaPipe landmarks."""
        try:
            if len(eye_landmarks) < 6:
                return 0.0
            
            # Use 6 key points for EAR calculation
            # Vertical distances
            A = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
            B = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
            
            # Horizontal distance
            C = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
            
            # Eye aspect ratio
            ear = (A + B) / (2.0 * C)
            
            return ear
            
        except Exception as e:
            logger.error(f"Error calculating EAR: {e}")
            return 0.0
    
    def _calculate_mouth_aspect_ratio_mediapipe(self, mouth_landmarks: List[tuple]) -> float:
        """Calculate mouth aspect ratio for MediaPipe landmarks."""
        try:
            if len(mouth_landmarks) < 6:
                return 0.0
            
            # Use key mouth points for MAR calculation
            # Vertical distance
            A = np.linalg.norm(np.array(mouth_landmarks[2]) - np.array(mouth_landmarks[6]))
            
            # Horizontal distance
            B = np.linalg.norm(np.array(mouth_landmarks[0]) - np.array(mouth_landmarks[4]))
            
            # Mouth aspect ratio
            mar = A / B if B > 0 else 0.0
            
            return mar
            
        except Exception as e:
            logger.error(f"Error calculating MAR: {e}")
            return 0.0
    
    def _update_ear_history(self, ear: float) -> None:
        """Update eye aspect ratio history."""
        self.ear_history.append(ear)
        if len(self.ear_history) > self.max_ear_history:
            self.ear_history.pop(0)
    
    def _update_mar_history(self, mar: float) -> None:
        """Update mouth aspect ratio history."""
        self.mar_history.append(mar)
        if len(self.mar_history) > self.max_mar_history:
            self.mar_history.pop(0)
    
    def _detect_blink(self, ear: float) -> bool:
        """Detect if the person is blinking."""
        is_blinking = ear < self.ear_threshold
        
        if is_blinking and len(self.ear_history) > 1:
            # Check if this is a new blink (ear was high, now low)
            if self.ear_history[-2] >= self.ear_threshold:
                self.blink_counter += 1
                self.last_blink_time = time.time()
        
        return is_blinking
    
    def _detect_yawn(self, mar: float) -> bool:
        """Detect if the person is yawning."""
        is_yawning = mar > self.mar_threshold
        
        if is_yawning and len(self.mar_history) > 1:
            # Check if this is a new yawn (mar was low, now high)
            if self.mar_history[-2] <= self.mar_threshold:
                self.yawn_counter += 1
                self.last_yawn_time = time.time()
        
        return is_yawning
    
    def _calculate_blink_rate(self) -> float:
        """Calculate blink rate (blinks per minute)."""
        current_time = time.time()
        time_window = 60.0  # 1 minute window
        
        # Count blinks in the last minute
        recent_blinks = 0
        for i in range(len(self.ear_history) - 1):
            if (self.ear_history[i] >= self.ear_threshold and 
                self.ear_history[i + 1] < self.ear_threshold):
                recent_blinks += 1
        
        # Estimate blink rate
        if len(self.ear_history) > 0:
            time_span = len(self.ear_history) / 30.0  # Assuming 30 FPS
            blink_rate = (recent_blinks / time_span) * 60.0
        else:
            blink_rate = 0.0
        
        return min(blink_rate, 60.0)  # Cap at 60 blinks per minute
    
    def _calculate_yawn_rate(self) -> float:
        """Calculate yawn rate (yawns per hour)."""
        current_time = time.time()
        time_window = 3600.0  # 1 hour window
        
        # Count yawns in the last hour
        recent_yawns = 0
        for i in range(len(self.mar_history) - 1):
            if (self.mar_history[i] <= self.mar_threshold and 
                self.mar_history[i + 1] > self.mar_threshold):
                recent_yawns += 1
        
        # Estimate yawn rate
        if len(self.mar_history) > 0:
            time_span = len(self.mar_history) / 30.0  # Assuming 30 FPS
            yawn_rate = (recent_yawns / time_span) * 3600.0
        else:
            yawn_rate = 0.0
        
        return min(yawn_rate, 10.0)  # Cap at 10 yawns per hour
    
    def _detect_movement_mediapipe(self, face_landmarks: Dict[str, Any]) -> float:
        """Detect movement level using MediaPipe landmarks."""
        try:
            landmarks = face_landmarks.get('landmarks', [])
            if len(landmarks) < 10:
                return 0.0
            
            # Use key facial landmarks for movement detection
            key_indices = [1, 10, 33, 263, 61, 291]  # Nose, chin, eyes, mouth
            current_positions = []
            
            for idx in key_indices:
                if idx < len(landmarks):
                    current_positions.append(landmarks[idx])
            
            if len(current_positions) < 3:
                return 0.0
            
            # Calculate movement magnitude
            if len(self.movement_history) > 0:
                last_positions = self.movement_history[-1]
                if len(last_positions) == len(current_positions):
                    movement = 0.0
                    for i in range(len(current_positions)):
                        if i < len(last_positions):
                            diff = np.linalg.norm(
                                np.array(current_positions[i]) - np.array(last_positions[i])
                            )
                            movement += diff
                    
                    movement /= len(current_positions)
                    movement = min(movement, 1.0)  # Normalize
                else:
                    movement = 0.0
            else:
                movement = 0.0
            
            # Update movement history
            self.movement_history.append(current_positions)
            if len(self.movement_history) > self.max_movement_history:
                self.movement_history.pop(0)
            
            return movement
            
        except Exception as e:
            logger.error(f"Error detecting movement: {e}")
            return 0.0
    
    def _calculate_fatigue_score(self, blink_rate: float, yawn_rate: float, movement_level: float) -> float:
        """Calculate fatigue score (0.0 = alert, 1.0 = very fatigued)."""
        # Normalize inputs
        normalized_blink_rate = min(blink_rate / 20.0, 1.0)  # Normal blink rate ~15-20/min
        normalized_yawn_rate = min(yawn_rate / 5.0, 1.0)     # Normal yawn rate ~0-5/hour
        normalized_movement = movement_level
        
        # Weighted combination
        fatigue_score = (
            0.4 * normalized_blink_rate +
            0.4 * normalized_yawn_rate +
            0.2 * normalized_movement
        )
        
        return min(fatigue_score, 1.0)
    
    def _detect_fatigue_level(self, fatigue_score: float) -> str:
        """Detect fatigue level based on fatigue score."""
        if fatigue_score > 0.7:
            return "High Fatigue"
        elif fatigue_score > 0.4:
            return "Moderate Fatigue"
        elif fatigue_score > 0.2:
            return "Low Fatigue"
        else:
            return "Alert"
    
    def _calculate_confidence(self, ear: float, mar: float, movement: float) -> float:
        """Calculate confidence in behavioral cues detection."""
        # Higher confidence when landmarks are well-defined
        confidence = 0.5  # Base confidence
        
        # Adjust based on landmark quality
        if ear > 0.1:  # Good eye landmarks
            confidence += 0.2
        if mar > 0.1:  # Good mouth landmarks
            confidence += 0.2
        if movement < 0.5:  # Stable face
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _get_empty_cues_result(self) -> Dict[str, Any]:
        """Return empty behavioral cues result."""
        return {
            'left_ear': 0.0,
            'right_ear': 0.0,
            'avg_ear': 0.0,
            'mar': 0.0,
            'is_blinking': False,
            'blink_rate': 0.0,
            'is_yawning': False,
            'yawn_rate': 0.0,
            'movement_level': 0.0,
            'fatigue_score': 0.0,
            'fatigue_level': 'No Face Detected',
            'eye_landmarks': None,
            'mouth_landmarks': None,
            'confidence': 0.0
        }
    
    def get_behavioral_summary(self, cues_result: Dict[str, Any]) -> str:
        """Get human-readable behavioral summary."""
        blink_rate = cues_result.get('blink_rate', 0.0)
        yawn_rate = cues_result.get('yawn_rate', 0.0)
        fatigue_score = cues_result.get('fatigue_score', 0.0)
        fatigue_level = cues_result.get('fatigue_level', 'Unknown')
        
        if fatigue_level == 'High Fatigue':
            return "High Fatigue - Frequent blinking and yawning detected"
        elif fatigue_level == 'Moderate Fatigue':
            return "Moderate Fatigue - Some signs of tiredness"
        elif fatigue_level == 'Low Fatigue':
            return "Low Fatigue - Slight signs of tiredness"
        elif blink_rate > 25:
            return "Frequent Blinking - May indicate tiredness"
        elif yawn_rate > 3:
            return "Frequent Yawning - May indicate fatigue"
        else:
            return "Normal Behavior - Alert and attentive" 