"""
Face Detection Module

This module provides multi-backend face detection capabilities using:
- MediaPipe (primary, recommended)
- DeepFace (fallback)
- OpenCV (fallback)

The module includes face tracking and visualization utilities.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
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
    # Suppress warning - use logger at ERROR level
    if False:  # Disable warnings
        logging.warning(f"MediaPipe not available: {e}. Using OpenCV only.")
    mp = None

# Optional imports
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    # Suppress warning
    pass

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    # Suppress warning
    pass

from ..utils.logger import get_logger

logger = get_logger(__name__)


class FaceDetector:
    """Multi-backend face detector with tracking capabilities."""
    
    def __init__(self, backend: str = "mediapipe", confidence_threshold: float = 0.5):
        """
        Initialize face detector.
        
        Args:
            backend: Detection backend ("mediapipe", "deepface", "opencv", "dlib")
            confidence_threshold: Minimum confidence for face detection
        """
        self.backend = backend
        self.confidence_threshold = confidence_threshold
        self.tracked_faces = {}  # face_id -> {bbox, confidence, frame_count}
        self.next_face_id = 0
        
        # Initialize MediaPipe
        if MP_AVAILABLE and mp is not None and backend == "mediapipe":
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,  # 0 for short-range, 1 for full-range
                min_detection_confidence=confidence_threshold
            )
        else:
            self.mp_face_detection = None
            self.mp_drawing = None
            self.face_detection = None
        
        # Initialize OpenCV cascade classifier
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        logger.info(f"Face detector initialized with backend: {backend}")
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in the frame using the specified backend.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of face detections with bounding boxes and confidence scores
        """
        if self.backend == "mediapipe":
            return self._detect_mediapipe(frame)
        elif self.backend == "deepface" and DEEPFACE_AVAILABLE:
            return self._detect_deepface(frame)
        elif self.backend == "opencv":
            return self._detect_opencv(frame)
        elif self.backend == "dlib" and DLIB_AVAILABLE:
            return self._detect_dlib(frame)
        else:
            # Fallback to MediaPipe
            logger.warning(f"Backend {self.backend} not available, using MediaPipe")
            return self._detect_mediapipe(frame)
    
    def _detect_mediapipe(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using MediaPipe."""
        # Check if MediaPipe is initialized
        if self.face_detection is None:
            logger.warning("MediaPipe not initialized, falling back to OpenCV")
            return self._detect_opencv(frame)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        faces = []
        if results.detections:
            h, w = frame.shape[:2]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                confidence = detection.score[0]
                
                if confidence >= self.confidence_threshold:
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    faces.append({
                        'bbox': (x, y, width, height),
                        'confidence': confidence,
                        'landmarks': self._extract_mediapipe_landmarks(detection, w, h)
                    })
        
        return faces
    
    def _detect_deepface(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using DeepFace."""
        try:
            results = DeepFace.extract_faces(
                frame, 
                detector_backend='opencv',
                enforce_detection=False
            )
            
            faces = []
            for result in results:
                bbox = result['facial_area']
                faces.append({
                    'bbox': (bbox['x'], bbox['y'], bbox['w'], bbox['h']),
                    'confidence': 0.8,  # DeepFace doesn't provide confidence
                    'landmarks': None
                })
            
            return faces
        except Exception as e:
            logger.error(f"DeepFace detection failed: {e}")
            return []
    
    def _detect_opencv(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using OpenCV Haar cascades."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        detected_faces = []
        for (x, y, w, h) in faces:
            detected_faces.append({
                'bbox': (x, y, w, h),
                'confidence': 0.7,  # OpenCV doesn't provide confidence
                'landmarks': None
            })
        
        return detected_faces
    
    def _detect_dlib(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using Dlib (if available)."""
        if not DLIB_AVAILABLE:
            return []
        
        try:
            # This would require a dlib face detector
            # For now, return empty list
            return []
        except Exception as e:
            logger.error(f"Dlib detection failed: {e}")
            return []
    
    def _extract_mediapipe_landmarks(self, detection, w: int, h: int) -> Dict[str, Tuple[int, int]]:
        """Extract facial landmarks from MediaPipe detection."""
        landmarks = {}
        if hasattr(detection.location_data, 'relative_keypoints'):
            keypoints = detection.location_data.relative_keypoints
            landmark_names = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear']
            
            for i, (name, keypoint) in enumerate(zip(landmark_names, keypoints)):
                x = int(keypoint.x * w)
                y = int(keypoint.y * h)
                landmarks[name] = (x, y)
        
        return landmarks
    
    def track_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect and track faces across frames.
        
        Args:
            frame: Input frame
            
        Returns:
            List of tracked faces with IDs
        """
        current_faces = self.detect_faces(frame)
        tracked_faces = []
        
        # Simple tracking by detection (can be improved with Kalman filters)
        for face in current_faces:
            bbox = face['bbox']
            
            # Find closest tracked face
            best_match = None
            best_iou = 0.5  # Minimum IoU threshold
            
            for face_id, tracked_face in self.tracked_faces.items():
                tracked_bbox = tracked_face['bbox']
                iou = self._calculate_iou(bbox, tracked_bbox)
                
                if iou > best_iou:
                    best_iou = iou
                    best_match = face_id
            
            if best_match is not None:
                # Update existing track
                self.tracked_faces[best_match].update({
                    'bbox': bbox,
                    'confidence': face['confidence'],
                    'frame_count': self.tracked_faces[best_match]['frame_count'] + 1
                })
                face['track_id'] = best_match
            else:
                # Create new track
                face['track_id'] = self.next_face_id
                self.tracked_faces[self.next_face_id] = {
                    'bbox': bbox,
                    'confidence': face['confidence'],
                    'frame_count': 1
                }
                self.next_face_id += 1
            
            tracked_faces.append(face)
        
        # Remove stale tracks
        stale_ids = []
        for face_id, tracked_face in self.tracked_faces.items():
            if tracked_face['frame_count'] > 30:  # Remove after 30 frames without detection
                stale_ids.append(face_id)
        
        for face_id in stale_ids:
            del self.tracked_faces[face_id]
        
        return tracked_faces
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                      bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def draw_faces(self, frame: np.ndarray, faces: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw face bounding boxes and landmarks on the frame.
        
        Args:
            frame: Input frame
            faces: List of detected faces
            
        Returns:
            Frame with visualizations
        """
        result_frame = frame.copy()
        
        for face in faces:
            bbox = face['bbox']
            confidence = face.get('confidence', 0.0)
            track_id = face.get('track_id', None)
            
            x, y, w, h = bbox
            
            # Draw bounding box
            color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255)
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw confidence and track ID
            label = f"Face {track_id}" if track_id is not None else "Face"
            label += f" ({confidence:.2f})"
            cv2.putText(result_frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw landmarks if available
            landmarks = face.get('landmarks', {})
            for landmark_name, (lx, ly) in landmarks.items():
                cv2.circle(result_frame, (lx, ly), 3, (255, 0, 0), -1)
        
        return result_frame
    
    def get_face_count(self, faces: List[Dict[str, Any]]) -> int:
        """Get the number of detected faces."""
        return len(faces)
    
    def get_largest_face(self, faces: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Get the largest face by bounding box area."""
        if not faces:
            return None
        
        largest_face = max(faces, key=lambda f: f['bbox'][2] * f['bbox'][3])
        return largest_face 