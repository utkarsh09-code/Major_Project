"""
Privacy utilities for data anonymization and PII protection.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple
import hashlib
import json
from datetime import datetime, timedelta
import os
from pathlib import Path

from .config import config
from .logger import logger


class PrivacyManager:
    """Manages data privacy and anonymization for the attentiveness detection system."""
    
    def __init__(self):
        """Initialize the privacy manager."""
        self.anonymization_enabled = config.privacy.anonymize_data
        self.data_retention_days = config.privacy.data_retention_days
        self.stored_data = {}
        self.data_hashes = {}
        
        # Create data directory if it doesn't exist
        self.data_dir = Path("data/anonymized")
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def anonymize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Anonymize a video frame by blurring faces and removing identifiable features."""
        if not self.anonymization_enabled:
            return frame
        
        try:
            # Create a copy to avoid modifying the original
            anonymized_frame = frame.copy()
            
            # Detect faces for anonymization
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(anonymized_frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Blur detected faces
            for (x, y, w, h) in faces:
                # Extract face region
                face_region = anonymized_frame[y:y+h, x:x+w]
                
                # Apply strong blur to face region
                blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
                
                # Replace face region with blurred version
                anonymized_frame[y:y+h, x:x+w] = blurred_face
            
            logger.debug(f"Anonymized frame with {len(faces)} faces detected")
            return anonymized_frame
            
        except Exception as e:
            logger.log_error_with_context(e, "frame_anonymization")
            return frame
    
    def anonymize_facial_data(self, facial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize facial landmark and feature data."""
        if not self.anonymization_enabled:
            return facial_data
        
        try:
            anonymized_data = {}
            
            for key, value in facial_data.items():
                if key in ['landmarks', 'face_bbox', 'eye_centers']:
                    # Hash the coordinates to remove direct PII
                    if isinstance(value, (list, tuple)):
                        anonymized_data[key] = [self._hash_coordinate(coord) for coord in value]
                    else:
                        anonymized_data[key] = self._hash_coordinate(value)
                elif key in ['gaze_direction', 'head_pose']:
                    # Normalize angles to remove absolute position information
                    if isinstance(value, (list, tuple)):
                        anonymized_data[key] = [self._normalize_angle(angle) for angle in value]
                    else:
                        anonymized_data[key] = self._normalize_angle(value)
                else:
                    # Keep non-PII data as is
                    anonymized_data[key] = value
            
            logger.debug("Anonymized facial data")
            return anonymized_data
            
        except Exception as e:
            logger.log_error_with_context(e, "facial_data_anonymization")
            return facial_data
    
    def _hash_coordinate(self, coord: Tuple[float, float]) -> str:
        """Hash coordinate data to remove direct PII."""
        coord_str = f"{coord[0]:.2f},{coord[1]:.2f}"
        return hashlib.sha256(coord_str.encode()).hexdigest()[:8]
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to remove absolute position information."""
        # Normalize to 0-360 range and discretize
        normalized = (angle % 360) // 10 * 10  # Discretize to 10-degree bins
        return normalized
    
    def store_anonymized_data(self, data: Dict[str, Any], data_type: str) -> str:
        """Store anonymized data with privacy protection."""
        if not config.privacy.store_facial_data:
            return ""
        
        try:
            # Create timestamp for data identification
            timestamp = datetime.now().isoformat()
            
            # Generate unique identifier
            data_id = hashlib.sha256(f"{timestamp}{data_type}".encode()).hexdigest()[:16]
            
            # Store data with metadata
            stored_data = {
                'id': data_id,
                'type': data_type,
                'timestamp': timestamp,
                'data': data,
                'anonymized': True
            }
            
            # Save to file
            filename = f"{data_id}_{data_type}.json"
            filepath = self.data_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(stored_data, f, indent=2)
            
            # Store reference for cleanup
            self.stored_data[data_id] = {
                'filepath': filepath,
                'timestamp': datetime.now(),
                'type': data_type
            }
            
            logger.log_privacy_event("store", data_type, True)
            return data_id
            
        except Exception as e:
            logger.log_error_with_context(e, "data_storage")
            return ""
    
    def cleanup_old_data(self) -> int:
        """Remove data older than retention period."""
        if not config.privacy.store_facial_data:
            return 0
        
        try:
            cutoff_date = datetime.now() - timedelta(days=self.data_retention_days)
            removed_count = 0
            
            for data_id, data_info in list(self.stored_data.items()):
                if data_info['timestamp'] < cutoff_date:
                    try:
                        # Remove file
                        if data_info['filepath'].exists():
                            data_info['filepath'].unlink()
                        
                        # Remove from tracking
                        del self.stored_data[data_id]
                        removed_count += 1
                        
                        logger.debug(f"Removed old data: {data_id}")
                        
                    except Exception as e:
                        logger.log_error_with_context(e, f"cleanup_data_{data_id}")
            
            logger.info(f"Cleaned up {removed_count} old data files")
            return removed_count
            
        except Exception as e:
            logger.log_error_with_context(e, "data_cleanup")
            return 0
    
    def get_privacy_summary(self) -> Dict[str, Any]:
        """Get privacy protection summary."""
        return {
            'anonymization_enabled': self.anonymization_enabled,
            'data_retention_days': self.data_retention_days,
            'stored_data_count': len(self.stored_data),
            'data_directory': str(self.data_dir),
            'privacy_compliance': self._check_privacy_compliance()
        }
    
    def _check_privacy_compliance(self) -> Dict[str, bool]:
        """Check privacy compliance status."""
        return {
            'data_anonymization': self.anonymization_enabled,
            'no_raw_video_storage': not config.privacy.store_raw_video,
            'limited_data_retention': self.data_retention_days <= 30,
            'logging_privacy_respect': config.privacy.enable_logging
        }
    
    def create_privacy_report(self) -> str:
        """Create a privacy compliance report."""
        summary = self.get_privacy_summary()
        compliance = summary['privacy_compliance']
        
        report = f"""
Privacy Protection Report
========================

Configuration:
- Data Anonymization: {'Enabled' if summary['anonymization_enabled'] else 'Disabled'}
- Data Retention: {summary['data_retention_days']} days
- Stored Data Files: {summary['stored_data_count']}
- Data Directory: {summary['data_directory']}

Compliance Status:
- Data Anonymization: {'✓' if compliance['data_anonymization'] else '✗'}
- No Raw Video Storage: {'✓' if compliance['no_raw_video_storage'] else '✗'}
- Limited Data Retention: {'✓' if compliance['limited_data_retention'] else '✗'}
- Privacy-Respectful Logging: {'✓' if compliance['logging_privacy_respect'] else '✗'}

Overall Compliance: {'✓' if all(compliance.values()) else '✗'}
        """
        
        return report


class DataAnonymizer:
    """Utility class for data anonymization operations."""
    
    @staticmethod
    def blur_sensitive_regions(image: np.ndarray, regions: list) -> np.ndarray:
        """Blur sensitive regions in an image."""
        anonymized_image = image.copy()
        
        for region in regions:
            x, y, w, h = region
            # Apply strong blur to sensitive region
            region_blurred = cv2.GaussianBlur(anonymized_image[y:y+h, x:x+w], (99, 99), 30)
            anonymized_image[y:y+h, x:x+w] = region_blurred
        
        return anonymized_image
    
    @staticmethod
    def remove_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove metadata that could contain PII."""
        sensitive_keys = ['timestamp', 'user_id', 'session_id', 'device_id', 'location']
        
        cleaned_data = {}
        for key, value in data.items():
            if key not in sensitive_keys:
                cleaned_data[key] = value
        
        return cleaned_data
    
    @staticmethod
    def quantize_coordinates(coordinates: list, precision: int = 2) -> list:
        """Quantize coordinate data to reduce precision and privacy risk."""
        return [[round(coord[0], precision), round(coord[1], precision)] for coord in coordinates]


# Global privacy manager instance
privacy_manager = PrivacyManager()


def anonymize_video_frame(frame: np.ndarray) -> np.ndarray:
    """Anonymize a video frame."""
    return privacy_manager.anonymize_frame(frame)


def anonymize_facial_features(facial_data: Dict[str, Any]) -> Dict[str, Any]:
    """Anonymize facial feature data."""
    return privacy_manager.anonymize_facial_data(facial_data)


def store_anonymized_data(data: Dict[str, Any], data_type: str) -> str:
    """Store anonymized data."""
    return privacy_manager.store_anonymized_data(data, data_type)


def cleanup_old_data() -> int:
    """Clean up old data files."""
    return privacy_manager.cleanup_old_data()


def get_privacy_summary() -> Dict[str, Any]:
    """Get privacy protection summary."""
    return privacy_manager.get_privacy_summary() 