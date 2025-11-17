"""
Logging utilities for the attentiveness detection system.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional
from pathlib import Path

from .config import config


class AttentivenessLogger:
    """Custom logger for the attentiveness detection system."""
    
    def __init__(self, name: str = "attentiveness_detection", log_file: Optional[str] = None):
        """Initialize the logger."""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Prevent duplicate handlers
        if self.logger.handlers:
            return
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Console handler - suppress warnings
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.ERROR)  # Only show errors, suppress warnings
        console_handler.setFormatter(simple_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if logging is enabled)
        if config.privacy.enable_logging:
            if log_file is None:
                # Create logs directory
                logs_dir = Path("logs")
                logs_dir.mkdir(exist_ok=True)
                
                # Create log file with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = logs_dir / f"attentiveness_detection_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(detailed_formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str) -> None:
        """Log critical message."""
        self.logger.critical(message)
    
    def log_performance(self, fps: float, latency: float, gpu_usage: Optional[float] = None) -> None:
        """Log performance metrics."""
        message = f"Performance - FPS: {fps:.2f}, Latency: {latency:.2f}ms"
        if gpu_usage is not None:
            message += f", GPU Usage: {gpu_usage:.1f}%"
        self.info(message)
    
    def log_attentiveness_score(self, score: float, gaze_score: float, pose_score: float, 
                               blink_score: float, yawn_score: float) -> None:
        """Log attentiveness scoring details."""
        self.debug(f"Attentiveness Score: {score:.3f} (Gaze: {gaze_score:.3f}, "
                  f"Pose: {pose_score:.3f}, Blink: {blink_score:.3f}, Yawn: {yawn_score:.3f})")
    
    def log_face_detection(self, num_faces: int, confidence: float, processing_time: float) -> None:
        """Log face detection results."""
        self.debug(f"Face Detection - Faces: {num_faces}, Confidence: {confidence:.3f}, "
                  f"Time: {processing_time:.2f}ms")
    
    def log_gaze_estimation(self, gaze_direction: tuple, confidence: float, processing_time: float) -> None:
        """Log gaze estimation results."""
        self.debug(f"Gaze Estimation - Direction: {gaze_direction}, Confidence: {confidence:.3f}, "
                  f"Time: {processing_time:.2f}ms")
    
    def log_pose_estimation(self, pitch: float, yaw: float, roll: float, processing_time: float) -> None:
        """Log pose estimation results."""
        self.debug(f"Pose Estimation - Pitch: {pitch:.2f}°, Yaw: {yaw:.2f}°, Roll: {roll:.2f}°, "
                  f"Time: {processing_time:.2f}ms")
    
    def log_behavioral_cues(self, blink_detected: bool, yawn_detected: bool, 
                           blink_rate: float, yawn_rate: float) -> None:
        """Log behavioral cues detection."""
        self.debug(f"Behavioral Cues - Blink: {blink_detected}, Yawn: {yawn_detected}, "
                  f"Blink Rate: {blink_rate:.2f}/min, Yawn Rate: {yawn_rate:.2f}/min")
    
    def log_privacy_event(self, event_type: str, data_type: str, anonymized: bool) -> None:
        """Log privacy-related events."""
        self.info(f"Privacy Event - Type: {event_type}, Data: {data_type}, Anonymized: {anonymized}")
    
    def log_error_with_context(self, error: Exception, context: str) -> None:
        """Log error with additional context."""
        self.error(f"Error in {context}: {str(error)}")
        if hasattr(error, '__traceback__'):
            import traceback
            self.debug(f"Traceback: {traceback.format_exc()}")
    
    def log_system_info(self) -> None:
        """Log system information."""
        import torch
        import cv2
        
        self.info("=== System Information ===")
        self.info(f"Python Version: {sys.version}")
        self.info(f"OpenCV Version: {cv2.__version__}")
        self.info(f"PyTorch Version: {torch.__version__}")
        self.info(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            self.info(f"CUDA Device Count: {torch.cuda.device_count()}")
            self.info(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        
        # Log configuration
        self.info("=== Configuration ===")
        self.info(f"Camera: {config.camera.width}x{config.camera.height} @ {config.camera.fps}fps")
        self.info(f"GPU Enabled: {config.gpu.use_gpu}")
        self.info(f"Face Detection Backend: {config.model.face_detection_backend}")
        self.info(f"Privacy Mode: {'Enabled' if config.privacy.anonymize_data else 'Disabled'}")


# Global logger instance
logger = AttentivenessLogger()


def get_logger(name: str = "attentiveness_detection") -> AttentivenessLogger:
    """Get a logger instance."""
    return AttentivenessLogger(name)


def log_function_call(func):
    """Decorator to log function calls."""
    def wrapper(*args, **kwargs):
        logger.debug(f"Calling {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Completed {func.__name__}")
            return result
        except Exception as e:
            logger.log_error_with_context(e, func.__name__)
            raise
    return wrapper


def log_performance_metrics(func):
    """Decorator to log performance metrics."""
    import time
    
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
            logger.debug(f"{func.__name__} took {processing_time:.2f}ms")
            return result
        except Exception as e:
            logger.log_error_with_context(e, func.__name__)
            raise
    return wrapper 