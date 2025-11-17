"""
Configuration management for the attentiveness detection system.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
import json


@dataclass
class CameraConfig:
    """Camera configuration settings."""
    device_id: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    buffer_size: int = 1


@dataclass
class GPUConfig:
    """GPU acceleration configuration."""
    use_gpu: bool = True
    cuda_device: int = 0
    enable_nvdec: bool = True
    memory_fraction: float = 0.8


@dataclass
class ModelConfig:
    """Model configuration settings."""
    face_detection_backend: str = "mediapipe"  # mediapipe, deepface, opencv
    gaze_estimation_model: str = "openface"    # openface, dlib
    pose_estimation_model: str = "dlib"        # dlib, openface
    confidence_threshold: float = 0.7
    max_faces: int = 1


@dataclass
class PrivacyConfig:
    """Privacy and data protection settings."""
    anonymize_data: bool = True
    store_raw_video: bool = False
    store_facial_data: bool = False
    data_retention_days: int = 7
    enable_logging: bool = True
    log_level: str = "INFO"


@dataclass
class PerformanceConfig:
    """Performance optimization settings."""
    enable_motion_estimation: bool = True
    enable_model_compression: bool = True
    quantization_bits: int = 8
    enable_pruning: bool = True
    target_fps: int = 30
    max_latency_ms: int = 100


@dataclass
class AttentivenessConfig:
    """Attentiveness scoring configuration."""
    gaze_weight: float = 0.4
    pose_weight: float = 0.3
    blink_weight: float = 0.2
    yawn_weight: float = 0.1
    attention_threshold: float = 0.6
    smoothing_window: int = 10
    update_interval_ms: int = 100


class Config:
    """Main configuration class for the attentiveness detection system."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration with optional config file."""
        self.camera = CameraConfig()
        self.gpu = GPUConfig()
        self.model = ModelConfig()
        self.privacy = PrivacyConfig()
        self.performance = PerformanceConfig()
        self.attentiveness = AttentivenessConfig()
        
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
    
    def load_from_file(self, config_file: str) -> None:
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update each config section
            for section_name, section_data in config_data.items():
                if hasattr(self, section_name):
                    section = getattr(self, section_name)
                    for key, value in section_data.items():
                        if hasattr(section, key):
                            setattr(section, key, value)
        except Exception as e:
            print(f"Warning: Could not load config file {config_file}: {e}")
    
    def save_to_file(self, config_file: str) -> None:
        """Save current configuration to JSON file."""
        config_data = {}
        
        # Convert each config section to dictionary
        for section_name in ['camera', 'gpu', 'model', 'privacy', 'performance', 'attentiveness']:
            section = getattr(self, section_name)
            config_data[section_name] = {
                key: getattr(section, key) 
                for key in section.__dataclass_fields__.keys()
            }
        
        try:
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save config file {config_file}: {e}")
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information and capabilities."""
        import torch
        
        gpu_info = {
            'available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
            'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'memory_allocated': torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0,
            'memory_reserved': torch.cuda.memory_reserved(0) if torch.cuda.is_available() else 0,
        }
        
        return gpu_info
    
    def validate_config(self) -> bool:
        """Validate configuration settings."""
        errors = []
        
        # Validate camera settings
        if self.camera.width <= 0 or self.camera.height <= 0:
            errors.append("Camera dimensions must be positive")
        
        if self.camera.fps <= 0:
            errors.append("Camera FPS must be positive")
        
        # Validate model settings
        if self.model.confidence_threshold < 0 or self.model.confidence_threshold > 1:
            errors.append("Confidence threshold must be between 0 and 1")
        
        # Validate attentiveness weights
        total_weight = (self.attentiveness.gaze_weight + 
                       self.attentiveness.pose_weight + 
                       self.attentiveness.blink_weight + 
                       self.attentiveness.yawn_weight)
        
        if abs(total_weight - 1.0) > 0.01:
            errors.append("Attentiveness weights must sum to 1.0")
        
        # Validate performance settings
        if self.performance.target_fps <= 0:
            errors.append("Target FPS must be positive")
        
        if self.performance.max_latency_ms <= 0:
            errors.append("Max latency must be positive")
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True
    
    def get_optimal_settings(self) -> Dict[str, Any]:
        """Get optimal settings based on system capabilities."""
        gpu_info = self.get_gpu_info()
        
        # Adjust settings based on GPU availability
        if not gpu_info['available']:
            self.gpu.use_gpu = False
            self.gpu.enable_nvdec = False
            self.performance.enable_model_compression = True
            self.model.face_detection_backend = "opencv"  # Fallback to CPU-optimized backend
        
        # Adjust camera settings for performance
        if self.performance.target_fps > 30:
            self.camera.fps = 30  # Cap at 30 FPS for stability
        
        # Adjust model settings for performance
        if not self.gpu.use_gpu:
            self.model.confidence_threshold = 0.8  # Higher threshold for CPU processing
        
        return {
            'gpu_available': gpu_info['available'],
            'recommended_backend': self.model.face_detection_backend,
            'recommended_fps': self.camera.fps,
            'performance_mode': 'GPU' if self.gpu.use_gpu else 'CPU'
        }


# Global configuration instance
config = Config()

# Default configuration file path
DEFAULT_CONFIG_FILE = "data/configs/default_config.json"

# Create default config directory if it doesn't exist
os.makedirs("data/configs", exist_ok=True)

# Load default configuration if available
if os.path.exists(DEFAULT_CONFIG_FILE):
    config.load_from_file(DEFAULT_CONFIG_FILE)
else:
    # Save default configuration
    config.save_to_file(DEFAULT_CONFIG_FILE) 