"""
Video capture module with GPU acceleration for real-time processing.
"""

import cv2
import numpy as np
import time
from typing import Optional, Tuple, Dict, Any
import threading
from queue import Queue
import torch

from ..utils.config import config
from ..utils.logger import logger, log_performance_metrics
from ..utils.privacy import anonymize_video_frame


class VideoCapture:
    """Enhanced video capture with GPU acceleration and real-time processing."""
    
    def __init__(self, device_id: int = 0):
        """Initialize video capture with specified device."""
        self.device_id = device_id
        self.cap = None
        self.is_running = False
        self.frame_queue = Queue(maxsize=10)  # Buffer for frames
        self.processing_thread = None
        self.gpu_available = torch.cuda.is_available() if config.gpu.use_gpu else False
        
        # Performance tracking
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        self.processing_times = []
        
        # Initialize camera
        self._initialize_camera()
        
        logger.info(f"Video capture initialized - Device: {device_id}, GPU: {self.gpu_available}")
    
    def _initialize_camera(self) -> None:
        """Initialize camera with optimal settings."""
        try:
            self.cap = cv2.VideoCapture(self.device_id)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open camera device {self.device_id}")
            
            # Set camera properties for optimal performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.height)
            self.cap.set(cv2.CAP_PROP_FPS, config.camera.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, config.camera.buffer_size)
            
            # Enable GPU acceleration if available
            if self.gpu_available and config.gpu.enable_nvdec:
                self._setup_gpu_acceleration()
            
            logger.info(f"Camera initialized: {config.camera.width}x{config.camera.height} @ {config.camera.fps}fps")
            
        except Exception as e:
            logger.log_error_with_context(e, "camera_initialization")
            raise
    
    def _setup_gpu_acceleration(self) -> None:
        """Setup GPU acceleration for video processing."""
        try:
            # Set CUDA device
            if torch.cuda.is_available():
                torch.cuda.set_device(config.gpu.cuda_device)
                logger.info(f"GPU acceleration enabled on device {config.gpu.cuda_device}")
            
            # Configure memory usage
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            
        except Exception as e:
            logger.log_error_with_context(e, "gpu_acceleration_setup")
            self.gpu_available = False
    
    @log_performance_metrics
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera with performance tracking."""
        if not self.cap or not self.cap.isOpened():
            return False, None
        
        try:
            ret, frame = self.cap.read()
            
            if not ret:
                logger.warning("Failed to read frame from camera")
                return False, None
            
            # Update FPS counter
            self._update_fps()
            
            # Apply privacy anonymization if enabled
            if config.privacy.anonymize_data:
                frame = anonymize_video_frame(frame)
            
            return True, frame
            
        except Exception as e:
            logger.log_error_with_context(e, "frame_reading")
            return False, None
    
    def _update_fps(self) -> None:
        """Update FPS calculation."""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def start_processing_thread(self) -> None:
        """Start background processing thread for continuous frame capture."""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        logger.info("Started video processing thread")
    
    def stop_processing_thread(self) -> None:
        """Stop background processing thread."""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
            logger.info("Stopped video processing thread")
    
    def _processing_loop(self) -> None:
        """Background processing loop for continuous frame capture."""
        while self.is_running:
            try:
                ret, frame = self.read_frame()
                
                if ret and frame is not None:
                    # Add frame to queue, remove oldest if full
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except:
                            pass
                    
                    self.frame_queue.put(frame)
                
                # Small delay to prevent excessive CPU usage
                time.sleep(1.0 / config.camera.fps)
                
            except Exception as e:
                logger.log_error_with_context(e, "processing_loop")
                time.sleep(0.1)  # Brief pause on error
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame from the processing queue."""
        try:
            if not self.frame_queue.empty():
                return self.frame_queue.get_nowait()
        except:
            pass
        return None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0.0
        
        gpu_usage = None
        if self.gpu_available:
            try:
                gpu_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
            except:
                pass
        
        return {
            'fps': self.current_fps,
            'avg_processing_time_ms': avg_processing_time * 1000,
            'gpu_usage_percent': gpu_usage,
            'queue_size': self.frame_queue.qsize(),
            'is_running': self.is_running
        }
    
    def set_camera_properties(self, width: int = None, height: int = None, fps: int = None) -> bool:
        """Update camera properties dynamically."""
        try:
            if width is not None:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                config.camera.width = width
            
            if height is not None:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                config.camera.height = height
            
            if fps is not None:
                self.cap.set(cv2.CAP_PROP_FPS, fps)
                config.camera.fps = fps
            
            logger.info(f"Updated camera properties: {config.camera.width}x{config.camera.height} @ {config.camera.fps}fps")
            return True
            
        except Exception as e:
            logger.log_error_with_context(e, "camera_properties_update")
            return False
    
    def get_camera_info(self) -> Dict[str, Any]:
        """Get camera information and capabilities."""
        if not self.cap or not self.cap.isOpened():
            return {}
        
        try:
            info = {
                'device_id': self.device_id,
                'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': self.cap.get(cv2.CAP_PROP_FPS),
                'brightness': self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
                'contrast': self.cap.get(cv2.CAP_PROP_CONTRAST),
                'gpu_acceleration': self.gpu_available,
                'backend': self.cap.getBackendName()
            }
            
            return info
            
        except Exception as e:
            logger.log_error_with_context(e, "camera_info_retrieval")
            return {}
    
    def release(self) -> None:
        """Release camera resources."""
        self.stop_processing_thread()
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        logger.info("Camera resources released")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()


class GPUFrameProcessor:
    """GPU-accelerated frame processing utilities."""
    
    def __init__(self):
        """Initialize GPU frame processor."""
        self.gpu_available = torch.cuda.is_available() if config.gpu.use_gpu else False
        
        if self.gpu_available:
            self.device = torch.device(f'cuda:{config.gpu.cuda_device}')
            logger.info(f"GPU frame processor initialized on {self.device}")
        else:
            self.device = torch.device('cpu')
            logger.info("GPU frame processor initialized on CPU")
    
    def to_gpu(self, frame: np.ndarray) -> torch.Tensor:
        """Convert frame to GPU tensor."""
        if not self.gpu_available:
            return torch.from_numpy(frame).to(self.device)
        
        try:
            # Convert BGR to RGB and normalize
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
            return frame_tensor.to(self.device)
        except Exception as e:
            logger.log_error_with_context(e, "gpu_frame_conversion")
            return torch.from_numpy(frame).to(self.device)
    
    def to_cpu(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert GPU tensor back to CPU numpy array."""
        try:
            # Convert back to BGR format
            frame_cpu = tensor.cpu().numpy()
            if frame_cpu.shape[-1] == 3:  # RGB to BGR
                frame_bgr = cv2.cvtColor(frame_cpu, cv2.COLOR_RGB2BGR)
                return (frame_bgr * 255).astype(np.uint8)
            return (frame_cpu * 255).astype(np.uint8)
        except Exception as e:
            logger.log_error_with_context(e, "cpu_frame_conversion")
            return tensor.cpu().numpy()
    
    def resize_frame(self, frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize frame with GPU acceleration if available."""
        if not self.gpu_available:
            return cv2.resize(frame, target_size)
        
        try:
            # Convert to GPU tensor
            frame_tensor = self.to_gpu(frame)
            
            # Resize on GPU
            frame_tensor = torch.nn.functional.interpolate(
                frame_tensor.unsqueeze(0).permute(0, 3, 1, 2),
                size=target_size,
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1).squeeze(0)
            
            # Convert back to CPU
            return self.to_cpu(frame_tensor)
            
        except Exception as e:
            logger.log_error_with_context(e, "gpu_frame_resize")
            return cv2.resize(frame, target_size)


# Global instances
video_capture = None
gpu_processor = GPUFrameProcessor()


def initialize_video_capture(device_id: int = 0) -> VideoCapture:
    """Initialize and return video capture instance."""
    global video_capture
    if video_capture is None:
        video_capture = VideoCapture(device_id)
    return video_capture


def get_video_capture() -> Optional[VideoCapture]:
    """Get the global video capture instance."""
    return video_capture


def release_video_capture() -> None:
    """Release the global video capture instance."""
    global video_capture
    if video_capture:
        video_capture.release()
        video_capture = None 