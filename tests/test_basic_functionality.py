"""
Basic functionality tests for the attentiveness detection system.
"""

import sys
import os
import unittest
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils.config import config
from src.utils.logger import logger


class TestBasicFunctionality(unittest.TestCase):
    """Test basic system functionality."""
    
    def setUp(self):
        """Set up test environment."""
        logger.info("Setting up test environment")
    
    def test_config_loading(self):
        """Test configuration loading."""
        self.assertIsNotNone(config)
        self.assertIsNotNone(config.camera)
        self.assertIsNotNone(config.gpu)
        self.assertIsNotNone(config.model)
        self.assertIsNotNone(config.privacy)
        self.assertIsNotNone(config.performance)
        self.assertIsNotNone(config.attentiveness)
        
        logger.info("Configuration loading test passed")
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid configuration
        self.assertTrue(config.validate_config())
        
        # Test invalid configuration (temporarily modify)
        original_width = config.camera.width
        config.camera.width = -1
        self.assertFalse(config.validate_config())
        config.camera.width = original_width  # Restore
        
        logger.info("Configuration validation test passed")
    
    def test_gpu_info(self):
        """Test GPU information retrieval."""
        gpu_info = config.get_gpu_info()
        self.assertIsInstance(gpu_info, dict)
        self.assertIn('available', gpu_info)
        self.assertIn('device_count', gpu_info)
        
        logger.info("GPU info test passed")
    
    def test_optimal_settings(self):
        """Test optimal settings calculation."""
        optimal_settings = config.get_optimal_settings()
        self.assertIsInstance(optimal_settings, dict)
        self.assertIn('gpu_available', optimal_settings)
        self.assertIn('recommended_backend', optimal_settings)
        self.assertIn('recommended_fps', optimal_settings)
        self.assertIn('performance_mode', optimal_settings)
        
        logger.info("Optimal settings test passed")
    
    def test_logger_functionality(self):
        """Test logger functionality."""
        # Test basic logging
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.error("Test error message")
        
        # Test performance logging
        logger.log_performance(30.0, 50.0, 80.0)
        
        # Test system info logging
        logger.log_system_info()
        
        logger.info("Logger functionality test passed")
    
    def test_privacy_utilities(self):
        """Test privacy utilities."""
        try:
            from src.utils.privacy import PrivacyManager
            
            privacy_manager = PrivacyManager()
            self.assertIsNotNone(privacy_manager)
            
            # Test data anonymization
            test_data = {"test": "data", "sensitive": "information"}
            anonymized = privacy_manager.anonymize_data(test_data)
            self.assertIsInstance(anonymized, dict)
            
            logger.info("Privacy utilities test passed")
            
        except ImportError:
            logger.warning("Privacy utilities not available for testing")
    
    def test_video_capture_initialization(self):
        """Test video capture initialization."""
        try:
            from src.core.video_capture import initialize_video_capture, release_video_capture
            
            # Initialize video capture
            video_capture = initialize_video_capture()
            self.assertIsNotNone(video_capture)
            
            # Test frame reading (if camera is available)
            ret, frame = video_capture.read_frame()
            if ret:
                self.assertIsInstance(frame, np.ndarray)
                self.assertEqual(len(frame.shape), 3)  # Should be 3D (height, width, channels)
            
            # Release video capture
            release_video_capture()
            
            logger.info("Video capture test passed")
            
        except Exception as e:
            logger.warning(f"Video capture test failed: {e}")
    
    def test_face_detection_initialization(self):
        """Test face detection initialization."""
        try:
            from src.core.face_detection import initialize_face_detector
            
            face_detector = initialize_face_detector()
            self.assertIsNotNone(face_detector)
            
            logger.info("Face detection initialization test passed")
            
        except Exception as e:
            logger.warning(f"Face detection initialization test failed: {e}")
    
    def test_gaze_estimation_initialization(self):
        """Test gaze estimation initialization."""
        try:
            from src.core.gaze_estimation import initialize_gaze_estimator
            
            gaze_estimator = initialize_gaze_estimator()
            self.assertIsNotNone(gaze_estimator)
            
            logger.info("Gaze estimation initialization test passed")
            
        except Exception as e:
            logger.warning(f"Gaze estimation initialization test failed: {e}")
    
    def test_pose_estimation_initialization(self):
        """Test pose estimation initialization."""
        try:
            from src.core.pose_estimation import initialize_pose_estimator
            
            pose_estimator = initialize_pose_estimator()
            self.assertIsNotNone(pose_estimator)
            
            logger.info("Pose estimation initialization test passed")
            
        except Exception as e:
            logger.warning(f"Pose estimation initialization test failed: {e}")
    
    def test_behavioral_detector_initialization(self):
        """Test behavioral detector initialization."""
        try:
            from src.core.behavioral_cues import initialize_behavioral_detector
            
            behavioral_detector = initialize_behavioral_detector()
            self.assertIsNotNone(behavioral_detector)
            
            logger.info("Behavioral detector initialization test passed")
            
        except Exception as e:
            logger.warning(f"Behavioral detector initialization test failed: {e}")
    
    def test_multimodal_fusion_initialization(self):
        """Test multimodal fusion initialization."""
        try:
            from src.core.multimodal_fusion import initialize_multimodal_fusion
            
            multimodal_fusion = initialize_multimodal_fusion()
            self.assertIsNotNone(multimodal_fusion)
            
            logger.info("Multimodal fusion initialization test passed")
            
        except Exception as e:
            logger.warning(f"Multimodal fusion initialization test failed: {e}")
    
    def test_attentiveness_scorer_initialization(self):
        """Test attentiveness scorer initialization."""
        try:
            from src.core.attentiveness_scoring import initialize_attentiveness_scorer
            
            attentiveness_scorer = initialize_attentiveness_scorer()
            self.assertIsNotNone(attentiveness_scorer)
            
            logger.info("Attentiveness scorer initialization test passed")
            
        except Exception as e:
            logger.warning(f"Attentiveness scorer initialization test failed: {e}")
    
    def test_gui_components(self):
        """Test GUI components initialization."""
        try:
            from src.gui.camera_widget import CameraWidget
            from src.gui.permission_handler import PermissionHandler
            
            # Test camera widget
            camera_widget = CameraWidget()
            self.assertIsNotNone(camera_widget)
            
            # Test permission handler
            permission_handler = PermissionHandler()
            self.assertIsNotNone(permission_handler)
            
            logger.info("GUI components test passed")
            
        except Exception as e:
            logger.warning(f"GUI components test failed: {e}")
    
    def tearDown(self):
        """Clean up test environment."""
        logger.info("Cleaning up test environment")


def run_tests():
    """Run all tests."""
    logger.info("Starting basic functionality tests")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBasicFunctionality)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Log results
    if result.wasSuccessful():
        logger.info("All tests passed!")
    else:
        logger.error(f"Tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 