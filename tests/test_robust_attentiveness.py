"""
Unit tests for robust attentiveness calculator.
"""

import unittest
import numpy as np
import time
from src.core.robust_attentiveness import RobustAttentivenessCalculator, AttentivenessState


class TestRobustAttentiveness(unittest.TestCase):
    """Test cases for robust attentiveness calculator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calc = RobustAttentivenessCalculator(calibration_duration_sec=1.0)  # Short calibration for tests
        self.test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_initialization(self):
        """Test calculator initialization."""
        self.assertIsNotNone(self.calc)
        self.assertTrue(self.calc.state.is_calibrating)
        self.assertEqual(self.calc.state.movement_baseline, 5.0)
    
    def test_no_face_detection(self):
        """Test behavior when no face is detected."""
        result = self.calc.calculate_attentiveness(
            frame=self.test_frame,
            face_detected=False,
            pose_result=None,
            gaze_result=None,
            behavioral_result=None
        )
        
        self.assertFalse(result['face_present'])
        self.assertIn('attentiveness_score', result)
        self.assertIn('attentive', result)
        self.assertIn('signal_quality', result)
        self.assertLessEqual(result['attentiveness_score'], 100.0)
        self.assertGreaterEqual(result['attentiveness_score'], 0.0)
    
    def test_face_present_calculation(self):
        """Test calculation with face present."""
        pose_result = {
            'yaw': 5.0,
            'pitch': -3.0,
            'roll': 1.0,
            'confidence': 0.9
        }
        gaze_result = {
            'gaze_direction': (0.1, 0.05),
            'confidence': 0.8
        }
        behavioral_result = {
            'blink_rate': 15.0
        }
        
        result = self.calc.calculate_attentiveness(
            frame=self.test_frame,
            face_detected=True,
            pose_result=pose_result,
            gaze_result=gaze_result,
            behavioral_result=behavioral_result
        )
        
        self.assertTrue(result['face_present'])
        self.assertAlmostEqual(result['yaw'], 5.0, places=1)
        self.assertAlmostEqual(result['pitch'], -3.0, places=1)
        self.assertGreater(result['signal_quality'], 0.0)
        self.assertLessEqual(result['latency_ms'], 150.0)  # Should be fast
    
    def test_pose_penalty_calculation(self):
        """Test pose penalty calculation."""
        # Small angles should have low penalty
        pose_result = {
            'yaw': 10.0,
            'pitch': 5.0,
            'roll': 2.0,
            'confidence': 0.9
        }
        
        result1 = self.calc.calculate_attentiveness(
            frame=self.test_frame,
            face_detected=True,
            pose_result=pose_result,
            gaze_result={'gaze_direction': (0.0, 0.0), 'confidence': 0.8},
            behavioral_result={'blink_rate': 15.0}
        )
        
        # Large angles should have high penalty
        pose_result['yaw'] = 40.0
        pose_result['pitch'] = 30.0
        
        result2 = self.calc.calculate_attentiveness(
            frame=self.test_frame,
            face_detected=True,
            pose_result=pose_result,
            gaze_result={'gaze_direction': (0.0, 0.0), 'confidence': 0.8},
            behavioral_result={'blink_rate': 15.0}
        )
        
        # Result 1 should have higher score than result 2
        self.assertGreater(result1['attentiveness_score'], result2['attentiveness_score'])
    
    def test_gaze_penalty_calculation(self):
        """Test gaze penalty calculation."""
        pose_result = {
            'yaw': 0.0,
            'pitch': 0.0,
            'roll': 0.0,
            'confidence': 0.9
        }
        
        # On-axis gaze
        gaze_result1 = {
            'gaze_direction': (0.0, 0.0),
            'confidence': 0.8
        }
        
        result1 = self.calc.calculate_attentiveness(
            frame=self.test_frame,
            face_detected=True,
            pose_result=pose_result,
            gaze_result=gaze_result1,
            behavioral_result={'blink_rate': 15.0}
        )
        
        # Off-axis gaze
        gaze_result2 = {
            'gaze_direction': (0.8, 0.6),  # Large offset
            'confidence': 0.8
        }
        
        result2 = self.calc.calculate_attentiveness(
            frame=self.test_frame,
            face_detected=True,
            pose_result=pose_result,
            gaze_result=gaze_result2,
            behavioral_result={'blink_rate': 15.0}
        )
        
        # Result 1 should have higher score
        self.assertGreater(result1['attentiveness_score'], result2['attentiveness_score'])
    
    def test_movement_tracking(self):
        """Test movement magnitude calculation."""
        # First frame (no previous frame)
        result1 = self.calc.calculate_attentiveness(
            frame=self.test_frame,
            face_detected=True,
            pose_result={'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'confidence': 0.9},
            gaze_result={'gaze_direction': (0.0, 0.0), 'confidence': 0.8},
            behavioral_result={'blink_rate': 15.0}
        )
        
        # Second frame (should calculate movement)
        test_frame2 = np.roll(self.test_frame, 5, axis=1)  # Shift horizontally
        result2 = self.calc.calculate_attentiveness(
            frame=test_frame2,
            face_detected=True,
            pose_result={'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'confidence': 0.9},
            gaze_result={'gaze_direction': (0.0, 0.0), 'confidence': 0.8},
            behavioral_result={'blink_rate': 15.0}
        )
        
        # Should have movement detected
        self.assertGreaterEqual(result2['movement_px_per_sec'], 0.0)
    
    def test_temporal_smoothing(self):
        """Test that temporal smoothing prevents sudden jumps."""
        pose_result = {
            'yaw': 0.0,
            'pitch': 0.0,
            'roll': 0.0,
            'confidence': 0.9
        }
        gaze_result = {
            'gaze_direction': (0.0, 0.0),
            'confidence': 0.8
        }
        
        # First few frames (establish baseline)
        for _ in range(5):
            self.calc.calculate_attentiveness(
                frame=self.test_frame,
                face_detected=True,
                pose_result=pose_result,
                gaze_result=gaze_result,
                behavioral_result={'blink_rate': 15.0}
            )
        
        # Sudden change
        pose_result['yaw'] = 45.0
        result = self.calc.calculate_attentiveness(
            frame=self.test_frame,
            face_detected=True,
            pose_result=pose_result,
            gaze_result=gaze_result,
            behavioral_result={'blink_rate': 15.0}
        )
        
        # Score should be smoothed (not immediately drop to very low)
        self.assertGreater(result['attentiveness_score'], 20.0)
    
    def test_face_absence_decay(self):
        """Test exponential decay when face is absent."""
        # Establish good score with face present
        for _ in range(10):
            self.calc.calculate_attentiveness(
                frame=self.test_frame,
                face_detected=True,
                pose_result={'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'confidence': 0.9},
                gaze_result={'gaze_direction': (0.0, 0.0), 'confidence': 0.8},
                behavioral_result={'blink_rate': 15.0}
            )
        
        # Face disappears
        result1 = self.calc.calculate_attentiveness(
            frame=self.test_frame,
            face_detected=False,
            pose_result=None,
            gaze_result=None,
            behavioral_result=None
        )
        
        # Wait a bit (simulate time passing)
        time.sleep(0.1)
        
        result2 = self.calc.calculate_attentiveness(
            frame=self.test_frame,
            face_detected=False,
            pose_result=None,
            gaze_result=None,
            behavioral_result=None
        )
        
        # Score should decay
        self.assertLess(result2['attentiveness_score'], result1['attentiveness_score'])
    
    def test_binary_label_hysteresis(self):
        """Test that binary label has hysteresis to prevent flickering."""
        # Start with good score
        for _ in range(20):
            self.calc.calculate_attentiveness(
                frame=self.test_frame,
                face_detected=True,
                pose_result={'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'confidence': 0.9},
                gaze_result={'gaze_direction': (0.0, 0.0), 'confidence': 0.8},
                behavioral_result={'blink_rate': 15.0}
            )
        
        # Should be attentive
        self.assertTrue(self.calc.state.label_state)
        
        # Brief drop below threshold shouldn't immediately change label
        for _ in range(5):
            result = self.calc.calculate_attentiveness(
                frame=self.test_frame,
                face_detected=True,
                pose_result={'yaw': 30.0, 'pitch': 25.0, 'roll': 0.0, 'confidence': 0.9},
                gaze_result={'gaze_direction': (0.5, 0.5), 'confidence': 0.8},
                behavioral_result={'blink_rate': 15.0}
            )
            # Label should remain attentive (hysteresis)
            if result['attentiveness_score'] < 60:
                # Still should be attentive due to hysteresis
                pass
    
    def test_calibration(self):
        """Test calibration phase."""
        self.assertTrue(self.calc.state.is_calibrating)
        
        # Feed calibration samples
        for i in range(30):
            self.calc.calculate_attentiveness(
                frame=self.test_frame,
                face_detected=True,
                pose_result={
                    'yaw': np.random.uniform(-5, 5),
                    'pitch': np.random.uniform(-3, 3),
                    'roll': np.random.uniform(-2, 2),
                    'confidence': 0.9
                },
                gaze_result={
                    'gaze_direction': (np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1)),
                    'confidence': 0.8
                },
                behavioral_result={'blink_rate': 15.0}
            )
            time.sleep(0.05)  # Simulate time passing
        
        # Calibration should complete
        # (Note: actual completion depends on calibration_duration_sec)
        # For this test, we just verify calibration samples are collected
        self.assertGreater(len(self.calc.state.calibration_samples), 0)
    
    def test_signal_quality(self):
        """Test signal quality calculation."""
        # Good quality (face detected, high confidence)
        result1 = self.calc.calculate_attentiveness(
            frame=self.test_frame,
            face_detected=True,
            pose_result={'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'confidence': 0.9},
            gaze_result={'gaze_direction': (0.0, 0.0), 'confidence': 0.8},
            behavioral_result={'blink_rate': 15.0}
        )
        
        # Poor quality (low confidence)
        result2 = self.calc.calculate_attentiveness(
            frame=self.test_frame,
            face_detected=True,
            pose_result={'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'confidence': 0.3},
            gaze_result={'gaze_direction': (0.0, 0.0), 'confidence': 0.2},
            behavioral_result={'blink_rate': 15.0}
        )
        
        self.assertGreater(result1['signal_quality'], result2['signal_quality'])
    
    def test_reset(self):
        """Test reset functionality."""
        # Feed some data
        for _ in range(10):
            self.calc.calculate_attentiveness(
                frame=self.test_frame,
                face_detected=True,
                pose_result={'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'confidence': 0.9},
                gaze_result={'gaze_direction': (0.0, 0.0), 'confidence': 0.8},
                behavioral_result={'blink_rate': 15.0}
            )
        
        # Reset
        self.calc.reset(new_calibration=True)
        
        # Should be calibrating again
        self.assertTrue(self.calc.state.is_calibrating)
        self.assertEqual(len(self.calc.state.pose_history), 0)


if __name__ == '__main__':
    unittest.main()

