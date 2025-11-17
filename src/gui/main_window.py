"""
Modern Professional GUI for Attentiveness Detection System
"""

import sys
import os
import time
import threading
from typing import Optional, Dict, Any
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QProgressBar,
                               QTextEdit, QGroupBox, QGridLayout, QMessageBox,
                               QSlider, QCheckBox, QComboBox, QSpinBox, QTabWidget,
                               QFrame, QGraphicsDropShadowEffect)
from PySide6.QtCore import Qt, QTimer, Signal, QThread, QPropertyAnimation, QEasingCurve, QRect
from PySide6.QtGui import QPixmap, QImage, QFont, QPalette, QColor, QLinearGradient, QPainter, QBrush
import cv2
import numpy as np

from src.core.face_detection import FaceDetector
from src.core.gaze_estimation import GazeEstimator
from src.core.pose_estimation import PoseEstimator
from src.core.behavioral_cues import BehavioralCuesDetector
from src.core.multimodal_fusion import MultimodalFusion
from src.core.attentiveness_scoring import AttentivenessScorer
from .camera_widget import CameraWidget
from .permission_handler import PermissionHandler


class ProcessingThread(QThread):
    """Thread for processing video frames."""
    
    frame_processed = Signal(np.ndarray, dict)
    status_updated = Signal(dict)
    error_occurred = Signal(str)
    
    def __init__(self, camera_index=0, width=640, height=480, fps=30, backend="mediapipe"):
        super().__init__()
        self.is_running = False
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.backend = backend
        
        self.video_capture = None
        self.face_detector = None
        self.gaze_estimator = None
        self.pose_estimator = None
        self.behavioral_detector = None
        self.fusion = None
        self.scorer = None
    
    def initialize_components(self):
        """Initialize all processing components."""
        try:
            # Initialize camera
            self.video_capture = cv2.VideoCapture(self.camera_index)
            if not self.video_capture.isOpened():
                self.error_occurred.emit(f"Could not open camera {self.camera_index}")
                return False
            
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.video_capture.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Initialize components
            self.face_detector = FaceDetector(backend=self.backend)
            self.gaze_estimator = GazeEstimator(backend=self.backend)
            self.pose_estimator = PoseEstimator(backend=self.backend)
            self.behavioral_detector = BehavioralCuesDetector()
            self.fusion = MultimodalFusion(fusion_strategy="weighted_sum")
            self.scorer = AttentivenessScorer()
            
            return True
        except Exception as e:
            self.error_occurred.emit(f"Failed to initialize components: {str(e)}")
            return False
    
    def start_processing(self):
        """Start the processing loop."""
        self.is_running = True
        self.start()
    
    def stop_processing(self):
        """Stop the processing loop."""
        self.is_running = False
        self.wait()
    
    def run(self):
        """Main processing loop."""
        frame_count = 0
        start_time = time.time()
        
        while self.is_running:
            try:
                ret, frame = self.video_capture.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
                
                # Process frame
                results = self._process_frame(frame)
                
                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                results['fps'] = fps
                
                # Emit results
                self.frame_processed.emit(frame, results)
                
                # Update status
                status = {
                    'face_detected': results.get('face_detected', False),
                    'attention_score': results.get('attention_score', 0.0),
                    'attention_level': results.get('attention_level', 'Unknown'),
                    'fps': fps,
                    'frame_count': frame_count
                }
                self.status_updated.emit(status)
                
                # Control processing rate
                time.sleep(1.0 / self.fps)
                
            except Exception as e:
                self.error_occurred.emit(f"Processing error: {str(e)}")
                time.sleep(0.1)
    
    def _process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process a single frame."""
        results = {
            'face_detected': False,
            'attention_score': 0.0,
            'attention_level': 'Very Low',
            'gaze_score': 0.0,
            'pose_score': 0.0,
            'behavioral_score': 0.0,
            'scoring_result': {
                'attention_score': 0.0,
                'attention_level': 'Very Low',
                'confidence': 0.0
            }
        }
        
        try:
            # Detect faces
            faces = self.face_detector.detect_faces(frame)
            
            if not faces:
                return results
            
            # Get largest face
            largest_face = self.face_detector.get_largest_face(faces)
            if not largest_face:
                return results
            
            bbox = largest_face['bbox']
            
            # Estimate gaze
            gaze_result = self.gaze_estimator.estimate_gaze(frame, bbox)
            
            # Estimate pose
            pose_result = self.pose_estimator.estimate_pose(frame, bbox)
            
            # Detect behavioral cues
            behavioral_result = self.behavioral_detector.detect_cues(frame, {
                'landmarks': largest_face.get('landmarks', [])
            })
            
            # Fuse features
            fusion_result = self.fusion.fuse_features(gaze_result, pose_result, behavioral_result)
            
            # Calculate attention score
            scoring_result = self.scorer.calculate_attention_score(gaze_result, pose_result, behavioral_result)
            
            results = {
                'face_detected': True,
                'faces': faces,
                'gaze_result': gaze_result,
                'pose_result': pose_result,
                'behavioral_result': behavioral_result,
                'fusion_result': fusion_result,
                'scoring_result': scoring_result,
                'attention_score': scoring_result.get('attention_score', 0.0),
                'attention_level': scoring_result.get('attention_level', 'Very Low'),
                'gaze_score': scoring_result.get('gaze_score', 0.0),
                'pose_score': scoring_result.get('pose_score', 0.0),
                'behavioral_score': scoring_result.get('behavioral_score', 0.0),
            }
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.video_capture:
                self.video_capture.release()
        except:
            pass


class MainWindow(QMainWindow):
    """Modern Professional Main Window."""
    
    def __init__(self):
        super().__init__()
        self.processing_thread = None
        self.permission_handler = PermissionHandler()
        self.current_attention_score = 0.0
        
        self.init_ui()
        self.setup_connections()
    
    def init_ui(self):
        """Initialize the modern UI."""
        self.setWindowTitle("AI Attentiveness Detection System - Professional Edition")
        self.setGeometry(100, 100, 1600, 950)
        
        # Apply modern professional styling
        self.apply_modern_style()
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout with better spacing
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(25, 25, 25, 25)
        
        # Left panel (controls and metrics)
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Right panel (camera and visualizations)
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 2)
    
    def apply_modern_style(self):
        """Apply modern professional styling."""
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0f1419, stop:0.5 #1a2332, stop:1 #0f1419);
            }
            QWidget {
                color: #e8eaed;
                font-family: 'Segoe UI', 'Roboto', 'Inter', Arial, sans-serif;
                font-size: 13px;
            }
            QGroupBox {
                font-weight: 600;
                font-size: 14px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 12px;
                margin-top: 18px;
                padding-top: 18px;
                background-color: rgba(255, 255, 255, 0.03);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 18px;
                padding: 0 12px 0 12px;
                color: #8ab4f8;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4285f4, stop:1 #1a73e8);
                border: none;
                border-radius: 10px;
                padding: 14px 28px;
                color: #ffffff;
                font-weight: 600;
                font-size: 14px;
                min-height: 24px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #5a95f5, stop:1 #2979ff);
                transform: scale(1.02);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1a73e8, stop:1 #1557b0);
            }
            QPushButton:disabled {
                background: rgba(255, 255, 255, 0.1);
                color: rgba(255, 255, 255, 0.4);
            }
            QPushButton#stopButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ea4335, stop:1 #c5221f);
            }
            QPushButton#stopButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f15c4f, stop:1 #d93025);
            }
            QLabel {
                color: #ffffff;
            }
            QProgressBar {
                border: 2px solid rgba(255, 255, 255, 0.15);
                border-radius: 12px;
                text-align: center;
                font-weight: 600;
                font-size: 13px;
                height: 36px;
                background-color: rgba(0, 0, 0, 0.4);
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #34a853, stop:0.5 #66bb6a, stop:1 #34a853);
                border-radius: 10px;
            }
            QComboBox, QSpinBox {
                background-color: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 5px;
                padding: 5px;
                color: #ffffff;
                min-height: 25px;
            }
            QComboBox:hover, QSpinBox:hover {
                background-color: rgba(255, 255, 255, 0.15);
            }
            QCheckBox {
                color: #ffffff;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid rgba(255, 255, 255, 0.5);
                border-radius: 3px;
                background-color: rgba(255, 255, 255, 0.1);
            }
            QCheckBox::indicator:checked {
                background-color: #42a5f5;
                border-color: #42a5f5;
            }
            QTextEdit {
                background-color: rgba(0, 0, 0, 0.3);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 5px;
                color: #ffffff;
                padding: 10px;
            }
        """)
    
    def create_left_panel(self) -> QWidget:
        """Create the left control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(20)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Title with modern design
        title = QLabel("AI Attentiveness\nDetection System")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                font-size: 26px;
                font-weight: 700;
                color: #8ab4f8;
                padding: 24px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(138, 180, 248, 0.1), stop:1 rgba(138, 180, 248, 0.05));
                border: 1px solid rgba(138, 180, 248, 0.2);
                border-radius: 14px;
            }
        """)
        layout.addWidget(title)
        
        # Control buttons with better spacing
        control_group = QGroupBox("System Controls")
        control_layout = QVBoxLayout(control_group)
        control_layout.setSpacing(12)
        control_layout.setContentsMargins(15, 25, 15, 15)
        
        self.start_button = QPushButton("Start")
        self.start_button.setMinimumHeight(55)
        self.start_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #34a853, stop:1 #2d8f47);
                border: none;
                border-radius: 12px;
                padding: 16px 32px;
                color: #ffffff;
                font-weight: 700;
                font-size: 16px;
                min-height: 24px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4caf50, stop:1 #388e3c);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2d8f47, stop:1 #1b5e20);
            }
        """)
        self.start_button.clicked.connect(self.start_monitoring)
        control_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.setMinimumHeight(55)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ea4335, stop:1 #c5221f);
                border: none;
                border-radius: 12px;
                padding: 16px 32px;
                color: #ffffff;
                font-weight: 700;
                font-size: 16px;
                min-height: 24px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f15c4f, stop:1 #d93025);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #c5221f, stop:1 #a50e0e);
            }
        """)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_monitoring)
        control_layout.addWidget(self.stop_button)
        
        self.screenshot_button = QPushButton("📷 Screenshot")
        self.screenshot_button.setMinimumHeight(50)
        self.screenshot_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #8ab4f8, stop:1 #5a95f5);
                border: none;
                border-radius: 12px;
                padding: 14px 28px;
                color: #ffffff;
                font-weight: 600;
                font-size: 14px;
                min-height: 24px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #9ab4f8, stop:1 #6aa5f5);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #5a95f5, stop:1 #4285f4);
            }
        """)
        self.screenshot_button.clicked.connect(self.take_screenshot)
        control_layout.addWidget(self.screenshot_button)
        
        layout.addWidget(control_group)
        
        # Real-time Metrics with better spacing
        metrics_group = QGroupBox("Real-time Metrics")
        metrics_layout = QVBoxLayout(metrics_group)
        metrics_layout.setSpacing(20)
        metrics_layout.setContentsMargins(15, 25, 15, 15)
        
        # Attention Score with enhanced display - more spacing
        attention_frame = QFrame()
        attention_frame.setStyleSheet("""
            QFrame {
                background: rgba(138, 180, 248, 0.05);
                border: 1px solid rgba(138, 180, 248, 0.2);
                border-radius: 12px;
                padding: 16px;
            }
        """)
        attention_layout = QVBoxLayout(attention_frame)
        attention_layout.setSpacing(12)
        
        self.attention_label = QLabel("📊 Attention Score")
        self.attention_label.setStyleSheet("font-size: 14px; font-weight: 600; color: #9aa0a6; padding-bottom: 8px;")
        attention_layout.addWidget(self.attention_label)
        
        self.attention_progress = QProgressBar()
        self.attention_progress.setRange(0, 100)
        self.attention_progress.setValue(0)
        self.attention_progress.setFormat("  %p%  ")
        self.attention_progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid rgba(255, 255, 255, 0.15);
                border-radius: 12px;
                text-align: center;
                font-weight: 700;
                font-size: 15px;
                height: 45px;
                background-color: rgba(0, 0, 0, 0.4);
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #34a853, stop:0.5 #66bb6a, stop:1 #34a853);
                border-radius: 10px;
            }
        """)
        attention_layout.addWidget(self.attention_progress)
        
        self.attention_level_label = QLabel("Level: Not Started")
        self.attention_level_label.setStyleSheet("font-size: 18px; font-weight: 700; color: #8ab4f8; padding-top: 10px;")
        self.attention_level_label.setAlignment(Qt.AlignCenter)
        attention_layout.addWidget(self.attention_level_label)
        
        metrics_layout.addWidget(attention_frame)
        
        # Detailed scores with better spacing - Grid layout to prevent overlap
        scores_frame = QFrame()
        scores_frame.setStyleSheet("""
            QFrame {
                background: rgba(138, 180, 248, 0.05);
                border: 1px solid rgba(138, 180, 248, 0.2);
                border-radius: 12px;
                padding: 16px;
            }
        """)
        scores_layout = QGridLayout(scores_frame)
        scores_layout.setSpacing(16)
        scores_layout.setContentsMargins(10, 10, 10, 10)
        
        # Gaze Score
        gaze_label = QLabel("👁 Gaze:")
        gaze_label.setStyleSheet("font-size: 13px; font-weight: 600; color: #9aa0a6;")
        scores_layout.addWidget(gaze_label, 0, 0)
        self.gaze_score_label = QLabel("--")
        self.gaze_score_label.setStyleSheet("font-size: 13px; font-weight: 600; color: #e8eaed;")
        self.gaze_score_label.setAlignment(Qt.AlignRight)
        scores_layout.addWidget(self.gaze_score_label, 0, 1)
        
        # Pose Score
        pose_label = QLabel("🎯 Head Pose:")
        pose_label.setStyleSheet("font-size: 13px; font-weight: 600; color: #9aa0a6;")
        scores_layout.addWidget(pose_label, 1, 0)
        self.pose_score_label = QLabel("--")
        self.pose_score_label.setStyleSheet("font-size: 13px; font-weight: 600; color: #e8eaed;")
        self.pose_score_label.setAlignment(Qt.AlignRight)
        scores_layout.addWidget(self.pose_score_label, 1, 1)
        
        # Behavioral Score
        behavior_label = QLabel("😊 Behavior:")
        behavior_label.setStyleSheet("font-size: 13px; font-weight: 600; color: #9aa0a6;")
        scores_layout.addWidget(behavior_label, 2, 0)
        self.behavioral_score_label = QLabel("--")
        self.behavioral_score_label.setStyleSheet("font-size: 13px; font-weight: 600; color: #e8eaed;")
        self.behavioral_score_label.setAlignment(Qt.AlignRight)
        scores_layout.addWidget(self.behavioral_score_label, 2, 1)
        
        metrics_layout.addWidget(scores_frame)
        
        # Status indicators with better spacing - Grid layout
        status_frame = QFrame()
        status_frame.setStyleSheet("""
            QFrame {
                background: rgba(138, 180, 248, 0.05);
                border: 1px solid rgba(138, 180, 248, 0.2);
                border-radius: 12px;
                padding: 16px;
            }
        """)
        status_layout = QGridLayout(status_frame)
        status_layout.setSpacing(16)
        status_layout.setContentsMargins(10, 10, 10, 10)
        
        # FPS
        fps_title = QLabel("⚡ FPS:")
        fps_title.setStyleSheet("font-size: 13px; font-weight: 600; color: #9aa0a6;")
        status_layout.addWidget(fps_title, 0, 0)
        self.fps_label = QLabel("--")
        self.fps_label.setStyleSheet("font-size: 13px; font-weight: 600; color: #e8eaed;")
        self.fps_label.setAlignment(Qt.AlignRight)
        status_layout.addWidget(self.fps_label, 0, 1)
        
        # Face Status
        face_title = QLabel("👤 Face:")
        face_title.setStyleSheet("font-size: 13px; font-weight: 600; color: #9aa0a6;")
        status_layout.addWidget(face_title, 1, 0)
        self.face_status_label = QLabel("Not Detected")
        self.face_status_label.setStyleSheet("font-size: 13px; font-weight: 600; color: #ea4335;")
        self.face_status_label.setAlignment(Qt.AlignRight)
        status_layout.addWidget(self.face_status_label, 1, 1)
        
        metrics_layout.addWidget(status_frame)
        
        layout.addWidget(metrics_group)
        
        # Settings with modern styling - better spacing
        settings_group = QGroupBox("⚙️ Settings")
        settings_layout = QGridLayout(settings_group)
        settings_layout.setSpacing(16)
        settings_layout.setContentsMargins(15, 25, 15, 15)
        
        camera_label = QLabel("📷 Camera:")
        camera_label.setStyleSheet("font-size: 12px; font-weight: 500; color: #e8eaed;")
        settings_layout.addWidget(camera_label, 0, 0)
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["Default Camera", "Camera 1", "Camera 2"])
        self.camera_combo.setStyleSheet("""
            QComboBox {
                background-color: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(138, 180, 248, 0.3);
                border-radius: 8px;
                padding: 8px;
                color: #e8eaed;
                min-height: 28px;
                font-size: 12px;
            }
            QComboBox:hover {
                background-color: rgba(255, 255, 255, 0.15);
                border-color: rgba(138, 180, 248, 0.5);
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #8ab4f8;
                margin-right: 8px;
            }
        """)
        settings_layout.addWidget(self.camera_combo, 0, 1)
        
        resolution_label = QLabel("📐 Resolution:")
        resolution_label.setStyleSheet("font-size: 12px; font-weight: 500; color: #e8eaed;")
        settings_layout.addWidget(resolution_label, 1, 0)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["640x480", "1280x720", "1920x1080"])
        self.resolution_combo.setStyleSheet(self.camera_combo.styleSheet())
        settings_layout.addWidget(self.resolution_combo, 1, 1)
        
        self.privacy_checkbox = QCheckBox("🔒 Enable Privacy Mode")
        self.privacy_checkbox.setChecked(True)
        self.privacy_checkbox.setStyleSheet("""
            QCheckBox {
                color: #e8eaed;
                font-size: 12px;
                font-weight: 500;
                spacing: 10px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border: 2px solid rgba(138, 180, 248, 0.5);
                border-radius: 4px;
                background-color: rgba(255, 255, 255, 0.1);
            }
            QCheckBox::indicator:checked {
                background-color: #4285f4;
                border-color: #4285f4;
            }
            QCheckBox::indicator:hover {
                border-color: #8ab4f8;
            }
        """)
        settings_layout.addWidget(self.privacy_checkbox, 2, 0, 1, 2)
        
        layout.addWidget(settings_group)
        
        # Session info with better layout
        session_group = QGroupBox("Session Information")
        session_layout = QGridLayout(session_group)
        session_layout.setSpacing(16)
        session_layout.setContentsMargins(15, 25, 15, 15)
        
        duration_title = QLabel("⏱ Duration:")
        duration_title.setStyleSheet("font-size: 13px; font-weight: 600; color: #9aa0a6;")
        session_layout.addWidget(duration_title, 0, 0)
        self.session_time_label = QLabel("00:00:00")
        self.session_time_label.setStyleSheet("font-size: 13px; font-weight: 600; color: #e8eaed;")
        self.session_time_label.setAlignment(Qt.AlignRight)
        session_layout.addWidget(self.session_time_label, 0, 1)
        
        frames_title = QLabel("📊 Frames:")
        frames_title.setStyleSheet("font-size: 13px; font-weight: 600; color: #9aa0a6;")
        session_layout.addWidget(frames_title, 1, 0)
        self.frame_count_label = QLabel("0")
        self.frame_count_label.setStyleSheet("font-size: 13px; font-weight: 600; color: #e8eaed;")
        self.frame_count_label.setAlignment(Qt.AlignRight)
        session_layout.addWidget(self.frame_count_label, 1, 1)
        
        layout.addWidget(session_group)
        
        layout.addStretch()
        
        return panel
    
    def create_right_panel(self) -> QWidget:
        """Create the right panel with camera view."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        
        # Camera widget with modern styling
        camera_group = QGroupBox("Live Camera Feed")
        camera_layout = QVBoxLayout(camera_group)
        camera_layout.setContentsMargins(10, 10, 10, 10)
        
        self.camera_widget = CameraWidget()
        camera_layout.addWidget(self.camera_widget)
        
        layout.addWidget(camera_group)
        
        # Modern status bar at bottom
        status_frame = QFrame()
        status_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(138, 180, 248, 0.1), stop:1 rgba(138, 180, 248, 0.05));
                border: 1px solid rgba(138, 180, 248, 0.2);
                border-radius: 10px;
                padding: 12px 16px;
            }
        """)
        status_layout = QHBoxLayout(status_frame)
        status_layout.setSpacing(15)
        
        self.status_label = QLabel("● Ready")
        self.status_label.setStyleSheet("font-size: 13px; font-weight: 600; color: #34a853;")
        status_layout.addWidget(self.status_label)
        
        status_layout.addStretch()
        
        self.time_label = QLabel(time.strftime("%H:%M:%S"))
        self.time_label.setStyleSheet("font-size: 13px; font-weight: 500; color: #9aa0a6;")
        status_layout.addWidget(self.time_label)
        
        layout.addWidget(status_frame)
        
        return panel
    
    def setup_connections(self):
        """Set up signal connections."""
        # Timer for UI updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_ui)
        self.update_timer.start(100)  # Update every 100ms
        
        # Timer for session time
        self.session_timer = QTimer()
        self.session_timer.timeout.connect(self.update_session_time)
        self.session_start_time = None
    
    def start_monitoring(self):
        """Start the attentiveness monitoring."""
        try:
            if not self.permission_handler.check_camera_permission():
                QMessageBox.warning(self, "Camera Permission", 
                                  "Camera permission is required to start monitoring.")
                return
            
            # Get settings
            resolution = self.resolution_combo.currentText().split('x')
            width, height = int(resolution[0]), int(resolution[1])
            
            # Initialize processing thread
            self.processing_thread = ProcessingThread(
                camera_index=0,
                width=width,
                height=height,
                fps=30,
                backend="mediapipe"
            )
            
            if not self.processing_thread.initialize_components():
                QMessageBox.critical(self, "Initialization Error", 
                                   "Failed to initialize system components.")
                return
            
            # Connect signals
            self.processing_thread.frame_processed.connect(self.on_frame_processed)
            self.processing_thread.status_updated.connect(self.update_status)
            self.processing_thread.error_occurred.connect(self.handle_error)
            
            # Start processing
            self.processing_thread.start_processing()
            
            # Update UI
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.status_label.setText("● Monitoring Active")
            self.status_label.setStyleSheet("font-size: 13px; font-weight: 600; color: #34a853;")
            
            # Start session timer
            self.session_start_time = time.time()
            self.session_timer.start(1000)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start monitoring: {str(e)}")
    
    def stop_monitoring(self):
        """Stop the attentiveness monitoring."""
        try:
            if self.processing_thread:
                self.processing_thread.stop_processing()
                self.processing_thread.cleanup()
                self.processing_thread = None
            
            # Update UI
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.status_label.setText("● Ready")
            self.status_label.setStyleSheet("font-size: 13px; font-weight: 600; color: #34a853;")
            
            # Stop session timer
            self.session_timer.stop()
            self.session_start_time = None
            
            # Reset metrics
            self.attention_progress.setValue(0)
            self.attention_level_label.setText("Level: Not Started")
            self.face_status_label.setText("Not Detected")
            self.face_status_label.setStyleSheet("font-size: 13px; font-weight: 600; color: #ea4335;")
            self.gaze_score_label.setText("--")
            self.gaze_score_label.setStyleSheet("font-size: 13px; font-weight: 600; color: #e8eaed;")
            self.pose_score_label.setText("--")
            self.pose_score_label.setStyleSheet("font-size: 13px; font-weight: 600; color: #e8eaed;")
            self.behavioral_score_label.setText("--")
            self.behavioral_score_label.setStyleSheet("font-size: 13px; font-weight: 600; color: #e8eaed;")
            self.fps_label.setText("--")
            self.frame_count_label.setText("0")
            
            self.camera_widget.clear_display()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to stop monitoring: {str(e)}")
    
    def take_screenshot(self):
        """Take a screenshot of the current frame."""
        try:
            if self.camera_widget.current_frame is not None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"screenshots/attentiveness_{timestamp}.jpg"
                os.makedirs("screenshots", exist_ok=True)
                cv2.imwrite(filename, self.camera_widget.current_frame)
                QMessageBox.information(self, "Screenshot", f"Screenshot saved:\n{filename}")
            else:
                QMessageBox.information(self, "Screenshot", "No frame available for screenshot.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to take screenshot: {str(e)}")
    
    def update_status(self, status: Dict[str, Any]):
        """Update status indicators."""
        try:
            # Update attention score
            attention_score = status.get('attention_score', 0.0)
            self.current_attention_score = attention_score
            self.attention_progress.setValue(int(attention_score * 100))
            
            # Update attention level with enhanced colors
            attention_level = status.get('attention_level', 'Unknown')
            level_colors = {
                'High': '#34a853',      # Google Green
                'Medium': '#fbbc04',    # Google Yellow
                'Low': '#ea4335',       # Google Red
                'Very Low': '#ea4335'   # Google Red
            }
            color = level_colors.get(attention_level, '#8ab4f8')
            self.attention_level_label.setText(f"Level: {attention_level}")
            self.attention_level_label.setStyleSheet(f"font-size: 16px; font-weight: 700; color: {color}; padding-top: 8px;")
            
            # Update progress bar color based on attention level
            if attention_level == 'High':
                progress_color = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #34a853, stop:1 #66bb6a);"
            elif attention_level == 'Medium':
                progress_color = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #fbbc04, stop:1 #fdd663);"
            else:
                progress_color = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ea4335, stop:1 #f28b82);"
            
            self.attention_progress.setStyleSheet(f"""
                QProgressBar {{
                    border: 2px solid rgba(255, 255, 255, 0.15);
                    border-radius: 12px;
                    text-align: center;
                    font-weight: 700;
                    font-size: 14px;
                    height: 40px;
                    background-color: rgba(0, 0, 0, 0.4);
                }}
                QProgressBar::chunk {{
                    background: {progress_color}
                    border-radius: 10px;
                }}
            """)
            
            # Update face status
            if status.get('face_detected', False):
                self.face_status_label.setText("Detected")
                self.face_status_label.setStyleSheet("font-size: 13px; font-weight: 600; color: #34a853;")
            else:
                self.face_status_label.setText("Not Detected")
                self.face_status_label.setStyleSheet("font-size: 13px; font-weight: 600; color: #ea4335;")
            
            # Update FPS
            fps = status.get('fps', 0.0)
            self.fps_label.setText(f"{fps:.1f}")
            
            # Update frame count
            frame_count = status.get('frame_count', 0)
            self.frame_count_label.setText(f"{frame_count}")
            
            # Update detailed scores from processing thread results
            # These will be updated when frame_processed signal is received
            
        except Exception as e:
            pass
    
    def update_ui(self):
        """Update UI elements."""
        # Update time
        self.time_label.setText(time.strftime("%H:%M:%S"))
    
    def update_session_time(self):
        """Update session duration."""
        if self.session_start_time:
            elapsed = time.time() - self.session_start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            self.session_time_label.setText(f"Duration: {hours:02d}:{minutes:02d}:{seconds:02d}")
    
    def on_frame_processed(self, frame: np.ndarray, results: Dict[str, Any]):
        """Handle processed frame and update detailed scores."""
        # Update camera widget
        self.camera_widget.update_frame(frame, results)
        
        # Update detailed scores in UI
        scoring_result = results.get('scoring_result', {})
        if scoring_result:
            gaze_score = scoring_result.get('gaze_score', 0.0)
            pose_score = scoring_result.get('pose_score', 0.0)
            behavioral_score = scoring_result.get('behavioral_score', 0.0)
            
            # Update gaze score with color coding
            if gaze_score > 0.7:
                color = "#34a853"
            elif gaze_score > 0.4:
                color = "#fbbc04"
            else:
                color = "#ea4335"
            self.gaze_score_label.setText(f"{gaze_score:.1%}")
            self.gaze_score_label.setStyleSheet(f"font-size: 13px; font-weight: 600; color: {color};")
            
            # Update pose score with color coding
            if pose_score > 0.7:
                color = "#34a853"
            elif pose_score > 0.4:
                color = "#fbbc04"
            else:
                color = "#ea4335"
            self.pose_score_label.setText(f"{pose_score:.1%}")
            self.pose_score_label.setStyleSheet(f"font-size: 13px; font-weight: 600; color: {color};")
            
            # Update behavioral score with color coding
            if behavioral_score > 0.7:
                color = "#34a853"
            elif behavioral_score > 0.4:
                color = "#fbbc04"
            else:
                color = "#ea4335"
            self.behavioral_score_label.setText(f"{behavioral_score:.1%}")
            self.behavioral_score_label.setStyleSheet(f"font-size: 13px; font-weight: 600; color: {color};")
    
    def handle_error(self, error_message: str):
        """Handle processing errors."""
        self.status_label.setText(f"● Error: {error_message[:30]}")
        self.status_label.setStyleSheet("font-size: 12px; font-weight: bold; color: #ea4335;")
    
    def closeEvent(self, event):
        """Handle application close event."""
        try:
            if self.processing_thread:
                self.stop_monitoring()
            event.accept()
        except:
            event.accept()


def run_gui():
    """Run the GUI application."""
    try:
        app = QApplication(sys.argv)
        
        # Set application properties
        app.setApplicationName("AI Attentiveness Detection System")
        app.setApplicationVersion("2.0.0")
        
        # Create and show main window
        window = MainWindow()
        window.show()
        
        # Run application
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"Error running GUI: {e}")
        return 1


if __name__ == "__main__":
    run_gui()
