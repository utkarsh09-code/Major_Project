"""
Enhanced Camera Widget with Professional Visualizations
"""

import cv2
import numpy as np
from typing import Optional, Dict, Any
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CameraWidget(QWidget):
    """Enhanced widget for displaying camera feed with professional visualizations."""
    
    def __init__(self):
        super().__init__()
        self.current_frame = None
        self.current_results = {}
        
        self.init_ui()
        
        # Timer for frame updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(33)  # ~30 FPS
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create camera display label
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(800, 600)
        self.camera_label.setStyleSheet("""
            QLabel {
                border: 2px solid rgba(138, 180, 248, 0.3);
                border-radius: 16px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0f1419, stop:1 #1a2332);
                padding: 4px;
            }
        """)
        
        # Set placeholder text with modern design
        self.camera_label.setText(
            "<div style='text-align: center; color: #8ab4f8; font-size: 20px; font-weight: 600; padding: 40px;'>"
            "📹 Camera Feed<br><br>"
            "<span style='font-size: 15px; color: #9aa0a6; font-weight: 400;'>Click 'Start Monitoring' to begin</span>"
            "</div>"
        )
        
        layout.addWidget(self.camera_label)
    
    def update_frame(self, frame: np.ndarray, results: Dict[str, Any]):
        """Update the camera frame and processing results."""
        try:
            self.current_frame = frame.copy()
            self.current_results = results
            
            # Apply professional visualizations
            if results.get('face_detected', False):
                self.current_frame = self._apply_visualizations(frame, results)
            
        except Exception as e:
            logger.error(f"Error updating frame: {e}")
    
    def _apply_visualizations(self, frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """Apply professional visualizations to the frame."""
        try:
            vis_frame = frame.copy()
            h, w = vis_frame.shape[:2]
            
            # Draw face detection with modern style
            if 'faces' in results and results['faces']:
                vis_frame = self._draw_face_detection(vis_frame, results['faces'])
            
            # Draw attention overlay
            if 'scoring_result' in results:
                vis_frame = self._draw_attention_overlay(vis_frame, results['scoring_result'], w, h)
            
            # Draw gaze visualization
            if 'gaze_result' in results:
                vis_frame = self._draw_gaze_visualization(vis_frame, results['gaze_result'], w, h)
            
            # Draw pose information
            if 'pose_result' in results:
                vis_frame = self._draw_pose_info(vis_frame, results['pose_result'], w, h)
            
            # Draw metrics panel
            vis_frame = self._draw_metrics_panel(vis_frame, results, w, h)
            
            return vis_frame
            
        except Exception as e:
            logger.error(f"Error applying visualizations: {e}")
            return frame
    
    def _draw_face_detection(self, frame: np.ndarray, faces: list) -> np.ndarray:
        """Draw modern professional face detection visualization."""
        try:
            if not faces:
                return frame
            
            face = faces[0]  # Use first face
            bbox = face.get('bbox', (0, 0, 100, 100))
            confidence = face.get('confidence', 0.0)
            
            x, y, w, h = bbox
            
            # Draw modern rounded rectangle with professional styling
            # Outer glow effect
            cv2.rectangle(frame, (x-3, y-3), (x+w+3, y+h+3), (138, 180, 248), 2)
            # Main rectangle with thicker border
            cv2.rectangle(frame, (x, y), (x+w, y+h), (66, 133, 244), 4)
            # Inner highlight for depth
            cv2.rectangle(frame, (x+3, y+3), (x+w-3, y+h-3), (138, 180, 248), 1)
            
            # Draw corner indicators for modern look
            corner_length = 20
            thickness = 3
            # Top-left
            cv2.line(frame, (x, y), (x+corner_length, y), (138, 180, 248), thickness)
            cv2.line(frame, (x, y), (x, y+corner_length), (138, 180, 248), thickness)
            # Top-right
            cv2.line(frame, (x+w, y), (x+w-corner_length, y), (138, 180, 248), thickness)
            cv2.line(frame, (x+w, y), (x+w, y+corner_length), (138, 180, 248), thickness)
            # Bottom-left
            cv2.line(frame, (x, y+h), (x+corner_length, y+h), (138, 180, 248), thickness)
            cv2.line(frame, (x, y+h), (x, y+h-corner_length), (138, 180, 248), thickness)
            # Bottom-right
            cv2.line(frame, (x+w, y+h), (x+w-corner_length, y+h), (138, 180, 248), thickness)
            cv2.line(frame, (x+w, y+h), (x+w, y+h-corner_length), (138, 180, 248), thickness)
            
            # Draw modern confidence badge with rounded corners effect
            conf_text = f"{confidence:.0%}"
            font_scale = 0.7
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            badge_x = x + w - text_w - 15
            badge_y = y + text_h + 20
            
            # Badge background with padding
            padding = 8
            cv2.rectangle(frame, (badge_x-padding, badge_y-text_h-padding), 
                         (badge_x+text_w+padding, badge_y+baseline+padding), (66, 133, 244), -1)
            # Badge border
            cv2.rectangle(frame, (badge_x-padding, badge_y-text_h-padding), 
                         (badge_x+text_w+padding, badge_y+baseline+padding), (255, 255, 255), 1)
            # Badge text
            cv2.putText(frame, conf_text, (badge_x, badge_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error drawing face: {e}")
            return frame
    
    def _draw_attention_overlay(self, frame: np.ndarray, scoring_result: Dict[str, Any], w: int, h: int) -> np.ndarray:
        """Draw modern professional attention level overlay."""
        try:
            attention_score = scoring_result.get('attention_score', 0.0)
            attention_level = scoring_result.get('attention_level', 'Unknown')
            
            # Enhanced color scheme based on attention level
            colors = {
                'High': (52, 168, 83),      # Green
                'Medium': (251, 188, 5),    # Yellow/Orange
                'Low': (234, 67, 53),       # Red-Orange
                'Very Low': (234, 67, 53)   # Red
            }
            color = colors.get(attention_level, (138, 180, 248))
            
            # Draw modern attention indicator at top center
            indicator_y = 30
            indicator_w = 380
            indicator_h = 70
            
            # Create semi-transparent background panel with rounded corners effect
            panel = np.zeros((indicator_h, indicator_w, 3), dtype=np.uint8)
            panel[:] = (15, 20, 25)  # Dark background
            overlay = frame.copy()
            
            x_offset = (w - indicator_w) // 2
            overlay[indicator_y:indicator_y+indicator_h, x_offset:x_offset+indicator_w] = panel
            
            # Blend with original (more transparent for modern look)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
            
            # Draw border for modern card effect
            cv2.rectangle(frame, (x_offset, indicator_y), (x_offset+indicator_w, indicator_y+indicator_h),
                         color, 2)
            
            # Draw attention level text with larger, bolder font
            text = f"{attention_level} Attention"
            font_scale = 1.0
            thickness = 3
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_x = x_offset + (indicator_w - text_w) // 2
            text_y = indicator_y + 35
            
            # Text shadow for depth
            cv2.putText(frame, text, (text_x+2, text_y+2),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
            # Main text
            cv2.putText(frame, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            
            # Draw score with modern styling
            score_text = f"{attention_score:.1%}"
            score_font_scale = 0.75
            (score_w, score_h), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, score_font_scale, 2)
            score_x = x_offset + (indicator_w - score_w) // 2
            score_y = indicator_y + indicator_h - 12
            
            cv2.putText(frame, score_text, (score_x, score_y),
                       cv2.FONT_HERSHEY_SIMPLEX, score_font_scale, (255, 255, 255), 2)
            
            # Draw progress bar indicator
            bar_w = indicator_w - 40
            bar_h = 6
            bar_x = x_offset + 20
            bar_y = indicator_y + indicator_h - 25
            
            # Background bar
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (50, 50, 50), -1)
            # Progress bar
            progress_w = int(bar_w * attention_score)
            if progress_w > 0:
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x+progress_w, bar_y+bar_h), color, -1)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error drawing attention overlay: {e}")
            return frame
    
    def _draw_gaze_visualization(self, frame: np.ndarray, gaze_result: Dict[str, Any], w: int, h: int) -> np.ndarray:
        """Draw simple, clean gaze visualization."""
        try:
            left_eye = gaze_result.get('left_eye_landmarks', [])
            right_eye = gaze_result.get('right_eye_landmarks', [])
            gaze_direction = gaze_result.get('gaze_direction', (0.0, 0.0))
            confidence = gaze_result.get('confidence', 0.0)
            
            if not left_eye or not right_eye:
                return frame
            
            # Calculate eye centers
            left_center = np.mean(left_eye, axis=0).astype(int)
            right_center = np.mean(right_eye, axis=0).astype(int)
            
            # Simple eye center dots
            cv2.circle(frame, tuple(left_center), 4, (66, 133, 244), -1)
            cv2.circle(frame, tuple(right_center), 4, (66, 133, 244), -1)
            
            # Calculate center between eyes
            eye_center_x = int((left_center[0] + right_center[0]) / 2)
            eye_center_y = int((left_center[1] + right_center[1]) / 2)
            
            # Simple gaze direction indicator
            arrow_length = 60
            end_x = int(eye_center_x + gaze_direction[0] * arrow_length)
            end_y = int(eye_center_y + gaze_direction[1] * arrow_length)
            
            # Color based on confidence
            if confidence > 0.6:
                arrow_color = (52, 168, 83)  # Green
            elif confidence > 0.3:
                arrow_color = (251, 188, 5)  # Yellow
            else:
                arrow_color = (234, 67, 53)  # Red
            
            # Simple arrow
            cv2.arrowedLine(frame, (eye_center_x, eye_center_y), (end_x, end_y), 
                          arrow_color, 2, tipLength=0.2, lineType=cv2.LINE_AA)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error drawing gaze: {e}")
            return frame
    
    def _draw_pose_info(self, frame: np.ndarray, pose_result: Dict[str, Any], w: int, h: int) -> np.ndarray:
        """Draw modern professional pose information."""
        try:
            pitch = pose_result.get('pitch', 0.0)
            yaw = pose_result.get('yaw', 0.0)
            roll = pose_result.get('roll', 0.0)
            
            # Draw pose info at bottom left with modern styling
            y_start = h - 110
            x_start = 20
            
            # Background panel with modern design
            panel_h = 90
            panel_w = 220
            panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
            panel[:] = (15, 20, 25)  # Dark background
            overlay = frame.copy()
            overlay[y_start:y_start+panel_h, x_start:x_start+panel_w] = panel
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            
            # Draw border
            cv2.rectangle(frame, (x_start, y_start), (x_start+panel_w, y_start+panel_h),
                         (138, 180, 248), 2)
            
            # Draw title
            title = "Head Pose"
            (title_w, title_h), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.putText(frame, title, (x_start + (panel_w - title_w) // 2, y_start + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (138, 180, 248), 2)
            
            # Draw pose angles with color coding
            y_offset = y_start + 40
            x_offset = x_start + 10
            
            # Pitch
            pitch_color = (232, 234, 237) if abs(pitch) < 30 else (251, 188, 5) if abs(pitch) < 60 else (234, 67, 53)
            cv2.putText(frame, f"Pitch: {pitch:+.1f}°", (x_offset, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, pitch_color, 2)
            
            # Yaw
            yaw_color = (232, 234, 237) if abs(yaw) < 30 else (251, 188, 5) if abs(yaw) < 60 else (234, 67, 53)
            cv2.putText(frame, f"Yaw: {yaw:+.1f}°", (x_offset, y_offset+25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, yaw_color, 2)
            
            # Roll
            roll_color = (232, 234, 237) if abs(roll) < 15 else (251, 188, 5) if abs(roll) < 30 else (234, 67, 53)
            cv2.putText(frame, f"Roll: {roll:+.1f}°", (x_offset, y_offset+50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, roll_color, 2)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error drawing pose: {e}")
            return frame
    
    def _draw_metrics_panel(self, frame: np.ndarray, results: Dict[str, Any], w: int, h: int) -> np.ndarray:
        """Draw modern professional metrics panel on the right side."""
        try:
            panel_w = 220
            panel_h = 240
            panel_x = w - panel_w - 20
            panel_y = 20
            
            # Create modern semi-transparent panel with darker background
            panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
            panel[:] = (15, 20, 25)  # Darker background
            overlay = frame.copy()
            overlay[panel_y:panel_y+panel_h, panel_x:panel_x+panel_w] = panel
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            
            # Draw modern border with color
            cv2.rectangle(frame, (panel_x, panel_y), (panel_x+panel_w, panel_y+panel_h),
                         (138, 180, 248), 2)
            
            # Draw title
            title = "Metrics"
            (title_w, title_h), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            title_x = panel_x + (panel_w - title_w) // 2
            cv2.putText(frame, title, (title_x, panel_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (138, 180, 248), 2)
            
            # Draw metrics with better spacing
            y_offset = panel_y + 50
            x_offset = panel_x + 15
            
            # Gaze score
            gaze_score = results.get('gaze_score', 0.0)
            cv2.putText(frame, f"Gaze:", (x_offset, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (232, 234, 237), 2)
            score_text = f"{gaze_score:.0%}"
            (score_w, _), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.putText(frame, score_text, (panel_x + panel_w - score_w - 15, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (52, 168, 83), 2)
            
            # Pose score
            pose_score = results.get('pose_score', 0.0)
            cv2.putText(frame, f"Pose:", (x_offset, y_offset+35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (232, 234, 237), 2)
            score_text = f"{pose_score:.0%}"
            (score_w, _), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.putText(frame, score_text, (panel_x + panel_w - score_w - 15, y_offset+35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (251, 188, 5), 2)
            
            # Behavioral score
            behavioral_score = results.get('behavioral_score', 0.0)
            cv2.putText(frame, f"Behavior:", (x_offset, y_offset+70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (232, 234, 237), 2)
            score_text = f"{behavioral_score:.0%}"
            (score_w, _), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.putText(frame, score_text, (panel_x + panel_w - score_w - 15, y_offset+70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (138, 180, 248), 2)
            
            # Draw enhanced progress bars
            bar_w = 190
            bar_h = 10
            bar_y = y_offset + 95
            
            # Gaze bar with modern styling
            cv2.rectangle(frame, (x_offset, bar_y), (x_offset+bar_w, bar_y+bar_h), (40, 40, 40), -1)
            gaze_w = int(bar_w * gaze_score)
            if gaze_w > 0:
                cv2.rectangle(frame, (x_offset, bar_y), (x_offset+gaze_w, bar_y+bar_h), (52, 168, 83), -1)
                # Highlight
                cv2.rectangle(frame, (x_offset, bar_y), (x_offset+gaze_w, bar_y+2), (102, 221, 170), -1)
            
            # Pose bar
            bar_y += 28
            cv2.rectangle(frame, (x_offset, bar_y), (x_offset+bar_w, bar_y+bar_h), (40, 40, 40), -1)
            pose_w = int(bar_w * pose_score)
            if pose_w > 0:
                cv2.rectangle(frame, (x_offset, bar_y), (x_offset+pose_w, bar_y+bar_h), (251, 188, 5), -1)
                # Highlight
                cv2.rectangle(frame, (x_offset, bar_y), (x_offset+pose_w, bar_y+2), (255, 214, 10), -1)
            
            # Behavioral bar
            bar_y += 28
            cv2.rectangle(frame, (x_offset, bar_y), (x_offset+bar_w, bar_y+bar_h), (40, 40, 40), -1)
            behavioral_w = int(bar_w * behavioral_score)
            if behavioral_w > 0:
                cv2.rectangle(frame, (x_offset, bar_y), (x_offset+behavioral_w, bar_y+bar_h), (138, 180, 248), -1)
                # Highlight
                cv2.rectangle(frame, (x_offset, bar_y), (x_offset+behavioral_w, bar_y+2), (174, 199, 252), -1)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error drawing metrics: {e}")
            return frame
    
    def update_display(self):
        """Update the camera display."""
        try:
            if self.current_frame is not None:
                # Convert frame to QPixmap
                pixmap = self._frame_to_pixmap(self.current_frame)
                
                # Scale pixmap to fit widget while maintaining aspect ratio
                scaled_pixmap = pixmap.scaled(
                    self.camera_label.size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                
                # Update label
                self.camera_label.setPixmap(scaled_pixmap)
                
        except Exception as e:
            logger.error(f"Error updating display: {e}")
    
    def _frame_to_pixmap(self, frame: np.ndarray) -> QPixmap:
        """Convert OpenCV frame to QPixmap."""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get frame dimensions
            height, width, channel = rgb_frame.shape
            bytes_per_line = 3 * width
            
            # Convert to QImage
            q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # Convert to QPixmap
            pixmap = QPixmap.fromImage(q_image)
            
            return pixmap
            
        except Exception as e:
            logger.error(f"Error converting frame: {e}")
            return QPixmap(800, 600)
    
    def clear_display(self):
        """Clear the camera display."""
        self.current_frame = None
        self.current_results = {}
        self.camera_label.setText(
            "<div style='text-align: center; color: #8ab4f8; font-size: 20px; font-weight: 600; padding: 40px;'>"
            "📹 Camera Feed<br><br>"
            "<span style='font-size: 15px; color: #9aa0a6; font-weight: 400;'>Click 'Start Monitoring' to begin</span>"
            "</div>"
        )
