"""
Attentiveness Scoring Module

This module calculates the final attentiveness score and manages session-level metrics
for comprehensive attention assessment.
"""

import numpy as np
import time
from typing import Dict, Any, List, Optional
import json
import os

from ..utils.logger import get_logger

logger = get_logger(__name__)


class AttentivenessScorer:
    """Calculates and manages attentiveness scores and session metrics."""
    
    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize attentiveness scorer.
        
        Args:
            session_id: Unique session identifier
        """
        self.session_id = session_id or f"session_{int(time.time())}"
        
        # Session metrics
        self.session_start_time = time.time()
        self.total_frames = 0
        self.attention_scores = []
        self.attention_levels = []
        self.session_events = []
        
        # Performance tracking
        self.scoring_times = []
        
        # Temporal smoothing for attention scores (prevents sudden jumps)
        self.attention_history = []
        self.smoothing_window = 10  # Number of frames for smoothing
        self.smoothing_alpha = 0.3  # Exponential smoothing factor (0-1, lower = more smoothing)
        
        # Attention thresholds (stricter for better accuracy)
        self.attention_thresholds = {
            'high': 0.70,      # High attention requires good pose + gaze
            'medium': 0.50,    # Medium requires at least moderate attention
            'low': 0.30        # Low threshold for minimal attention
        }
        
        # Session statistics
        self.session_stats = {
            'avg_attention_score': 0.0,
            'attention_distribution': {'high': 0, 'medium': 0, 'low': 0, 'very_low': 0},
            'total_duration': 0.0,
            'focus_percentage': 0.0
        }
        
        logger.info(f"Attentiveness scorer initialized for session: {self.session_id}")
    
    def calculate_attention_score(self, gaze_result: Dict[str, Any], 
                                pose_result: Dict[str, Any], 
                                behavioral_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comprehensive attention score from multiple modalities.
        
        Args:
            gaze_result: Gaze estimation results
            pose_result: Pose estimation results
            behavioral_result: Behavioral cues results
            
        Returns:
            Dictionary containing attention score and metrics
        """
        start_time = time.time()
        
        try:
            # Extract individual scores
            gaze_score = self._calculate_gaze_score(gaze_result)
            pose_score = self._calculate_pose_score(pose_result)
            behavioral_score = self._calculate_behavioral_score(behavioral_result)
            
            # Calculate weighted attention score (pose is critical indicator)
            # Increased pose weight since head angle is a strong attention signal
            attention_score = (
                0.35 * gaze_score +
                0.45 * pose_score +  # Increased from 0.3 to 0.45
                0.2 * behavioral_score  # Reduced from 0.3 to 0.2
            )
            
            # REMOVED: Hard caps that create barriers and cause extreme jumps
            # The weighted sum naturally handles all cases - no artificial restrictions
            
            # Ensure score is in [0, 1] range
            attention_score = max(0.0, min(1.0, attention_score))
            
            # Apply temporal smoothing to prevent sudden jumps
            attention_score = self._apply_temporal_smoothing(attention_score)
            
            # Determine attention level
            attention_level = self._determine_attention_level(attention_score)
            
            # Calculate confidence
            confidence = self._calculate_overall_confidence(gaze_result, pose_result, behavioral_result)
            
            # Update session metrics
            self._update_session_metrics(attention_score, attention_level)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.scoring_times.append(processing_time)
            if len(self.scoring_times) > 100:
                self.scoring_times.pop(0)
            
            return {
                'attention_score': attention_score,
                'attention_level': attention_level,
                'confidence': confidence,
                'gaze_score': gaze_score,
                'pose_score': pose_score,
                'behavioral_score': behavioral_score,
                'session_duration': time.time() - self.session_start_time,
                'processing_time_ms': processing_time * 1000
            }
            
        except Exception as e:
            logger.error(f"Error calculating attention score: {e}")
            return self._get_empty_scoring_result()
    
    def _calculate_gaze_score(self, gaze_result: Dict[str, Any]) -> float:
        """Calculate gaze-based attention score with smooth, continuous transitions."""
        gaze_direction = gaze_result.get('gaze_direction', (0.0, 0.0))
        gaze_confidence = gaze_result.get('confidence', 0.0)
        is_blinking = gaze_result.get('is_blinking', False)
        
        # Calculate gaze magnitude (distance from center, normalized to 0-1)
        gaze_magnitude = np.sqrt(gaze_direction[0]**2 + gaze_direction[1]**2)
        
        # Smooth, continuous gaze scoring (no hard thresholds)
        # Use exponential decay for natural, gradual transitions
        # At 0.0: score = 1.0 (perfect focus)
        # At 1.0: score approaches 0.0 (completely off-center)
        # This creates smooth transitions without barriers
        
        # Exponential decay function for smooth scoring
        # Parameters tuned for realistic attention curve
        gaze_score = np.exp(-2.5 * gaze_magnitude)  # Smooth exponential decay
        
        # Ensure score is in valid range
        gaze_score = max(0.0, min(1.0, gaze_score))
        
        # Apply confidence weighting
        gaze_score *= gaze_confidence
        
        # Minimal penalty for blinking (natural behavior)
        if is_blinking:
            gaze_score *= 0.95  # Reduced penalty for more realistic scoring
        
        return gaze_score
    
    def _calculate_pose_score(self, pose_result: Dict[str, Any]) -> float:
        """Calculate pose-based attention score with smooth, continuous transitions."""
        pitch = abs(pose_result.get('pitch', 0.0))
        yaw = abs(pose_result.get('yaw', 0.0))
        roll = abs(pose_result.get('roll', 0.0))
        pose_confidence = pose_result.get('confidence', 0.0)
        is_moving = pose_result.get('is_moving', False)
        
        # Calculate pose stability - use maximum angle deviation
        max_angle = max(pitch, yaw, roll)
        
        # Smooth, continuous pose scoring using exponential decay
        # No hard thresholds - creates gradual transitions
        # At 0 degrees: score = 1.0 (perfect forward-facing)
        # At 180 degrees: score approaches 0.0 (completely turned away)
        
        # Normalize angle to 0-1 range (0-180 degrees)
        normalized_angle = min(max_angle / 180.0, 1.0)
        
        # Exponential decay for smooth, realistic scoring
        # Tuned to allow normal head movements (0-60 deg) while penalizing extreme turns
        pose_score = np.exp(-2.0 * normalized_angle)  # Smooth exponential decay
        
        # Ensure score is in valid range
        pose_score = max(0.0, min(1.0, pose_score))
        
        # Apply confidence weighting
        pose_score *= pose_confidence
        
        # Minimal penalty for movement (natural behavior)
        if is_moving:
            pose_score *= 0.97  # Reduced penalty for more realistic scoring
        
        return pose_score
    
    def _calculate_behavioral_score(self, behavioral_result: Dict[str, Any]) -> float:
        """Calculate behavioral-based attention score."""
        blink_rate = behavioral_result.get('blink_rate', 0.0)
        yawn_rate = behavioral_result.get('yawn_rate', 0.0)
        fatigue_score = behavioral_result.get('fatigue_score', 0.0)
        movement_level = behavioral_result.get('movement_level', 0.0)
        
        # Calculate behavioral score (inverse of fatigue)
        behavioral_score = 1.0 - fatigue_score
        
        # More lenient movement penalty
        if movement_level > 0.5:  # Increased threshold
            behavioral_score *= 0.85
        
        # Normalize blink rate (more lenient)
        if blink_rate > 30:  # Increased threshold
            behavioral_score *= 0.9
        
        # Normalize yawn rate (more lenient)
        if yawn_rate > 8:  # Increased threshold
            behavioral_score *= 0.8
        
        return behavioral_score
    
    def _apply_temporal_smoothing(self, attention_score: float) -> float:
        """
        Apply temporal smoothing to prevent sudden jumps in attention scores.
        Uses exponential moving average for responsive yet stable smoothing.
        """
        # Add current score to history
        self.attention_history.append(attention_score)
        
        # Maintain smoothing window size
        if len(self.attention_history) > self.smoothing_window:
            self.attention_history.pop(0)
        
        # If we don't have enough history, return current score
        if len(self.attention_history) < 2:
            return attention_score
        
        # Apply exponential moving average for smooth transitions
        # Lower alpha = more smoothing (less responsive to sudden changes)
        smoothed_score = self.attention_history[0]
        
        for score in self.attention_history[1:]:
            smoothed_score = self.smoothing_alpha * score + (1 - self.smoothing_alpha) * smoothed_score
        
        return smoothed_score
    
    def _determine_attention_level(self, attention_score: float) -> str:
        """Determine attention level based on score (more lenient thresholds)."""
        if attention_score >= self.attention_thresholds['high']:
            return "High"
        elif attention_score >= self.attention_thresholds['medium']:
            return "Medium"
        elif attention_score >= self.attention_thresholds['low']:
            return "Low"
        else:
            return "Very Low"
    
    def _calculate_overall_confidence(self, gaze_result: Dict[str, Any], 
                                    pose_result: Dict[str, Any], 
                                    behavioral_result: Dict[str, Any]) -> float:
        """Calculate overall confidence in the attention assessment."""
        gaze_conf = gaze_result.get('confidence', 0.0)
        pose_conf = pose_result.get('confidence', 0.0)
        
        # Behavioral confidence is implicit
        behavioral_conf = 0.6  # Default confidence
        
        # Weighted average
        total_confidence = (
            0.4 * gaze_conf +
            0.3 * pose_conf +
            0.3 * behavioral_conf
        )
        
        return min(total_confidence, 1.0)
    
    def _update_session_metrics(self, attention_score: float, attention_level: str) -> None:
        """Update session-level metrics."""
        self.total_frames += 1
        self.attention_scores.append(attention_score)
        self.attention_levels.append(attention_level)
        
        # Update attention distribution
        if attention_level == "High":
            self.session_stats['attention_distribution']['high'] += 1
        elif attention_level == "Medium":
            self.session_stats['attention_distribution']['medium'] += 1
        elif attention_level == "Low":
            self.session_stats['attention_distribution']['low'] += 1
        else:
            self.session_stats['attention_distribution']['very_low'] += 1
        
        # Update average attention score
        if self.attention_scores:
            self.session_stats['avg_attention_score'] = np.mean(self.attention_scores)
        
        # Update session duration
        self.session_stats['total_duration'] = time.time() - self.session_start_time
        
        # Calculate focus percentage (time spent with medium or high attention)
        focus_frames = (self.session_stats['attention_distribution']['high'] + 
                       self.session_stats['attention_distribution']['medium'])
        if self.total_frames > 0:
            self.session_stats['focus_percentage'] = (focus_frames / self.total_frames) * 100.0
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session summary."""
        return {
            'session_id': self.session_id,
            'session_duration': self.session_stats['total_duration'],
            'total_frames': self.total_frames,
            'avg_attention_score': self.session_stats['avg_attention_score'],
            'focus_percentage': self.session_stats['focus_percentage'],
            'attention_distribution': self.session_stats['attention_distribution'],
            'avg_processing_time_ms': np.mean(self.scoring_times) * 1000 if self.scoring_times else 0.0
        }
    
    def get_attention_trend(self, window_size: int = 30) -> str:
        """Get attention trend over recent frames."""
        if len(self.attention_scores) < window_size:
            return "Insufficient Data"
        
        recent_scores = self.attention_scores[-window_size:]
        trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
        
        if trend > 0.01:
            return "Improving"
        elif trend < -0.01:
            return "Declining"
        else:
            return "Stable"
    
    def detect_attention_issues(self, scoring_result: Dict[str, Any]) -> List[str]:
        """Detect potential attention issues."""
        issues = []
        
        attention_score = scoring_result.get('attention_score', 0.0)
        gaze_score = scoring_result.get('gaze_score', 0.0)
        pose_score = scoring_result.get('pose_score', 0.0)
        behavioral_score = scoring_result.get('behavioral_score', 0.0)
        
        # Check for low attention score
        if attention_score < 0.3:  # More lenient threshold
            issues.append("Low attention level")
        
        # Check for gaze issues
        if gaze_score < 0.2:  # More lenient threshold
            issues.append("Poor gaze focus")
        
        # Check for pose issues
        if pose_score < 0.2:  # More lenient threshold
            issues.append("Head turned away")
        
        # Check for behavioral issues
        if behavioral_score < 0.3:  # More lenient threshold
            issues.append("Fatigue indicators")
        
        return issues
    
    def save_session_data(self, filepath: Optional[str] = None) -> str:
        """Save session data to file."""
        if filepath is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = f"session_data_{self.session_id}_{timestamp}.json"
        
        session_data = {
            'session_id': self.session_id,
            'session_start_time': self.session_start_time,
            'session_end_time': time.time(),
            'summary': self.get_session_summary(),
            'attention_scores': self.attention_scores,
            'attention_levels': self.attention_levels,
            'performance_metrics': {
                'avg_processing_time_ms': np.mean(self.scoring_times) * 1000 if self.scoring_times else 0.0,
                'total_scoring_operations': len(self.scoring_times)
            }
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(session_data, f, indent=2)
            logger.info(f"Session data saved to: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving session data: {e}")
            return ""
    
    def _get_empty_scoring_result(self) -> Dict[str, Any]:
        """Return empty scoring result."""
        return {
            'attention_score': 0.0,
            'attention_level': "Very Low",
            'confidence': 0.0,
            'gaze_score': 0.0,
            'pose_score': 0.0,
            'behavioral_score': 0.0,
            'session_duration': 0.0,
            'processing_time_ms': 0.0
        }
    
    def get_attention_summary(self, scoring_result: Dict[str, Any]) -> str:
        """Get human-readable attention summary."""
        attention_level = scoring_result.get('attention_level', "Very Low")
        attention_score = scoring_result.get('attention_score', 0.0)
        confidence = scoring_result.get('confidence', 0.0)
        
        if confidence < 0.3:
            return f"{attention_level} Attention (Low Confidence)"
        else:
            return f"{attention_level} Attention ({attention_score:.2f})"
    
    def reset_session(self, new_session_id: Optional[str] = None) -> None:
        """Reset session metrics for a new session."""
        if new_session_id:
            self.session_id = new_session_id
        else:
            self.session_id = f"session_{int(time.time())}"
        
        self.session_start_time = time.time()
        self.total_frames = 0
        self.attention_scores.clear()
        self.attention_levels.clear()
        self.session_events.clear()
        self.scoring_times.clear()
        self.attention_history.clear()  # Reset smoothing history
        
        # Reset session statistics
        self.session_stats = {
            'avg_attention_score': 0.0,
            'attention_distribution': {'high': 0, 'medium': 0, 'low': 0, 'very_low': 0},
            'total_duration': 0.0,
            'focus_percentage': 0.0
        }
        
        logger.info(f"Session reset: {self.session_id}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get scoring performance metrics."""
        avg_processing_time = np.mean(self.scoring_times) if self.scoring_times else 0.0
        
        return {
            'avg_processing_time_ms': avg_processing_time * 1000,
            'total_scoring_operations': len(self.scoring_times),
            'session_duration': time.time() - self.session_start_time,
            'attention_thresholds': self.attention_thresholds
        } 