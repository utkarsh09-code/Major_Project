"""
Multimodal Fusion Module

This module fuses features from gaze estimation, pose estimation, and behavioral cues
into a comprehensive attention score using various fusion strategies.
"""

import numpy as np
from typing import Dict, Any, List, Optional
import time

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MultimodalFusion:
    """Fuses multiple modalities for comprehensive attention assessment."""
    
    def __init__(self, fusion_strategy: str = "weighted_sum"):
        """
        Initialize multimodal fusion.
        
        Args:
            fusion_strategy: Fusion strategy ("weighted_sum", "attention", "ensemble")
        """
        self.fusion_strategy = fusion_strategy
        
        # Feature weights for weighted sum fusion (pose is critical)
        self.gaze_weight = 0.35
        self.pose_weight = 0.45  # Increased - head pose is strong indicator
        self.behavioral_weight = 0.2  # Reduced - less critical
        
        # Attention thresholds (stricter for accuracy)
        self.attention_thresholds = {
            'high': 0.70,
            'medium': 0.50,
            'low': 0.30
        }
        
        # History for smoothing
        self.attention_history = []
        self.max_history_length = 30
        
        # Performance tracking
        self.fusion_times = []
        
        logger.info(f"Multimodal fusion initialized with strategy: {fusion_strategy}")
    
    def fuse_features(self, gaze_result: Dict[str, Any], 
                     pose_result: Dict[str, Any], 
                     behavioral_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fuse features from multiple modalities.
        
        Args:
            gaze_result: Gaze estimation results
            pose_result: Pose estimation results
            behavioral_result: Behavioral cues results
            
        Returns:
            Dictionary containing fused attention score and confidence
        """
        start_time = time.time()
        
        try:
            # Extract features
            gaze_features = self._extract_gaze_features(gaze_result)
            pose_features = self._extract_pose_features(pose_result)
            behavioral_features = self._extract_behavioral_features(behavioral_result)
            
            # Apply fusion strategy
            if self.fusion_strategy == "weighted_sum":
                attention_score = self._weighted_sum_fusion(gaze_features, pose_features, behavioral_features)
            elif self.fusion_strategy == "attention":
                attention_score = self._attention_fusion(gaze_features, pose_features, behavioral_features)
            elif self.fusion_strategy == "ensemble":
                attention_score = self._ensemble_fusion(gaze_features, pose_features, behavioral_features)
            else:
                attention_score = self._weighted_sum_fusion(gaze_features, pose_features, behavioral_features)
            
            # Calculate confidence
            confidence = self._calculate_fusion_confidence(gaze_result, pose_result, behavioral_result)
            
            # Update history for smoothing
            self._update_attention_history(attention_score)
            
            # Smooth attention score
            smoothed_score = self._smooth_attention_score()
            
            # Determine attention level
            attention_level = self._determine_attention_level(smoothed_score)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.fusion_times.append(processing_time)
            if len(self.fusion_times) > 100:
                self.fusion_times.pop(0)
            
            return {
                'attention_score': smoothed_score,
                'attention_level': attention_level,
                'confidence': confidence,
                'gaze_contribution': gaze_features['contribution'],
                'pose_contribution': pose_features['contribution'],
                'behavioral_contribution': behavioral_features['contribution'],
                'processing_time_ms': processing_time * 1000,
                'fusion_strategy': self.fusion_strategy
            }
            
        except Exception as e:
            logger.error(f"Error in multimodal fusion: {e}")
            return self._get_empty_fusion_result()
    
    def _extract_gaze_features(self, gaze_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from gaze estimation results with improved sensitivity."""
        gaze_direction = gaze_result.get('gaze_direction', (0.0, 0.0))
        gaze_confidence = gaze_result.get('confidence', 0.0)
        is_blinking = gaze_result.get('is_blinking', False)
        
        # Calculate gaze magnitude (distance from center, normalized to 0-1)
        gaze_magnitude = np.sqrt(gaze_direction[0]**2 + gaze_direction[1]**2)
        
        # Improved gaze scoring (same as attentiveness_scoring)
        if gaze_magnitude <= 0.2:
            gaze_score = 1.0 - (gaze_magnitude / 0.2) * 0.1  # 1.0 to 0.9
        elif gaze_magnitude <= 0.4:
            gaze_score = 0.9 - ((gaze_magnitude - 0.2) / 0.2) * 0.2  # 0.9 to 0.7
        elif gaze_magnitude <= 0.6:
            gaze_score = 0.7 - ((gaze_magnitude - 0.4) / 0.2) * 0.3  # 0.7 to 0.4
        elif gaze_magnitude <= 0.8:
            gaze_score = 0.4 - ((gaze_magnitude - 0.6) / 0.2) * 0.25  # 0.4 to 0.15
        else:
            gaze_score = max(0.0, 0.15 - (gaze_magnitude - 0.8) * 0.5)  # < 0.15
        
        gaze_score = max(0.0, min(1.0, gaze_score))
        gaze_score *= gaze_confidence
        
        # Penalty for blinking
        if is_blinking:
            gaze_score *= 0.92
        
        return {
            'score': gaze_score,
            'confidence': gaze_confidence,
            'direction': gaze_direction,
            'magnitude': gaze_magnitude,
            'is_blinking': is_blinking,
            'contribution': gaze_score * self.gaze_weight
        }
    
    def _extract_pose_features(self, pose_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from pose estimation - lenient for normal angles, strict for extreme."""
        pitch = abs(pose_result.get('pitch', 0.0))
        yaw = abs(pose_result.get('yaw', 0.0))
        roll = abs(pose_result.get('roll', 0.0))
        pose_confidence = pose_result.get('confidence', 0.0)
        is_moving = pose_result.get('is_moving', False)
        
        # Calculate pose stability - use maximum angle deviation
        max_angle = max(pitch, yaw, roll)
        
        # Lenient scoring for normal head movements, strict for extreme angles (same as attentiveness_scoring)
        if max_angle <= 30.0:
            pose_score = 1.0 - (max_angle / 30.0) * 0.3  # 1.0 to 0.7
        elif max_angle <= 60.0:
            pose_score = 0.7 - ((max_angle - 30.0) / 30.0) * 0.4  # 0.7 to 0.3
        elif max_angle <= 90.0:
            pose_score = 0.3 - ((max_angle - 60.0) / 30.0) * 0.2  # 0.3 to 0.1
        elif max_angle <= 120.0:
            normalized = (max_angle - 90.0) / 30.0
            pose_score = 0.1 * (1.0 - normalized)  # 0.1 to 0.0
        elif max_angle <= 180.0:
            normalized = (max_angle - 120.0) / 60.0
            pose_score = 0.01 * (1.0 - normalized)  # Near zero
        else:
            pose_score = 0.0
        
        pose_score = max(0.0, min(1.0, pose_score))
        pose_score *= pose_confidence
        
        # Minimal penalty for movement
        if is_moving:
            pose_score *= 0.95
        
        return {
            'score': pose_score,
            'confidence': pose_confidence,
            'angles': (pitch, yaw, roll),
            'max_angle': max_angle,
            'is_moving': is_moving,
            'contribution': pose_score * self.pose_weight
        }
    
    def _extract_behavioral_features(self, behavioral_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from behavioral cues results."""
        blink_rate = behavioral_result.get('blink_rate', 0.0)
        yawn_rate = behavioral_result.get('yawn_rate', 0.0)
        fatigue_score = behavioral_result.get('fatigue_score', 0.0)
        movement_level = behavioral_result.get('movement_level', 0.0)
        
        # Normalize behavioral features
        normalized_blink_rate = min(blink_rate / 20.0, 1.0)  # Normal blink rate ~15-20/min
        normalized_yawn_rate = min(yawn_rate / 5.0, 1.0)     # Normal yawn rate ~0-5/hour
        
        # Calculate behavioral score (inverse of fatigue)
        behavioral_score = 1.0 - fatigue_score
        
        # Penalize excessive movement
        if movement_level > 0.3:
            behavioral_score *= 0.8
        
        return {
            'score': behavioral_score,
            'blink_rate': blink_rate,
            'yawn_rate': yawn_rate,
            'fatigue_score': fatigue_score,
            'movement_level': movement_level,
            'contribution': behavioral_score * self.behavioral_weight
        }
    
    def _weighted_sum_fusion(self, gaze_features: Dict[str, Any], 
                            pose_features: Dict[str, Any], 
                            behavioral_features: Dict[str, Any]) -> float:
        """Fuse features using weighted sum."""
        attention_score = (
            gaze_features['contribution'] +
            pose_features['contribution'] +
            behavioral_features['contribution']
        )
        
        return min(max(attention_score, 0.0), 1.0)
    
    def _attention_fusion(self, gaze_features: Dict[str, Any], 
                         pose_features: Dict[str, Any], 
                         behavioral_features: Dict[str, Any]) -> float:
        """Fuse features using attention mechanism."""
        # Calculate attention weights based on confidence
        total_confidence = (gaze_features['confidence'] + 
                          pose_features['confidence'] + 
                          0.5)  # Behavioral confidence is implicit
        
        if total_confidence > 0:
            gaze_weight = gaze_features['confidence'] / total_confidence
            pose_weight = pose_features['confidence'] / total_confidence
            behavioral_weight = 0.5 / total_confidence
        else:
            gaze_weight = pose_weight = behavioral_weight = 1.0 / 3.0
        
        # Apply attention weights
        attention_score = (
            gaze_features['score'] * gaze_weight +
            pose_features['score'] * pose_weight +
            behavioral_features['score'] * behavioral_weight
        )
        
        return min(max(attention_score, 0.0), 1.0)
    
    def _ensemble_fusion(self, gaze_features: Dict[str, Any], 
                        pose_features: Dict[str, Any], 
                        behavioral_features: Dict[str, Any]) -> float:
        """Fuse features using ensemble method."""
        scores = [
            gaze_features['score'],
            pose_features['score'],
            behavioral_features['score']
        ]
        
        # Use median for robustness
        attention_score = np.median(scores)
        
        return min(max(attention_score, 0.0), 1.0)
    
    def _calculate_fusion_confidence(self, gaze_result: Dict[str, Any], 
                                   pose_result: Dict[str, Any], 
                                   behavioral_result: Dict[str, Any]) -> float:
        """Calculate confidence in the fusion result."""
        gaze_conf = gaze_result.get('confidence', 0.0)
        pose_conf = pose_result.get('confidence', 0.0)
        
        # Behavioral confidence is implicit (based on detection quality)
        behavioral_conf = 0.5  # Default confidence
        
        # Weighted average of confidences
        total_confidence = (
            gaze_conf * self.gaze_weight +
            pose_conf * self.pose_weight +
            behavioral_conf * self.behavioral_weight
        )
        
        return min(total_confidence, 1.0)
    
    def _update_attention_history(self, attention_score: float) -> None:
        """Update attention score history for smoothing."""
        self.attention_history.append(attention_score)
        if len(self.attention_history) > self.max_history_length:
            self.attention_history.pop(0)
    
    def _smooth_attention_score(self) -> float:
        """Smooth attention score using moving average."""
        if not self.attention_history:
            return 0.0
        
        # Use exponential moving average for more responsive smoothing
        alpha = 0.3  # Smoothing factor
        smoothed_score = self.attention_history[0]
        
        for score in self.attention_history[1:]:
            smoothed_score = alpha * score + (1 - alpha) * smoothed_score
        
        return smoothed_score
    
    def _determine_attention_level(self, attention_score: float) -> str:
        """Determine attention level based on score."""
        if attention_score >= self.attention_thresholds['high']:
            return "High"
        elif attention_score >= self.attention_thresholds['medium']:
            return "Medium"
        elif attention_score >= self.attention_thresholds['low']:
            return "Low"
        else:
            return "Very Low"
    
    def _get_empty_fusion_result(self) -> Dict[str, Any]:
        """Return empty fusion result."""
        return {
            'attention_score': 0.0,
            'attention_level': "Very Low",
            'confidence': 0.0,
            'gaze_contribution': 0.0,
            'pose_contribution': 0.0,
            'behavioral_contribution': 0.0,
            'processing_time_ms': 0.0,
            'fusion_strategy': self.fusion_strategy
        }
    
    def get_attention_summary(self, fusion_result: Dict[str, Any]) -> str:
        """Get human-readable attention summary."""
        attention_level = fusion_result.get('attention_level', "Very Low")
        confidence = fusion_result.get('confidence', 0.0)
        
        if confidence < 0.3:
            return f"{attention_level} Attention (Low Confidence)"
        else:
            return f"{attention_level} Attention"
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get fusion performance metrics."""
        avg_processing_time = np.mean(self.fusion_times) if self.fusion_times else 0.0
        
        return {
            'fusion_strategy': self.fusion_strategy,
            'avg_processing_time_ms': avg_processing_time * 1000,
            'attention_history_length': len(self.attention_history),
            'gaze_weight': self.gaze_weight,
            'pose_weight': self.pose_weight,
            'behavioral_weight': self.behavioral_weight
        } 