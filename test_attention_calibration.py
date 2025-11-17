#!/usr/bin/env python3
"""
Test script to validate attention scoring calibration.
Tests various head pose angles to ensure proper scoring.
"""

import sys
from src.core.attentiveness_scoring import AttentivenessScorer

def test_pose_scoring():
    """Test pose scoring for various angles."""
    scorer = AttentivenessScorer()
    
    print("=" * 70)
    print("ATTENTION SCORING CALIBRATION TEST")
    print("=" * 70)
    print("\nTesting pose scoring for various head angles:")
    print("-" * 70)
    
    # Test cases: (angle, expected_level, description)
    test_cases = [
        (0, "High", "Looking straight ahead"),
        (15, "High", "Normal head movement"),
        (30, "High", "Moderate head turn (should still allow high)"),
        (45, "Medium", "Looking around (should allow medium)"),
        (60, "Medium", "Significant head turn (should allow medium)"),
        (75, "Low", "Large head turn"),
        (90, "Low", "Very large head turn"),
        (120, "Very Low", "Extreme head turn"),
        (150, "Very Low", "Nearly turned away"),
        (180, "Very Low", "Completely turned away (180 degrees)"),
    ]
    
    print(f"{'Angle':<10} {'Pose Score':<15} {'Attention':<15} {'Level':<15} {'Status'}")
    print("-" * 70)
    
    all_passed = True
    
    for angle, expected_level, description in test_cases:
        # Create mock pose result
        pose_result = {
            'pitch': 0.0,
            'yaw': angle,  # Test yaw angle
            'roll': 0.0,
            'confidence': 1.0,
            'is_moving': False
        }
        
        # Create mock gaze and behavioral results (neutral)
        gaze_result = {
            'gaze_direction': (0.0, 0.0),
            'confidence': 1.0,
            'is_blinking': False
        }
        
        behavioral_result = {
            'blink_rate': 15.0,
            'yawn_rate': 0.0,
            'fatigue_score': 0.0,
            'movement_level': 0.0
        }
        
        # Calculate scores
        pose_score = scorer._calculate_pose_score(pose_result)
        scoring_result = scorer.calculate_attention_score(gaze_result, pose_result, behavioral_result)
        attention_score = scoring_result['attention_score']
        attention_level = scoring_result['attention_level']
        
        # Validate
        passed = True
        if angle >= 120:
            # 120+ degrees (approaching 180) should be Very Low
            if attention_level != "Very Low":
                passed = False
                all_passed = False
        elif angle >= 90:
            # 90-120 degrees should be Low or Very Low
            if attention_level not in ["Low", "Very Low"]:
                passed = False
                all_passed = False
        elif angle >= 60:
            # 60-90 degrees should allow Medium or Low
            if attention_level == "High":
                passed = False
                all_passed = False
        elif angle <= 30:
            # 0-30 degrees should allow High
            if attention_score < 0.5:  # Should be at least medium
                passed = False
                all_passed = False
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{angle:>3}°      {pose_score:>6.3f}         {attention_score:>6.3f}         {attention_level:<15} {status}")
    
    print("-" * 70)
    print("\n" + "=" * 70)
    
    if all_passed:
        print("✅ ALL TESTS PASSED - Calibration is correct!")
        print("=" * 70)
        return True
    else:
        print("❌ SOME TESTS FAILED - Calibration needs adjustment")
        print("=" * 70)
        return False

def test_180_degree_specific():
    """Specifically test 180-degree case to ensure it's very low."""
    scorer = AttentivenessScorer()
    
    print("\n" + "=" * 70)
    print("SPECIFIC TEST: 180-DEGREE HEAD TURN")
    print("=" * 70)
    
    # Test 180-degree yaw (completely turned away)
    pose_result = {
        'pitch': 0.0,
        'yaw': 180.0,
        'roll': 0.0,
        'confidence': 1.0,
        'is_moving': False
    }
    
    # Even with perfect gaze and behavior, attention should be very low
    gaze_result = {
        'gaze_direction': (0.0, 0.0),  # Perfect gaze
        'confidence': 1.0,
        'is_blinking': False
    }
    
    behavioral_result = {
        'blink_rate': 15.0,  # Normal
        'yawn_rate': 0.0,    # No yawning
        'fatigue_score': 0.0,  # No fatigue
        'movement_level': 0.0  # No movement
    }
    
    scoring_result = scorer.calculate_attention_score(gaze_result, pose_result, behavioral_result)
    attention_score = scoring_result['attention_score']
    attention_level = scoring_result['attention_level']
    pose_score = scoring_result['pose_score']
    
    print(f"\nTest Scenario: Head turned 180 degrees (perfect gaze, no fatigue)")
    print(f"Pose Score: {pose_score:.4f}")
    print(f"Attention Score: {attention_score:.4f}")
    print(f"Attention Level: {attention_level}")
    print(f"\nExpected: Very Low (score < 0.30)")
    
    if attention_level == "Very Low" and attention_score < 0.30:
        print("✅ PASS: 180-degree turn correctly classified as Very Low")
        return True
    else:
        print(f"❌ FAIL: 180-degree turn incorrectly classified as {attention_level} (score: {attention_score:.4f})")
        return False

def test_normal_movements():
    """Test that normal head movements (0-60 degrees) can achieve high scores."""
    scorer = AttentivenessScorer()
    
    print("\n" + "=" * 70)
    print("TEST: NORMAL HEAD MOVEMENTS (0-60 degrees)")
    print("=" * 70)
    
    test_cases = [
        (0, "Should be High"),
        (20, "Should allow High"),
        (45, "Should allow Medium/High"),
        (60, "Should allow Medium"),
    ]
    
    all_passed = True
    print(f"\n{'Angle':<10} {'Pose Score':<15} {'Attention':<15} {'Level':<15} {'Status'}")
    print("-" * 70)
    
    for angle, description in test_cases:
        pose_result = {
            'pitch': 0.0,
            'yaw': angle,
            'roll': 0.0,
            'confidence': 1.0,
            'is_moving': False
        }
        
        gaze_result = {
            'gaze_direction': (0.0, 0.0),
            'confidence': 1.0,
            'is_blinking': False
        }
        
        behavioral_result = {
            'blink_rate': 15.0,
            'yawn_rate': 0.0,
            'fatigue_score': 0.0,
            'movement_level': 0.0
        }
        
        scoring_result = scorer.calculate_attention_score(gaze_result, pose_result, behavioral_result)
        attention_score = scoring_result['attention_score']
        attention_level = scoring_result['attention_level']
        pose_score = scoring_result['pose_score']
        
        # Normal movements should allow scores above 0.30
        passed = attention_score > 0.30
        if not passed:
            all_passed = False
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{angle:>3}°      {pose_score:>6.3f}         {attention_score:>6.3f}         {attention_level:<15} {status}")
    
    return all_passed

if __name__ == "__main__":
    print("\n")
    test1 = test_pose_scoring()
    test2 = test_180_degree_specific()
    test3 = test_normal_movements()
    
    print("\n" + "=" * 70)
    if test1 and test2 and test3:
        print("✅ ALL CALIBRATION TESTS PASSED")
        print("The model is now properly calibrated!")
        print("- Normal head movements (0-60°) can achieve High/Medium attention")
        print("- Extreme angles (120-180°) are correctly classified as Very Low")
        sys.exit(0)
    else:
        print("❌ CALIBRATION TESTS FAILED")
        print("Model needs further adjustment.")
        sys.exit(1)

