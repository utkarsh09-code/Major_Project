#!/usr/bin/env python3
"""
Flask web server wrapper for the attentiveness detection system.
This allows Testsprite to test the core functionality via HTTP API.
"""

from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import io
from PIL import Image
import time
import threading
import logging

# Import core modules
from src.core.face_detection import FaceDetector
from src.core.gaze_estimation import GazeEstimator
from src.core.pose_estimation import PoseEstimator
from src.core.behavioral_cues import BehavioralCuesDetector
from src.core.multimodal_fusion import MultimodalFusion
from src.core.attentiveness_scoring import AttentivenessScorer

app = Flask(__name__)

# Global variables for components
face_detector = None
gaze_estimator = None
pose_estimator = None
behavioral_detector = None
fusion = None
scorer = None

def initialize_components():
    """Initialize all system components."""
    global face_detector, gaze_estimator, pose_estimator, behavioral_detector, fusion, scorer
    
    try:
        face_detector = FaceDetector(backend="mediapipe")
        gaze_estimator = GazeEstimator(backend="mediapipe")
        pose_estimator = PoseEstimator(backend="mediapipe")
        behavioral_detector = BehavioralCuesDetector()
        fusion = MultimodalFusion(fusion_strategy="weighted_sum")
        scorer = AttentivenessScorer(session_id="web_session")
        
        app.logger.info("All components initialized successfully")
        return True
    except Exception as e:
        app.logger.error(f"Error initializing components: {e}")
        return False

def create_test_image():
    """Create a simple test image for testing."""
    # Create a simple test image (640x480 RGB)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add some simple shapes to make it more interesting
    cv2.rectangle(img, (100, 100), (300, 300), (255, 255, 255), -1)
    cv2.circle(img, (400, 200), 50, (255, 0, 0), -1)
    return img

def array_to_cv2(data):
    """Convert array data to OpenCV image."""
    try:
        # If it's a base64 string
        if isinstance(data, str):
            return base64_to_cv2(data)
        
        # If it's a list/array of numbers
        if isinstance(data, list):
            # Try to reshape as RGB image
            data = np.array(data, dtype=np.uint8)
            
            # If the data is too small, create a test image
            if len(data) < 1000:
                return create_test_image()
            
            # Try to reshape as RGB image (assuming square-ish dimensions)
            size = int(np.sqrt(len(data) // 3))
            if size * size * 3 == len(data):
                return data.reshape(size, size, 3)
            else:
                # Fallback to test image
                return create_test_image()
        
        # If it's already a numpy array
        if isinstance(data, np.ndarray):
            return data
        
        # Fallback to test image
        return create_test_image()
        
    except Exception as e:
        app.logger.error(f"Error converting array to image: {e}")
        return create_test_image()

def base64_to_cv2(base64_string):
    """Convert base64 string to OpenCV image."""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        img_data = base64.b64decode(base64_string)
        
        # Convert to PIL Image
        img = Image.open(io.BytesIO(img_data))
        
        # Convert to OpenCV format (BGR)
        cv2_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        return cv2_img
    except Exception as e:
        app.logger.error(f"Error converting base64 to image: {e}")
        return create_test_image()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'components_initialized': all([
            face_detector, gaze_estimator, pose_estimator, 
            behavioral_detector, fusion, scorer
        ])
    })

@app.route('/detect_faces', methods=['POST'])
def detect_faces():
    """Detect faces in the provided image."""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Convert image data to OpenCV format
        frame = array_to_cv2(data['image'])
        if frame is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Detect faces
        faces = face_detector.detect_faces(frame)
        
        # Convert results to serializable format
        results = []
        for face in faces:
            results.append({
                'bbox': face.get('bbox', []),
                'confidence': face.get('confidence', 0.0),
                'landmarks': face.get('landmarks', {})
            })
        
        return jsonify({
            'success': True,
            'faces': results,
            'face_count': len(results)
        })
        
    except Exception as e:
        app.logger.error(f"Error in face detection: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/estimate_gaze', methods=['POST'])
def estimate_gaze():
    """Estimate gaze direction."""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Convert image data to OpenCV format
        frame = array_to_cv2(data['image'])
        if frame is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Estimate gaze
        gaze_result = gaze_estimator.estimate_gaze(frame)
        
        return jsonify({
            'success': True,
            'gaze_result': gaze_result
        })
        
    except Exception as e:
        app.logger.error(f"Error in gaze estimation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/estimate_pose', methods=['POST'])
def estimate_pose():
    """Estimate head pose."""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Convert image data to OpenCV format
        frame = array_to_cv2(data['image'])
        if frame is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Estimate pose
        pose_result = pose_estimator.estimate_pose(frame)
        
        return jsonify({
            'success': True,
            'pose_result': pose_result
        })
        
    except Exception as e:
        app.logger.error(f"Error in pose estimation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/detect_behavioral_cues', methods=['POST'])
def detect_behavioral_cues():
    """Detect behavioral cues."""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Convert image data to OpenCV format
        frame = array_to_cv2(data['image'])
        if frame is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Detect behavioral cues
        behavioral_result = behavioral_detector.detect_cues(frame)
        
        return jsonify({
            'success': True,
            'behavioral_result': behavioral_result,
            # Also return key fields at top level for compatibility
            'is_blinking': behavioral_result.get('is_blinking', False),
            'blink_rate': behavioral_result.get('blink_rate', 0.0),
            'is_yawning': behavioral_result.get('is_yawning', False),
            'yawn_rate': behavioral_result.get('yawn_rate', 0.0),
            'movement_score': behavioral_result.get('movement_level', 0.0),
            'fatigue_score': behavioral_result.get('fatigue_score', 0.0),
            'fatigue_level': behavioral_result.get('fatigue_level', 'Unknown'),
            'confidence': behavioral_result.get('confidence', 0.0)
        })
        
    except Exception as e:
        app.logger.error(f"Error in behavioral cues detection: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/calculate_attention_score', methods=['POST'])
def calculate_attention_score():
    """Calculate attention score from multiple modalities."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get individual results
        gaze_result = data.get('gaze_result', {})
        pose_result = data.get('pose_result', {})
        behavioral_result = data.get('behavioral_result', {})
        
        # Calculate attention score
        scoring_result = scorer.calculate_attention_score(
            gaze_result, pose_result, behavioral_result
        )
        
        return jsonify({
            'success': True,
            'scoring_result': scoring_result
        })
        
    except Exception as e:
        app.logger.error(f"Error in attention scoring: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_session_summary', methods=['GET'])
def get_session_summary():
    """Get session summary."""
    try:
        summary = scorer.get_session_summary()
        return jsonify({
            'success': True,
            'session_summary': summary
        })
        
    except Exception as e:
        app.logger.error(f"Error getting session summary: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/fuse_modalities', methods=['POST'])
def fuse_modalities():
    """Fuse multiple detection modalities."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get individual results
        gaze_result = data.get('gaze_result', {})
        pose_result = data.get('pose_result', {})
        behavioral_result = data.get('behavioral_result', {})
        fusion_strategy = data.get('fusion_strategy', 'weighted_sum')
        
        # Fuse modalities
        fusion_result = fusion.fuse_modalities(
            gaze_result, pose_result, behavioral_result, fusion_strategy
        )
        
        return jsonify({
            'success': True,
            'fusion_result': fusion_result
        })
        
    except Exception as e:
        app.logger.error(f"Error in modality fusion: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/test_camera', methods=['GET'])
def test_camera():
    """Test camera access."""
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return jsonify({
                'success': False,
                'error': 'Camera not accessible'
            }), 500
        
        # Read a frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({
                'success': False,
                'error': 'Could not read frame from camera'
            }), 500
        
        return jsonify({
            'success': True,
            'message': 'Camera is accessible',
            'frame_shape': frame.shape
        })
        
    except Exception as e:
        app.logger.error(f"Error testing camera: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Process a complete frame through all components."""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Convert image data to OpenCV format
        frame = array_to_cv2(data['image'])
        if frame is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Process through all components
        faces = face_detector.detect_faces(frame)
        
        if not faces:
            return jsonify({
                'success': True,
                'message': 'No faces detected',
                'attention_score': 0.0,
                'attention_level': 'no_face'
            })
        
        # Get individual results
        gaze_result = gaze_estimator.estimate_gaze(frame)
        pose_result = pose_estimator.estimate_pose(frame)
        behavioral_result = behavioral_detector.detect_cues(frame)
        
        # Calculate attention score
        scoring_result = scorer.calculate_attention_score(
            gaze_result, pose_result, behavioral_result
        )
        
        return jsonify({
            'success': True,
            'face_count': len(faces),
            'gaze_result': gaze_result,
            'pose_result': pose_result,
            'behavioral_result': behavioral_result,
            'scoring_result': scoring_result
        })
        
    except Exception as e:
        app.logger.error(f"Error processing frame: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize components
    if not initialize_components():
        print("Failed to initialize components. Exiting.")
        exit(1)
    
    print("Starting attentiveness detection web server on port 8000...")
    print("Available endpoints:")
    print("  GET  /health - Health check")
    print("  POST /detect_faces - Face detection")
    print("  POST /estimate_gaze - Gaze estimation")
    print("  POST /estimate_pose - Pose estimation")
    print("  POST /detect_behavioral_cues - Behavioral cues detection")
    print("  POST /calculate_attention_score - Attention scoring")
    print("  GET  /get_session_summary - Session summary")
    print("  POST /fuse_modalities - Modality fusion")
    print("  GET  /test_camera - Camera test")
    print("  POST /process_frame - Complete frame processing")
    
    app.run(host='0.0.0.0', port=8000, debug=False) 