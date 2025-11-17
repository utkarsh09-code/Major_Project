import requests
import json

BASE_URL = "http://localhost:8000"
TIMEOUT = 30

def test_gaze_estimation_accuracy_and_response():
    # First, create a sample frame and face_landmarks by detecting a face using /detect_faces
    detect_faces_url = f"{BASE_URL}/detect_faces"
    estimate_gaze_url = f"{BASE_URL}/estimate_gaze"

    # Sample dummy frame data (e.g., flattened grayscale or RGB pixel values)
    # For testing, use a small dummy array; in real scenario, this should be actual frame data
    sample_frame = [0.0] * 100  # minimal dummy frame data

    # Detect faces to get face_landmarks for gaze estimation
    detect_faces_payload = {
        "frame": sample_frame,
        "backend": "mediapipe"
    }

    try:
        detect_faces_resp = requests.post(detect_faces_url, json=detect_faces_payload, timeout=TIMEOUT)
        detect_faces_resp.raise_for_status()
        faces = detect_faces_resp.json()
        assert isinstance(faces, list), "Face detection response should be a list"
        assert len(faces) > 0, "No faces detected to test gaze estimation"
        face_landmarks = faces[0].get("landmarks")
        assert face_landmarks is not None, "Face landmarks missing in detection response"
    except Exception as e:
        raise AssertionError(f"Face detection failed or returned invalid data: {e}")

    # Prepare payload for gaze estimation
    gaze_payload = {
        "frame": sample_frame,
        "face_landmarks": face_landmarks
    }

    try:
        gaze_resp = requests.post(estimate_gaze_url, json=gaze_payload, timeout=TIMEOUT)
        gaze_resp.raise_for_status()
        gaze_result = gaze_resp.json()
    except Exception as e:
        raise AssertionError(f"Gaze estimation request failed: {e}")

    # Validate gaze estimation response structure and values
    assert isinstance(gaze_result, dict), "Gaze estimation response should be a dictionary"
    assert "gaze_direction" in gaze_result, "Missing 'gaze_direction' in response"
    assert "confidence" in gaze_result, "Missing 'confidence' in response"
    assert "eye_landmarks" in gaze_result, "Missing 'eye_landmarks' in response"

    gaze_direction = gaze_result["gaze_direction"]
    confidence = gaze_result["confidence"]
    eye_landmarks = gaze_result["eye_landmarks"]

    assert isinstance(gaze_direction, list), "'gaze_direction' should be a list"
    assert all(isinstance(x, (int, float)) for x in gaze_direction), "'gaze_direction' items should be numbers"
    assert len(gaze_direction) > 0, "'gaze_direction' should not be empty"

    assert isinstance(confidence, (int, float)), "'confidence' should be a number"
    assert 0.0 <= confidence <= 1.0, "'confidence' should be between 0 and 1"

    assert isinstance(eye_landmarks, dict), "'eye_landmarks' should be a dictionary"

test_gaze_estimation_accuracy_and_response()