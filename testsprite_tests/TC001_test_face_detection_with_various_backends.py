import requests
import json

BASE_URL = "http://localhost:8000"
TIMEOUT = 30

def test_face_detection_with_various_backends():
    url = f"{BASE_URL}/detect_faces"
    headers = {"Content-Type": "application/json"}

    # Sample dummy frame data (e.g., flattened grayscale or RGB pixel values)
    # For testing, we use a small array of numbers to simulate a frame.
    sample_frame = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    backends = ["mediapipe", "deepface", "opencv", "dlib"]

    for backend in backends:
        payload = {
            "frame": sample_frame,
            "backend": backend
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
        except requests.RequestException as e:
            assert False, f"Request to /detect_faces with backend '{backend}' failed: {e}"

        assert response.status_code == 200, f"Expected status 200 but got {response.status_code} for backend '{backend}'"

        try:
            data = response.json()
        except json.JSONDecodeError:
            assert False, f"Response is not valid JSON for backend '{backend}'"

        assert isinstance(data, list), f"Response data should be a list for backend '{backend}'"

        for face in data:
            assert isinstance(face, dict), f"Each face detection result should be a dict for backend '{backend}'"
            # Validate bbox
            assert "bbox" in face, f"'bbox' missing in face detection result for backend '{backend}'"
            bbox = face["bbox"]
            assert isinstance(bbox, list), f"'bbox' should be a list for backend '{backend}'"
            assert all(isinstance(coord, (int, float)) for coord in bbox), f"All bbox coordinates should be numbers for backend '{backend}'"
            # Validate confidence
            assert "confidence" in face, f"'confidence' missing in face detection result for backend '{backend}'"
            confidence = face["confidence"]
            assert isinstance(confidence, (int, float)), f"'confidence' should be a number for backend '{backend}'"
            assert 0.0 <= confidence <= 1.0, f"'confidence' should be between 0 and 1 for backend '{backend}'"
            # Validate landmarks
            assert "landmarks" in face, f"'landmarks' missing in face detection result for backend '{backend}'"
            landmarks = face["landmarks"]
            assert isinstance(landmarks, dict), f"'landmarks' should be an object/dict for backend '{backend}'"

test_face_detection_with_various_backends()