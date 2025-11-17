import requests

BASE_URL = "http://localhost:8000"
TIMEOUT = 30

def test_head_pose_estimation_results_consistency():
    # Sample dummy frame data (e.g., flattened pixel values or simplified numeric array)
    sample_frame = [0.0, 0.1, 0.2, 0.3, 0.4]  # Minimal example frame data
    # Sample dummy face landmarks data (as an object/dict)
    sample_face_landmarks = {
        "landmark_0": {"x": 0.5, "y": 0.5, "z": 0.0},
        "landmark_1": {"x": 0.6, "y": 0.5, "z": 0.0},
        "landmark_2": {"x": 0.5, "y": 0.6, "z": 0.0}
    }

    url = f"{BASE_URL}/estimate_pose"
    headers = {"Content-Type": "application/json"}
    payload = {
        "frame": sample_frame,
        "face_landmarks": sample_face_landmarks
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=TIMEOUT)
    except requests.RequestException as e:
        assert False, f"Request to /estimate_pose failed: {e}"

    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"

    try:
        data = response.json()
    except ValueError:
        assert False, "Response is not valid JSON"

    # Validate presence and types of required fields
    for key in ["pitch", "yaw", "roll", "confidence"]:
        assert key in data, f"Response JSON missing '{key}' field"
        assert isinstance(data[key], (int, float)), f"Field '{key}' should be a number"

    # Validate confidence score range
    confidence = data["confidence"]
    assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} out of expected range [0.0, 1.0]"

    # Validate pitch, yaw, roll reasonable ranges (typical head pose angles in degrees)
    pitch = data["pitch"]
    yaw = data["yaw"]
    roll = data["roll"]
    # Assuming pitch, yaw, roll in degrees roughly between -180 and 180
    for angle_name, angle_value in [("pitch", pitch), ("yaw", yaw), ("roll", roll)]:
        assert -180.0 <= angle_value <= 180.0, f"{angle_name} angle {angle_value} out of range [-180, 180]"

test_head_pose_estimation_results_consistency()