import requests

BASE_URL = "http://localhost:8000"
TIMEOUT = 30

def test_behavioral_cues_detection_functionality():
    # Prepare a dummy frame and face_landmarks for testing
    # Flatten the 2D frame to a 1D list as per API schema
    frame_2d = [[0.0 for _ in range(10)] for _ in range(10)]  # 10x10 zero frame
    frame = [pixel for row in frame_2d for pixel in row]
    face_landmarks = {
        "landmark_0": {"x": 0.5, "y": 0.5},
        "landmark_1": {"x": 0.6, "y": 0.5},
        "landmark_2": {"x": 0.4, "y": 0.6}
    }

    url = f"{BASE_URL}/detect_behavioral_cues"
    headers = {"Content-Type": "application/json"}
    payload = {
        "frame": frame,
        "face_landmarks": face_landmarks
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as e:
        assert False, f"Request to /detect_behavioral_cues failed: {e}"

    data = response.json()

    # Validate response keys and types
    assert isinstance(data, dict), "Response should be a JSON object"
    assert "blink_rate" in data, "Response missing 'blink_rate'"
    assert "yawn_detected" in data, "Response missing 'yawn_detected'"
    assert "movement_score" in data, "Response missing 'movement_score'"
    assert "confidence" in data, "Response missing 'confidence'"

    assert isinstance(data["blink_rate"], (int, float)), "'blink_rate' should be a number"
    assert isinstance(data["yawn_detected"], bool), "'yawn_detected' should be a boolean"
    assert isinstance(data["movement_score"], (int, float)), "'movement_score' should be a number"
    assert isinstance(data["confidence"], (int, float)), "'confidence' should be a number"

    # Validate reasonable ranges for numeric values
    assert 0 <= data["blink_rate"] <= 100, "'blink_rate' out of expected range (0-100)"
    assert 0 <= data["movement_score"] <= 1, "'movement_score' out of expected range (0-1)"
    assert 0 <= data["confidence"] <= 1, "'confidence' out of expected range (0-1)"


test_behavioral_cues_detection_functionality()
