import requests

BASE_URL = "http://localhost:8000"
TIMEOUT = 30

def test_attention_score_calculation_and_session_metrics():
    headers = {"Content-Type": "application/json"}

    # Step 1: Prepare a dummy frame data (flattened list of pixels representing a small grayscale image)
    dummy_frame_2d = [[0.0 for _ in range(10)] for _ in range(10)]  # 10x10 dummy frame data
    dummy_frame = [pixel for row in dummy_frame_2d for pixel in row]  # Flatten to 1D list

    # Step 2: Detect faces using mediapipe backend
    face_detection_payload = {
        "frame": dummy_frame,
        "backend": "mediapipe"
    }
    try:
        face_resp = requests.post(
            f"{BASE_URL}/detect_faces",
            json=face_detection_payload,
            headers=headers,
            timeout=TIMEOUT
        )
        face_resp.raise_for_status()
        faces = face_resp.json()
        assert isinstance(faces, list), "Face detection response should be a list"
        assert len(faces) > 0, "No faces detected, cannot proceed with test"
        face_landmarks = faces[0].get("landmarks")
        assert face_landmarks is not None, "Face landmarks missing in detection result"
    except Exception as e:
        raise AssertionError(f"Face detection failed: {e}")

    # Step 3: Estimate gaze using detected face landmarks
    gaze_payload = {
        "frame": dummy_frame,
        "face_landmarks": face_landmarks
    }
    try:
        gaze_resp = requests.post(
            f"{BASE_URL}/estimate_gaze",
            json=gaze_payload,
            headers=headers,
            timeout=TIMEOUT
        )
        gaze_resp.raise_for_status()
        gaze_result = gaze_resp.json()
        assert "gaze_direction" in gaze_result, "Gaze direction missing in response"
        assert "confidence" in gaze_result, "Gaze confidence missing in response"
        assert isinstance(gaze_result["gaze_direction"], list), "Gaze direction should be a list"
    except Exception as e:
        raise AssertionError(f"Gaze estimation failed: {e}")

    # Step 4: Estimate head pose using detected face landmarks
    pose_payload = {
        "frame": dummy_frame,
        "face_landmarks": face_landmarks
    }
    try:
        pose_resp = requests.post(
            f"{BASE_URL}/estimate_pose",
            json=pose_payload,
            headers=headers,
            timeout=TIMEOUT
        )
        pose_resp.raise_for_status()
        pose_result = pose_resp.json()
        for key in ["pitch", "yaw", "roll", "confidence"]:
            assert key in pose_result, f"{key} missing in pose estimation response"
    except Exception as e:
        raise AssertionError(f"Pose estimation failed: {e}")

    # Step 5: Detect behavioral cues using detected face landmarks
    behavioral_payload = {
        "frame": dummy_frame,
        "face_landmarks": face_landmarks
    }
    try:
        behavioral_resp = requests.post(
            f"{BASE_URL}/detect_behavioral_cues",
            json=behavioral_payload,
            headers=headers,
            timeout=TIMEOUT
        )
        behavioral_resp.raise_for_status()
        behavioral_result = behavioral_resp.json()
        for key in ["blink_rate", "yawn_detected", "movement_score", "confidence"]:
            assert key in behavioral_result, f"{key} missing in behavioral cues response"
    except Exception as e:
        raise AssertionError(f"Behavioral cues detection failed: {e}")

    # Step 6: Calculate attention score using gaze, pose, and behavioral results
    attention_payload = {
        "gaze_result": gaze_result,
        "pose_result": pose_result,
        "behavioral_result": behavioral_result
    }
    try:
        attention_resp = requests.post(
            f"{BASE_URL}/calculate_attention_score",
            json=attention_payload,
            headers=headers,
            timeout=TIMEOUT
        )
        attention_resp.raise_for_status()
        attention_result = attention_resp.json()
        # Validate presence and types of expected fields
        assert "attention_score" in attention_result, "attention_score missing in response"
        assert isinstance(attention_result["attention_score"], (int, float)), "attention_score should be numeric"
        assert "attention_level" in attention_result, "attention_level missing in response"
        assert isinstance(attention_result["attention_level"], str), "attention_level should be string"
        assert "confidence" in attention_result, "confidence missing in response"
        assert isinstance(attention_result["confidence"], (int, float)), "confidence should be numeric"
        assert "session_metrics" in attention_result, "session_metrics missing in response"
        assert isinstance(attention_result["session_metrics"], dict), "session_metrics should be an object"
    except Exception as e:
        raise AssertionError(f"Attention score calculation failed: {e}")


test_attention_score_calculation_and_session_metrics()
