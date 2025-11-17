import requests

BASE_URL = "http://localhost:8000"
TIMEOUT = 30

def test_multimodal_fusion_with_different_strategies():
    headers = {"Content-Type": "application/json"}

    # Use a dummy frame and minimal landmarks for testing
    dummy_frame = [0] * 100  # changed to integer zero values

    # Step 1: Detect faces using mediapipe backend to get face landmarks
    face_detect_payload = {
        "frame": dummy_frame,
        "backend": "mediapipe"
    }
    try:
        face_detect_resp = requests.post(
            f"{BASE_URL}/detect_faces",
            json=face_detect_payload,
            headers=headers,
            timeout=TIMEOUT
        )
        face_detect_resp.raise_for_status()
        faces = face_detect_resp.json()
        assert isinstance(faces, list), "Face detection response is not a list"
        assert len(faces) > 0, "No faces detected for fusion test"
        face_landmarks = faces[0].get("landmarks")
        assert face_landmarks is not None, "Face landmarks missing in detection result"
    except Exception as e:
        raise AssertionError(f"Face detection failed: {e}")

    # Step 2: Estimate gaze
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
        assert "gaze_direction" in gaze_result and isinstance(gaze_result["gaze_direction"], list), "Invalid gaze_direction"
        assert "confidence" in gaze_result and isinstance(gaze_result["confidence"], (float, int)), "Missing gaze confidence"
    except Exception as e:
        raise AssertionError(f"Gaze estimation failed: {e}")

    # Step 3: Estimate pose
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
            assert key in pose_result, f"Pose result missing {key}"
            assert isinstance(pose_result[key], (float, int)), f"Pose {key} is not a number"
    except Exception as e:
        raise AssertionError(f"Pose estimation failed: {e}")

    # Step 4: Detect behavioral cues
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
        assert "blink_rate" in behavioral_result and isinstance(behavioral_result["blink_rate"], (float, int)), "Missing blink_rate"
        assert "yawn_detected" in behavioral_result and isinstance(behavioral_result["yawn_detected"], bool), "Missing yawn_detected"
        assert "movement_score" in behavioral_result and isinstance(behavioral_result["movement_score"], (float, int)), "Missing movement_score"
        assert "confidence" in behavioral_result and isinstance(behavioral_result["confidence"], (float, int)), "Missing behavioral confidence"
    except Exception as e:
        raise AssertionError(f"Behavioral cues detection failed: {e}")

    # Step 5: Test fusion strategies
    fusion_strategies = ["weighted_sum", "attention", "ensemble"]
    for strategy in fusion_strategies:
        fusion_payload = {
            "gaze_result": gaze_result,
            "pose_result": pose_result,
            "behavioral_result": behavioral_result,
            "fusion_strategy": strategy
        }
        try:
            fusion_resp = requests.post(
                f"{BASE_URL}/fuse_modalities",
                json=fusion_payload,
                headers=headers,
                timeout=TIMEOUT
            )
            fusion_resp.raise_for_status()
            fusion_result = fusion_resp.json()
            assert "fused_score" in fusion_result and isinstance(fusion_result["fused_score"], (float, int)), f"Fused score missing or invalid for strategy {strategy}"
            assert "confidence" in fusion_result and isinstance(fusion_result["confidence"], (float, int)), f"Confidence missing or invalid for strategy {strategy}"
            assert "modality_weights" in fusion_result and isinstance(fusion_result["modality_weights"], dict), f"Modality weights missing or invalid for strategy {strategy}"
            # modality_weights keys should include gaze_result, pose_result, behavioral_result
            for modality in ["gaze_result", "pose_result", "behavioral_result"]:
                assert modality in fusion_result["modality_weights"], f"Modality weight for {modality} missing in strategy {strategy}"
        except Exception as e:
            raise AssertionError(f"Fusion failed for strategy {strategy}: {e}")

test_multimodal_fusion_with_different_strategies()
