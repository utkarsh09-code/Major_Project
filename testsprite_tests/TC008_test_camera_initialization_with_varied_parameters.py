import requests

BASE_URL = "http://localhost:8000"
TIMEOUT = 30

def test_camera_initialization_with_varied_parameters():
    url = f"{BASE_URL}/initialize_camera"
    test_params = [
        {"camera_index": 0, "width": 640, "height": 480, "fps": 30},
        {"camera_index": 1, "width": 1280, "height": 720, "fps": 60},
        {"camera_index": 0, "width": 1920, "height": 1080, "fps": 15},
        {"camera_index": 2, "width": 800, "height": 600, "fps": 24},
        {"camera_index": 0, "width": 320, "height": 240, "fps": 10}
    ]

    for params in test_params:
        try:
            response = requests.post(url, json=params, timeout=TIMEOUT)
            assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
            data = response.json()
            assert "success" in data, "Response missing 'success' field"
            assert data["success"] is True, f"Camera initialization failed for params: {params}"
            assert "camera_info" in data, "Response missing 'camera_info' field"
            camera_info = data["camera_info"]
            assert isinstance(camera_info, dict), "'camera_info' should be an object"
            # Optionally check that camera_info contains expected keys like width, height, fps, camera_index
            for key in ["camera_index", "width", "height", "fps"]:
                assert key in camera_info, f"'camera_info' missing key: {key}"
                assert camera_info[key] == params[key], f"Mismatch in '{key}': expected {params[key]}, got {camera_info[key]}"
        except requests.exceptions.RequestException as e:
            assert False, f"Request failed for params {params}: {e}"

test_camera_initialization_with_varied_parameters()