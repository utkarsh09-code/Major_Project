import requests

BASE_URL = "http://localhost:8000"
TIMEOUT = 30

def test_start_monitoring_api_functionality():
    url = f"{BASE_URL}/start_monitoring"
    headers = {"Content-Type": "application/json"}
    payload = {
        "camera_index": 0,
        "resolution": {"width": 640, "height": 480},
        "backend": "mediapipe"
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as e:
        assert False, f"Request to /start_monitoring failed: {e}"

    data = response.json()
    assert isinstance(data, dict), "Response is not a JSON object"
    assert "success" in data, "'success' key missing in response"
    assert data["success"] is True, "Monitoring did not start successfully"
    assert "session_id" in data, "'session_id' key missing in response"
    assert isinstance(data["session_id"], str) and data["session_id"], "Invalid session_id returned"

test_start_monitoring_api_functionality()