import requests

BASE_URL = "http://localhost:8000"
TIMEOUT = 30

def test_session_summary_retrieval():
    url = f"{BASE_URL}/get_session_summary"
    headers = {
        "Accept": "application/json"
    }
    try:
        response = requests.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as e:
        assert False, f"Request to /get_session_summary failed: {e}"

    data = response.json()

    # Validate presence and types of expected fields
    assert isinstance(data, dict), "Response is not a JSON object"
    assert "avg_attention_score" in data, "Missing avg_attention_score in response"
    assert "attention_distribution" in data, "Missing attention_distribution in response"
    assert "total_duration" in data, "Missing total_duration in response"
    assert "focus_percentage" in data, "Missing focus_percentage in response"

    assert isinstance(data["avg_attention_score"], (int, float)), "avg_attention_score is not a number"
    assert isinstance(data["attention_distribution"], dict), "attention_distribution is not an object"
    assert isinstance(data["total_duration"], (int, float)), "total_duration is not a number"
    assert isinstance(data["focus_percentage"], (int, float)), "focus_percentage is not a number"

    # Validate value ranges where applicable
    assert 0.0 <= data["avg_attention_score"] <= 1.0, "avg_attention_score out of expected range 0.0-1.0"
    assert data["total_duration"] >= 0, "total_duration should be non-negative"
    assert 0.0 <= data["focus_percentage"] <= 100.0, "focus_percentage out of expected range 0.0-100.0"

    # Optionally validate attention_distribution contents if keys are known
    # Here just check it's not empty
    assert len(data["attention_distribution"]) > 0, "attention_distribution is empty"

test_session_summary_retrieval()