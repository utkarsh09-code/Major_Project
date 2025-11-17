import requests

BASE_URL = "http://localhost:8000"
TIMEOUT = 30
HEADERS = {"Content-Type": "application/json"}

def test_privacy_data_anonymization_levels():
    endpoint = f"{BASE_URL}/anonymize_data"
    sample_sensitive_data = {
        "name": "John Doe",
        "email": "john.doe@example.com",
        "phone": "+1234567890",
        "address": "1234 Elm Street, Springfield",
        "ssn": "123-45-6789"
    }
    anonymization_levels = ["low", "medium", "high"]

    for level in anonymization_levels:
        payload = {
            "data": sample_sensitive_data,
            "anonymization_level": level
        }
        try:
            response = requests.post(endpoint, json=payload, headers=HEADERS, timeout=TIMEOUT)
            response.raise_for_status()
        except requests.RequestException as e:
            assert False, f"Request failed for anonymization level '{level}': {e}"

        json_resp = response.json()
        assert "anonymized_data" in json_resp, f"Missing 'anonymized_data' in response for level '{level}'"
        assert isinstance(json_resp["anonymized_data"], dict), f"'anonymized_data' should be a dict for level '{level}'"
        assert "privacy_score" in json_resp, f"Missing 'privacy_score' in response for level '{level}'"
        privacy_score = json_resp["privacy_score"]
        assert isinstance(privacy_score, (int, float)), f"'privacy_score' should be a number for level '{level}'"
        assert 0 <= privacy_score <= 1, f"'privacy_score' should be between 0 and 1 for level '{level}'"

        # Additional checks: anonymized data should differ from original for medium and high levels
        if level in ["medium", "high"]:
            assert json_resp["anonymized_data"] != sample_sensitive_data, f"Data not anonymized for level '{level}'"
        # For low level, anonymized data may be similar or slightly altered, so no strict inequality check

test_privacy_data_anonymization_levels()