# test_api.py

import requests
import json

# URL of local Flask API
API_URL = 'http://127.0.0.1:5000/predict'

# Example email data to test
sample_email = {
    "sender": "alert@secure-paypal.com",
    "subject": "URGENT: Verify your account now!",
    "body": "Dear customer, please click here to verify your account immediately."
}

# Send POST request to API
response = requests.post(API_URL, json=sample_email)

# Check the response
if response.status_code == 200:
    result = response.json()
    print("\n=== API Prediction ===")
    print(f"Prediction: {result['prediction']} (1 = Phishing, 0 = Legitimate)")
else:
    print("\n[ERROR] Failed to get a valid response from API.")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
