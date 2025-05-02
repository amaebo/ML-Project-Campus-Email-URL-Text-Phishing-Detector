# app.py

from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
from app import preprocessing

# ======================
# Initialize Flask App
# ======================

app = Flask(__name__)

# ======================
# Load Saved Assets
# ======================

MODEL_PATH = 'models/best_logistic_regression_model.pkl'
VECTORIZER_PATH = 'models/tfidf_vectorizer.pkl'
ENCODER_PATH = 'models/sender_encoder.pkl'
FEATURE_NAMES_PATH = 'models/feature_names.pkl'

# Check that all required files exist
for path, label in [
    (MODEL_PATH, "Model"),
    (VECTORIZER_PATH, "Vectorizer"),
    (ENCODER_PATH, "Sender encoder"),
    (FEATURE_NAMES_PATH, "Feature names")
]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found at {path}. Please train and save your model pipeline.")

# Load all necessary objects
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
sender_encoder = joblib.load(ENCODER_PATH)
feature_names = joblib.load(FEATURE_NAMES_PATH)

# ======================
# Predict Endpoint
# ======================

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict whether a submitted email is phishing or legitimate.
    Expects JSON input with 'sender', 'subject', 'body'.
    Returns prediction as JSON: 0 = legitimate, 1 = phishing.
    """
    data = request.get_json()

    # Validate required fields
    required_fields = ['sender', 'subject', 'body']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f"Missing field: {field}"}), 400

    # Build a one-row DataFrame from input
    email_df = pd.DataFrame([{
        'sender': data['sender'],
        'receiver': '',
        'date': '',
        'subject': data['subject'],
        'body': data['body'],
        'label': 0,
        'urls': ''
    }])

    # Preprocess input using loaded encoder and vectorizer
    X = preprocessing.preprocess(email_df, encoder=sender_encoder, vectorizer=vectorizer)
    X = X.reindex(columns=feature_names, fill_value=0)

    # Predict using the trained model
    prediction = model.predict(X)[0]

    return jsonify({'prediction': int(prediction)})

# ======================
# Run App
# ======================

if __name__ == '__main__':
    app.run(debug=True)
