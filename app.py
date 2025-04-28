# app.py

from flask import Flask, request, jsonify
import pandas as pd
import joblib
from app import preprocessing
import os

# ======================
# Initialize Flask App
# ======================

app = Flask(__name__)

# ======================
# Load Trained Model and TF-IDF Vectorizer
# ======================

MODEL_PATH = 'models/best_logistic_regression_model.pkl'
VECTORIZER_PATH = 'models/tfidf_vectorizer.pkl'

# Ensure paths exist
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train and save your model first.")

if not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError(f"Vectorizer not found at {VECTORIZER_PATH}. Please train and save your vectorizer first.")

# Load model and vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# ======================
# Define Predict Endpoint
# ======================

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to predict if an email is phishing or legitimate.
    Expects JSON input with keys: 'sender', 'subject', 'body'.
    Returns a prediction: 0 = legitimate, 1 = phishing.
    """

    # Get JSON data
    data = request.get_json()

    # Validate input
    required_fields = ['sender', 'subject', 'body']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f"Missing field: {field}"}), 400

    # Construct a one-row DataFrame from input
    email_df = pd.DataFrame([{
        'sender': data['sender'],
        'receiver': '',
        'date': '',
        'subject': data['subject'],
        'body': data['body'],
        'label': 0,
        'urls': ''
    }])

    # Preprocess using the pre-loaded vectorizer
    X = preprocessing.preprocess(email_df, vectorizer=vectorizer)
    X = X.fillna(0)

    # Predict
    prediction = model.predict(X)[0]

    return jsonify({'prediction': int(prediction)})

# ======================
# Run Flask App
# ======================

if __name__ == '__main__':
    app.run(debug=True)
