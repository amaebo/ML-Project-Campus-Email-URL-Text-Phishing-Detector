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
# Load Trained Model
# ======================

# Make sure model path exists
MODEL_PATH = 'models/best_logistic_regression_model.pkl'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train and save your model first.")

# Load the saved Logistic Regression model
model = joblib.load(MODEL_PATH)

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

    # Validate input fields
    required_fields = ['sender', 'subject', 'body']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f"Missing field: {field}"}), 400

    # Construct a one-row DataFrame from input
    email_df = pd.DataFrame([{
        'sender': data['sender'],
        'receiver': '',  # Placeholder (for preprocessing)
        'date': '',      # Placeholder
        'subject': data['subject'],
        'body': data['body'],
        'label': 0,      # Placeholder
        'urls': ''       # Placeholder
    }])

    # Preprocess the email
    X = preprocessing.preprocess(email_df)
    X = X.fillna(0)

    # Make prediction
    prediction = model.predict(X)[0]

    # Return prediction
    return jsonify({'prediction': int(prediction)})

# ======================
# Run the App
# ======================

if __name__ == '__main__':
    app.run(debug=True)
