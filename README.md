# Campus Email URL/Text Phishing Detector

## Overview

This project is a **phishing email detector** that analyzes different components of an email—such as the **sender, subject, body, and hyperlinks**—to identify and prevent phishing attacks before they cause harm. The model is deployed as an API to allow real-time phishing detection.

## Features

- **Email Text Analysis**: Checks for suspicious keywords, urgency cues, and phishing-related text.
- **URL Inspection**: Extracts and analyzes links for potential phishing threats.
- **Machine Learning & Deep Learning Models**: Uses **Logistic Regression, Random Forest, BERT**, and other approaches.
- **API for Deployment**: Deployed using **Flask** or **FastAPI** for real-time detection.

---

## Installation

### **1. Set Up the Virtual Environment in Python 3.10.0**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### **2. Install Dependencies**

```bash
pip install -r requirements.txt
```

---
### **3. Training the Models**

To train the phishing detection models, run the following scripts in order.

> ⚠️ **Disclaimer:** Depending on your machine and dataset size, model training — especially the optimization step — may take up to **45 minutes** to complete.

#### 1. Baseline Model

This script trains a basic logistic regression model using default hyperparameters and the preprocessed features.

```bash
python baseline_model.py
```
#### 2. Optimized Model
This script performs hyperparameter tuning using RandomizedSearchCV to improve model performance.
```bash
python optimize_model.py
```

#### 3. Saved Models and Evaluation Metrics
The models are saved to the following files under /models:
   - baseline_logistic_regression_model.pkl
   - best_logistic_regression_model.pkl
     
We evaluate both the baseline and optimized models using the following metrics:
   - Accuracy
   - Precision
   - Recall
   - F1 Score

Each score is printed to the console and saved to the CSV files: 
   - baseline_evaluation_results.csv
   - optimized_evaluation_results.csv

### **4. Running the API**

#### **Step 1: Start the Flask API**

1. Ensure that all required model files are present in the `models/` directory:
   - `best_logistic_regression_model.pkl`
   - `tfidf_vectorizer.pkl`
   - `sender_encoder.pkl`
   - `feature_names.pkl`

2. Start the Flask application:
   ```bash
   python -m app.app
   ```

3. The API will start on `http://127.0.0.1:5000/` by default.

---

#### **Step 2: Test the API**

You can test the API in two ways:

##### **Option 1: Using cURL**
Send a POST request to the `/predict` endpoint with a JSON payload:
```bash
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{
    "sender": "alert@secure-paypal.com",
    "subject": "URGENT: Verify your account now!",
    "body": "Dear customer, please click here to verify your account immediately."
}'
```

- **Response Example**:
  ```json
  {
    "prediction": 1
  }
  ```
  - `1` indicates phishing, `0` indicates legitimate.

##### **Option 2: Using the Provided `test_api.py` Script**
1. Open a new terminal window while the Flask API is running.
2. Run the `test_api.py` script:
   ```bash
   python test_api.py
   ```
3. The script will send a sample email to the `/predict` endpoint and display the prediction result:
   - Example Output:
     ```
     === API Prediction ===
     Prediction: 1 (1 = Phishing, 0 = Legitimate)
     ```

---

#### **Testing with Custom Data**

To test with your own email data:
1. If using cURL, modify the JSON payload in the `-d` flag.
2. If using `test_api.py`, open the file and update the `sample_email` dictionary with your own `sender`, `subject`, and `body` values:
   ```python
   sample_email = {
       "sender": "your_email@example.com",
       "subject": "Your custom subject",
       "body": "Your custom email body"
   }
   ```
3. Save the file and rerun the script:
   ```bash
   python test_api.py
   ```

## Dataset Sources
We used the following publicly available datasets for training and evaluation:

- **Phishing Email Dataset by Naser Abdullah Alam (Kaggle)**  
  A curated dataset specifically targeting phishing email detection, containing labeled examples suitable for training and testing classification models.  
  Link: https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
---

## Contributors

- **Ama Ebong**
- **Mai Nguyen**
