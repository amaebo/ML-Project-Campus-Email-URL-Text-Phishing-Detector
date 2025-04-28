# Optimization of the Logistic Regression model for phishing detection - Mai Nguyen

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from scipy.stats import loguniform
import preprocessing
import joblib
import os

# ======================
# Load and Preprocess Data
# ======================

# Load dataset
raw_df = pd.read_csv('data/CEAS-08.csv')

# Preprocess features
X = preprocessing.preprocess(raw_df.copy())
X = X.fillna(0)

# Correct y to match surviving rows
y = raw_df.loc[X.index, 'label']

# ======================
# Train-Validation-Test Split
# ======================

# Step 1: Split into Train+Val and Test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 2: Split Train+Val into Train and Validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=0.25,  # 0.25 of 0.8 = 0.2 → final: 60% train, 20% val, 20% test
    random_state=42,
    stratify=y_train_val
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# ======================
# Baseline Model Cross-Validation
# ======================

baseline_model = LogisticRegression(max_iter=1000)
baseline_cv_scores = cross_val_score(
    baseline_model,
    X_train_val,
    y_train_val,
    cv=3,
    scoring='accuracy'
)

print("\n=== Baseline Logistic Regression CV Results ===")
print("Cross-validation scores:", baseline_cv_scores)
print("Mean CV Accuracy:", baseline_cv_scores.mean())

# ======================
# Hyperparameter Tuning (Randomized Search)
# ======================

param_distributions = {
    'C': loguniform(0.01, 10),
    'solver': ['liblinear', 'lbfgs']
}

random_search = RandomizedSearchCV(
    LogisticRegression(max_iter=1000),
    param_distributions=param_distributions,
    n_iter=8,  # Only try 8 random combinations
    cv=3,
    scoring='accuracy',
    n_jobs=1,
    random_state=42
)

random_search.fit(X_train_val, y_train_val)

print("\n=== Hyperparameter Tuning Results ===")
print("Best Parameters:", random_search.best_params_)
print("Best Cross-Validation Score:", random_search.best_score_)

# ======================
# Retrain Best Model
# ======================

best_lr_model = random_search.best_estimator_

# Train on Train set
best_lr_model.fit(X_train, y_train)

# ======================
# Evaluate on Validation Set
# ======================

y_val_pred = best_lr_model.predict(X_val)

print("\n=== Validation Set Classification Report ===")
print(classification_report(y_val, y_val_pred))

print("\n=== Validation Set Confusion Matrix ===")
val_cm = confusion_matrix(y_val, y_val_pred)
print(val_cm)

# ======================
# Final Evaluation on Test Set
# ======================

y_test_pred = best_lr_model.predict(X_test)

print("\n=== Test Set Classification Report ===")
print(classification_report(y_test, y_test_pred))

print("\n=== Test Set Confusion Matrix ===")
test_cm = confusion_matrix(y_test, y_test_pred)
print(test_cm)

# ======================
# False Positives / False Negatives on Test Set
# ======================

TN, FP, FN, TP = test_cm.ravel()

print(f"\nFalse Positives (Legitimate misclassified as Phishing): {FP}")
print(f"False Negatives (Phishing misclassified as Legitimate): {FN}")

print("\n=== Test Set Accuracy Score ===")
print(accuracy_score(y_test, y_test_pred))

# Save the trained best model
os.makedirs('models', exist_ok=True)

joblib.dump(best_lr_model, 'models/best_logistic_regression_model.pkl')

print("\n Model saved successfully as 'best_logistic_regression_model.pkl'")

# === Save Evaluation Metrics to Table ===
os.makedirs('evaluations', exist_ok=True)

accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

# Create simple evaluation table
optimized_evaluation_results = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Score': [accuracy, precision, recall, f1]
})

# Save to CSV inside 'evaluations' folder
optimized_evaluation_results.to_csv('evaluations/optimized_evaluation_results.csv', index=False)