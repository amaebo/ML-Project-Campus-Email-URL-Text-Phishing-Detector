# baseline_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import preprocessing

# ======================
# Load and Preprocess Data
# ======================

# Load raw dataset
raw_df = pd.read_csv('data/CEAS-08.csv')

# Preprocess features
X = preprocessing.preprocess(raw_df.copy())
X = X.fillna(0)

# Correct y to match surviving rows
y = raw_df.loc[X.index, 'label']

# ======================
# Train-Test Split
# ======================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# ======================
# Train Baseline Logistic Regression
# ======================

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# ======================
# Save Baseline Model
# ======================

joblib.dump(lr_model, 'baseline_logistic_regression_model.pkl')

print("\n Baseline model saved successfully as 'baseline_logistic_regression_model.pkl'")

# ======================
# Evaluate on Test Set
# ======================

y_test_pred = lr_model.predict(X_test)

print("\n=== Test Set Classification Report ===")
print(classification_report(y_test, y_test_pred))

print("\n=== Test Set Confusion Matrix ===")
test_cm = confusion_matrix(y_test, y_test_pred)
print(test_cm)

TN, FP, FN, TP = test_cm.ravel()
print(f"\nFalse Positives: {FP}")
print(f"False Negatives: {FN}")

print("\n=== Test Set Accuracy Score ===")
print(accuracy_score(y_test, y_test_pred))
