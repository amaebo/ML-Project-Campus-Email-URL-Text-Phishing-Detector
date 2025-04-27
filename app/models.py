# baseline_model_v2.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import preprocessing

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
# Train Model
# ======================

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# ======================
# Evaluate on Validation Set
# ======================

y_val_pred = lr_model.predict(X_val)

print("\n=== Validation Set Classification Report ===")
print(classification_report(y_val, y_val_pred))

print("\n=== Validation Set Confusion Matrix ===")
val_cm = confusion_matrix(y_val, y_val_pred)
print(val_cm)

# ======================
# Final Test Set Evaluation
# ======================

y_test_pred = lr_model.predict(X_test)

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

# ======================
# (Optional) Accuracy Score
# ======================

print("\n=== Test Set Accuracy Score ===")
print(accuracy_score(y_test, y_test_pred))
