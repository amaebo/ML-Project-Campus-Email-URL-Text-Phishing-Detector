# Baseline Model for Phishing Detection using Logistic Regression - Mai Nguyen

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

# Load raw dataset
raw_df = pd.read_csv('data/CEAS-08.csv')

# Preprocess
X = preprocessing.preprocess(raw_df.copy())
X = X.fillna(0)

# Correct y to match surviving rows
y = raw_df.loc[X.index, 'label']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Predict & Evaluate
y_pred = lr_model.predict(X_test)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== Accuracy Score ===")
print(accuracy_score(y_test, y_pred))

# Plot Confusion Matrix
ConfusionMatrixDisplay.from_estimator(
    lr_model,
    X_test,
    y_test,
    cmap="Blues",
    display_labels=["Legitimate", "Phishing"]
)
plt.title("Confusion Matrix - Logistic Regression")
plt.show()
