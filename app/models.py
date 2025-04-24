import pandas as pd
import numpy as np
import preprocessing

# Load raw dataset
raw_df = pd.read_csv('data/CEAS-08.csv')

# Preprocess a copy
X_features = preprocessing.preprocess(raw_df.copy())

# ====== Debugging Output ======

# View extracted feature names
print("\n[INFO] Feature Columns:")
print(X_features.columns.tolist()[:10], "...")  # Print only first 10 to keep it short

# View one-hot sender example (should not include 'sender' column anymore)
print("\n[INFO] Sample Sender Features:")
print(X_features.iloc[:5, :10])  # Print first 5 rows and first 10 columns

# If you want to check sender domains pre-extraction:
print("\n[DEBUG] Raw Sender Domains:")
print(raw_df['sender'].head())


