# All the necessary functions for preprocessing the data
# Data Handling
import pandas as pd
import numpy as np

# Text Processing
import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import tldextract  # For URL domain parsing

# Feature Engineering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder


def preprocess(dataset):
    ''' 
    Preprocesses the entire dataset, handles missing data, and extracts features 
    for further processing and model training.

    Parameters:
    - dataset (pd.DataFrame): Full unprocessed dataset containing columns such as 
      'sender', 'receiver', 'date', 'subject', 'body', 'label', 'urls'.
    
    Returns:
    - processed_df (pd.DataFrame): Processed data frame with extracted features.
    
    Raises:
    - ValueError: If the dataset is not a pandas DataFrame or is empty.
    '''
    
    # Check if the dataset is a pandas DataFrame
    if not isinstance(dataset, pd.DataFrame):
        raise ValueError("Input dataset must be a pandas DataFrame.")
    
    # Check if the dataset is empty
    if dataset.empty:
        raise ValueError("Dataset is empty.")
    
    # Check for missing values
    missing_values = dataset.isnull().sum()
    if missing_values.any():
        print("Warning: Missing values found in the following columns:")
        print(missing_values[missing_values > 0])

    # Check for expected columns 
    required_columns = ['sender', 'receiver', 'date', 'subject', 'body', 'label', 'urls']
    missing_columns = [col for col in required_columns if col not in dataset.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {', '.join(missing_columns)}")
    
    # Sender Preprocessing 
    dataset = sender_encoding(dataset)
    
    # Body Preprocessing (placeholder for actual implementation)
    
    # URL Preprocessing (placeholder for actual implementation)

    return dataset


# ====== Sender Extraction and Preprocessing ======
def sender_encoding(df):
    ''' 
    Encodes the domains of the senders using a hybrid of one-hot encoding and frequency encoding.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame containing a 'sender' column with email addresses.

    Returns:
    - df (pd.DataFrame): DataFrame with new columns 'sender_email' and 'sender_domain'.
    
    Raises:
    - ValueError: If the 'sender' column is not found in the DataFrame.
    '''
    
    # Get the sender email (the main part of the feature)
    df['sender_email'] = df['sender'].str.extract(r'<(.*?)>')[0].fillna(df['sender']) # Extract the email (whether inside < > or as plain email)

    # Extract the domain from the sender_email
    df['sender_domain'] = df['sender_email'].str.extract(r'@([a-zA-Z0-9.-]+)')[0]
    df['sender'] = df['sender_domain'].str.lower()

    #Remove extra columns so only 'sender' column exists
    df.drop(columns=['sender_email','sender_domain'], inplace=True)

     

    # Top 100 Encoding (placeholder for actual implementation)
    
    return df