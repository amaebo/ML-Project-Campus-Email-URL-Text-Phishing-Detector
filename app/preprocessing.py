# All the necessary functions for preprocessing the data
# Data Handling
import pandas as pd
import numpy as np

# Text Processing
import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import tldextract  # For URL domain parsing
from scipy.sparse import hstack

# Feature Engineering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.base import BaseEstimator, TransformerMixin

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
    
    # Debugging: Check for missing values
    missing_values = dataset.isnull().sum()
    if missing_values.any():
        print("Warning: Missing values found in the following columns:")
        print(missing_values[missing_values > 0])
    total_rows = len(dataset)
    print(f"Null subjects: {28} out of {total_rows} rows ({(28/total_rows)*100:.2f}%)")

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
    print(df['sender'].nunique())


    # Step 1: One-Hot Encode Top Domains
    top_k = 100
    top_senders = df['sender'].value_counts().nlargest(top_k).index.tolist()
    
    encoder = OneHotEncoder(
    categories=[top_senders],  # Explicit categories for consistency
    sparse_output=True,        # Memory efficiency
    handle_unknown='ignore'    # Treat new domains as all zeros
    )
    sender_encoded= encoder.fit_transform(df[['sender']])
    # Step 2: Add 'domain_other' flag
    df['domain_other'] = (~df['sender'].isin(top_senders)).astype(int)
    final_features = hstack([sender_encoded, df[['domain_other']].values])

    
    
    return df


class URLFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    This is a transformer that uses scikit-learn to extract features
    from URLs found in email text (subject + body) which may be suspicious.
    """

    def __init__(self,
                 suspicious_tlds=None,
                 suspicious_keywords=None):
        """
        :param suspicious_tlds: A list of TLDs (e.g., ['ru', 'cn', 'xyz']) that
                                are often associated with phishing or malicious sites.
        :param suspicious_keywords: A list of substrings often associated with
                                   phishing or malicious sites (e.g., ['login', 'verify']).
        """
        # Default sets if not provided
        if suspicious_tlds is None:
            suspicious_tlds = {'ru', 'cn', 'xyz', 'tk', 'pw', 'top'}
        if suspicious_keywords is None:
            suspicious_keywords = {'login', 'verify', 'update', 'secure', 'account'}

        self.suspicious_tlds = suspicious_tlds
        self.suspicious_keywords = suspicious_keywords

    def fit(self, X, y=None):
        # Nothing to fit in this transformer
        return self

    def transform(self, X):
        """
        :param X: A pandas Series or list of raw email text (subject + body).
        :return: A pandas DataFrame with features about URLs.
        """
        features = []
        for text in X:
            urls = self._extract_urls(str(text))

            # Basic counts
            num_urls = len(urls)
            num_suspicious_urls = 0
            num_ip_urls = 0
            num_suspicious_tld = 0
            num_suspicious_keyword = 0

            for url in urls:
                # Check if URL is IP-based
                if self._is_ip_address(url):
                    num_ip_urls += 1

                # Check TLD
                if self._has_suspicious_tld(url):
                    num_suspicious_tld += 1

                # Check for suspicious keywords in the domain/path
                if self._has_suspicious_keyword(url):
                    num_suspicious_keyword += 1

            # if it meets ANY of the criteria above it is suspicious
            num_suspicious_urls = num_ip_urls + num_suspicious_tld + num_suspicious_keyword

            features.append([
                num_urls,
                num_ip_urls,
                num_suspicious_tld,
                num_suspicious_keyword,
                num_suspicious_urls
            ])

        return pd.DataFrame(features, columns=[
            'num_urls',
            'num_ip_urls',
            'num_suspicious_tld',
            'num_suspicious_keyword',
            'num_suspicious_urls'
        ])

    def _extract_urls(self, text):
        """
        Extract all URLs using a regex.
        """
        return re.findall(r'(https?://[^\s]+)', text)

    def _is_ip_address(self, url):
        """
        Checks if the URL uses an IP address rather than a domain name.
        For example: http://192.168.0.1/ or http://8.8.8.8/
        """
        # Extract just the domain part
        domain = self._get_domain(url)
        # A simple check for 1-3 digit groups separated by '.'
        return bool(re.match(r'^(\d{1,3}\.){3}\d{1,3}$', domain))

    def _has_suspicious_tld(self, url):
        """
        Checks if the URL's top-level domain (TLD) is in the suspicious list.
        """
        ext = tldextract.extract(url)
        return ext.suffix.lower() in self.suspicious_tlds

    def _has_suspicious_keyword(self, url):
        """
        Checks if the URL contains any of the suspicious keywords in the domain or path.
        """
        url_lower = url.lower()
        return any(keyword in url_lower for keyword in self.suspicious_keywords)

    def _get_domain(self, url):
        """
        Extracts the domain portion of the URL using tldextract.
        If you only want the registered domain, combine .domain + .suffix.
        """
        ext = tldextract.extract(url)
        # For instance, if ext.domain = 'google' and ext.suffix = 'com',
        # domain would be 'google.com'
        return f"{ext.domain}.{ext.suffix}"
