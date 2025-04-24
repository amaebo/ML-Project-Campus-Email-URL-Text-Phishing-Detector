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

PHISHING_KEYWORDS = {"urgent", "verify", "click", "account", "login", "security", "alert", "confirm"}

def preprocess(dataset):
    """
    Master pipeline to preprocess dataset:
    - Extracts sender domain features
    - Extracts phishing-aware TF-IDF body features
    - Returns model-ready DataFrame

    Parameters
    ----------
    dataset : pd.DataFrame
        Full email dataset with at least 'sender' and 'body' columns.

    Returns
    -------
    pd.DataFrame
        Combined feature matrix of encoded sender domains and body TF-IDF features.
    
    Raises
    ------
    ValueError
        If the input is not a DataFrame or required columns are missing.
    """
    # Validate input
    if not isinstance(dataset, pd.DataFrame):
        raise ValueError("Input dataset must be a pandas DataFrame.")
    if dataset.empty:
        raise ValueError("Dataset is empty.")
    
    # Check for required columns
    required_columns = ['sender', 'body']
    missing_columns = [col for col in required_columns if col not in dataset.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {', '.join(missing_columns)}")

    # Extract Features: sender,body, subject, urls. --- NOT DONE YET
    sender_features_df = sender_extraction(dataset)
    body_features_df = body_extraction(dataset)

    # Merge horizontally --- NOT DONE YET
    final_features_df = pd.concat([
        sender_features_df.reset_index(drop=True),
        body_features_df.reset_index(drop=True)
    ], axis=1)

    return final_features_df


# ====== Sender Extraction and Preprocessing ======
def sender_extraction(df, top_k=100):
    """
    Extracts sender features using:
    - One-hot encoding for top-K domains
    - A binary 'domain_other' flag for rare domains
    - Log-frequency encoding for all domains

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'sender' column with email addresses.
    
    top_k : int, optional
        Number of most frequent domains to one-hot encode. Default is 100.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with three types of features:
        - One-hot columns for top domains
        - Binary 'domain_other' flag
        - Continuous 'domain_frequency' column (log-scaled)
    
    Raises
    ------
    ValueError
        If 'sender' column is missing.
    """
    if 'sender' not in df.columns:
        raise ValueError("The DataFrame must contain a 'sender' column.")

    # Step 1: Extract sender domain
    df['sender_email'] = df['sender'].str.extract(r'<(.*?)>')[0].fillna(df['sender'])
    df['sender_domain'] = df['sender_email'].str.extract(r'@([a-zA-Z0-9.-]+)')[0].str.lower()

    # Step 2: Calculate frequency (and log-frequency)
    domain_counts = df['sender_domain'].value_counts()
    df['domain_frequency'] = df['sender_domain'].map(domain_counts)
    df['domain_frequency'] = np.log1p(df['domain_frequency'])  # log(1 + count)

    # Step 3: One-hot encode top-K domains
    top_domains = domain_counts.nlargest(top_k).index.tolist()
    df['domain_other'] = (~df['sender_domain'].isin(top_domains)).astype(int)

    encoder = OneHotEncoder(categories=[top_domains], sparse_output=False, handle_unknown='ignore')
    domain_ohe = encoder.fit_transform(df[['sender_domain']])
    domain_ohe_df = pd.DataFrame(domain_ohe, columns=encoder.get_feature_names_out(['sender_domain']))

    # Step 4: Build final feature set
    sender_features_df = pd.concat([
        domain_ohe_df.reset_index(drop=True),
        df[['domain_other', 'domain_frequency']].reset_index(drop=True)
    ], axis=1)

    return sender_features_df


# ====== Body Cleaning and Extraction (TF-IDF) ======
def body_extraction(df):
    """
    Extracts TF-IDF features from the 'body' column of emails.

    Steps:
    - Cleans HTML, URLs, and special characters
    - Tokenizes while boosting phishing keywords
    - Applies TF-IDF vectorization using unigrams and bigrams

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a 'body' column.

    Returns
    -------
    pd.DataFrame
        TF-IDF feature DataFrame.
    
    Raises
    ------
    ValueError
        If 'body' column is missing from the input DataFrame.
    """
    if 'body' not in df.columns:
        raise ValueError("The DataFrame must contain a 'body' column.")

    # Inner function to clean text
    def clean(text):
        if pd.isnull(text):
            return ""
        text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML
        text = re.sub(r'\S+@\S+', 'emailaddr', text)          # Mask emails
        text = re.sub(r'http\S+|www\S+|https\S+', 'urladdr', text)  # Mask URLs
        text = text.lower()                                   # Lowercase for uniformity
        text = re.sub(r'[^a-z\s]', '', text)                  # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()              # Normalize whitespace
        return text

    # Tokenizer that repeats phishing keywords to increase their TF-IDF weight
    def phishing_tokenizer(text):
        tokens = text.split()
        return tokens + [kw for kw in tokens if kw in PHISHING_KEYWORDS] * 2

    # Clean the body column
    df['clean_body'] = df['body'].apply(clean)

    # TF-IDF vectorizer setup
    vectorizer = TfidfVectorizer(
        max_features=15000,
        min_df=5,
        max_df=0.8,
        ngram_range=(1, 2),
        tokenizer=phishing_tokenizer
    )

    # Fit and transform text into TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(df['clean_body'])

    # Convert to DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    return tfidf_df


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
