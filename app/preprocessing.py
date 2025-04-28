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

import joblib  # For saving and loading models
import os

PHISHING_KEYWORDS = {"urgent", "verify", "click", "account", "login", "security", "alert", "confirm"}

def preprocess(dataset, vectorizer=None):
    """
    Master pipeline to preprocess dataset:
    - Extracts sender features
    - Extracts body features (requires optional TF-IDF vectorizer)
    - Extracts subject phishing score
    - Extracts URL features
    - Returns model-ready feature DataFrame

    Parameters
    ----------
    dataset : pd.DataFrame
        Raw email dataset.
    
    vectorizer : TfidfVectorizer or None
        If None, fits a new vectorizer (training mode).
        If provided, uses the existing vectorizer (inference mode).

    Returns
    -------
    pd.DataFrame
        Combined feature set.
    """
    # ====== Input Validation ======
    if not isinstance(dataset, pd.DataFrame):
        raise ValueError("Input dataset must be a pandas DataFrame.")
    if dataset.empty:
        raise ValueError("Dataset is empty.")

    required_columns = ['sender', 'body', 'subject']
    missing_columns = [col for col in required_columns if col not in dataset.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {', '.join(missing_columns)}")

    # ====== Feature Extraction ======
    sender_features_df = sender_extraction(dataset)
    body_features_df = body_extraction(dataset, vectorizer=vectorizer)
    subject_features_df = subject_extraction(dataset)

    # Combine subject + body text for URL extraction
    combined_text = dataset['subject'].fillna('') + ' ' + dataset['body'].fillna('')
    url_extractor = URLFeatureExtractor()
    url_features_df = url_extractor.transform(combined_text)

    # ====== Combine Features ======
    final_features_df = pd.concat([
        sender_features_df.reset_index(drop=True),
        body_features_df.reset_index(drop=True),
        subject_features_df.reset_index(drop=True),
        url_features_df.reset_index(drop=True)
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
def body_extraction(df, vectorizer=None):
    """
    Extracts TF-IDF features from the 'body' column of emails.
    If vectorizer is provided, applies only transform (for inference).
    If vectorizer is None, fits a new TF-IDF vectorizer (for training).

    Steps:
    - Cleans HTML, URLs, and special characters
    - Tokenizes while boosting phishing keywords
    - Applies TF-IDF vectorization using unigrams and bigrams

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a 'body' column.
    
    vectorizer : TfidfVectorizer or None
        If None, a new vectorizer will be created and fitted (for training).
        If provided, this vectorizer will be used to transform (for inference).

    Returns
    -------
    tuple:
        pd.DataFrame
            TF-IDF feature DataFrame.
        vectorizer
            The fitted or provided TF-IDF vectorizer (useful for saving later).
    
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
        text = text.lower()                                   # Lowercase
        text = re.sub(r'[^a-z\s]', '', text)                  # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()              # Normalize whitespace
        return text

    # Clean the body column
    df['clean_body'] = df['body'].apply(clean)

    # Training: Fit new vectorizer
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=15000,
            min_df=5,
            max_df=0.8,
            ngram_range=(1, 2),
            tokenizer=phishing_tokenizer
        )
        tfidf_matrix = vectorizer.fit_transform(df['clean_body'])

        # Save the vectorizer 
        os.makedirs('models', exist_ok=True)
        joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
        print("[INFO] TF-IDF vectorizer saved to 'models/tfidf_vectorizer.pkl'")
    else:
        # Inference: Use provided vectorizer
        tfidf_matrix = vectorizer.transform(df['clean_body'])

    # Convert to DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    return tfidf_df

def phishing_tokenizer(text):
    """
    Custom tokenizer that repeats phishing keywords to boost TF-IDF weight.
    """
    tokens = text.split()
    return tokens + [kw for kw in tokens if kw in PHISHING_KEYWORDS] * 2

# ====== Subject Feature Extraction ======
def subject_extraction(df):
    """
    Extracts phishing-related features from the subject line of emails and computes a subject_phish_score.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'subject' column.

    Returns
    -------
    pd.DataFrame
        DataFrame containing a single 'subject_phish_score' column.

    Raises
    ------
    ValueError
        If 'subject' column is missing from the input DataFrame.
    """
    if 'subject' not in df.columns:
        raise ValueError("The DataFrame must contain a 'subject' column.")

    # Define urgency keywords
    urgency_keywords = ['urgent', 'immediate', 'important', 'verify', 'action required', 'attention']

    # Define punctuation weighting
    heavy_punctuations = {
        '!': 2,
        '?': 1.5,
        '.': 0.5
    }

    def extract_subject_features(subject):
        if pd.isnull(subject):
            return 0.0  # Null subjects treated as no phishing signals

        # Clean text
        subject = subject.lower().strip()

        # Calculate urgency score
        urgency_score = sum(subject.count(word) for word in urgency_keywords)

        # Calculate punctuation score
        punctuation_score = sum(subject.count(punct) * weight for punct, weight in heavy_punctuations.items())

        # Detect fake thread
        fake_thread_flag = 1 if subject.startswith('re:') or subject.startswith('fwd:') else 0

        # Aggregate raw score
        raw_score = urgency_score * 2 + punctuation_score + fake_thread_flag * 1.5

        return raw_score

    # Apply feature extraction
    df['subject_raw_score'] = df['subject'].apply(extract_subject_features)

    # Normalize to 0–1 range
    scaler = MinMaxScaler()
    df['subject_phish_score'] = scaler.fit_transform(df[['subject_raw_score']])

    # Only return the final normalized score
    return df[['subject_phish_score']]

# ====== URL Feature Extraction ======
import re
import pandas as pd
import tldextract
from sklearn.base import BaseEstimator, TransformerMixin

class URLFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer to extract URL-related features from email text.
    Produces both detailed risk counts and a binary suspicious flag.
    """

    def __init__(self,
                 suspicious_tlds=None,
                 suspicious_keywords=None):
        """
        Initializes the extractor with optional custom suspicious TLDs and keywords.
        
        Parameters
        ----------
        suspicious_tlds : set of str, optional
            Top-Level Domains often associated with phishing (e.g., '.ru', '.cn').
        suspicious_keywords : set of str, optional
            Substrings in URLs that might indicate phishing intent (e.g., 'login', 'verify').
        """
        if suspicious_tlds is None:
            suspicious_tlds = {'ru', 'cn', 'xyz', 'tk', 'pw', 'top'}
        if suspicious_keywords is None:
            suspicious_keywords = {'login', 'verify', 'update', 'secure', 'account'}

        self.suspicious_tlds = suspicious_tlds
        self.suspicious_keywords = suspicious_keywords

    def fit(self, X, y=None):
        """
        Required by sklearn API. No fitting necessary for this stateless transformer.
        """
        return self

    def transform(self, X):
        """
        Transforms a Series or list of email texts into URL risk features.

        Parameters
        ----------
        X : list or pandas.Series
            Raw email body/subject text.

        Returns
        -------
        pd.DataFrame
            DataFrame with the following columns:
            - num_urls
            - num_ip_urls
            - num_suspicious_tld
            - num_suspicious_keyword
            - num_suspicious_urls
            - suspicious_url_flag
        """
        features = []

        for text in X:
            urls = self._extract_urls(str(text))  # Extract all URLs from email text

            # Initialize feature counts
            num_urls = len(urls)
            num_ip_urls = 0
            num_suspicious_tld = 0
            num_suspicious_keyword = 0

            # Analyze each URL
            for url in urls:
                #Check for IP address urls, TLD, and keywords
                if self._is_ip_address(url):
                    num_ip_urls += 1
                if self._has_suspicious_tld(url):
                    num_suspicious_tld += 1
                if self._has_suspicious_keyword(url):
                    num_suspicious_keyword += 1

            # Calculate total number of suspicious triggers
            num_suspicious_urls = num_ip_urls + num_suspicious_tld + num_suspicious_keyword

            # Create a binary suspicious flag
            suspicious_url_flag = 1 if num_suspicious_urls > 0 else 0

            features.append([
                num_urls,
                num_ip_urls,
                num_suspicious_tld,
                num_suspicious_keyword,
                num_suspicious_urls,
                suspicious_url_flag
            ])

        return pd.DataFrame(features, columns=[
            'num_urls',
            'num_ip_urls',
            'num_suspicious_tld',
            'num_suspicious_keyword',
            'num_suspicious_urls',
            'suspicious_url_flag'
        ])

    def _extract_urls(self, text):
        """
        Extracts all URLs from a text using regex matching 'http://' or 'https://'.

        Parameters
        ----------
        text : str
            Raw email text.

        Returns
        -------
        list
            List of extracted URL strings.
        """
        return re.findall(r'(https?://[^\s]+)', text)

    def _is_ip_address(self, url):
        """
        Determines if a URL uses an IP address rather than a domain name.

        Parameters
        ----------
        url : str
            A URL string.

        Returns
        -------
        bool
            True if URL domain is an IP address, False otherwise.
        """
        domain = self._get_domain(url)
        return bool(re.match(r'^(\d{1,3}\.){3}\d{1,3}$', domain))

    def _has_suspicious_tld(self, url):
        """
        Checks if the URL's top-level domain is considered suspicious.

        Parameters
        ----------
        url : str
            A URL string.

        Returns
        -------
        bool
            True if TLD is suspicious, False otherwise.
        """
        ext = tldextract.extract(url)
        return ext.suffix.lower() in self.suspicious_tlds

    def _has_suspicious_keyword(self, url):
        """
        Checks if the URL contains any known suspicious keywords.

        Parameters
        ----------
        url : str
            A URL string.

        Returns
        -------
        bool
            True if suspicious keyword found, False otherwise.
        """
        url_lower = url.lower()
        return any(keyword in url_lower for keyword in self.suspicious_keywords)

    def _get_domain(self, url):
        """
        Extracts and reconstructs the domain portion of a URL.

        Parameters
        ----------
        url : str
            A URL string.

        Returns
        -------
        str
            Registered domain + suffix (e.g., 'example.com').
        """
        ext = tldextract.extract(url)
        return f"{ext.domain}.{ext.suffix}"
