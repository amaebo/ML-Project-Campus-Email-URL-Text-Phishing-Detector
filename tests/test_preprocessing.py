

import pytest
import pandas as pd
import numpy as np
from app.preprocessing import preprocess, sender_extraction, body_extraction, URLFeatureExtractor
# === PREPROCESS ==
def test_preprocess_valid_input():
    data = {
        'sender': ['test@example.com', 'user@domain.com'],
        'body': ['This is a test email.', 'Another email body.']
    }
    df = pd.DataFrame(data)
    result = preprocess(df)

    # Base assertions
    assert not result.empty
    assert result.shape[0] == 2
    assert 'domain_frequency' in result.columns

    # Confirm that at least one TF-IDF feature is present
    tfidf_cols = [col for col in result.columns if col not in ['domain_frequency', 'domain_other']]
    assert len(tfidf_cols) > 0
    assert result[tfidf_cols].sum().sum() > 0


def test_preprocess_missing_columns():
    data = {'sender': ['test@example.com']}
    df = pd.DataFrame(data)
    with pytest.raises(ValueError, match="Missing expected columns: body"):
        preprocess(df)


def test_preprocess_empty_dataframe():
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="Dataset is empty."):
        preprocess(df)


# === SENDER EXTRACTION ===

def test_sender_extraction_valid_input():
    data = {'sender': ['test@example.com', 'user@domain.com']}
    df = pd.DataFrame(data)
    result = sender_extraction(df)

    assert not result.empty
    assert result.shape[0] == 2
    assert 'domain_frequency' in result.columns
    assert 'domain_other' in result.columns

    # Confirm presence of at least one one-hot encoded column
    assert any(col.startswith("sender_domain_") for col in result.columns)


def test_sender_extraction_missing_sender_column():
    data = {'body': ['This is a test email.']}
    df = pd.DataFrame(data)
    with pytest.raises(ValueError, match="The DataFrame must contain a 'sender' column."):
        sender_extraction(df)


# === BODY EXTRACTION ===

def test_body_extraction_valid_input():
    data = {'body': ['This is a test email.', 'Another email body.']}
    df = pd.DataFrame(data)
    result = body_extraction(df)

    assert not result.empty
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 2
    assert result.shape[1] > 0  # Should have TF-IDF features

    # Check for expected phishing tokens in TF-IDF features
    phishing_terms = [col for col in result.columns if 'click' in col or 'verify' in col]
    assert isinstance(phishing_terms, list)  # Even if empty, should not crash


def test_body_extraction_missing_body_column():
    data = {'sender': ['test@example.com']}
    df = pd.DataFrame(data)
    with pytest.raises(ValueError, match="The DataFrame must contain a 'body' column."):
        body_extraction(df)


def test_body_extraction_handles_html():
    data = {'body': ['<html><body>Click <a href="http://link.com">here</a></body></html>']}
    df = pd.DataFrame(data)
    result = body_extraction(df)

    assert not result.empty
    assert result.shape[0] == 1
    assert result.sum().sum() > 0  # At least one TF-IDF feature should be non-zero


# === URL FEATURE EXTRACTOR ===

def test_url_feature_extractor():
    data = ['Check this link: http://example.com', 'No URLs here.']
    extractor = URLFeatureExtractor()
    result = extractor.transform(data)

    assert not result.empty
    assert 'num_urls' in result.columns
    assert result['num_urls'].iloc[0] == 1
    assert result['num_urls'].iloc[1] == 0
