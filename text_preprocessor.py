import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import pipeline
import logging


class AdvancedTextPreprocessor:
    def __init__(self):
        # Download necessary NLTK resources
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.sentiment_analyzer = pipeline('sentiment-analysis')

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def clean_text(self, text):
        """Advanced text cleaning"""
        if pd.isna(text):
            return ""

        # Convert to lowercase
        text = str(text).lower()

        # Remove URLs and special characters
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        return text.strip()

    def remove_stopwords(self, text):
        """Remove stopwords"""
        words = word_tokenize(text)
        return ' '.join([w for w in words if w not in self.stop_words])

    def lemmatize_text(self, text):
        """Lemmatize text"""
        words = word_tokenize(text)
        return ' '.join([self.lemmatizer.lemmatize(word) for word in words])

    def preprocess(self, text):
        # Check if text is NaN or not a string
        if pd.isna(text) or not isinstance(text, str):
            return ""

        # Truncate text to first 512 tokens
        truncated_text = text[:512]

        # Clean text
        cleaned_text = self.clean_text(truncated_text)

        # Remove stopwords
        text_without_stopwords = self.remove_stopwords(cleaned_text)

        # Lemmatize
        lemmatized_text = self.lemmatize_text(text_without_stopwords)

        # Sentiment analysis on truncated text
        try:
            sentiment_result = self.sentiment_analyzer(truncated_text)[0]
            # Append sentiment information
            processed_text = f"{lemmatized_text} [SENTIMENT:{sentiment_result['label']}:{sentiment_result['score']}]"
        except Exception as e:
            # Fallback if sentiment analysis fails
            processed_text = lemmatized_text
            print(f"Sentiment analysis failed: {e}")

        return processed_text