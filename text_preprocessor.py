import re
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure necessary NLTK resources are downloaded
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

class TextPreprocessor:
    def __init__(self):
        logger.info("Initializing text preprocessor...")
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        """Remove special characters and normalize text"""
        if pd.isna(text):
            return ""

        text = str(text).lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        # Remove special characters
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def remove_stopwords(self, text):
        """Remove stop words from text"""
        words = word_tokenize(text)
        words = [w for w in words if w not in self.stop_words]
        return ' '.join(words)

    def lemmatize_text(self, text):
        """Lemmatize text"""
        words = word_tokenize(text)
        words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)

    def preprocess(self, text):
        """Apply all preprocessing steps"""
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        text = self.lemmatize_text(text)
        return text
