from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import pandas as pd


class AdvancedFeatureExtractor:
    def __init__(self, max_features=1000):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2
        )

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def extract_features(self, texts):
        """Extract TF-IDF features from texts"""
        self.logger.info(f"Extracting TF-IDF features (max_features={self.max_features})...")

        # Transform texts to TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(texts)

        # Convert to DataFrame for easier manipulation
        return pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=self.vectorizer.get_feature_names_out()
        )