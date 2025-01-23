import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self, max_features=1000):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2
        )

    def extract_features(self, texts):
        """Extract TF-IDF features from texts"""
        logger.info(f"Extracting TF-IDF features (max_features={self.max_features})...")
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        return pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=self.vectorizer.get_feature_names_out()
        )
