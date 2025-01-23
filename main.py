import logging
import os
import pandas as pd
from text_preprocessor import TextPreprocessor
from feature_extractor import FeatureExtractor
from sentiment_model import SentimentModel
from data_loader import DataLoader
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SentimentAnalysisPipeline:
    def __init__(self, data_path, max_features=1000):
        self.data_loader = DataLoader(data_path)
        self.text_preprocessor = TextPreprocessor()
        self.feature_extractor = FeatureExtractor(max_features)
        self.sentiment_model = SentimentModel()

    def load_and_preprocess_data(self):
        # Load data
        logger.info("Loading data...")
        df = self.data_loader.load_data()

        # Display initial dataset details
        print("\n--- Dataset Initial Information ---")
        print(df.info())
        print("\nFirst 5 rows:\n", df.head())

        # Preprocess text
        logger.info("Preprocessing text data...")
        df['processed_review'] = df['reviewText'].apply(self.text_preprocessor.preprocess)
        df['processed_summary'] = df['summary'].apply(self.text_preprocessor.preprocess)

        # Display preprocessed text samples
        print("\n--- Preprocessed Text Samples ---")
        comparison_df = pd.DataFrame({
            'Original Review': df['reviewText'].head(),
            'Preprocessed Review': df['processed_review'].head()
        })
        print(comparison_df)

        # Extract features
        logger.info("Extracting features...")
        tfidf_features = self.feature_extractor.extract_features(df['processed_review'])

        # Optional: Display feature matrix
        print("\n--- Feature Matrix Information ---")
        print(f"Feature matrix shape: {tfidf_features.shape}")
        print("First few feature column names:", list(tfidf_features.columns[:10]))

        return df, tfidf_features

    def save_preprocessed_data(self, df):
        """Save preprocessed data with clear visibility"""
        output_dir = './sample_data'
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, 'preprocessed_amazon_reviews.csv')
        df.to_csv(output_file, index=False)

        print(f"\n--- Preprocessed Data Saved ---")
        print(f"File: {output_file}")
        print(f"Total rows saved: {len(df)}")
        print("\nSample of saved data:")
        print(df.head())


def main():
    data_path = './sample_data/amazon_reviews.csv'
    pipeline = SentimentAnalysisPipeline(data_path)

    # Load and preprocess data
    df, tfidf_features = pipeline.load_and_preprocess_data()

    # Save preprocessed data
    pipeline.save_preprocessed_data(df)


if __name__ == "__main__":
    main()