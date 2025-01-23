import streamlit as st
import nltk
import matplotlib.pyplot as plt

from data_loader import AdvancedDataLoader
from text_preprocessor import AdvancedTextPreprocessor
from feature_extractor import AdvancedFeatureExtractor
from theme_extractor import ThemeExtractor
from sentiment_analyzer import SentimentAnalyzer

class SentimentAnalysisPipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data_loader = AdvancedDataLoader(data_path)
        self.text_preprocessor = AdvancedTextPreprocessor()
        self.feature_extractor = AdvancedFeatureExtractor()
        self.theme_extractor = ThemeExtractor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.df = None
        self.features = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.topics = None

    def run_analysis(self):
        # Load data
        self.df = self.data_loader.load_data()

        # Preprocess text
        self.df['processed_review'] = self.df['reviewText'].apply(self.text_preprocessor.preprocess)

        # Extract features
        self.features = self.feature_extractor.extract_features(self.df['processed_review'])

        # Prepare data for sentiment analysis
        X = self.features
        y = self.df['overall'].apply(lambda x: 'positive' if x > 3 else 'negative')

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = self.sentiment_analyzer.prepare_data(X, y)

        # Train and evaluate model
        self.sentiment_analyzer.train_model(self.X_train, self.y_train)
        model_metrics = self.sentiment_analyzer.evaluate_model(self.X_test, self.y_test)

        # Extract themes
        processed_texts = [' '.join(nltk.word_tokenize(text.lower())) for text in self.df['processed_review']]
        self.topics = self.extract_and_parse_topics(processed_texts)

        return model_metrics

    def extract_and_parse_topics(self, processed_texts):
        # Ensure texts are properly tokenized
        if not processed_texts:
            return []

        # Tokenize texts if they aren't already tokenized
        tokenized_texts = []
        for text in processed_texts:
            if isinstance(text, str):
                # Tokenize string texts
                tokens = nltk.word_tokenize(text.lower())
            elif isinstance(text, list):
                # If already a list of tokens, use as-is
                tokens = text
            else:
                # Skip invalid text types
                continue

            tokenized_texts.append(tokens)

        try:
            topics = self.theme_extractor.extract_topics(tokenized_texts)

            # Debug print
            print("Raw topics:", topics)

            # Handle different topic formats
            if not topics:
                return []

            # If topics are a list of strings
            if isinstance(topics, list) and all(isinstance(t, str) for t in topics):
                return topics[:5]

            # If topics are a list of tuples
            if isinstance(topics[0], tuple):
                return [topic for topic, _ in topics[:5]]

            # If topics is a single string
            if isinstance(topics, str):
                import re
                unique_topics = set(re.findall(r'\b\w+\b', topics))
                return list(unique_topics)[:5]

            # Fallback
            return []

        except Exception as e:
            print(f"Error extracting topics: {e}")
            return []
def create_dashboard():
    # Streamlit dashboard configuration
    st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")
    st.title("Sentiment Analysis Dashboard")

    # Sidebar for data input
    st.sidebar.header("Data Input")
    data_path = st.sidebar.text_input("Data Path", value="./sample_data/amazon_reviews.csv")

    # Analysis button
    if st.sidebar.button("Run Analysis"):
        try:
            # Initialize and run pipeline
            pipeline = SentimentAnalysisPipeline(data_path)
            model_metrics = pipeline.run_analysis()

            # Dashboard sections
            col1, col2 = st.columns(2)

            with col1:
                # Sentiment Distribution
                st.subheader("Sentiment Distribution")
                fig, ax = plt.subplots()
                pipeline.sentiment_analyzer.visualize_sentiment_distribution(pipeline.df, ax)
                st.pyplot(fig)

            with col2:
                st.subheader("Top Topics")
                if pipeline.topics:
                    print("Topics to display:", pipeline.topics)
                    for topic in pipeline.topics:
                        st.write(f"- {topic}")
                else:
                    st.write("No topics found")

            # Model Performance
            st.header("Model Performance")
            st.write("Evaluation Metrics:")
            st.write(model_metrics)

        except Exception as e:
            st.error(f"Error in analysis: {e}")

def main():
    create_dashboard()

if __name__ == "__main__":
    main()