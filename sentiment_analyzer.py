from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import plotly.express as px
import logging
import pandas as pd


class SentimentAnalyzer:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def prepare_data(self, X, y, test_size=0.2):
        """Split data into training and testing sets"""
        return train_test_split(X, y, test_size=test_size, random_state=42)

    def train_model(self, X_train, y_train):
        """Train sentiment classification model"""
        self.model.fit(X_train, y_train)
        self.logger.info("Model training completed")

    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred)
        self.logger.info("Model Evaluation:\n" + report)
        return report

    def visualize_sentiment_distribution(self, df, ax=None):
        import matplotlib.pyplot as plt

        # If no axis is provided, create a new figure
        if ax is None:
            fig, ax = plt.subplots()

        # Count sentiment distribution
        sentiment_counts = df['overall'].apply(lambda x: 'Positive' if x > 3 else 'Negative').value_counts()

        # Create pie chart
        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
        ax.set_title('Sentiment Distribution')

        return fig if ax is None else ax