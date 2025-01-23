import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class SentimentModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def train_test_split(self, X, y, test_size=0.2, random_state=42):
        """Split the data into training and testing sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    def train(self):
        """Train the sentiment model"""
        if self.X_train is not None and self.y_train is not None:
            self.model.fit(self.X_train, self.y_train)
            print("Model training completed.")
        else:
            print("Error: Train-test split not performed. Please run train_test_split first.")

    def evaluate(self):
        """Evaluate the model on the test set"""
        if self.X_test is not None and self.y_test is not None:
            y_pred = self.model.predict(self.X_test)

            # Classification report
            print("Classification Report:")
            print(classification_report(self.y_test, y_pred))

            # Confusion matrix
            self.plot_confusion_matrix(y_pred)
        else:
            print("Error: Train-test split not performed. Please run train_test_split first.")

    def plot_confusion_matrix(self, y_pred):
        """Plot confusion matrix for model evaluation"""
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Neutral', 'Positive'],
                    yticklabels=['Negative', 'Neutral', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

    def visualize_top_features(self, vectorizer):
        """Visualize the top features (important words for sentiment prediction)"""
        if hasattr(self.model, 'coef_'):
            coef = self.model.coef_.toarray().flatten()
            top_indices = np.argsort(coef)[-10:]  # Top 10 most important words
            top_features = np.array(vectorizer.get_feature_names_out())[top_indices]

            plt.barh(top_features, coef[top_indices])
            plt.title('Top 10 Most Important Features')
            plt.xlabel('Coefficient Value')
            plt.show()
        else:
            print("Error: Model not trained yet. Please train the model first.")

