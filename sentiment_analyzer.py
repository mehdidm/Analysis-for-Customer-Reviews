from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


class SentimentAnalyzer:
    def __init__(self):
        self.model = LogisticRegression(class_weight='balanced', max_iter=1000)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        return classification_report(y_test, y_pred, output_dict=True)