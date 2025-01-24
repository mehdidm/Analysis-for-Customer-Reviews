from sklearn.feature_extraction.text import TfidfVectorizer


class AdvancedFeatureExtractor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            min_df=5,
            max_df=0.85,
            ngram_range=(1, 2),
            stop_words='english'
        )

    def fit(self, texts):
        self.vectorizer.fit(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()
