import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import pipeline
class AdvancedTextPreprocessor:
    def __init__(self):
        nltk.download(['punkt', 'stopwords', 'wordnet'], quiet=True)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.custom_stopwords = {'nook', 'kindle', 'get', 'one', 'like'}
        self.sentiment_pipeline = pipeline('sentiment-analysis')

    def preprocess(self, text):
        """Pipeline complet de prétraitement"""
        try:
            # Nettoyage de base
            text = re.sub(r'http\S+|www\S+|[^a-zA-Z\s]', '', str(text).lower())

            # Tokenization et filtrage
            tokens = nltk.word_tokenize(text)
            filtered = [
                self.lemmatizer.lemmatize(w)
                for w in tokens
                if w not in self.stop_words and w not in self.custom_stopwords
            ]

            # Analyse de sentiment
            sentiment = self.sentiment_pipeline(text[:512])[0]
            return ' '.join(filtered) + f" [SENTIMENT:{sentiment['label']}:{sentiment['score']:.2f}]"
        except:
            return ""
