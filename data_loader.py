import pandas as pd
import logging
import spacy


class AdvancedDataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.nlp = spacy.load('en_core_web_sm')
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        """Load and preprocess Amazon reviews dataset"""
        self.logger.info("Loading Amazon reviews dataset...")
        self.df = pd.read_csv(self.filepath)

        # Additional feature engineering
        self.df['text_length'] = self.df['reviewText'].str.len()
        self.df['word_count'] = self.df['reviewText'].apply(lambda x: len(str(x).split()))

        return self.df

    def extract_named_entities(self, text):
        """Extract named entities using spaCy"""
        doc = self.nlp(text)
        return {
            'PRODUCT': [ent.text for ent in doc.ents if ent.label_ == 'PRODUCT'],
            'ORG': [ent.text for ent in doc.ents if ent.label_ == 'ORG'],
            'PERSON': [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
        }