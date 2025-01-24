import logging

import pandas as pd
import spacy


class AdvancedDataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.nlp = spacy.load("en_core_web_sm")
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        """Charge les données et calcule des métriques basiques"""
        try:
            self.df = pd.read_csv(self.filepath)
            self.df['text_length'] = self.df['reviewText'].str.len()
            self.df['word_count'] = self.df['reviewText'].str.split().str.len()
            return self.df
        except Exception as e:
            self.logger.error(f"Erreur de chargement: {str(e)}")
            raise