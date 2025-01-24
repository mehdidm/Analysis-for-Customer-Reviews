from gensim import corpora, models
import logging


class ThemeExtractor:
    def __init__(self, num_topics=5):
        self.num_topics = num_topics
        self.dictionary = None
        self.lda_model = None
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    def extract_topics(self, processed_texts):
        """Extrait des thèmes avec LDA"""
        texts = [text.split() for text in processed_texts]
        self.dictionary = corpora.Dictionary(texts)
        corpus = [self.dictionary.doc2bow(text) for text in texts]

        self.lda_model = models.LdaModel(
            corpus=corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            passes=15,
            alpha='asymmetric',
            random_state=42
        )
        return self.lda_model.print_topics()
