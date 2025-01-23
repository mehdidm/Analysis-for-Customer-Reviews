from gensim import corpora
from gensim.models import LdaModel
import logging


class ThemeExtractor:
    def __init__(self, num_topics=5):
        self.num_topics = num_topics
        self.dictionary = None
        self.corpus = None
        self.lda_model = None

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def extract_topics(self, documents):
        """Extract key themes using Latent Dirichlet Allocation"""
        self.logger.info(f"Extracting {self.num_topics} topics")

        # Create dictionary
        self.dictionary = corpora.Dictionary(documents)

        # Create corpus
        self.corpus = [self.dictionary.doc2bow(doc) for doc in documents]

        # Train LDA model
        self.lda_model = LdaModel(
            self.corpus,
            num_topics=self.num_topics,
            id2word=self.dictionary
        )

        return self.lda_model.print_topics()