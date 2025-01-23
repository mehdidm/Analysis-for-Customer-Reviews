import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

    def load_data(self):
        """Load and prepare Amazon reviews dataset"""
        logger.info("Loading Amazon reviews dataset...")
        self.df = pd.read_csv(self.filepath)
        return self.organize_data()

    def organize_data(self):
        """Organize and structure the loaded data"""
        logger.info("Organizing data structure...")

        # Convert helpful column from string to list
        self.df['helpful'] = self.df['helpful'].apply(eval)

        # Calculate helpfulness metrics
        self.df['helpful_ratio'] = self.df.apply(
            lambda x: x['helpful'][0]/x['helpful'][1]
            if x['helpful'][1] > 0 else 0, axis=1
        )

        # Convert time-related columns
        self.df['reviewTime'] = pd.to_datetime(self.df['reviewTime'])
        self.df['year'] = self.df['reviewTime'].dt.year
        self.df['month'] = self.df['reviewTime'].dt.month

        # Create review length features
        self.df['review_length'] = self.df['reviewText'].str.len()
        self.df['summary_length'] = self.df['summary'].str.len()

        return self.df

    def generate_data_report(self):
        """Generate a comprehensive data report"""
        report = {
            "total_reviews": len(self.df),
            "unique_products": self.df['asin'].nunique(),
            "unique_reviewers": self.df['reviewerID'].nunique(),
            "average_rating": self.df['overall'].mean(),
            "rating_distribution": self.df['overall'].value_counts().to_dict(),
            "time_span": {
                "start": self.df['reviewTime'].min(),
                "end": self.df['reviewTime'].max()
            },
            "average_review_length": self.df['review_length'].mean()
        }
        return report
