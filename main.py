import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langdetect import detect
from collections import Counter
import re
import matplotlib.pyplot as plt

class SentimentAnalysis:
    def __init__(self):
        """
        Initialize the SentimentAnalysis class.

        Attributes:
        - analyzer: Instance of SentimentIntensityAnalyzer from vaderSentiment library.
        - custom_labels: Dictionary mapping sentiment categories to custom labels.
        """
        self.analyzer = SentimentIntensityAnalyzer()
        self.custom_labels = {'positive': '😊', 'neutral': '😐', 'negative': '😢'}

    def analyze_sentiment(self, text):
        """
        Analyze the sentiment of the given text.

        Args:
        - text (str): The input text to analyze.

        Returns:
        A dictionary containing sentiment analysis results, including sentiment scores, category,
        custom label, language, top keywords, and the original and processed text.
        """
        # Language detection
        language = self.detect_language(text)
        if language != 'en':
            return {'text': text, 'error': 'Language not supported for sentiment analysis'}

        # Text preprocessing
        processed_text = self.preprocess_text(text)

        # Get sentiment scores
        sentiment_scores = self.analyzer.polarity_scores(processed_text)

        # Determine sentiment category based on compound score
        sentiment = self.get_sentiment_category(sentiment_scores['compound'])

        # Extract top keywords
        keywords = self.extract_top_keywords(processed_text)

        return {
            'text': text,
            'processed_text': processed_text,
            'sentiment_scores': sentiment_scores,
            'sentiment': sentiment,
            'custom_label': self.get_custom_label(sentiment),
            'language': language,
            'top_keywords': keywords
        }

    def analyze_batch_sentiment(self, texts):
        """
        Analyze the sentiment of a batch of texts.

        Args:
        - texts (list): List of input texts to analyze.

        Returns:
        A list of dictionaries, each containing sentiment analysis results for a specific text.
        """
        results = [self.analyze_sentiment(text) for text in texts]
        return results

    @staticmethod
    def detect_language(text):
        """
        Detect the language of the input text.

        Args:
        - text (str): The input text to detect language.

        Returns:
        A string representing the detected language.
        """
        try:
            return detect(text)
        except Exception as e:
            return 'unknown'

    @staticmethod
    def preprocess_text(text):
        """
        Perform basic text preprocessing.

        Args:
        - text (str): The input text to preprocess.

        Returns:
        The preprocessed text after converting to lowercase and removing special characters and numbers.
        """
        # Convert to lowercase
        text = text.lower()

        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        return text

    @staticmethod
    def get_sentiment_category(compound_score):
        """
        Determine the sentiment category based on the compound score.

        Args:
        - compound_score (float): The compound score from sentiment analysis.

        Returns:
        A string representing the sentiment category ('positive', 'negative', or 'neutral').
        """
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'

    def get_custom_label(self, sentiment):
        """
        Get the custom label associated with a sentiment category.

        Args:
        - sentiment (str): The sentiment category ('positive', 'negative', or 'neutral').

        Returns:
        The custom label associated with the sentiment category.
        """
        return self.custom_labels.get(sentiment, 'Unknown')

    @staticmethod
    def extract_top_keywords(text, num_keywords=3):
        """
        Extract the top keywords from the given text.

        Args:
        - text (str): The input text to extract keywords from.
        - num_keywords (int): The number of top keywords to extract (default is 3).

        Returns:
        A list of tuples containing the top keywords and their frequencies.
        """
        # Tokenize the text (simple word-based tokenizer)
        words = re.findall(r'\b\w+\b', text.lower())

        # Count word occurrences
        word_counts = Counter(words)

        # Get the top keywords
        top_keywords = word_counts.most_common(num_keywords)

        return top_keywords

    def visualize_sentiment(self, sentiment_scores):
        """
        Visualize sentiment scores using a bar chart.

        Args:
        - sentiment_scores (dict): Dictionary containing sentiment scores.

        Returns:
        Displays a bar chart showing the sentiment scores.
        """
        labels = list(sentiment_scores.keys())
        values = list(sentiment_scores.values())

        plt.bar(labels, values, color=['green', 'grey', 'red'])
        plt.title('Sentiment Scores')
        plt.xlabel('Sentiment Category')
        plt.ylabel('Score')
        plt.show()

    def save_model(self, filename='sentiment_model.pkl'):
        """
        Save the SentimentAnalysis model to a file.

        Args:
        - filename (str): The filename to save the model to (default is 'sentiment_model.pkl').

        Returns:
        Saves the model to the specified file.
        """
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_model(filename='sentiment_model.pkl'):
        """
        Load a SentimentAnalysis model from a file.

        Args:
        - filename (str): The filename to load the model from (default is 'sentiment_model.pkl').

        Returns:
        An instance of the loaded SentimentAnalysis model.
        """
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        return model


# Interactive User Interface
def interactive_ui():
    """
    Run an interactive command-line interface for users to input text and get sentiment analysis results.
    """
    sa = SentimentAnalysis()

    while True:
        user_input = input("Enter text (type 'exit' to quit): ")
        
        if user_input.lower() == 'exit':
            break

        result = sa.analyze_sentiment(user_input)
        print("Sentiment Analysis Result:")
        print(result)
        print("\n")

        # Visualize sentiment scores
        sa.visualize_sentiment(result['sentiment_scores'])


# Example usage
if __name__ == "__main__":
    interactive_ui()
