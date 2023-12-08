import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langdetect import detect_langs, lang_detect_exception
from langid.langid import LanguageIdentifier, model
from collections import Counter
import re
import matplotlib.pyplot as plt
import spacy

class SentimentAnalysis:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.custom_labels = {'positive': '😊', 'neutral': '😐', 'negative': '😢'}
        self.nlp = spacy.load("en_core_web_sm")
        self.identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

    def analyze_sentiment(self, text, sentiment_threshold=0.1, language_threshold=0.8):
        # Language detection
        language = self.detect_language(text, threshold=language_threshold)
        if language == 'unknown':
            return {'text': text, 'error': 'Language not supported for sentiment analysis'}

        # Text preprocessing
        processed_text = self.preprocess_text(text, language)

        # Get sentiment scores
        sentiment_scores = self.analyzer.polarity_scores(processed_text)

        # Determine sentiment category based on custom threshold
        sentiment = self.get_sentiment_category(sentiment_scores['compound'], threshold=sentiment_threshold)

        # Extract top keywords
        keywords = self.extract_top_keywords(processed_text)

        # Named Entity Recognition (NER)
        entities = self.extract_named_entities(processed_text, language)

        return {
            'text': text,
            'processed_text': processed_text,
            'sentiment_scores': sentiment_scores,
            'sentiment': sentiment,
            'custom_label': self.get_custom_label(sentiment),
            'language': language,
            'top_keywords': keywords,
            'named_entities': entities
        }

    def analyze_batch_sentiment(self, texts, sentiment_threshold=0.1, language_threshold=0.8):
        results = [self.analyze_sentiment(text, sentiment_threshold, language_threshold) for text in texts]
        return results

    @staticmethod
    def detect_language(text, threshold=0.8):
        try:
            # Use langdetect for languages with confidence above the threshold
            langs = detect_langs(text)
            if langs[0].lang == 'unknown' or langs[0].prob < threshold:
                return 'unknown'
            return langs[0].lang
        except lang_detect_exception.LangDetectException:
            # Use langid as a fallback if langdetect fails
            lang, confidence = model.predict(text)
            return lang if confidence > threshold else 'unknown'

    @staticmethod
    def preprocess_text(text, language, custom_stopwords=None):
        # Convert to lowercase
        text = text.lower()

        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove custom stopwords
        if custom_stopwords:
            text = ' '.join([word for word in text.split() if word not in custom_stopwords])

        # Lemmatization using spaCy
        if language == 'en':
            doc = nlp(text)
            text = ' '.join([token.lemma_ for token in doc])

        return text

    @staticmethod
    def get_sentiment_category(compound_score, threshold=0.1):
        if compound_score >= threshold:
            return 'positive'
        elif compound_score <= -threshold:
            return 'negative'
        else:
            return 'neutral'

    def get_custom_label(self, sentiment):
        return self.custom_labels.get(sentiment, 'Unknown')

    @staticmethod
    def extract_top_keywords(text, num_keywords=3):
        words = re.findall(r'\b\w+\b', text.lower())
        word_counts = Counter(words)
        top_keywords = word_counts.most_common(num_keywords)
        return top_keywords

    def extract_named_entities(self, text, language):
        # Named Entity Recognition (NER) using spaCy
        if language == 'en':
            doc = self.nlp(text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            return entities
        else:
            return []

    def visualize_sentiment(self, sentiment_scores):
        labels = list(sentiment_scores.keys())
        values = list(sentiment_scores.values())

        plt.bar(labels, values, color=['green', 'grey', 'red'])
        plt.title('Sentiment Scores')
        plt.xlabel('Sentiment Category')
        plt.ylabel('Score')
        plt.show()

    def save_model(self, filename='sentiment_model.pkl'):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_model(filename='sentiment_model.pkl'):
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        return model

# Interactive User Interface
def interactive_ui():
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
