# PySentiment
Certainly! Below is a textual documentation for the provided code. This document describes the purpose, functionality, and usage of each class, method, and function in the `sentiment_analysis.py` file.

---

# Sentiment Analysis Library Documentation

## Overview

The Sentiment Analysis Library provides a set of functionalities for analyzing the sentiment of textual input. It utilizes the VADER sentiment analysis tool, language detection, text preprocessing, and keyword extraction. Additionally, the library offers an interactive user interface and the ability to save and load the sentiment analysis model.

## Classes

### `SentimentAnalysis`

#### Attributes:

- `analyzer`: Instance of `SentimentIntensityAnalyzer` from the `vaderSentiment` library.
- `custom_labels`: Dictionary mapping sentiment categories to custom labels.

#### Methods:

1. `__init__(self)`: Initialize the `SentimentAnalysis` class.

2. `analyze_sentiment(self, text) -> dict`: Analyze the sentiment of the given text.

    - Args:
        - `text (str)`: The input text to analyze.

    - Returns:
        A dictionary containing sentiment analysis results, including sentiment scores, category, custom label, language, top keywords, and the original and processed text.

3. `analyze_batch_sentiment(self, texts) -> list`: Analyze the sentiment of a batch of texts.

    - Args:
        - `texts (list)`: List of input texts to analyze.

    - Returns:
        A list of dictionaries, each containing sentiment analysis results for a specific text.

4. `detect_language(text) -> str`: Detect the language of the input text.

    - Args:
        - `text (str)`: The input text to detect language.

    - Returns:
        A string representing the detected language.

5. `preprocess_text(text) -> str`: Perform basic text preprocessing.

    - Args:
        - `text (str)`: The input text to preprocess.

    - Returns:
        The preprocessed text after converting to lowercase and removing special characters and numbers.

6. `get_sentiment_category(compound_score) -> str`: Determine the sentiment category based on the compound score.

    - Args:
        - `compound_score (float)`: The compound score from sentiment analysis.

    - Returns:
        A string representing the sentiment category ('positive', 'negative', or 'neutral').

7. `get_custom_label(sentiment) -> str`: Get the custom label associated with a sentiment category.

    - Args:
        - `sentiment (str)`: The sentiment category ('positive', 'negative', or 'neutral').

    - Returns:
        The custom label associated with the sentiment category.

8. `extract_top_keywords(text, num_keywords=3) -> list`: Extract the top keywords from the given text.

    - Args:
        - `text (str)`: The input text to extract keywords from.
        - `num_keywords (int)`: The number of top keywords to extract (default is 3).

    - Returns:
        A list of tuples containing the top keywords and their frequencies.

9. `visualize_sentiment(sentiment_scores) -> None`: Visualize sentiment scores using a bar chart.

    - Args:
        - `sentiment_scores (dict)`: Dictionary containing sentiment scores.

    - Returns:
        Displays a bar chart showing the sentiment scores.

10. `save_model(filename='sentiment_model.pkl') -> None`: Save the `SentimentAnalysis` model to a file.

    - Args:
        - `filename (str)`: The filename to save the model to (default is 'sentiment_model.pkl').

    - Returns:
        Saves the model to the specified file.

11. `load_model(filename='sentiment_model.pkl') -> SentimentAnalysis`: Load a `SentimentAnalysis` model from a file.

    - Args:
        - `filename (str)`: The filename to load the model from (default is 'sentiment_model.pkl').

    - Returns:
        An instance of the loaded `SentimentAnalysis` model.

### `interactive_ui()`

- Function: Run an interactive command-line interface for users to input text and get sentiment analysis results.

- Usage:
  - Execute `interactive_ui()` to start the interactive interface.

## Example Usage

```python
# Create an instance of SentimentAnalysis
sa = SentimentAnalysis()

# Analyze sentiment for a single text
result = sa.analyze_sentiment("I love programming!")

# Analyze sentiment for a batch of texts
batch_results = sa.analyze_batch_sentiment(["I feel great!", "This is not good."])

# Visualize sentiment scores
sa.visualize_sentiment(result['sentiment_scores'])

# Save and load the model
sa.save_model()
loaded_model = SentimentAnalysis.load_model()
```

---

This documentation provides a comprehensive guide on the purpose, usage, and functionality of the Sentiment Analysis Library. Users can refer to this document to understand how to use the library and its various features. Adjustments can be made based on specific documentation standards or requirements.
