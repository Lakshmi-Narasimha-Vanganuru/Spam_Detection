import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class SentimentAnalyzer:
    """
    A class to perform sentiment analysis on text using VADER.
    """

    def __init__(self):
        """
        Initializes the SentimentIntensityAnalyzer.
        Downloads the 'vader_lexicon' if not already present.
        """
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except nltk.downloader.DownloadError:
            nltk.download('vader_lexicon')
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, text: str) -> dict:
        """
        Analyzes the sentiment of a given text.

        Args:
            text: The input string to analyze.

        Returns:
            A dictionary containing the input text, sentiment scores (compound,
            positive, negative, neutral), and overall sentiment.
            Returns an error message dictionary if the input is not a string.

        Example:
            {
                'text': 'This is a great movie!',
                'sentiment': {
                    'compound': 0.85,
                    'positive': 0.6,
                    'negative': 0.0,
                    'neutral': 0.4
                },
                'overall_sentiment': 'positive'
            }
        """
        if not isinstance(text, str):
            return {
                'error': 'Input must be a string.',
                'text': text
            }

        sentiment_scores = self.analyzer.polarity_scores(text)
        compound_score = sentiment_scores['compound']

        if compound_score >= 0.05:
            overall_sentiment = 'positive'
        elif compound_score <= -0.05:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'

        return {
            'text': text,
            'sentiment': {
                'compound': compound_score,
                'positive': sentiment_scores['pos'],
                'negative': sentiment_scores['neg'],
                'neutral': sentiment_scores['neu']
            },
            'overall_sentiment': overall_sentiment
        }

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()

    # Example usage:
    text1 = "This is a great movie! I loved it."
    analysis1 = analyzer.analyze_sentiment(text1)
    print(f"Analysis for '{analysis1['text']}':")
    print(f"  Sentiment Scores: {analysis1['sentiment']}")
    print(f"  Overall Sentiment: {analysis1['overall_sentiment']}")
    print("-" * 20)

    text2 = "This is a terrible experience. I hated it."
    analysis2 = analyzer.analyze_sentiment(text2)
    print(f"Analysis for '{analysis2['text']}':")
    print(f"  Sentiment Scores: {analysis2['sentiment']}")
    print(f"  Overall Sentiment: {analysis2['overall_sentiment']}")
    print("-" * 20)

    text3 = "The weather is okay today."
    analysis3 = analyzer.analyze_sentiment(text3)
    print(f"Analysis for '{analysis3['text']}':")
    print(f"  Sentiment Scores: {analysis3['sentiment']}")
    print(f"  Overall Sentiment: {analysis3['overall_sentiment']}")
    print("-" * 20)

    text4 = 12345 # Non-string input
    analysis4 = analyzer.analyze_sentiment(text4)
    print(f"Analysis for '{analysis4['text']}':")
    if 'error' in analysis4:
        print(f"  Error: {analysis4['error']}")
    else:
        print(f"  Sentiment Scores: {analysis4['sentiment']}")
        print(f"  Overall Sentiment: {analysis4['overall_sentiment']}")
    print("-" * 20)

    text5 = "VADER is smart, handsome, and funny." # Example from NLTK VADER documentation
    analysis5 = analyzer.analyze_sentiment(text5)
    print(f"Analysis for '{analysis5['text']}':")
    print(f"  Sentiment Scores: {analysis5['sentiment']}")
    print(f"  Overall Sentiment: {analysis5['overall_sentiment']}")
    print("-" * 20)

    text6 = "The food was good, but the service was horrible."
    analysis6 = analyzer.analyze_sentiment(text6)
    print(f"Analysis for '{analysis6['text']}':")
    print(f"  Sentiment Scores: {analysis6['sentiment']}")
    print(f"  Overall Sentiment: {analysis6['overall_sentiment']}")
    print("-" * 20)
