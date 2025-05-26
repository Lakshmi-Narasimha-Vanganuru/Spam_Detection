# This file will contain the content suggestion module.
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# Ensure necessary NLTK resources are available
# These will be downloaded if not found when _extract_keywords is first called
# or when the script is run directly (see if __name__ == "__main__")
_NLTK_RESOURCES_DOWNLOADED = False

def _ensure_nltk_resources():
    global _NLTK_RESOURCES_DOWNLOADED
    if _NLTK_RESOURCES_DOWNLOADED:
        return
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK punkt...")
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK averaged_perceptron_tagger...")
        nltk.download('averaged_perceptron_tagger', quiet=True)
    _NLTK_RESOURCES_DOWNLOADED = True


class ContentSuggestor:
    """
    A class to provide content suggestions based on sentiment analysis results.
    Suggestions are more specific and aim to be actionable.
    """

    def _extract_keywords(self, text: str, num_keywords: int = 1) -> list[str]:
        """
        Extracts simple keywords from the text.
        Prioritizes nouns, then other significant words if nouns are scarce.
        """
        _ensure_nltk_resources() # Ensure resources are downloaded before use

        if not text:
            return []

        # Tokenize and convert to lowercase
        tokens = word_tokenize(text.lower())

        # Remove punctuation and stopwords
        stop_words = set(stopwords.words('english'))
        punct = set(string.punctuation)
        filtered_tokens = [
            token for token in tokens 
            if token not in stop_words and token not in punct and len(token) > 2 # Min word length
        ]

        if not filtered_tokens:
            return []

        # Part-of-speech tagging
        tagged_tokens = nltk.pos_tag(filtered_tokens)

        # Prioritize nouns
        nouns = [word for word, tag in tagged_tokens if tag.startswith('NN')]
        
        if nouns:
            # Most common nouns
            noun_counts = Counter(nouns)
            top_nouns = [word for word, count in noun_counts.most_common(num_keywords)]
            if top_nouns:
                return top_nouns

        # Fallback: most common words if no nouns found or preferred nouns are not enough
        # This part can be refined if nouns don't give good results
        if not nouns: # Or if you want to supplement nouns
            word_counts = Counter(filtered_tokens)
            top_words = [word for word, count in word_counts.most_common(num_keywords)]
            return top_words
            
        return [] # Should not be reached if filtered_tokens is not empty

    def suggest_content(self, sentiment_analysis_result: dict) -> dict:
        """
        Generates specific content suggestions based on the overall sentiment of a text,
        incorporating extracted keywords where possible.

        Args:
            sentiment_analysis_result: A dictionary containing the output from
                                       SentimentAnalyzer.analyze_sentiment.
                                       Expected keys: 'overall_sentiment', 'text'.

        Returns:
            A dictionary containing the original sentiment analysis result
            and a list of more specific, actionable suggestions.
        """
        if not isinstance(sentiment_analysis_result, dict):
            return {
                'original_analysis': sentiment_analysis_result,
                'suggestions': ["Error: Input must be a dictionary."]
            }

        required_keys = ['overall_sentiment', 'text']
        for key in required_keys:
            if key not in sentiment_analysis_result:
                return {
                    'original_analysis': sentiment_analysis_result,
                    'suggestions': [f"Error: Input dictionary missing '{key}' key."]
                }

        overall_sentiment = sentiment_analysis_result['overall_sentiment']
        original_text = sentiment_analysis_result['text']
        
        keywords = self._extract_keywords(original_text, num_keywords=1)
        keyword_topic = keywords[0] if keywords else "this topic"
        keyword_aspect = keywords[0] if keywords else "this point"


        suggestions = []

        if overall_sentiment == 'positive':
            suggestions.extend([
                f"Amplify this! Try: 'This is great! Fully agree with the point about {keyword_aspect}.'",
                "Share the positivity: 'Love this perspective! What does everyone else think?'",
                f"Engage further: 'Awesome point! Could you tell us more about {keyword_aspect}?'",
                "Consider adding a relevant positive emoji to your response! e.g., 👍, 🎉, 😊"
            ])
        elif overall_sentiment == 'negative':
            suggestions.extend([
                f"Acknowledge and offer help: 'We're sorry to hear about your experience with {keyword_topic}. Please DM us your details so we can assist.'",
                f"Show understanding: 'Thanks for bringing this to our attention. We understand your frustration regarding {keyword_topic} and are looking into it.'",
                "Offer to take it private: 'This is important. To resolve it, could you please contact our support at [support@example.com/link] or DM us?'"
            ])
        elif overall_sentiment == 'neutral':
            suggestions.extend([
                f"Spark discussion: 'Interesting point. What are your thoughts on how this impacts {keyword_topic} or a related area?'",
                "Add a call to action: 'Good overview. For those interested, learn more here: [your_link_here] or What's your key takeaway?'",
                "Invite perspectives: 'This is a balanced view. We'd love to hear different perspectives on this!'"
            ])
        else:
            suggestions.append(f"Warning: Unknown sentiment '{overall_sentiment}'. No specific suggestions available.")

        return {
            'original_analysis': sentiment_analysis_result,
            'suggestions': suggestions
        }

if __name__ == "__main__":
    print("--- Ensuring NLTK Resources for ContentSuggestor ---")
    _ensure_nltk_resources() # Download resources if running standalone
    print("NLTK resources check/download complete.")
    print("----------------------------------------------------")


    # For standalone execution, we'll mock the SentimentAnalyzer and its output.
    class MockSentimentAnalyzer:
        def analyze_sentiment(self, text: str) -> dict:
            # Simplified mock logic for demonstration
            if not text: # Handle empty text case for keyword extraction test
                 return {
                    'text': text,
                    'sentiment': {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 0.0},
                    'overall_sentiment': 'neutral'
                }
            if "great" in text.lower() or "loved" in text.lower() or "excellent" in text.lower():
                return {
                    'text': text,
                    'sentiment': {'compound': 0.85, 'positive': 0.6, 'negative': 0.0, 'neutral': 0.4},
                    'overall_sentiment': 'positive'
                }
            elif "terrible" in text.lower() or "hated" in text.lower() or "awful" in text.lower():
                return {
                    'text': text,
                    'sentiment': {'compound': -0.8, 'positive': 0.0, 'negative': 0.7, 'neutral': 0.3},
                    'overall_sentiment': 'negative'
                }
            elif "okay" in text.lower() or "report" in text.lower(): # Added "report" for neutral keyword
                 return {
                    'text': text,
                    'sentiment': {'compound': 0.0, 'positive': 0.1, 'negative': 0.1, 'neutral': 0.8},
                    'overall_sentiment': 'neutral'
                }
            else:
                return {
                    'text': text,
                    'sentiment': {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0},
                    'overall_sentiment': 'neutral', 
                    'error': 'Mock sentiment for unknown input pattern'
                }

    # --- Example Usage ---
    suggestor = ContentSuggestor()
    mock_analyzer = MockSentimentAnalyzer()

    print("\n--- Content Suggestion Examples (Enhanced) ---")

    # Positive Sentiment Example
    positive_text = "The new software update is excellent and very user-friendly."
    positive_analysis = mock_analyzer.analyze_sentiment(positive_text)
    positive_suggestions = suggestor.suggest_content(positive_analysis)
    print(f"\nOriginal Analysis (Positive): {positive_suggestions['original_analysis']['text']}")
    print("  Overall Sentiment:", positive_suggestions['original_analysis']['overall_sentiment'])
    print("Suggestions:")
    for sugg in positive_suggestions['suggestions']:
        print(f"- {sugg}")

    # Negative Sentiment Example
    negative_text = "The customer support was awful regarding the billing issue."
    negative_analysis = mock_analyzer.analyze_sentiment(negative_text)
    negative_suggestions = suggestor.suggest_content(negative_analysis)
    print(f"\nOriginal Analysis (Negative): {negative_suggestions['original_analysis']['text']}")
    print("  Overall Sentiment:", negative_suggestions['original_analysis']['overall_sentiment'])
    print("Suggestions:")
    for sugg in negative_suggestions['suggestions']:
        print(f"- {sugg}")

    # Neutral Sentiment Example
    neutral_text = "The company released its quarterly report today."
    neutral_analysis = mock_analyzer.analyze_sentiment(neutral_text)
    neutral_suggestions = suggestor.suggest_content(neutral_analysis)
    print(f"\nOriginal Analysis (Neutral): {neutral_suggestions['original_analysis']['text']}")
    print("  Overall Sentiment:", neutral_suggestions['original_analysis']['overall_sentiment'])
    print("Suggestions:")
    for sugg in neutral_suggestions['suggestions']:
        print(f"- {sugg}")
        
    # Neutral Sentiment Example with less obvious keywords
    neutral_text_vague = "It is what it is."
    neutral_analysis_vague = mock_analyzer.analyze_sentiment(neutral_text_vague)
    neutral_suggestions_vague = suggestor.suggest_content(neutral_analysis_vague)
    print(f"\nOriginal Analysis (Neutral - Vague): {neutral_suggestions_vague['original_analysis']['text']}")
    print("  Overall Sentiment:", neutral_suggestions_vague['original_analysis']['overall_sentiment'])
    print("Suggestions:")
    for sugg in neutral_suggestions_vague['suggestions']:
        print(f"- {sugg}")

    # Empty text input
    empty_text = ""
    empty_analysis = mock_analyzer.analyze_sentiment(empty_text)
    empty_suggestions = suggestor.suggest_content(empty_analysis)
    print(f"\nOriginal Analysis (Empty Text): '{empty_suggestions['original_analysis']['text']}'")
    print("  Overall Sentiment:", empty_suggestions['original_analysis']['overall_sentiment'])
    print("Suggestions:")
    for sugg in empty_suggestions['suggestions']:
        print(f"- {sugg}")


    # Example with missing 'overall_sentiment' key
    invalid_analysis_missing_key = {'text': "Some text here, but no sentiment."}
    error_suggestions_missing = suggestor.suggest_content(invalid_analysis_missing_key)
    print(f"\nOriginal Analysis (Error - Missing Key): {error_suggestions_missing['original_analysis']}")
    print("Suggestions (Error Handling):")
    for sugg in error_suggestions_missing['suggestions']:
        print(f"- {sugg}")

    # Example with non-dict input
    invalid_input_type = "This is just a string, not a dict"
    error_suggestions_type = suggestor.suggest_content(invalid_input_type)
    print(f"\nOriginal Analysis (Error - Wrong Type): {error_suggestions_type['original_analysis']}")
    print("Suggestions (Error Handling):")
    for sugg in error_suggestions_type['suggestions']:
        print(f"- {sugg}")
