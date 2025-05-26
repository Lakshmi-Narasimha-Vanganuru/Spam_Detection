import unittest
import sys
import os

# Adjust sys.path to allow imports from the 'src' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sentiment_analysis import SentimentAnalyzer

class TestSentimentAnalyzer(unittest.TestCase):
    """
    Unit tests for the SentimentAnalyzer class.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up the SentimentAnalyzer instance once for all tests.
        This is efficient as SentimentIntensityAnalyzer initialization (and VADER lexicon loading)
        can take a moment and doesn't need to be repeated for each test method.
        """
        cls.analyzer = SentimentAnalyzer()

    def test_positive_sentiment(self):
        """Test with clearly positive text."""
        text = "I love this product! It's amazing and fantastic."
        result = self.analyzer.analyze_sentiment(text)

        self.assertNotIn('error', result, "Analysis should not produce an error for valid text.")
        self.assertEqual(result['overall_sentiment'], 'positive')
        self.assertGreater(result['sentiment']['positive'], result['sentiment']['negative'], "Positive score should be greater than negative.")
        # For very positive text, positive score is often also greater than neutral.
        self.assertGreater(result['sentiment']['positive'], result['sentiment']['neutral'], "Positive score should be greater than neutral for strongly positive text.")
        self.assertGreaterEqual(result['sentiment']['compound'], 0.05, "Compound score should be >= 0.05 for positive sentiment.")
        self.assertEqual(result['text'], text)

    def test_negative_sentiment(self):
        """Test with clearly negative text."""
        text = "This is terrible. I am very unhappy and disappointed."
        result = self.analyzer.analyze_sentiment(text)

        self.assertNotIn('error', result)
        self.assertEqual(result['overall_sentiment'], 'negative')
        self.assertGreater(result['sentiment']['negative'], result['sentiment']['positive'], "Negative score should be greater than positive.")
        # For very negative text, negative score is often also greater than neutral.
        self.assertGreater(result['sentiment']['negative'], result['sentiment']['neutral'], "Negative score should be greater than neutral for strongly negative text.")
        self.assertLessEqual(result['sentiment']['compound'], -0.05, "Compound score should be <= -0.05 for negative sentiment.")
        self.assertEqual(result['text'], text)

    def test_neutral_sentiment(self):
        """Test with neutral text."""
        text = "The sky is blue and the clouds are white." # A more descriptive neutral text
        result = self.analyzer.analyze_sentiment(text)

        self.assertNotIn('error', result)
        self.assertEqual(result['overall_sentiment'], 'neutral')
        self.assertTrue(-0.05 < result['sentiment']['compound'] < 0.05, "Compound score should be between -0.05 and 0.05 for neutral sentiment.")
        self.assertEqual(result['text'], text)

    def test_mixed_sentiment(self):
        """Test with text containing both positive and negative elements."""
        text = "The food was great, but the service was slow and disappointing."
        result = self.analyzer.analyze_sentiment(text)

        self.assertNotIn('error', result)
        # The overall sentiment for mixed text can vary. VADER might lean one way or stay neutral.
        # For "The food was great, but the service was slow and disappointing.", VADER often gives negative.
        # Let's check that scores are present and compound reflects some nuance.
        # We are primarily checking the structure and that it runs.
        self.assertIn('sentiment', result)
        self.assertIn('overall_sentiment', result)
        self.assertTrue(result['sentiment']['positive'] > 0, "Positive score should exist for mixed text.")
        self.assertTrue(result['sentiment']['negative'] > 0, "Negative score should exist for mixed text.")
        # Example: VADER scores for "The food was great, but the service was slow and disappointing."
        # {'neg': 0.268, 'neu': 0.525, 'pos': 0.207, 'compound': -0.2263} -> negative
        self.assertEqual(result['overall_sentiment'], 'negative', "VADER often rates this specific mixed example as negative.")
        self.assertEqual(result['text'], text)

    def test_empty_string_input(self):
        """Test with an empty string."""
        text = ""
        result = self.analyzer.analyze_sentiment(text)

        self.assertNotIn('error', result)
        self.assertEqual(result['overall_sentiment'], 'neutral', "Empty string should be treated as neutral.")
        self.assertEqual(result['sentiment']['compound'], 0.0, "Compound score for empty string should be 0.0.")
        self.assertEqual(result['sentiment']['positive'], 0.0)
        self.assertEqual(result['sentiment']['negative'], 0.0)
        self.assertEqual(result['sentiment']['neutral'], 0.0) # VADER 0.0 for empty, older versions might give 1.0
        self.assertEqual(result['text'], text)


    def test_non_string_input_integer(self):
        """Test with integer input."""
        text = 12345
        result = self.analyzer.analyze_sentiment(text)
        self.assertIn('error', result, "Should return an error for non-string input.")
        self.assertEqual(result['error'], 'Input must be a string.')
        self.assertEqual(result['text'], text)

    def test_non_string_input_list(self):
        """Test with list input."""
        text = ["this", "is", "a", "list"]
        result = self.analyzer.analyze_sentiment(text)
        self.assertIn('error', result, "Should return an error for non-string input.")
        self.assertEqual(result['error'], 'Input must be a string.')
        self.assertEqual(result['text'], text)
        
    def test_non_string_input_none(self):
        """Test with None input."""
        text = None
        result = self.analyzer.analyze_sentiment(text)
        self.assertIn('error', result, "Should return an error for None input.")
        self.assertEqual(result['error'], 'Input must be a string.')
        self.assertEqual(result['text'], text)

    def test_emoji_positive_sentiment(self):
        """Test with text containing positive emojis."""
        text = "I love this 😊" # VADER recognizes 😊 as positive
        result = self.analyzer.analyze_sentiment(text)
        
        self.assertNotIn('error', result)
        self.assertEqual(result['overall_sentiment'], 'positive')
        self.assertGreaterEqual(result['sentiment']['compound'], 0.05)
        self.assertGreater(result['sentiment']['positive'], result['sentiment']['negative'])
        self.assertEqual(result['text'], text)

    def test_emoji_negative_sentiment(self):
        """Test with text containing negative emojis."""
        text = "This is bad 😠" # VADER recognizes 😠 as negative
        result = self.analyzer.analyze_sentiment(text)

        self.assertNotIn('error', result)
        self.assertEqual(result['overall_sentiment'], 'negative')
        self.assertLessEqual(result['sentiment']['compound'], -0.05)
        self.assertGreater(result['sentiment']['negative'], result['sentiment']['positive'])
        self.assertEqual(result['text'], text)

    def test_emoji_mixed_sentiment_with_text(self):
        """Test with text and mixed emojis."""
        text = "The movie was okay 🤔 but the acting was good 👍" # 🤔 is neutral, 👍 is positive
        result = self.analyzer.analyze_sentiment(text)

        self.assertNotIn('error', result)
        # Depending on VADER's weighting, this could be neutral or positive.
        # "The movie was okay 🤔 but the acting was good 👍" -> compound around 0.5 -> positive
        self.assertEqual(result['overall_sentiment'], 'positive')
        self.assertTrue(result['sentiment']['positive'] > result['sentiment']['negative'])
        self.assertEqual(result['text'], text)

    def test_sentence_with_punctuation_intensity(self):
        """Test sentence with exclamation marks for intensity."""
        text = "This is great!!!"
        result_normal = self.analyzer.analyze_sentiment("This is great.")
        result_intense = self.analyzer.analyze_sentiment(text)

        self.assertNotIn('error', result_intense)
        self.assertEqual(result_intense['overall_sentiment'], 'positive')
        # VADER increases intensity for "!!!"
        self.assertGreater(result_intense['sentiment']['compound'], result_normal['sentiment']['compound'])
        self.assertEqual(result_intense['text'], text)

    def test_degree_modifiers(self):
        """Test effect of degree modifiers (e.g., 'very', 'slightly')."""
        text_very_good = "This is very good."
        text_slightly_good = "This is slightly good."
        result_very_good = self.analyzer.analyze_sentiment(text_very_good)
        result_slightly_good = self.analyzer.analyze_sentiment(text_slightly_good)

        self.assertNotIn('error', result_very_good)
        self.assertNotIn('error', result_slightly_good)
        self.assertEqual(result_very_good['overall_sentiment'], 'positive')
        self.assertEqual(result_slightly_good['overall_sentiment'], 'positive')
        self.assertGreater(result_very_good['sentiment']['compound'], result_slightly_good['sentiment']['compound'])
        self.assertEqual(result_very_good['text'], text_very_good)
        self.assertEqual(result_slightly_good['text'], text_slightly_good)


if __name__ == '__main__':
    unittest.main()
