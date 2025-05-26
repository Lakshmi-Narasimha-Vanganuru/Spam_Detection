import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Adjust sys.path to allow imports from the 'src' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.content_suggestion import ContentSuggestor

# To prevent actual NLTK downloads during tests and to control NLTK function outputs
# Patching where the names are looked up in the 'src.content_suggestion' module.
@patch('src.content_suggestion._ensure_nltk_resources', MagicMock()) 
@patch('src.content_suggestion.stopwords.words', MagicMock(return_value=['is', 'a', 'the', 'it', 'what'])) 
@patch('src.content_suggestion.nltk.pos_tag') 
@patch('src.content_suggestion.word_tokenize') 
class TestContentSuggestor(unittest.TestCase):
    """
    Unit tests for the enhanced ContentSuggestor class.
    NLTK functions used by _extract_keywords are mocked.
    """

    def setUp(self): # Corrected selfself to self
        """
        Set up the ContentSuggestor instance and mock sentiment data for each test.
        """
        self.suggestor = ContentSuggestor()
        # Updated mock data for better keyword testing
        self.positive_sentiment_data = {
            'text': "The new AI model is excellent and shows great promise.",
            'sentiment': {'compound': 0.85, 'positive': 0.7, 'negative': 0.0, 'neutral': 0.3},
            'overall_sentiment': 'positive'
        }
        self.negative_sentiment_data = {
            'text': "The recent data breach is a terrible disaster for user trust.",
            'sentiment': {'compound': -0.75, 'positive': 0.0, 'negative': 0.6, 'neutral': 0.4},
            'overall_sentiment': 'negative'
        }
        self.neutral_sentiment_data = {
            'text': "The company will announce its quarterly earnings next week.",
            'sentiment': {'compound': 0.0, 'positive': 0.1, 'negative': 0.0, 'neutral': 0.9},
            'overall_sentiment': 'neutral'
        }
        self.neutral_sentiment_no_keyword_data = {
            'text': "It is what it is.", 
            'sentiment': {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0},
            'overall_sentiment': 'neutral'
        }
        self.empty_text_data = {
            'text': "",
            'sentiment': {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 0.0},
            'overall_sentiment': 'neutral'
        }

    def test_positive_sentiment_suggestions(self, mock_word_tokenize, mock_pos_tag): # Corrected selfself to self
        """Test enhanced suggestions for positive sentiment with keyword."""
        mock_word_tokenize.return_value = ['the', 'new', 'ai', 'model', 'is', 'excellent', 'and', 'shows', 'great', 'promise', '.']
        mock_pos_tag.return_value = [('new', 'JJ'), ('ai', 'NN'), ('model', 'NN'), ('excellent', 'JJ'), ('shows', 'VBZ'), ('great', 'JJ'), ('promise', 'NN')]
        
        result = self.suggestor.suggest_content(self.positive_sentiment_data)
        self.assertIn('suggestions', result)
        suggestions = result['suggestions']
        self.assertTrue(len(suggestions) >= 3)

        self.assertTrue(any("Amplify this! Try:" in s for s in suggestions))
        self.assertTrue(any("Share the positivity:" in s for s in suggestions))
        self.assertTrue(any("Engage further:" in s for s in suggestions))
        self.assertTrue(any("ai" in s.lower() or "model" in s.lower() or "promise" in s.lower() for s in suggestions),
                        f"Keyword not found in positive suggestions: {suggestions}")
        self.assertTrue(any("emoji" in s.lower() for s in suggestions))


    def test_negative_sentiment_suggestions(self, mock_word_tokenize, mock_pos_tag): # Corrected selfself to self
        """Test enhanced suggestions for negative sentiment with keyword."""
        mock_word_tokenize.return_value = ['the', 'recent', 'data', 'breach', 'is', 'a', 'terrible', 'disaster', 'for', 'user', 'trust', '.']
        mock_pos_tag.return_value = [('recent', 'JJ'), ('data', 'NN'), ('breach', 'NN'), ('terrible', 'JJ'), ('disaster', 'NN'), ('user', 'NN'), ('trust', 'NN')]
        
        result = self.suggestor.suggest_content(self.negative_sentiment_data)
        self.assertIn('suggestions', result)
        suggestions = result['suggestions']
        self.assertTrue(len(suggestions) >= 3)

        self.assertTrue(any("Acknowledge and offer help:" in s for s in suggestions))
        self.assertTrue(any("Show understanding:" in s for s in suggestions))
        self.assertTrue(any("Offer to take it private:" in s for s in suggestions))
        self.assertTrue(any("data" in s.lower() or "breach" in s.lower() or "disaster" in s.lower() for s in suggestions),
                        f"Keywords like 'data', 'breach', 'disaster' not found in negative suggestions: {suggestions}")

    def test_neutral_sentiment_suggestions(self, mock_word_tokenize, mock_pos_tag): # Corrected selfself to self
        """Test enhanced suggestions for neutral sentiment with keyword."""
        mock_word_tokenize.return_value = ['the', 'company', 'will', 'announce', 'its', 'quarterly', 'earnings', 'next', 'week', '.']
        mock_pos_tag.return_value = [('company', 'NN'), ('announce', 'VB'), ('quarterly', 'JJ'), ('earnings', 'NNS'), ('next', 'JJ'), ('week', 'NN')]

        result = self.suggestor.suggest_content(self.neutral_sentiment_data)
        self.assertIn('suggestions', result)
        suggestions = result['suggestions']
        self.assertTrue(len(suggestions) >= 3)

        self.assertTrue(any("Spark discussion:" in s for s in suggestions))
        self.assertTrue(any("Add a call to action:" in s for s in suggestions))
        self.assertTrue(any("Invite perspectives:" in s for s in suggestions))
        self.assertTrue(any("company" in s.lower() or "earnings" in s.lower() or "week" in s.lower() for s in suggestions),
                        f"Keywords 'company', 'earnings', or 'week' not found in neutral suggestions: {suggestions}")

    def test_keyword_extraction_fallback_no_clear_keyword(self, mock_word_tokenize, mock_pos_tag): # Corrected selfself to self
        """Test suggestions use fallback when no clear keyword is extracted."""
        mock_word_tokenize.return_value = ['it', 'is', 'what', 'it', 'is', '.'] 
        mock_pos_tag.return_value = [] 
        
        result = self.suggestor.suggest_content(self.neutral_sentiment_no_keyword_data)
        self.assertIn('suggestions', result)
        suggestions = result['suggestions']
        self.assertTrue(len(suggestions) > 0)
        
        suggestions_text = " ".join(suggestions).lower()
        self.assertTrue("this topic" in suggestions_text or "this point" in suggestions_text,
                        f"Fallback keyword not found in suggestions for vague text: {suggestions}")

    def test_empty_text_input_suggestions(self, mock_word_tokenize, mock_pos_tag): # Corrected selfself to self
        """Test suggestions for empty text input, ensuring fallback keywords are used."""
        mock_word_tokenize.return_value = [] 
        mock_pos_tag.return_value = []
        
        result = self.suggestor.suggest_content(self.empty_text_data)
        self.assertIn('suggestions', result)
        suggestions = result['suggestions']
        self.assertTrue(len(suggestions) > 0)

        suggestions_text = " ".join(suggestions).lower()
        self.assertTrue("this topic" in suggestions_text or "this point" in suggestions_text,
                        f"Fallback keyword not found in suggestions for empty text: {suggestions}")
        self.assertTrue(any("spark discussion" in s.lower() for s in suggestions) or \
                        any("add a call to action" in s.lower() for s in suggestions) or \
                        any("invite perspectives" in s.lower() for s in suggestions))


    def test_missing_overall_sentiment_key(self, mock_word_tokenize, mock_pos_tag): # Corrected selfself to self
        invalid_data = {
            'text': "Some text here",
            'sentiment': {'compound': 0.1, 'positive': 0.2, 'negative': 0.0, 'neutral': 0.8}
        }
        result = self.suggestor.suggest_content(invalid_data)
        self.assertIn("Error: Input dictionary missing 'overall_sentiment' key.", result['suggestions'][0])

    def test_missing_text_key(self, mock_word_tokenize, mock_pos_tag): # Corrected selfself to self
        invalid_data = {
            'sentiment': {'compound': 0.1, 'positive': 0.2, 'negative': 0.0, 'neutral': 0.8},
            'overall_sentiment': 'neutral'
        }
        result = self.suggestor.suggest_content(invalid_data)
        self.assertIn("Error: Input dictionary missing 'text' key.", result['suggestions'][0])

    def test_unknown_sentiment_value(self, mock_word_tokenize, mock_pos_tag): # Corrected selfself to self
        mock_word_tokenize.return_value = ['this', 'is', 'something', 'new', 'and', 'exciting', '!']
        mock_pos_tag.return_value = [('something', 'NN'), ('new', 'JJ'), ('exciting', 'JJ')]

        unknown_sentiment_data = {
            'text': "This is something new and exciting!",
            'sentiment': {'compound': 0.7, 'positive': 0.6, 'negative': 0.0, 'neutral': 0.4},
            'overall_sentiment': 'excited' 
        }
        result = self.suggestor.suggest_content(unknown_sentiment_data)
        self.assertIn("Warning: Unknown sentiment 'excited'. No specific suggestions available.", result['suggestions'][0])

    def test_input_not_dictionary_string(self, mock_word_tokenize, mock_pos_tag): # Corrected selfself to self
        non_dict_input = "This is just a string."
        result = self.suggestor.suggest_content(non_dict_input)
        self.assertIn("Error: Input must be a dictionary.", result['suggestions'][0])

    def test_input_not_dictionary_list(self, mock_word_tokenize, mock_pos_tag): # Corrected selfself to self
        non_dict_input = ["this", "is", "a", "list"]
        result = self.suggestor.suggest_content(non_dict_input)
        self.assertIn("Error: Input must be a dictionary.", result['suggestions'][0])
        
    def test_input_none(self, mock_word_tokenize, mock_pos_tag): # Corrected selfself to self
        non_dict_input = None
        result = self.suggestor.suggest_content(non_dict_input)
        self.assertIn("Error: Input must be a dictionary.", result['suggestions'][0])


if __name__ == '__main__':
    unittest.main()
