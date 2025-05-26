import unittest
import os
import sys
from unittest.mock import patch, MagicMock

# Adjust sys.path to allow imports from the 'src' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Conditional import for TweepyException to avoid issues if tweepy is not installed
# In a real test environment, tweepy would be a dev dependency.
try:
    import tweepy
    TweepyException = tweepy.TweepyException
except ImportError:
    # Create a dummy TweepyException if tweepy is not installed
    # This allows tests to be defined, though they might not run correctly without tweepy
    class TweepyException(Exception):
        pass
    # Mock tweepy module itself if not present
    sys.modules['tweepy'] = MagicMock()


from src.twitter_client import TwitterClient


class TestTwitterClient(unittest.TestCase):
    """
    Unit tests for the TwitterClient class.
    """

    def setUp(self):
        """
        Set up dummy environment variables for Twitter API keys before each test.
        """
        self.original_env = os.environ.copy()
        os.environ['TWITTER_API_KEY'] = 'test_api_key'
        os.environ['TWITTER_API_SECRET_KEY'] = 'test_api_secret_key'
        os.environ['TWITTER_ACCESS_TOKEN'] = 'test_access_token'
        os.environ['TWITTER_ACCESS_TOKEN_SECRET'] = 'test_access_token_secret'

    def tearDown(self):
        """
        Clean up environment variables set in setUp after each test.
        """
        os.environ.clear()
        os.environ.update(self.original_env)

    @patch('src.twitter_client.tweepy.API')
    @patch('src.twitter_client.tweepy.OAuth1UserHandler')
    def test_initialization_success(self, MockOAuth1UserHandler, MockAPI):
        """Test successful initialization of TwitterClient."""
        # Mock the tweepy API and OAuthHandler
        mock_auth_handler_instance = MockOAuth1UserHandler.return_value
        mock_api_instance = MockAPI.return_value
        mock_api_instance.verify_credentials.return_value = True # Or a mock user object

        client = TwitterClient()

        MockOAuth1UserHandler.assert_called_once_with(
            'test_api_key', 'test_api_secret_key',
            'test_access_token', 'test_access_token_secret'
        )
        MockAPI.assert_called_once_with(mock_auth_handler_instance, wait_on_rate_limit=True)
        mock_api_instance.verify_credentials.assert_called_once()
        self.assertIsNotNone(client.api)

    def test_initialization_missing_one_credential(self):
        """Test initialization when one API credential is missing."""
        os.environ.pop('TWITTER_API_KEY') # Remove one key
        with self.assertRaisesRegex(ValueError, "Twitter API credentials not fully found"):
            TwitterClient()

    def test_initialization_missing_all_credentials(self):
        """Test initialization when all API credentials are missing."""
        keys_to_pop = [
            'TWITTER_API_KEY', 'TWITTER_API_SECRET_KEY',
            'TWITTER_ACCESS_TOKEN', 'TWITTER_ACCESS_TOKEN_SECRET'
        ]
        for key in keys_to_pop:
            os.environ.pop(key, None)
        
        with self.assertRaisesRegex(ValueError, "Twitter API credentials not fully found"):
            TwitterClient()

    @patch('src.twitter_client.tweepy.OAuth1UserHandler')
    def test_initialization_tweepy_auth_exception(self, MockOAuth1UserHandler):
        """Test initialization when tweepy.API().verify_credentials() raises an exception."""
        # Setup: Mock OAuth1UserHandler to return a mock object
        mock_auth_instance = MockOAuth1UserHandler.return_value
        
        # Patch tweepy.API separately to control its instance and verify_credentials
        with patch('src.twitter_client.tweepy.API') as MockAPI_local:
            mock_api_instance = MockAPI_local.return_value
            mock_api_instance.verify_credentials.side_effect = TweepyException("Authentication failed")

            with self.assertRaises(TweepyException) as context:
                TwitterClient()
            self.assertIn("Authentication failed", str(context.exception))

    @patch('src.twitter_client.tweepy.API') # Patch API at the class/module level for client instance
    def test_fetch_tweets_success(self, MockAPI): # MockAPI is passed but we use client.api
        """Test successfully fetching tweets."""
        # Setup client with a mocked API
        mock_api_instance = MockAPI.return_value
        mock_api_instance.verify_credentials.return_value = True # Ensure init passes
        
        client = TwitterClient() # Client now has the globally mocked API instance
        
        # Mock tweet objects
        mock_tweet1 = MagicMock()
        mock_tweet1.full_text = "This is tweet 1"
        mock_tweet2 = MagicMock()
        mock_tweet2.full_text = "This is tweet 2"
        
        client.api.search_tweets.return_value = [mock_tweet1, mock_tweet2]

        query = "test_query"
        count = 2
        lang = "en"
        tweet_mode = "extended"
        
        tweets = client.fetch_tweets(query=query, count=count, lang=lang, tweet_mode=tweet_mode)

        client.api.search_tweets.assert_called_once_with(
            q=query, lang=lang, count=count, tweet_mode=tweet_mode
        )
        self.assertEqual(len(tweets), 2)
        self.assertEqual(tweets[0], "This is tweet 1")
        self.assertEqual(tweets[1], "This is tweet 2")

    @patch('src.twitter_client.tweepy.API')
    def test_fetch_tweets_success_compat_mode(self, MockAPI):
        """Test successfully fetching tweets in compatibility mode."""
        mock_api_instance = MockAPI.return_value
        mock_api_instance.verify_credentials.return_value = True
        
        client = TwitterClient()
        
        mock_tweet1 = MagicMock()
        mock_tweet1.text = "Compat tweet 1" # No full_text in compat mode
        # Ensure full_text is not present or different to test the logic path
        del mock_tweet1.full_text 

        client.api.search_tweets.return_value = [mock_tweet1]

        tweets = client.fetch_tweets(query="test", count=1, tweet_mode="compat")

        client.api.search_tweets.assert_called_once_with(
            q="test", lang="en", count=1, tweet_mode="compat"
        )
        self.assertEqual(len(tweets), 1)
        self.assertEqual(tweets[0], "Compat tweet 1")


    @patch('src.twitter_client.tweepy.API')
    @patch('src.twitter_client.logging') # Mock logging to check error logs
    def test_fetch_tweets_api_error(self, mock_logging, MockAPI):
        """Test fetching tweets when API call raises an error."""
        mock_api_instance = MockAPI.return_value
        mock_api_instance.verify_credentials.return_value = True
        
        client = TwitterClient()
        client.api.search_tweets.side_effect = TweepyException("API Error")

        tweets = client.fetch_tweets("test_query")

        self.assertEqual(tweets, [])
        # Check if logging.error was called (basic check)
        mock_logging.error.assert_called()
        # More specific check for the log message content if desired:
        # mock_logging.error.assert_any_call("Error fetching tweets for query 'test_query': API Error")


    @patch('src.twitter_client.tweepy.API')
    def test_fetch_tweets_empty_result(self, MockAPI):
        """Test fetching tweets when API returns an empty list."""
        mock_api_instance = MockAPI.return_value
        mock_api_instance.verify_credentials.return_value = True
        
        client = TwitterClient()
        client.api.search_tweets.return_value = [] # Simulate no tweets found

        tweets = client.fetch_tweets("test_query")

        self.assertEqual(tweets, [])
        client.api.search_tweets.assert_called_once_with(
            q="test_query", lang="en", count=10, tweet_mode="extended" # Default values
        )

if __name__ == '__main__':
    unittest.main()
