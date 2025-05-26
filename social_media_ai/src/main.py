# This file will contain the main application logic.

# Adjusting sys.path to allow direct execution and imports from 'src'
import sys
import os
import logging # Added logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sentiment_analysis import SentimentAnalyzer
from src.content_suggestion import ContentSuggestor
try:
    from src.twitter_client import TwitterClient
except ImportError:
    logging.warning("Tweepy library not found or src.twitter_client missing. Twitter functionality will be unavailable.")
    TwitterClient = None # Ensure TwitterClient is defined, even if None
except Exception as e:
    logging.warning(f"An unexpected error occurred importing TwitterClient: {e}. Twitter functionality may be unavailable.")
    TwitterClient = None


# Configure basic logging for the main application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def process_text_and_suggest(text, sentiment_analyzer, content_suggestor):
    """
    Helper function to analyze sentiment for a given text and provide suggestions.
    """
    if not text.strip():
        logging.info("Received empty text for processing.")
        print("  Input text is empty. Skipping analysis and suggestions.")
        return

    print("\n--- Sentiment Analysis ---")
    try:
        sentiment_result = sentiment_analyzer.analyze_sentiment(text)
        if 'error' in sentiment_result:
            print(f"  Error in sentiment analysis: {sentiment_result['error']}")
            return # Don't proceed if sentiment analysis itself had an error
        
        print(f"  Text: \"{sentiment_result.get('text')}\"") # Added quotes for clarity
        print(f"  Overall Sentiment: {sentiment_result.get('overall_sentiment', 'N/A')}")
        if 'sentiment' in sentiment_result:
            scores = sentiment_result['sentiment']
            print(f"  Scores:")
            print(f"    Positive: {scores.get('positive', 0.0):.3f}")
            print(f"    Negative: {scores.get('negative', 0.0):.3f}")
            print(f"    Neutral:  {scores.get('neutral', 0.0):.3f}")
            print(f"    Compound: {scores.get('compound', 0.0):.3f}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during sentiment analysis: {e}")
        print(f"  An unexpected error occurred during sentiment analysis: {e}")
        return # Don't proceed if analysis fails

    print("\n--- Content Suggestions ---")
    try:
        suggestion_result = content_suggestor.suggest_content(sentiment_result)
        if 'suggestions' in suggestion_result and isinstance(suggestion_result['suggestions'], list):
            if any("Error:" in s for s in suggestion_result['suggestions']):
                print("  Could not generate suggestions due to an issue:")
            for suggestion in suggestion_result['suggestions']:
                print(f"  - {suggestion}")
        else:
            print("  Could not retrieve suggestions or suggestions are not in the expected format.")
            if 'original_analysis' in suggestion_result and 'error' in suggestion_result['original_analysis']:
                print(f"  Reason from suggestor: {suggestion_result['original_analysis']['error']}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during content suggestion: {e}")
        print(f"  An unexpected error occurred during content suggestion: {e}")


def run_app(test_inputs=None):
    """
    Runs the Social Media AI application.
    Initializes components, then enters a loop for user interaction:
    manual text input, fetching tweets, or exiting.
    """
    logging.info("Initializing Social Media AI...")
    sentiment_analyzer = None
    content_suggestor = None
    twitter_client_instance = None

    try:
        sentiment_analyzer = SentimentAnalyzer()
        content_suggestor = ContentSuggestor()
        logging.info("SentimentAnalyzer and ContentSuggestor initialized successfully.")
    except Exception as e:
        logging.critical(f"Critical error initializing SentimentAnalyzer or ContentSuggestor: {e}")
        print(f"Critical error during core component initialization: {e}. Application cannot continue.")
        return

    if TwitterClient: # Only try to initialize if the class was imported
        try:
            twitter_client_instance = TwitterClient()
            logging.info("TwitterClient initialized successfully.")
        except ValueError as ve: # Raised by TwitterClient if keys are missing
            logging.warning(f"TwitterClient initialization failed: {ve}. Twitter functionality will be unavailable.")
            twitter_client_instance = None
        except Exception as e: # Catch other potential Tweepy or unexpected errors
            logging.error(f"An unexpected error occurred during TwitterClient initialization: {e}. Twitter functionality will be unavailable.")
            twitter_client_instance = None
    else:
        logging.warning("TwitterClient class not available. Twitter functionality disabled.")

    print("\nInitialization complete. Welcome to Social Media AI!")

    input_source = test_inputs if test_inputs else []
    input_idx = 0
    is_test_mode = bool(test_inputs)

    while True:
        print("\n" + "="*50)
        choice = ""

        if is_test_mode:
            if input_idx < len(input_source):
                # In test mode, we assume all inputs are for manual text processing
                # or an 'exit' command. We'll prefix them with '1' to simulate manual choice.
                test_action = input_source[input_idx]
                input_idx += 1
                if test_action.lower() == 'exit':
                    choice = '3' # Exit
                else:
                    choice = '1' # Manual input
                    # We'll use test_action as the actual text input later
                    print(f"Test Mode: Simulating manual text input with: '{test_action}'")
            else:
                logging.info("Finished processing all predefined test inputs.")
                break # Exit loop if no more test inputs
        else: # pragma: no cover
            print("Choose an option:")
            print("  (1) Enter text manually")
            print("  (2) Fetch tweets")
            print("  (3) Exit")
            try:
                choice = input("Enter your choice (1, 2, or 3): ").strip()
            except EOFError:
                logging.warning("EOFError encountered while reading user choice. Exiting.")
                break

        if choice == '1': # Manual text input
            manual_text = ""
            if is_test_mode:
                manual_text = test_action # Use the text from test_feed
                if manual_text.lower() == 'exit': # Should have been caught by choice '3' already
                    continue
            else: # pragma: no cover
                try:
                    manual_text = input("Enter your social media post/text:\n> ")
                except EOFError:
                    logging.warning("EOFError encountered while reading manual text. Returning to main menu.")
                    continue
            
            process_text_and_suggest(manual_text, sentiment_analyzer, content_suggestor)

        elif choice == '2': # Fetch tweets
            if is_test_mode: # pragma: no cover
                print("Test Mode: Skipping Twitter fetching.")
                continue

            if not twitter_client_instance: # pragma: no cover
                print("Twitter client not available. Please check API credentials and ensure 'tweepy' is installed.")
                logging.warning("Attempted to use Twitter client, but it's not available.")
                continue
            
            # pragma: no cover (interactive part)
            search_query = input("Enter keyword/hashtag to search on Twitter: ").strip()
            if not search_query:
                print("Search query cannot be empty.")
                continue
            
            num_tweets_str = input("How many tweets to fetch? (default 10, max 100): ").strip()
            num_tweets = 10 # Default
            if num_tweets_str:
                try:
                    num_tweets = int(num_tweets_str)
                    if not (1 <= num_tweets <= 100):
                        print("Number of tweets must be between 1 and 100. Using default (10).")
                        num_tweets = 10
                except ValueError:
                    print("Invalid number. Using default (10).")
                    num_tweets = 10
            
            print(f"\nFetching {num_tweets} tweets for query: '{search_query}'...")
            try:
                fetched_tweets = twitter_client_instance.fetch_tweets(query=search_query, count=num_tweets)
                if not fetched_tweets:
                    print("No tweets found for your query, or an error occurred during fetching.")
                else:
                    print(f"--- Processing {len(fetched_tweets)} Fetched Tweets ---")
                    for i, tweet_text in enumerate(fetched_tweets):
                        print(f"\n\n--- Tweet {i+1}/{len(fetched_tweets)} ---")
                        print(f"Original Tweet: \"{tweet_text}\"")
                        process_text_and_suggest(tweet_text, sentiment_analyzer, content_suggestor)
                        print("-" * 30) # Separator for each tweet's full analysis
            except Exception as e: # Catch any error from fetch_tweets or subsequent processing
                logging.error(f"An error occurred during tweet fetching or processing: {e}")
                print(f"An error occurred: {e}")


        elif choice == '3': # Exit
            logging.info("User chose to exit.")
            print("Exiting Social Media AI. Goodbye!")
            break

        else: # Invalid choice
            if not is_test_mode: # pragma: no cover
                 print("Invalid choice. Please enter 1, 2, or 3.")
            # In test mode, an invalid choice means the test_action was not 'exit', so it's treated as text.


if __name__ == "__main__":
    # NLTK VADER lexicon check (remains important)
    try:
        import nltk
        nltk.data.find('sentiment/vader_lexicon.zip')
    except nltk.downloader.DownloadError: # pragma: no cover
        logging.warning("NLTK 'vader_lexicon' not found. Attempting to download...")
        try:
            nltk.download('vader_lexicon')
            logging.info("'vader_lexicon' downloaded successfully.")
        except Exception as e:
            logging.critical(f"Failed to download 'vader_lexicon': {e}. Application might not work correctly.")
            print(f"Failed to download 'vader_lexicon': {e}. Please ensure internet connection or install manually.")
            # sys.exit(1) # Decided not to exit, SentimentAnalyzer will also try to download
    except ImportError: # pragma: no cover
        logging.critical("NLTK library not found. Please install it: pip install nltk")
        print("NLTK library not found. Please install it using: pip install nltk")
        sys.exit(1) # NLTK is critical for core functionality

    # Tweepy check (optional, as TwitterClient handles its absence)
    try:
        import tweepy
    except ImportError: # pragma: no cover
        logging.warning("Tweepy library not found. Twitter functionality will be disabled. Install with: pip install tweepy")
        # No sys.exit here as Twitter functionality is optional

    # Test feed for non-interactive mode (manual input path)
    # The test_feed will only test choice '1' (manual input) and '3' (exit)
    test_feed = [
        "I love this new product! It's amazing.",
        "This is the worst service I have ever received.",
        "The weather is quite neutral today.",
        "", # Test empty input
        "exit" # Test exit condition
    ]
    # To run interactively, call run_app()
    # To run with test_feed for manual path: run_app(test_inputs=test_feed)
    
    # Defaulting to interactive mode if run directly for now.
    # If you want to automatically run test_feed, uncomment the line below
    # run_app(test_inputs=test_feed) 
    run_app() # Runs in interactive mode by default.
    # For automated testing in a CI/CD, you might pass a special arg or env var to trigger test_feed.
