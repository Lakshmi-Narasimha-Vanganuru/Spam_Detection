import os
import tweepy
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class TwitterClient:
    """
    A client to interact with the Twitter API using Tweepy.
    Handles authentication and fetching tweets.
    """

    def __init__(self):
        """
        Initializes the TwitterClient.

        Loads Twitter API credentials from environment variables and authenticates with Tweepy.

        Environment Variables Expected:
            - TWITTER_API_KEY: Your Twitter application's API key.
            - TWITTER_API_SECRET_KEY: Your Twitter application's API secret key.
            - TWITTER_ACCESS_TOKEN: Your account's access token.
            - TWITTER_ACCESS_TOKEN_SECRET: Your account's access token secret.

        Raises:
            ValueError: If any of the required API credentials are not found in environment variables.
            tweepy.TweepyException: If authentication with Twitter fails.
        """
        self.api_key = os.getenv("TWITTER_API_KEY")
        self.api_secret_key = os.getenv("TWITTER_API_SECRET_KEY")
        self.access_token = os.getenv("TWITTER_ACCESS_TOKEN")
        self.access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")

        if not all([self.api_key, self.api_secret_key, self.access_token, self.access_token_secret]):
            error_msg = "Twitter API credentials not fully found in environment variables. "\
                        "Please set TWITTER_API_KEY, TWITTER_API_SECRET_KEY, "\
                        "TWITTER_ACCESS_TOKEN, and TWITTER_ACCESS_TOKEN_SECRET."
            logging.error(error_msg)
            raise ValueError(error_msg)

        try:
            auth = tweepy.OAuth1UserHandler(
                self.api_key, self.api_secret_key,
                self.access_token, self.access_token_secret
            )
            self.api = tweepy.API(auth, wait_on_rate_limit=True)
            # Verify credentials to ensure authentication is successful
            self.api.verify_credentials()
            logging.info("TwitterClient initialized and authenticated successfully.")
        except tweepy.TweepyException as e:
            logging.error(f"Error during Twitter authentication: {e}")
            raise  # Re-raise the TweepyException to be handled by the caller

    def fetch_tweets(self, query: str, count: int = 10, lang: str = "en", tweet_mode: str = "extended") -> list[str]:
        """
        Fetches recent tweets based on a search query.

        Args:
            query: The search query (e.g., keyword, hashtag).
            count: The maximum number of tweets to fetch (default is 10).
                   Note: Twitter API limitations might return fewer tweets than requested.
            lang: The language of tweets to search for (e.g., "en" for English). Default is "en".
            tweet_mode: Specifies whether to fetch full-length tweets ("extended") or 
                        standard (potentially truncated) tweets ("compat"). Default is "extended".

        Returns:
            A list of strings, where each string is the text of a fetched tweet.
            Returns an empty list if an error occurs or no tweets are found.
        """
        tweets_text = []
        try:
            # Using cursor for pagination if needed, though search_tweets itself handles 'count' for simple cases.
            # Tweepy's search_tweets is for the v1.1 API. For v2, it would be different.
            searched_tweets = self.api.search_tweets(q=query, lang=lang, count=count, tweet_mode=tweet_mode)
            
            for status in searched_tweets:
                if tweet_mode == "extended":
                    if hasattr(status, 'full_text'):
                        tweets_text.append(status.full_text)
                    elif hasattr(status, 'text'): # Fallback for safety, though extended should have full_text
                        tweets_text.append(status.text)
                else: # tweet_mode == "compat" or not specified for older API versions
                     if hasattr(status, 'text'):
                        tweets_text.append(status.text)
            
            logging.info(f"Fetched {len(tweets_text)} tweets for query: '{query}'")

        except tweepy.TweepyException as e:
            logging.error(f"Error fetching tweets for query '{query}': {e}")
            # Depending on desired behavior, could raise a custom exception here
            # For now, returning an empty list on error.
        except Exception as e:
            logging.error(f"An unexpected error occurred while fetching tweets: {e}")
        
        return tweets_text

if __name__ == "__main__":
    print("Twitter Client Example Usage")
    print("----------------------------")
    print("Note: This example requires Twitter API credentials to be set as environment variables:")
    print("TWITTER_API_KEY, TWITTER_API_SECRET_KEY, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET")
    print("If these are not set, initialization will fail.")

    try:
        client = TwitterClient()
        
        # Example: Fetch 5 recent English tweets containing #Python
        search_query = "#Python"
        num_tweets = 5
        print(f"\nFetching {num_tweets} tweets for query: '{search_query}'...")
        
        fetched_tweets = client.fetch_tweets(query=search_query, count=num_tweets)
        
        if fetched_tweets:
            print(f"\nSuccessfully fetched {len(fetched_tweets)} tweets:")
            for i, tweet_text in enumerate(fetched_tweets):
                print(f"\nTweet {i+1}:")
                print(tweet_text)
        elif not fetched_tweets and client.api: # Check if client was initialized but no tweets found
             print(f"No tweets found for the query '{search_query}'.")
        # If client.api is None, it means initialization failed before fetch_tweets could be called.
        # This case is handled by the ValueError exception below.

    except ValueError as ve:
        # This catches the ValueError raised if credentials are not set.
        logging.warning(f"Could not run TwitterClient example: {ve}")
    except tweepy.TweepyException as te:
        # This catches errors during authentication or fetching if not handled within methods
        logging.error(f"A Twitter API error occurred: {te}")
    except Exception as e:
        # Catch any other unexpected errors
        logging.error(f"An unexpected error occurred in the example: {e}")

    print("\nExample usage finished.")
