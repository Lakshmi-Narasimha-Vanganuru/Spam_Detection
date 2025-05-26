# AI for Social Media Sentiment Analysis and Content Suggestion v2.0

## Description

This project provides an enhanced command-line tool to analyze the sentiment of social media text and offer content suggestions. Version 2.0 introduces Twitter integration to fetch and analyze recent tweets, alongside more specific and actionable content suggestions powered by basic keyword extraction. It identifies whether text is positive, negative, or neutral and provides advice for engagement.

## Features

*   **Sentiment Analysis**: Utilizes NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner) to perform sentiment analysis on input text.
*   **Enhanced Content Suggestions**: Offers more specific, actionable textual examples and templates based on the determined sentiment (positive, negative, or neutral).
    *   Uses basic keyword extraction (focusing on nouns) from the input text to personalize suggestions, making them more relevant.
*   **Twitter Integration**:
    *   Allows fetching recent tweets based on a keyword or hashtag.
    *   Fetched tweets can then be processed for sentiment analysis and content suggestions.
*   **Command-Line Interface (CLI)**: Allows users to interact with the application by inputting text manually or by fetching tweets.

## Project Structure

```
social_media_ai/
├── README.md               # This file
├── .gitignore              # Specifies intentionally untracked files that Git should ignore
├── requirements.txt        # Project dependencies
├── src/                    # Source code
│   ├── __init__.py         # Makes src a Python package
│   ├── sentiment_analysis.py # Core sentiment analysis logic
│   ├── content_suggestion.py # Enhanced content suggestion logic
│   ├── twitter_client.py   # Twitter API interaction client
│   └── main.py             # Main CLI application
├── data/                   # Placeholder for data files
│   └── .gitkeep
├── models/                 # Placeholder for trained models
│   └── .gitkeep
└── tests/                  # Unit tests
    ├── __init__.py         # Makes tests a Python package
    ├── test_sentiment_analysis.py
    ├── test_content_suggestion.py
    └── test_twitter_client.py
```
*(Note: `__init__.py` files are present in `src` and `tests` to make them recognizable as Python packages.)*

## Setup and Installation

### Prerequisites

*   Python 3.7 or newer.
*   Access to a Twitter Developer account and API keys/tokens for Twitter integration features.

### Instructions

1.  **Clone the repository** (Example):
    ```bash
    git clone https://your-repository-url/social_media_ai.git # Replace with actual URL
    cd social_media_ai
    ```

2.  **Create and configure a virtual environment (recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies**:
    This project uses NLTK (for sentiment analysis, tokenization, POS tagging) and Tweepy (for Twitter integration).
    Install all dependencies using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Twitter API Credentials (Optional, for Twitter features)**:
    To use features that interact with Twitter (e.g., fetching tweets), you need to set up Twitter API credentials.
    *   Apply for a [Twitter Developer account](https://developer.twitter.com/) and create an application to get your API keys and tokens.
    *   The application expects the following environment variables to be set:
        *   `TWITTER_API_KEY`
        *   `TWITTER_API_SECRET_KEY`
        *   `TWITTER_ACCESS_TOKEN`
        *   `TWITTER_ACCESS_TOKEN_SECRET`
    *   You can set these variables in your operating system or by creating a `.env` file in the project root directory (e.g., `social_media_ai/.env`) with the following format:
        ```
        TWITTER_API_KEY="your_api_key"
        TWITTER_API_SECRET_KEY="your_api_secret_key"
        TWITTER_ACCESS_TOKEN="your_access_token"
        TWITTER_ACCESS_TOKEN_SECRET="your_access_token_secret"
        ```
        Ensure `.env` is listed in your `.gitignore` file (it is by default). If these are not set, the Twitter fetching functionality will not be available, but manual text analysis will still work.

5.  **NLTK Resource Downloads**:
    The application requires several NLTK resources (`vader_lexicon` for sentiment analysis, `punkt` for tokenization, `stopwords` for keyword extraction, and `averaged_perceptron_tagger` for POS tagging).
    The application attempts to download these resources automatically if they are not found when a relevant module is first run or initialized. You may see download messages in the console on first use. If automatic download fails, you might need to download them manually in a Python interpreter:
    ```python
    import nltk
    nltk.download('vader_lexicon')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    ```

## How to Run

1.  Navigate to the root directory of the project (`social_media_ai`).
2.  Ensure your virtual environment is activated if you created one.
3.  Run the main application script:
    ```bash
    python src/main.py
    ```
4.  The application will display a welcome message and then an interactive menu:
    ```
    Initializing Social Media AI...
    Initialization complete. Welcome to Social Media AI!

    ==================================================
    Choose an option:
      (1) Enter text manually
      (2) Fetch tweets
      (3) Exit
    Enter your choice (1, 2, or 3): 
    ```
    *   **Option 1 (Enter text manually)**: Prompts you to enter any text. The application will then perform sentiment analysis and provide content suggestions for your input.
    *   **Option 2 (Fetch tweets)**:
        *   If Twitter credentials are set up correctly, this option will first prompt you for a search query (e.g., a keyword or hashtag like `#AI`).
        *   Then, it will ask for the number of recent tweets you want to fetch.
        *   Each fetched tweet will be displayed, followed by its sentiment analysis and content suggestions.
        *   If Twitter credentials are not set up or are invalid, this option will show a warning.
    *   **Option 3 (Exit)**: Terminates the application.

## How to Run Tests

Unit tests are provided to ensure the core components are working as expected.

1.  Navigate to the root directory of the project (`social_media_ai`).
2.  Ensure your virtual environment (with development dependencies like `tweepy` if testing Twitter client) is activated.
3.  To run all tests (recommended):
    ```bash
    python -m unittest discover tests
    ```
    Or, more explicitly:
    ```bash
    python -m unittest discover -s tests -p "test_*.py"
    ```
4.  To run specific test files:
    ```bash
    python -m unittest tests.test_sentiment_analysis
    python -m unittest tests.test_content_suggestion
    python -m unittest tests.test_twitter_client 
    ```
    *(Note: `tests.test_twitter_client` will mock out actual API calls but tests the client logic.)*

## Future Enhancements

*   **Advanced NLP Models**: Incorporate more sophisticated NLP models (e.g., Transformers like BERT or GPT) for more nuanced sentiment analysis and content generation.
*   **Custom Model Training**: Train models on domain-specific datasets for improved accuracy in particular niches.
*   **Web Interface**: Develop a user-friendly web interface using frameworks like Flask, Django, or Streamlit.
*   **Expanded Social Media API Integration**: Connect to more platforms or utilize more features of the Twitter API (e.g., posting, user lookups).
*   **More Sophisticated Keyword Extraction**: Improve keyword extraction with techniques like TF-IDF, RAKE, or spaCy entity recognition.
*   **Configuration Options**: Allow users to customize settings, such as sentiment thresholds, number of keywords, or suggestion templates.
*   **Batch Processing**: Allow analysis of multiple texts from a file or a list of tweet IDs.

## Known Issues / Limitations
*   Keyword extraction is basic and relies on simple noun extraction; it may not always identify the most salient topic.
*   Twitter fetching is subject to API rate limits. The `TwitterClient` is set to `wait_on_rate_limit=True`, which helps but might slow down operations if limits are hit.
```
