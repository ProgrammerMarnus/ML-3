"""
Example: Run Sentiment Analysis.
Demonstrates fetching news and calculating sentiment scores.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_predictor_ml.data.sentiment import SentimentAnalyzer


def main():
    print("=" * 60)
    print("Sentiment Analysis Example")
    print("=" * 60)
    
    # Initialize analyzer
    print("\n1. Initializing sentiment analyzer...")
    analyzer = SentimentAnalyzer(use_mock=True)  # Use mock for demo
    
    # Analyze tickers
    tickers = ["AAPL", "MSFT", "TSLA"]
    
    for ticker in tickers:
        print(f"\n2. Analyzing {ticker}...")
        
        # Get aggregate sentiment
        sentiment = analyzer.get_aggregate_sentiment(ticker, days=7)
        
        print(f"   Mean Sentiment: {sentiment['mean_sentiment']:.3f}")
        print(f"   Sentiment Volatility: {sentiment['sentiment_volatility']:.3f}")
        print(f"   News Count: {sentiment['news_count']}")
        print(f"   Positive Ratio: {sentiment['positive_ratio']*100:.1f}%")
        
        # Test individual sentiment calculation
        test_headlines = [
            f"{ticker} reports record earnings beat",
            f"{ticker} faces regulatory challenges",
            f"Analysts upgrade {ticker} price target",
        ]
        
        print("\n   Sample headline analysis:")
        for headline in test_headlines:
            polarity, subjectivity = analyzer.calculate_sentiment(headline)
            sentiment_label = "Positive" if polarity > 0 else ("Negative" if polarity < 0 else "Neutral")
            print(f"      '{headline[:50]}...' -> {sentiment_label} ({polarity:.2f})")
    
    print("\n" + "=" * 60)
    print("Sentiment analysis example completed!")
    print("=" * 60)
    
    print("\nNote: This demo used mock news data.")
    print("For real news, set use_mock=False and install textblob:")
    print("  pip install textblob")
    print("  python -m textblob.download_corpora")


if __name__ == "__main__":
    main()
