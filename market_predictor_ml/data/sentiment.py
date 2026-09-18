"""
Sentiment Analysis Module.
Fetches news and calculates sentiment scores for trading signals.
"""

import pandas as pd
import zlib

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class SentimentAnalyzer:
    """
    Analyzes news sentiment for financial assets.
    
    Uses TextBlob for NLP scoring with fallback mock data.
    """
    
    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock
        
        try:
            from textblob import TextBlob
            self.TextBlob = TextBlob
            self.nlp_available = True
        except ImportError:
            self.nlp_available = False
    
    def fetch_news(self, ticker: str, days: int = 7) -> List[Dict]:
        """Fetch news headlines for a ticker."""
        if self.use_mock or not self.nlp_available:
            return self._get_mock_news(ticker, days)
        
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            news_items = stock.news
            
            results = []
            for item in news_items[:50]:  # Limit to 50 items
                results.append({
                    "title": item.get("title", ""),
                    "publisher": item.get("publisher", ""),
                    "timestamp": datetime.fromtimestamp(item.get("providerPublishTime", 0)),
                    "link": item.get("link", ""),
                })
            
            return results
        except Exception:
            return self._get_mock_news(ticker, days)
    
    def _get_mock_news(self, ticker: str, days: int = 7) -> List[Dict]:
        """Generate mock news for testing."""
        # zlib.crc32 is stable across processes (str hash is salted per run)
        np.random.seed(zlib.crc32(ticker.encode("utf-8")) & 0xFFFFFFFF)
        
        base_titles = [
            f"{ticker} reports strong earnings beat",
            f"Analysts upgrade {ticker} price target",
            f"{ticker} faces regulatory scrutiny",
            f"Market volatility impacts {ticker}",
            f"{ticker} announces new product launch",
            f"Institutional investors increase stake in {ticker}",
            f"{ticker} CEO discusses growth strategy",
            f"Supply chain concerns affect {ticker}",
        ]
        
        news = []
        for i in range(days * 3):  # 3 articles per day
            timestamp = datetime.now() - timedelta(days=np.random.randint(0, days))
            title = np.random.choice(base_titles)
            
            news.append({
                "title": title,
                "publisher": "Mock News",
                "timestamp": timestamp,
                "link": f"https://example.com/news/{i}",
            })
        
        return sorted(news, key=lambda x: x["timestamp"], reverse=True)
    
    def calculate_sentiment(self, text: str) -> Tuple[float, float]:
        """
        Calculate sentiment polarity and subjectivity.
        
        Returns:
            Tuple of (polarity [-1, 1], subjectivity [0, 1])
        """
        if not self.nlp_available or self.use_mock:
            # Mock sentiment based on keywords
            text_lower = text.lower()
            positive_words = ["strong", "beat", "upgrade", "growth", "increase"]
            negative_words = ["scrutiny", "volatility", "concerns", "decline", "drop"]
            
            pos_count = sum(1 for w in positive_words if w in text_lower)
            neg_count = sum(1 for w in negative_words if w in text_lower)
            
            total = pos_count + neg_count
            if total == 0:
                return 0.0, 0.5
            
            polarity = (pos_count - neg_count) / total
            subjectivity = np.random.uniform(0.4, 0.8)
            
            return polarity, subjectivity
        
        blob = self.TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    
    def get_aggregate_sentiment(
        self,
        ticker: str,
        days: int = 7,
    ) -> Dict[str, float]:
        """
        Get aggregate sentiment metrics for a ticker.
        
        Returns:
            Dictionary with mean_sentiment, sentiment_volatility, news_count
        """
        news = self.fetch_news(ticker, days)
        
        if len(news) == 0:
            return {
                "mean_sentiment": 0.0,
                "sentiment_volatility": 0.0,
                "news_count": 0,
            }
        
        sentiments = []
        for item in news:
            polarity, _ = self.calculate_sentiment(item["title"])
            sentiments.append(polarity)
        
        sentiments = np.array(sentiments)
        
        return {
            "mean_sentiment": float(np.mean(sentiments)),
            "sentiment_volatility": float(np.std(sentiments)),
            "news_count": len(news),
            "positive_ratio": float(np.mean(sentiments > 0)),
        }
    
    def create_sentiment_features(
        self,
        ticker: str,
        price_df: pd.DataFrame,
        window: int = 5,
    ) -> pd.DataFrame:
        """
        Create sentiment features aligned with price data.
        
        Args:
            ticker: Stock ticker
            price_df: DataFrame with OHLCV data
            window: Rolling window for sentiment aggregation
            
        Returns:
            DataFrame with sentiment features
        """
        # Get sentiment over the period
        dates = price_df.index if hasattr(price_df.index, '__len__') else pd.date_range(
            start="2020-01-01", periods=len(price_df)
        )
        
        # Fetch/generate news ONCE for the whole span, then build a single
        # per-day sentiment series and walk it (O(n) instead of O(n^2)).
        span_days = 0
        try:
            span_days = max(0, int((dates[-1] - dates[0]).days)) if len(dates) > 1 else 0
        except Exception:
            span_days = 0
        news = sorted(
            self.fetch_news(ticker, days=span_days + 7),
            key=lambda n: n["timestamp"],
        )
        from collections import defaultdict as _dd
        per_day = _dd(list)
        for item in news:
            polarity, _ = self.calculate_sentiment(item["title"])
            ts = item["timestamp"]
            per_day[ts.date() if hasattr(ts, "date") else ts].append(polarity)
        sent_series = pd.Series(
            {d: float(np.mean(v)) for d, v in per_day.items()}
        )
        if not sent_series.empty:
            sent_series.index = pd.to_datetime(sent_series.index)
            sent_series = sent_series.sort_index()
        # Align to the price index (date-normalized), forward-fill gaps.
        norm_index = pd.DatetimeIndex(dates).normalize()
        sentiment_scores = list(
            sent_series.reindex(norm_index).ffill().fillna(0.0).values
        )

        # Create features
        df = pd.DataFrame({
            "sentiment_raw": sentiment_scores,
        }, index=price_df.index)
        
        # Rolling features
        df["sentiment_ma"] = df["sentiment_raw"].rolling(window=window, min_periods=1).mean()
        df["sentiment_std"] = df["sentiment_raw"].rolling(window=window, min_periods=1).std().fillna(0)
        df["sentiment_momentum"] = df["sentiment_raw"].diff().fillna(0)
        
        return df
