import os
import sys
import requests
import numpy as np
from typing import Dict, List
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from config.settings import Settings

class SentimentAnalyzer:
    def __init__(self):
        self.settings = Settings
        self.api_key = Settings.API_KEYS["news1"]  # will need to update settings.py
        
    def fetch_news(self, symbol: str) -> List[Dict]:
        """Fetch news headlines for a given symbol using NewsAPI"""
        url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={self.api_key}"
        try:
            response = requests.get(url)
            return response.json().get("articles", [])
        except Exception as e:
            print(f"Error fetching news for {symbol}: {e}")
            return []

    def compute_sentiment_scores(self, articles: List[Dict]) -> List[float]:
        """Compute sentiment polarity for each article"""
        sentiments = []
        for article in articles:
            headline = article.get("title", "")
            if headline:
                sentiment = TextBlob(headline).sentiment.polarity
                sentiments.append(sentiment)
        return sentiments

    def calculate_certainty(self, sentiments: List[float]) -> Tuple[float, float]:
        """Calculate certainty interval based on sentiment variability"""
        if not sentiments:
            return (0.0, 0.0)
            
        std_dev = np.std(sentiments) if len(sentiments) > 1 else 1
        confidence = 1 / (1 + std_dev)  # Higher variability -> Lower confidence
        width = (1 - confidence) / 2 * 0.5  # Scale width
        
        avg_sentiment = np.mean(sentiments)
        return (avg_sentiment - width, avg_sentiment + width)

    def analyze_ticker(self, ticker: str) -> Dict:
        """Analyze sentiment for a single ticker"""
        articles = self.fetch_news(ticker)
        sentiments = self.compute_sentiment_scores(articles)
        
        if not sentiments:
            return {
                'score': 0.0,
                'confidence_interval': (0.0, 0.0),
                'num_articles': 0
            }
            
        return {
            'score': np.mean(sentiments),
            'confidence_interval': self.calculate_certainty(sentiments),
            'num_articles': len(sentiments)
        }

    def normalize_scores(self, sentiment_scores: Dict[str, float], scale_factor: float = 0.5) -> Dict[str, float]:
        """Normalize sentiment scores to a specific range"""
        scores = np.array(list(sentiment_scores.values())).reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(-scale_factor, scale_factor))
        normalized = scaler.fit_transform(scores).flatten()
        return dict(zip(sentiment_scores.keys(), normalized))

    def get_views_and_confidences(self, tickers: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Get sentiment-based views and confidences for Black-Litterman model"""
        views = {}
        confidences = {}
        
        for ticker in tickers:
            analysis = self.analyze_ticker(ticker)
            if analysis['num_articles'] > 0:
                views[ticker] = analysis['score']
                # Convert confidence interval to confidence level
                interval_width = abs(analysis['confidence_interval'][1] - analysis['confidence_interval'][0])
                confidences[ticker] = 1 / (1 + interval_width)  # Wider interval = less confidence
        
        # Normalize views
        views = self.normalize_scores(views, self.settings.SENTIMENT_IMPACT_FACTOR)
        
        return views, confidences

if __name__ == "__main__":
    # Test the sentiment analyzer
    analyzer = SentimentAnalyzer()
    test_tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    views, confidences = analyzer.get_views_and_confidences(test_tickers)
    
    print("\nSentiment Analysis Results:")
    for ticker in test_tickers:
        if ticker in views:
            print(f"\n{ticker}:")
            print(f"Sentiment-Based View: {views[ticker]:.4f}")
            print(f"Confidence Level: {confidences[ticker]:.4f}")