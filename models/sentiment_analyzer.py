from typing import List, Dict
import numpy as np
import pandas as pd

class SentimentAnalyzer:
    def __init__(self):
        self.model = None  # for FinBERT
    
    def process_news_data(self, news_data: List[Dict]) -> List[float]:
        #process media data and return sentiment score
        pass
    
    def adjust_expected_returns(self, base_returns: np.ndarray, sentiment_scores: List[float]) -> np.ndarray:
        #adjust returns according to sentiment analysis
        pass
