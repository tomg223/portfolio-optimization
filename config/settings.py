from typing import Dict, Any
from pathlib import Path

class Settings:
    # api config
    API_KEYS: Dict[str, str] = {
        "news1": "",
        "news2": "",
        "news3": ""
    }
    
    #data params
    DATA_WINDOW: str = "1825d"  # 5 years of historical data
    DATA_FREQUENCY: str = "1d"  # daily data
    
    # Portfolio Parameters
    RISK_FREE_RATE: float = 0.0419  # 3% risk-free rate
    TARGET_RETURN: float = 0.10   # 10% target return
    MINIMUM_WEIGHT: float = 0.0   # no short selling
    MAXIMUM_WEIGHT: float = 1.0   # no leverage
    
    # Sentiment Analysis Parameters
    SENTIMENT_WINDOW: int = 30  # days to look back for news
    SENTIMENT_IMPACT_FACTOR: float = 0.2  # how much sentiment affects returns
    
    # Cache Settings
    CACHE_EXPIRY: int = 24 * 60 * 60  # 24 hours in seconds
    USE_CACHE: bool = True
    
    def get_all_settings(cls) -> Dict[str, Any]:
        """Return all settings as a dictionary"""
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('_') and k.isupper()}
