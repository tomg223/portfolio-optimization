from typing import Dict, List
import numpy as np
import pandas as pd

class PortfolioMetrics:

    def calculate_roi(initial_value, final_value):
        return ((final_value - initial_value) / initial_value) * 100

    def calculate_sharpe_ratio(returns: np.ndarray,weights: np.ndarray, risk_free_rate: float):
        #sharpe ratio compares investment returns with investment risk
        portfolio_return = np.sum(returns * weights)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(np.cov(returns) * 252, weights)))
        return (portfolio_return - risk_free_rate) / portfolio_std
    
    def calculate_sortino_ratio(returns: np.ndarray,weights: np.ndarray, risk_free_rate: float):
        #compares performance to downards volatility
        pass
    
    def calculate_maximum_drawdown(portfolio_values: np.ndarray) -> float:
        #largest peak to trough % change. Shows maximum % loss historically
        pass
    
    def calculate_information_ratio(portfolio_returns: np.ndarray,benchmark_returns: np.ndarray):
        #compares portfolio return to some benchmark like S&P 500
        pass