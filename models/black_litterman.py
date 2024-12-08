import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.optimize import minimize
import os
import sys
import yfinance as yf
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from config.settings import Settings
from data.data_loader import HistoricalDataLoader

class BlackLittermanOptimizer:
    def __init__(self):
        self.settings = Settings
        self.data = HistoricalDataLoader()
        
    def get_market_caps(self, tickers: List[str]) -> Dict[str, float]:
        
        mcaps = {}
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            mcaps[ticker] = stock.info.get("marketCap", 0)
        return mcaps
    
    def calculate_implied_returns(self, 
                                market_caps: Dict[str, float], 
                                cov_matrix: np.ndarray,
                                risk_aversion: float = 2.5) -> np.ndarray:
        # convert market caps to weights
        total_mcap = sum(market_caps.values())
        mkt_weights = np.array([cap/total_mcap for cap in market_caps.values()])
        
        # Calculate implied returns (π = λΣw)
        implied_returns = risk_aversion * cov_matrix @ mkt_weights
        return implied_returns
    
    def incorporate_views(self,
                         prior_returns: np.ndarray,
                         views: Dict[str, float],
                         confidences: Dict[str, float],
                         tickers: List[str],
                         cov_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_assets = len(tickers)
        
        #  view matrix (P)
        view_matrix = np.zeros((len(views), n_assets))
        view_vector = np.zeros(len(views))
        
        for i, (ticker, view) in enumerate(views.items()):
            ticker_idx = tickers.index(ticker)
            view_matrix[i, ticker_idx] = 1
            view_vector[i] = view
        
        # Create confidence matrix (Ω)
        confidence_matrix = np.diag([1/confidences[ticker] for ticker in views.keys()])
        
        #formula
        #[(τΣ)^(-1) + P'Ω^(-1)P]^(-1) [(τΣ)^(-1)π + P'Ω^(-1)Q]

        tau = 0.05  # Standard value for tau
        
        # Calculate posterior parameters
        temp1 = np.linalg.inv(tau * cov_matrix)
        temp2 = view_matrix.T @ np.linalg.inv(confidence_matrix) @ view_matrix
        posterior_covar = np.linalg.inv(temp1 + temp2)
        
        temp3 = temp1 @ prior_returns
        temp4 = view_matrix.T @ np.linalg.inv(confidence_matrix) @ view_vector
        posterior_returns = posterior_covar @ (temp3 + temp4)
        
        return posterior_returns, posterior_covar
    
    def optimize_portfolio(self,
                         returns: np.ndarray,
                         covar: np.ndarray,
                         tickers: List[str]) -> Dict:
        #optimize weights using posterioror
        n_assets = len(tickers)
        
        def objective(weights):
            port_return = np.sum(returns * weights)
            port_vol = np.sqrt(weights.T @ covar @ weights)
            sharpe = (port_return - self.settings.RISK_FREE_RATE) / port_vol
            return -sharpe
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        bounds = tuple(
            (self.settings.MINIMUM_WEIGHT, self.settings.MAXIMUM_WEIGHT) 
            for _ in range(n_assets)
        )
        
        result = minimize(
            objective,
            np.array([1/n_assets] * n_assets),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_weights = result.x
        port_return = np.sum(returns * optimal_weights)
        port_vol = np.sqrt(optimal_weights.T @ covar @ optimal_weights)
        sharpe = (port_return - self.settings.RISK_FREE_RATE) / port_vol
        
        return {
            'weights': dict(zip(tickers, optimal_weights)),
            'expected_return': port_return,
            'volatility': port_vol,
            'sharpe_ratio': sharpe
        }
    
    def run_optimization(self,
                        tickers: List[str],
                        views: Dict[str, float],
                        confidences: Dict[str, float]) -> Dict:
        #full optimization
        # Get market data
        historical_data = self.data.get_stock_data(tickers)
        returns_matrix = self.data.get_returns_matrix(historical_data)
        cov_matrix = returns_matrix.cov().values * 252  # Annualize
        
        market_caps = self.get_market_caps(tickers)
        prior_returns = self.calculate_implied_returns(market_caps, cov_matrix)
        
        posterior_returns, posterior_covar = self.incorporate_views(
            prior_returns, views, confidences, tickers, cov_matrix
        )
        
        results = self.optimize_portfolio(posterior_returns, posterior_covar, tickers)
        
        results['views'] = views
        results['confidences'] = confidences
        
        return results

if __name__ == "__main__":
    # Test the optimizer
    optimizer = BlackLittermanOptimizer()
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # Sample views and confidence levels
    views = {
        'AAPL': 0.15,#expect 15% return
        'MSFT': 0.12,# expect 12% return
        'GOOGL': 0.10#expect 10% return
    }
    
    confidences = {
        'AAPL': 0.6,#60% confident
        'MSFT': 0.5,#50% confident
        'GOOGL': 0.4#40% confident
    }
    
    results = optimizer.run_optimization(test_tickers, views, confidences)
    
    print("\nBlack-Litterman Portfolio Results:")
    print("\nOptimal Weights:")
    for ticker, weight in results['weights'].items():
        print(f"{ticker}: {weight:.4f}")
    
    print(f"\nExpected Annual Return: {results['expected_return']:.4f}")
    print(f"Annual Volatility: {results['volatility']:.4f}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
    
    print("\nViews Used:")
    for ticker, view in results['views'].items():
        print(f"{ticker}: {view:.4f} (Confidence: {results['confidences'][ticker]:.2f})")