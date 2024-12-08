import numpy as np
import os
import sys
import pandas as pd
from typing import Dict, Tuple
from scipy.optimize import minimize
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from config.settings import Settings
from data.data_loader import HistoricalDataLoader


class MarkowitzOptimizer:
    def __init__(self):
        self.settings = Settings
        self.data = HistoricalDataLoader()
    
    def calculate_portfolio_metrics(self, weights: np.ndarray, returns: np.ndarray, cov_matrix: np.ndarray) -> Tuple[float, float]:
        #returns portofolio return and volatility as a tuple
        portfolio_return = np.sum(returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return portfolio_return, portfolio_volatility
        
    def optimize_portfolio(
        self,
        tickers: list,
        objective: str = 'sharpe',
        target_return: float = None
    ):
        
        #tickers: list of stock tickers
        #objective: either sharpe or min_risk. Sharpe maximizes sharpe ratio min_risk minimizes
        #target_return: target_return required if objective is min_risk
        # Returns dictionary of optimal weights and returns
        
        # get  data
        historical_data = self.data.get_stock_data(tickers)
        returns_matrix = self.data.get_returns_matrix(historical_data)
        
        exp_returns = returns_matrix.mean() * 252  # Annualized returns
        cov_matrix = returns_matrix.cov() * 252    # Annualized covariance
        
        # intialize weights
        num_assets = len(tickers)
        init_weights = np.array([1.0/num_assets] * num_assets)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # weights sum to 1
        ]
        
        #minimum return constraint if specified
        if target_return is not None:
            constraints.append(
                {'type': 'eq', 
                 'fun': lambda w: np.sum(w * exp_returns) - target_return}
            )
        
        # bounds for each weight (no shorting or leverage)
        bounds = tuple(
            (self.settings.MINIMUM_WEIGHT, self.settings.MAXIMUM_WEIGHT) 
            for _ in range(num_assets)
        )
        
        if objective == 'sharpe':
            def objective_function(weights):
                ret, vol = self.calculate_portfolio_metrics(
                    weights, exp_returns, cov_matrix
                )
                sharpe = (ret - self.settings.RISK_FREE_RATE) / vol
                return -sharpe  # Minimize negative Sharpe ratio
        else:  # min_risk
            def objective_function(weights):
                _, vol = self.calculate_portfolio_metrics(
                    weights, exp_returns, cov_matrix
                )
                return vol
        
        # Optimize
        result = minimize(
            objective_function,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # metrics for optimal portfolio
        optimal_weights = result.x
        opt_return, opt_vol = self.calculate_portfolio_metrics(
            optimal_weights, exp_returns, cov_matrix
        )
        
        return {
            'weights': dict(zip(tickers, optimal_weights)),
            'expected_return': opt_return,
            'volatility': opt_vol,
            'sharpe_ratio': (opt_return - self.settings.RISK_FREE_RATE) / opt_vol
        }

if __name__ == "__main__":
    optimizer = MarkowitzOptimizer()
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # test maximizing sharpe ratio
    sharpe_result = optimizer.optimize_portfolio(tickers, objective='sharpe')
    print("\nOptimal Portfolio (Maximum Sharpe Ratio):")
    for ticker, weight in sharpe_result['weights'].items():
        print(f"{ticker}: {weight:.4f}")
    print(f"Expected Return: {sharpe_result['expected_return']:.4f}")
    print(f"Volatility: {sharpe_result['volatility']:.4f}")
    print(f"Sharpe Ratio: {sharpe_result['sharpe_ratio']:.4f}")
    
    # test minimum volatility with target return
    min_vol_result = optimizer.optimize_portfolio(
        tickers, 
        objective='min_risk',
        target_return=0.15  # 15% target return
    )
    print("\nOptimal Portfolio (Minimum Volatility with 15% target return):")
    for ticker, weight in min_vol_result['weights'].items():
        print(f"{ticker}: {weight:.4f}")
    print(f"Expected Return: {min_vol_result['expected_return']:.4f}")
    print(f"Volatility: {min_vol_result['volatility']:.4f}")
    print(f"Sharpe Ratio: {min_vol_result['sharpe_ratio']:.4f}")
    