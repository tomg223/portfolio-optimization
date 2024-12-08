import numpy as np
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
from models.markowitz import MarkowitzOptimizer

class PortfolioVisualizer:
    def __init__(self):
        self.optimizer = MarkowitzOptimizer()
        
    def plot_efficient_frontier(self, tickers, num_portfolios: int = 100):
        #plot the efficient frontier with the optimal portfolio point


        #generate random portfolios
        returns = []
        volatilities = []
        sharpe_ratios = []
        weights_list = []
        
        historical_data = self.optimizer.data.get_stock_data(tickers)
        returns_matrix = self.optimizer.data.get_returns_matrix(historical_data)
        exp_returns = returns_matrix.mean() * 252
        cov_matrix = returns_matrix.cov() * 252
        
        for _ in range(num_portfolios):
            # random weights
            weights = np.random.random(len(tickers))
            weights = weights / np.sum(weights)
            weights_list.append(weights)
            
            # portfolio metrics
            portfolio_return, portfolio_vol = self.optimizer.calculate_portfolio_metrics(
                weights, exp_returns, cov_matrix
            )
            returns.append(portfolio_return)
            volatilities.append(portfolio_vol)
            sharpe_ratios.append(
                (portfolio_return - self.optimizer.settings.RISK_FREE_RATE) / portfolio_vol
            )
        
        # get optimal portfolio
        optimal = self.optimizer.optimize_portfolio(tickers, objective='sharpe')
        
        #plot
        plt.figure(figsize=(10, 6))
        plt.scatter(volatilities, returns, c=sharpe_ratios, cmap='viridis', 
                   marker='o', s=10, alpha=0.3)
        plt.colorbar(label='Sharpe Ratio')
        
        # plot optimal portfolio point
        plt.scatter(optimal['volatility'], optimal['expected_return'], 
                   color='red', marker='*', s=200, label='Optimal Portfolio')
        
        plt.xlabel('Expected Volatility')
        plt.ylabel('Expected Return')
        plt.title('Efficient Frontier')
        plt.legend()
        
        return plt
    
    def plot_portfolio_composition(self, optimal_weights: Dict[str, float]):
        
        plt.figure(figsize=(10, 6))
        plt.pie(optimal_weights.values(), labels=optimal_weights.keys(), autopct='%1.1f%%')
        plt.title('Optimal Portfolio Composition')
        
        return plt
    
    def plot_performance_comparison(self, tickers: List[str], weights: Dict[str, float]):
        

        #historical data
        historical_data = self.optimizer.data.get_stock_data(tickers)
        
        #portfolio value over time
        portfolio_values = pd.DataFrame()
        
        #normalize each stock to start at 100
        for ticker in tickers:
            mask = historical_data['Ticker'] == ticker
            prices = historical_data[mask]['Close']
            portfolio_values[ticker] = 100 * (prices / prices.iloc[0])
        
        #calculate weighted portfolio value
        portfolio_values['Portfolio'] = sum(
            portfolio_values[ticker] * weight 
            for ticker, weight in weights.items()
        )
        

        plt.figure(figsize=(12, 6))
        for ticker in tickers:
            plt.plot(portfolio_values.index, portfolio_values[ticker], 
                    label=ticker, alpha=0.5)
        plt.plot(portfolio_values.index, portfolio_values['Portfolio'], 
                label='Portfolio', linewidth=3, color='black')
        
        plt.xlabel('Date')
        plt.ylabel('Value (Starting at 100)')
        plt.title('Performance Comparison')
        plt.legend()
        
        return plt

if __name__ == "__main__":
    # Test the visualizer
    visualizer = PortfolioVisualizer()
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # Get optimal portfolio
    optimal = visualizer.optimizer.optimize_portfolio(tickers, objective='sharpe')
    
    # Create and save all plots
    visualizer.plot_efficient_frontier(tickers)
    plt.savefig('efficient_frontier.png')
    
    visualizer.plot_portfolio_composition(optimal['weights'])
    plt.savefig('portfolio_composition.png')
    
    visualizer.plot_performance_comparison(tickers, optimal['weights'])
    plt.savefig('performance_comparison.png')
    
    plt.close('all')
