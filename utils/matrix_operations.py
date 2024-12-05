# utils/matrix_operations.py
import numpy as np

def calculate_portfolio_variance(weights: np.ndarray, covariance_matrix: np.ndarray) -> float:
    
    return float(weights.T @ covariance_matrix @ weights)

def calculate_portfolio_return(weights: np.ndarray, expected_returns: np.ndarray) -> float:
    #Calculate  expected return
    return float(weights.T @ expected_returns)