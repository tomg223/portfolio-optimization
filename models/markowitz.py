import numpy as np
from typing import Dict
import pandas as pd

class MarkowitzOptimizer:
    def __init__(self):
        self.returns = None
        self.weights = None
        self.covariance_matrix = None
    
    def calculate_expected_returns(self, historical_data: pd.DataFrame) -> np.ndarray:
        #use markowitz to calculate retunrs
        pass
    
    def calculate_covariance_matrix(self, historical_data: pd.DataFrame) -> np.ndarray:
        #find covar matrix
        pass
    
    def optimize_portfolio(self, target_return: float = None) -> Dict:
        #optimize holdings
        pass
