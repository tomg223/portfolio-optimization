from typing import List, Optional, Dict
import pandas as pd
import yfinance as yf
from datetime import datetime
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from config.settings import Settings


class HistoricalDataLoader:
    def __init__(self):
        self.data_window = Settings.DATA_WINDOW
        self.data_freq = Settings.DATA_FREQUENCY

    def get_stock_data(self,
                           ticker_list, #list of companies to look at
                           start_date = None, #start date of data retrieval
                           end_date = None): # end date of data retrieval
            
            
        end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start = datetime.now() - pd.Timedelta(self.data_window)
            start_date = start.strftime('%Y-%m-%d')
            
        # get all ticker data
        data = yf.download(
            ticker_list,
            start=start_date,
            end=end_date,
            interval=self.data_freq,
            group_by='ticker',
            auto_adjust=True
        )
            
        # for 1 tick
        #if len(ticker_list) == 1:
            #data.columns = pd.MultiIndex.from_product([ticker_list, data.columns])
            
        # convert data to more digestible data fram
        df = self._restructure_data(data, ticker_list)
            
        # find returns
        df['Returns'] = df.groupby('Ticker')['Close'].pct_change()
            
        return df.dropna(subset=['Returns'])
                           
        pass

    def _restructure_data(self, data: pd.DataFrame, ticker_list) -> pd.DataFrame:
        
        dataframes = []#store individual ticker dataframes

        
        for ticker in ticker_list:
            
            if len(ticker_list) == 1:
                ticker_data = data.copy()
            else:
                ticker_data = data[ticker].copy()
            
           
            ticker_data['Ticker'] = ticker
            
            #adding date column
            ticker_data.reset_index(inplace=True)
            ticker_data.rename(columns={'index': 'Date'}, inplace=True)
            
            dataframes.append(ticker_data)
        
        # return nice looking data
        return pd.concat(dataframes, ignore_index=True)
    
    def get_returns_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert the historical data into a returns matrix"""
        returns_matrix = data.pivot(
            index='Date',
            columns='Ticker',
            values='Returns'
        )
        return returns_matrix
    
    def get_covariance_matrix(self, returns_matrix: pd.DataFrame) -> pd.DataFrame:
        """Calculate the covariance matrix of returns"""
        # annualize covariance matrix(252 trading days in a year on avg)
        return returns_matrix.cov() * 252
    
    def get_expected_returns(self, returns_matrix: pd.DataFrame) -> pd.Series:
        # annualized returns
        return returns_matrix.mean() * 252
    
    
class NewsDataLoader:
    def __init__(self):
        pass
    
    def fetch_news_data(self, keywords: List[str], date_range: str) -> List[Dict]:
        """Fetch news data for sentiment analysis"""
        pass

if __name__ == "__main__":
    
    loader = HistoricalDataLoader()
    
    # Test with some stocks
    test_tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    try:
        # Fetch data
        historical_data = loader.get_stock_data(test_tickers)
        print("\nFirst few rows of historical data:")
        print(historical_data.head())
        
        # Get returns matrix
        returns_matrix = loader.get_returns_matrix(historical_data)
        print("\nFirst few rows of returns matrix:")
        print(returns_matrix.head())
        
        # Get covariance matrix
        cov_matrix = loader.get_covariance_matrix(returns_matrix)
        print("\nCovariance matrix:")
        print(cov_matrix)
        
        # Get expected returns
        exp_returns = loader.get_expected_returns(returns_matrix)
        print("\nExpected returns:")
        print(exp_returns)
        
    except Exception as e:
        print(f"Error during testing: {e}")