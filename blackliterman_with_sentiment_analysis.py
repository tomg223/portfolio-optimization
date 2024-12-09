# !pip install yfinance
# !pip install PyPortfolioOpt

#Grab Data
import yfinance as yf

import requests
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import random

from pypfopt import EfficientFrontier, objective_functions
from pypfopt import black_litterman, risk_models
from pypfopt import BlackLittermanModel, plotting
from pypfopt.plotting import plot_weights
from pypfopt import DiscreteAllocation

# Default portfolio
symbols =  [
    'AAPL', 
    'MSFT', 
    'GOOGL', 
    ]

#Get the stock data
portfolio = yf.download(symbols, start="2019-01-01", end="2024-01-01")['Adj Close']
market_prices = yf.download("SPY", start='2019-01-01', end='2024-01-01')["Adj Close"]

# Data summary
# print(portfolio.info())
# print(portfolio.describe())

# Visualize adjusted closing prices
plt.figure(figsize=(14, 8))
for symbol in symbols:
    plt.plot(portfolio[symbol], label=symbol)
plt.title('ETF Adjusted Closing Prices (2019-2024)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(portfolio.pct_change().corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix of ETFs")
plt.show()

# ====================== GUYS PAY ATTENTION ======================
# Base Markowitz Optimization here (Baseline with no views) (TODO)
# Can you code this please?
# Include results of portfolio (Expected annual return, Annual volatility, Sharpe ratio)

# ================================================================

# BLACK Litterman

#Grab Market Capitalization for each stock in portfolio
mcaps = {}
for t in symbols:
    stock = yf.Ticker(t)
    mcaps[t] = stock.info["marketCap"]
mcaps

#Calculate Sigma and Delta to get implied market returns
#Ledoit-Wolf is a particular form of shrinkage, where the shrinkage coefficient is computed using O
S = risk_models.CovarianceShrinkage(portfolio).ledoit_wolf()

delta = black_litterman.market_implied_risk_aversion(market_prices)

market_prior = black_litterman.market_implied_prior_returns(mcaps, delta, S)

# Passing in views
# Generate random views for each symbol
# For example, views range from -0.3 to 0.5
viewdict = {symbol: random.uniform(-0.3, 0.5) for symbol in symbols}

bl = BlackLittermanModel(S, pi=market_prior, absolute_views=viewdict)

# Generate random intervals around each view
# For example, pick a random half-width between 0.05 and 0.3
intervals = []
for symbol in symbols:
    view = viewdict[symbol]
    half_width = random.uniform(0.05, 0.3)
    lower_bound = view - half_width
    upper_bound = view + half_width
    intervals.append((lower_bound, upper_bound))

variances = []
for lb, ub in intervals:
    sigma = (ub - lb)/2
    variances.append(sigma ** 2)

print(variances)
omega = np.diag(variances)

bl = BlackLittermanModel(S, pi="market", market_caps=mcaps, risk_aversion=delta,
                        absolute_views=viewdict, omega=omega)

ret_bl = bl.bl_returns()

rets_df = pd.DataFrame([market_prior, ret_bl, pd.Series(viewdict)],
             index=["Prior", "Posterior", "Views"]).T

S_bl = bl.bl_cov()

# Maximum Sharpe
ef = EfficientFrontier(ret_bl, S_bl)
ef.add_objective(objective_functions.L2_reg)
ef.max_sharpe()
weights = ef.clean_weights()

plot_weights(weights)

# Randomly generated portfolio outcome
print("Random Views:")
for s, v in viewdict.items():
    print(f"{s}: {v:.4f}")

print("\nRandom Certainty Intervals:")
for s, interval in zip(symbols, intervals):
    print(f"{s}: {interval}")

print(f"\Weights for randomized data: {weights}")

print("Optimized portfolio with random views and weights")
ef.portfolio_performance(verbose = True)

# From NewsAPI
NEWS_API_KEY = "684b174a63e046cabe4017e0fceced1e"  
SYMBOLS = [
    'AAPL', 
    'MSFT', 
    'GOOGL', 
    ]
MARKET_SYMBOL = "SPY"
START_DATE = "2019-01-01"
END_DATE = "2024-01-01"
VIEW_SCALING_FACTOR = 0.5  # Controls how much sentiment impacts returns

# Functions
def fetch_stock_data(symbols, start_date, end_date):
    return yf.download(symbols, start=start_date, end=end_date)['Adj Close']

def fetch_news(symbol, api_key):
    """Fetch news headlines for a given symbol using NewsAPI."""
    url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={api_key}"
    response = requests.get(url)
    return response.json().get("articles", [])

def compute_sentiment_scores(articles):
    """Compute average sentiment polarity for a list of articles."""
    sentiments = []
    for article in articles:
        headline = article.get("title", "")
        if headline:
            sentiment = TextBlob(headline).sentiment.polarity
            sentiments.append(sentiment)
    return sentiments

def calculate_certainty_interval(view, sentiments):
    """ Adjust certainty intervals centered around the view based on sentiment variability. """
    std_dev = np.std(sentiments) if sentiments else 1  # Prevent zero division
    confidence = 1 / (1 + std_dev)  # Higher variability -> Lower confidence
    width = (1 - confidence) / 2 * 0.5 # Interval width shrinks with higher confidence
    # Added 0.5 for width scaling

    lower_bound = view - width
    upper_bound = view + width
    return (lower_bound, upper_bound)

def normalize_views(sentiment_scores, baseline_returns, scaling_factor):
    """Combine baseline returns with normalized sentiment scores."""
    scaler = MinMaxScaler(feature_range=(-scaling_factor, scaling_factor))
    normalized_sentiments = scaler.fit_transform(np.array(sentiment_scores).reshape(-1, 1)).flatten()
    return {symbol: baseline + sentiment for symbol, baseline, sentiment in zip(SYMBOLS, baseline_returns, normalized_sentiments)}

# Main
if __name__ == "__main__":
    print("Fetching stock and market data")
    # Fetch historical stock data
    portfolio_prices = fetch_stock_data(SYMBOLS, START_DATE, END_DATE)
    market_prices = fetch_stock_data(MARKET_SYMBOL, START_DATE, END_DATE)

    # Calculate market-implied returns
    print("Calculating market-implied returns")
    market_caps = {ticker: yf.Ticker(ticker).info.get("marketCap", 0) for ticker in SYMBOLS}
    cov_matrix = risk_models.CovarianceShrinkage(portfolio_prices).ledoit_wolf()
    risk_aversion = black_litterman.market_implied_risk_aversion(market_prices)
    baseline_returns = black_litterman.market_implied_prior_returns(market_caps, risk_aversion, cov_matrix)

    # Analyze sentiment for each stock
    print("Analyzing sentiment for news articles")
    sentiment_results = {}
    sentiment_scores_list = []

    for symbol in SYMBOLS:
        articles = fetch_news(symbol, NEWS_API_KEY)
        sentiments = compute_sentiment_scores(articles)
        avg_sentiment = np.mean(sentiments) if sentiments else 0
        sentiment_results[symbol] = avg_sentiment
        sentiment_scores_list.append(avg_sentiment)

    # Generate views (expected returns) adjusted by sentiment
    adjusted_views = normalize_views(sentiment_scores_list, baseline_returns, VIEW_SCALING_FACTOR)
    print(adjusted_views)

    # Calculate certainty intervals centered around views
    print("Calculating certainty intervals...")
    certainty_intervals = {
        symbol: calculate_certainty_interval(view, compute_sentiment_scores(fetch_news(symbol, NEWS_API_KEY)))
        for symbol, view in adjusted_views.items()
    }

    # Black-Litterman Model with adjusted views and omega
    print("Running Black-Litterman model...")
    omega_matrix = np.diag([abs(interval[1] - interval[0]) / 2 for interval in certainty_intervals.values()])
    bl_model = BlackLittermanModel(cov_matrix, pi=baseline_returns, absolute_views=adjusted_views, omega=omega_matrix)
    posterior_returns = bl_model.bl_returns()
    posterior_covariance = bl_model.bl_cov()

    # Optimize portfolio
    print("Optimizing portfolio with sentiment-adjusted views...")
    ef = EfficientFrontier(posterior_returns, posterior_covariance)
    ef.add_objective(objective_functions.L2_reg)
    ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    ef.portfolio_performance(verbose=True)

    # Output results
    print("\nOptimal Portfolio Weights:")
    for stock, weight in cleaned_weights.items():
        print(f"{stock}: {weight:.4f}")

    print("\nSentiment-Adjusted Views:")
    for stock, view in adjusted_views.items():
      print(f"{stock}: {view:.4f}")

    print("\nCertainty Intervals (Centered Around Views):")
    for stock, interval in certainty_intervals.items():
        print(f"{stock}: {interval}")
