from data.data_loader import HistoricalDataLoader, NewsDataLoader
from models.markowitz import MarkowitzOptimizer
from models.sentiment_analyzer import SentimentAnalyzer


def main():
    hist_loader = HistoricalDataLoader()
    news_loader = NewsDataLoader()
    markowitz = MarkowitzOptimizer()
    sentiment = SentimentAnalyzer()

    pass



if __name__ == "__main__":
    main()