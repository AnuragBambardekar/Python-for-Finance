import yfinance as yf
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from newsapi import NewsApiClient

# Initialize NLTK
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Initialize News API client
newsapi = NewsApiClient(api_key='API_KEY')

# Get the S&P 500 tickers
sp500url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
data_table = pd.read_html(sp500url)
tickers = data_table[0]['Symbol'].tolist()

# Loop over all the tickers and predict their trends
# for ticker in tickers:
# Fetch the company name for the ticker
ticker_obj = yf.Ticker("AAPL")
company_name = ticker_obj.info['longName']

# Download historical data for the stock
data = yf.download("AAPL", period='max')

# Get news articles related to the stock ticker
news = newsapi.get_everything(q=company_name, language='en', sort_by='relevancy')

print(f"Fetching news for {company_name} (AAPL)...")
print(news['articles'])

# Preprocess the news text and compute sentiment scores
sentiments = []
for article in news['articles']:
    title = article['title']
    description = article['description']
    if description is not None:
        text = title + ' ' + description
    else:
        text = title
    text = text.lower().replace('\n', ' ')
    sentiment = sia.polarity_scores(text)['compound']
    sentiments.append(sentiment)

# Compute the average sentiment score for the news data
if len(sentiments) > 0:
    avg_sentiment = sum(sentiments) / len(sentiments)
else:
    avg_sentiment = 0

# Predict the stock trend based on the sentiment score
if avg_sentiment >= 0.05:
    trend = 'Up'
elif avg_sentiment <= -0.05:
    trend = 'Down'
else:
    trend = 'Neutral'

# Print the predicted trend
print('Predicted trend for {} ({}) based on news sentiment: {}'.format(company_name, "AAPL", trend))

# import yfinance as yf
# symbol = 'AAPL'
# ticker = yf.Ticker(symbol)
# print(ticker.info)