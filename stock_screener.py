import yfinance as yf
import pandas as pd
from yahoo_fin import stock_info as si
import datetime as dt

tickers = si.tickers_sp500()

# compare performance of individual stocks with s&p500 as a whole

start = dt.datetime.now() - dt.timedelta(days=365)
end = dt.datetime.now()

#load sp500 data
sp500_df = yf.download('^GSPC', start=start, end=end)

# percentage change from day to day
sp500_df['Pct Change'] = sp500_df['Adj Close'].pct_change()
sp500_return = (sp500_df['Pct Change'] + 1).cumprod()[-1]

# do the same thing for individual stocks
return_list = []
final_df = pd.DataFrame(columns=['Ticker','Latest_Price','Score','PE_Ratio','PEG_Ratio','SMA_150','SMA_200','52_Week_Low','52_Week_High'])
# PE = Price to Earnings ratio
# PEG = Price:Earnings:Growth ratio
counter = 0

"""
In theory, a PEG ratio value of 1 represents a perfect correlation between the company's market value and its projected earnings growth. PEG ratios higher than 1.0 are generally considered unfavorable, suggesting a stock is overvalued. Conversely, ratios lower than 1.0 are considered better, indicating a stock is undervalued.
"""

for ticker in tickers:
    df = yf.download(ticker, start=start, end=end)
    df.to_csv(f'stock_data/{ticker}.csv')

    df['Pct Change'] = df['Adj Close'].pct_change()
    stock_return = (df['Pct Change'] + 1).cumprod()[-1]

    returns_compared = round((stock_return/sp500_return),2)
    return_list.append(returns_compared)

    counter += 1
    if counter == 10:
        break

# Filter out the best performance/worst performance
best_performers = pd.DataFrame(list(zip(tickers, return_list)), columns=['Ticker', 'Returns Compared'])
print(best_performers["Ticker"])
best_performers['Score'] = best_performers['Returns Compared'].rank(pct=True) * 100
best_performers = best_performers[best_performers['Score'] >= best_performers['Score'].quantile(0.7)]
# 0.8 means we pick the top 20% best
# 0.7 means we pick the top 30% best

for ticker in best_performers['Ticker']:
    try:
        df = pd.read_csv(f'stock_data/{ticker}.csv', index_col=0)
        moving_averages = [150,200]
        for ma in moving_averages:
            df['SMA_'+ str(ma)] = round(df['Adj Close'].rolling(window=ma).mean(), 2)

        latest_price = df['Adj Close'][-1]
        pe_ratio = yf.Ticker(ticker).info['trailingPE']
        print(pe_ratio)
        peg_ratio = yf.Ticker(ticker).info['pegRatio']
        moving_average_150 = df['SMA_150'][-1]
        moving_average_200 = df['SMA_200'][-1]
        low_52week = round(min(df['Low'][-(52*5):]), 2) # 52 weeks * 5 days(did not exclude special holidays)
        high_52week = round(max(df['High'][-(52*5):]), 2) # 52 weeks * 5 days(did not exclude special holidays)
        score = round(best_performers[best_performers['Ticker'] == ticker]['Score'].to_list()[0])

        condition_1 = latest_price > moving_average_150 > moving_average_200
        condition_2 = latest_price >= (1.3 * low_52week) # above 30% of low
        condition_3 = latest_price >= (0.75 * high_52week) # above 30% of low
        condition_4 = pe_ratio < 40
        condition_5 = peg_ratio < 2

        if condition_1 and condition_2 and condition_3 and condition_4 and condition_5:
            final_df = pd.concat([final_df, pd.DataFrame({'Ticker': [ticker], 'Latest_Price': [latest_price], 'Score': [score],
                                        'PE_Ratio':[pe_ratio], 'PEG_Ratio':[peg_ratio],
                                        'SMA_150':[moving_average_150], 'SMA_200':[moving_average_200],
                                        '52_Week_Low': [low_52week], '52_Week_High':[high_52week]})],
                                        ignore_index = True)
    except Exception as e:
        print(f"{e} for {ticker}")

final_df.sort_values(by='Score', ascending=False)
pd.set_option('display.max_columns',10)
print(final_df)
final_df.to_csv('final.csv')