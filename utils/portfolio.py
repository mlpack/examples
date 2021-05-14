import numpy as np
import pandas as pd
from pandas_datareader import data

def cportfolio(stocks, start, end, filename='portfolio.csv'):
  test = data.DataReader(stocks , 'yahoo', start=start, end=end)
  test = test['Adj Close']

  returns = np.log(test/test.shift(1))
  returns.reset_index(inplace=True)
  # Drop the first row since it contains NaN.
  returns = returns.iloc[1:]
  # Normalize dates.
  returns['Date'] = returns['Date'].apply(lambda x : x.strftime('%Y%m%d'))
  returns.to_csv(filename, header=False, index=False)