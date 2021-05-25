import numpy as np
import pandas as pd
import os
from pandas_datareader import data

def cportfolio(stocks, dataSource, start, end, filePath='portfolio.csv'):
  stocks = stocks.split(',')
  test = data.DataReader(stocks , dataSource, start=start, end=end)
  test = test['Adj Close']

  returns = np.log(test/test.shift(1))
  returns.reset_index(inplace=True)
  # Drop the first row since it contains NaN.
  returns = returns.iloc[1:]
  # Normalize dates.
  returns['Date'] = returns['Date'].apply(lambda x : x.strftime('%Y%m%d'))

  directory = os.path.dirname(filePath)

  # Create dir if the specified non-empty dir doesn't exist.
  if not os.path.exists(directory) and directory:
    os.makedirs(directory)

  returns.to_csv(filePath, header=False, index=False)