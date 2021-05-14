import numpy as np
import pandas as pd
from pandas_datareader import data

def cportfolio(stocks, start, end, filename='portfolio.csv'):
  test = data.DataReader(stocks , 'yahoo', start=start, end=end)
  print(test.head())

  test = test['Adj Close']

  returns = np.log(test/test.shift(1))
  returns.to_csv(filename)