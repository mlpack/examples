import numpy as np
import pandas as pd
from pandas_datareader import data

def cportfolio(stock0, stock1, stock2, stock3, start, end, filename='portfolio.csv'):
  test = data.DataReader([stock0, stock1, stock2, stock3], 'yahoo', start=start, end=end)
  print(test.head())

  test = test['Adj Close']

  returns = np.log(test/test.shift(1))
  returns.to_csv('portfolio.csv')