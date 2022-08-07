import yfinance as yf
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
start = datetime.datetime(2014, 1, 1)
end = datetime.datetime.now().strftime("%Y-%M-%d")
tcs=yf.download('TSLA',start,end)
infy = yf.download('INFY',start,end)
wipro=yf.download('WIPRO.NS',start,end)
tcs['Open'].plot(label='TSLA',figsize=(10,10))
infy['Open'].plot(label = "Infosys")
wipro['Open'].plot(label='WIPRO')
plt.title('TSLA and WIPRO')
plt.legend()
plt.show()

tcs['Volume'].plot(label = 'TCS', figsize = (10,10))
infy['Volume'].plot(label = "Infosys")
wipro['Volume'].plot(label = 'Wipro')
plt.title('Volume of Stock traded')
plt.legend()
plt.show()

#PERCENTAGE INCREASE IN STOCK VALUE
tcs['returns'] = (tcs['Close']/tcs['Close'].shift(1)) -1
infy['returns'] = (infy['Close']/infy['Close'].shift(1))-1
wipro['returns'] = (wipro['Close']/wipro['Close'].shift(1)) - 1
tcs['returns'].hist(bins = 100, label = 'TCS', alpha = 0.5, figsize = (10,10))
infy['returns'].hist(bins = 100, label = 'Infosys', alpha = 0.5)
wipro['returns'].hist(bins = 100, label = 'Wipro', alpha = 0.5)
plt.legend()
plt.show()
