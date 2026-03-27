import math
import numpy as np
import pandas as pd
import datetime
import scipy.stats as stats
import matplotlib.pyplot as plt
import yfinance as yf

def get_data(stocks, start, end):
    stockData = yf.download(stocks, start, end)
    if isinstance(stockData, pd.Series):    #pbm: si un seul stock, stockData['Close'] devient une serie au lieu d'un dataframe
        stockData = stockData.to_frame()
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix


stocks =['AAPL', 'MSFT', 'TSLA']
endDate = datetime.datetime.now()
startDate = endDate - datetime.timedelta(days=300)

meanReturns, covMatrix = get_data(stocks, startDate, endDate)
meanReturns = meanReturns.values

weights = np.random.dirichlet(np.ones(len(meanReturns)))

# Monte Carlo Metho
mc_sims = 400       # nb de simulations
T = 100             #periode (en jours)

meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
meanM = meanM.T             #transposé 

portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

initialPortfolio = 10000


L = np.linalg.cholesky(covMatrix) # decomposition de Cholesky (matrice triangulaire inf)
for m in range(0, mc_sims):
    Z = np.random.normal(size=(len(weights), T))        #VA décorrélé -> bruit 
    correlated_returns = L @ Z
    dailyReturns = meanReturns[:, np.newaxis] + correlated_returns
    portfolio_returns = weights @ dailyReturns
    portfolio_sims[:, m] = np.cumprod(1 + portfolio_returns) * initialPortfolio #maj colonne m 

plt.plot(portfolio_sims)
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Days')
plt.title('MC simulation of a stock portfolio')
plt.show()

def mcVaR(returns, alpha=5):
    """ Input: pandas series of returns
        Output: percentile on return distribution to a given confidence level alpha
    """
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    else:
        raise TypeError("Expected a pandas data series.")

def mcCVaR(returns, alpha=5):
    """ Input: pandas series of returns
        Output: CVaR or Expected Shortfall to a given confidence level alpha
    """
    if isinstance(returns, pd.Series):
        belowVaR = returns <= mcVaR(returns, alpha=alpha)
        return returns[belowVaR].mean()
    else:
        raise TypeError("Expected a pandas data series.")
    


portResults = pd.Series(portfolio_sims[-1,:])

VaR = initialPortfolio - mcVaR(portResults, alpha=5)
CVaR = initialPortfolio - mcCVaR(portResults, alpha=5)

print('VaR_5 ${}'.format(round(VaR,2)))
print('CVaR_5 ${}'.format(round(CVaR,2)))
