import pandas as pd 
import numpy as np
import yfinance as yf 
import datetime

from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit




stock = "TSLA"
endDate = datetime.datetime.now()
startDate = endDate - datetime.timedelta(days=365 * 5)

data = yf.download(stock, start=startDate, end=endDate)




# rendement futur
data['FutureReturn'] = data['Close'].pct_change(5).shift(-5)

# filtre bruit
threshold = 0.002  # 0.2%

data['Target'] = (data['FutureReturn'] > 0).astype(int)



# lags
for i in range(1, 6):
    data[f'Lag{i}'] = data['Close'].shift(i)

# moyennes mobiles
data['MA10'] = data['Close'].rolling(10).mean()
data['MA20'] = data['Close'].rolling(20).mean()

# différence de tendance
data['MA_diff'] = data['MA10'] - data['MA20']

# volatilité
data['Volatility'] = data['Close'].rolling(10).std()

# rendement
data['Return'] = data['Close'].pct_change()

# RSI
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

# volume
data['Volume_change'] = data['Volume'].pct_change()


data = data.dropna()



features = ['Lag1','Lag2','Lag3','Lag4','Lag5','MA10','MA20','MA_diff','Volatility','Return','RSI','Volume_change']

X = data[features]
y = data['Target']

mask = (abs(data['FutureReturn']) > threshold)
X = X[mask]
y = y[mask]

tscv = TimeSeriesSplit(n_splits=5)

direction_scores = []

for train_index, val_index in tscv.split(X):
    X_train, X_valid = X.iloc[train_index], X.iloc[val_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[val_index]

    model = XGBClassifier(n_estimators=500, learning_rate=0.05)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    accuracy = (preds == y_valid).mean()
    direction_scores.append(accuracy)



print("Direction accuracy moyenne :", (np.mean(direction_scores)*100).round(2),"%")
print("vraie hausse",(y.mean()*100).round(2),"%")

# Pour Apple, le score de precision d'environ 50% est inssufisant, cela revient à jeter une piece. 
# La cause peut etre le fait que le marché est très bruité et difficile à predire à court terme
# Cependant, la strategie 'naive' de predire unquement la hausse est plus efficace (55%), ce qui indique une perte d'information quelque part

# Pour Tesla, j'arrive a obtenir une precision de 51,2%, superieure à la hausse (49%)
