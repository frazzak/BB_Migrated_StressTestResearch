#Acquire mapping to Appropriate Features
    #Attention based or Feature selection based from source domain.

#Select relevant features from Source and target domain

#Review scaling and normalization
#Remember to invert scale for forecasting results.

#Convert Time-Series into supervised learning?
from pandas import DataFrame
from pandas import concat
import pandas as pd

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

#Univariate
values = [x for x in range(10)]
data = series_to_supervised(values, 2, 2)
print(data)

#Multi-variate
raw = DataFrame()
raw['ob1'] = [x for x in range(10)]
raw['ob2'] = [x for x in range(50, 60)]
values = raw.values
data = series_to_supervised(values)
print(data)


#Proper Deep and Linear Baselines. (MA, Holts Winters, Holts Linear Trend, SES, WMA, SA, NF, ARIMA)

#Use of AutoArima
#load the data
#data = pd.read_csv('international-airline-passengers.csv')

data = pd.read_csv(file , header =0, parse_dates = [0], index_col = 0, squeeze = True, date_parser = parser)

data = pd.DataFrame(data)


#divide into train and validation set
train = data[:int(0.7*(len(data)))]
valid = data[int(0.7*(len(data))):]

#preprocessing (since arima takes univariate series as input)

#train.drop('Month',axis=1,inplace=True)
#valid.drop('Month',axis=1,inplace=True)

#plotting the data
#train['International airline passengers'].plot()
#valid['International airline passengers'].plot()
#data.plot()
train['Sales'].plot()
valid['Sales'].plot()

#building the model

from pmdarima.arima import auto_arima
model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True)
model.fit(train)


forecast = model.predict(n_periods=len(valid))
forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])

#plot the predictions for validation set
from matplotlib import pyplot as plt
plt.plot(train, label='Train')
plt.plot(valid, label='Valid')
plt.plot(forecast, label='Prediction')
#plt.show()

#calculate rmse
from math import sqrt
from sklearn.metrics import mean_squared_error

rms = sqrt(mean_squared_error(valid,forecast))
print(rms)


#Use more Time-series Evaluation metrics, MAPE, RMSE, MAE, etc.


