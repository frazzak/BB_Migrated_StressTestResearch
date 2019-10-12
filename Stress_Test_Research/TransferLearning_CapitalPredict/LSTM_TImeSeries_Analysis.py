from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
import matplotlib
import numpy
from numpy import concatenate
import os,sys

def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')



#make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1,1, len(X))
    yhat = model.predict(X, batch_size = batch_size)
    return yhat[0,0]


#fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:,0:-1], train[:,-1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful = True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer = 'adam')
    for i in range(nb_epoch):
            model.fit(X,y, epochs = 1, batch_size = batch_size, verbose = 0, shuffle = False)
            model.reset_states()
    return model


def scale(train, test):
    #fit scaler
    scaler   = MinMaxScaler(feature_range=(-1,1))
    scaler = scaler.fit(train)
    #transfrom train
    train = train.reshape(train.shape[0],train.shape[1])
    train_scaled = scaler.transform(train)
    #transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


#frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag =1 ):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis = 1)
    return df

#create differenced series
def difference(dataset, interval = 1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] = dataset[i - interval]
        diff.append(value)
    return Series(diff)

def experiment(repeats, series, features):
    #transform data to be stationary
    raw_values = series.values
    diff_values = difference(raw_values, 1)

    #transfrom data to be supervised learning
    supervised = timeseries_to_supervised(diff_values, features)
    supervised_values = supervised.values[features:,:]

    #split data into train and test-sets
    train, test = supervised_values[0:-12,:], supervised_values[-12:,:]

    #transform the scale of the data
    scaler, train_scaled, test_scaled = scale(train,test)

    #run experiment
    error_scores = list()
    #fit the base model
    lstm_model = fit_lstm(train_scaled, 1, 500, 1)
    #forecast test dataset
    predictions = list()
    for i in range(len(test_scaled)):
        #predict
        X,y = test_Scaled[i,0:-1], test_scaled[i,-1]
        yhat = forecast_lstm(lstm_model, 1, X)

# execute experiment
def run():
        #load dataset
        series = read_csv('/Users/phn1x/icdm2018_research_BB/Stress_Test_Research/TransferLearning_CapitalPredict/data/shampoo.csv', header =0, parse_dates = [0], index_col = 0, squeeze = True, date_parser = parser)

        #summarize first few rows
        #print(series.head())
        #line plot
       # series.plot()


        #experiment
        repeats = 10
        results = DataFrame()
        #run experiment
        features = 1
        results['results'] = experiment(repeats, series, features)