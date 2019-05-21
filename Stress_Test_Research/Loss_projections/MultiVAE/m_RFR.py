# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 21:21:26 2019

@author: yifei
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.init as Init
import pandas as pd


from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
#from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error






def parser(x):
    return datetime.strptime('190' + x, '%Y-%m')



class mLinReg(nn.Module):

    # inputs.shape: t * n, t = dim_batch, n = dim_inputs
    def __init__(self, dim_batch, dim_inputs, dim_out, dim_hidden):
        super(mLinReg, self).__init__()
        #self.linreg1 = nn.Linear(dim_inputs, dim_hidden)
        #self.linreg2= nn.Linear(dim_hidden, dim_hidden)
        # self.hidden_linear = nn.Linear(dim_hidden, dim_inputs)
        self.linear = nn.Linear(dim_batch, dim_out)
        self.dim_out = dim_out
        self.dim_batch = dim_batch
        self.dim_inputs = dim_inputs
        self.dim_hidden = dim_hidden
        #nn.Linear.
        # t_constant = 1e-2
        # Init.constant_(self.linreg1.weight_hh.data, t_constant)
        # Init.constant_(self.linreg1.weight_ih.data, t_constant)
        # Init.constant_(self.linreg2.weight_hh.data, t_constant)
        # Init.constant_(self.linreg2.weight_ih.data, t_constant)
        # Init.constant_(self.hidden_linear.weight.data, t_constant)
        # Init.constant_(self.time_linear.weight.data, t_constant)

        # self.h_t1 = torch.zeros(dim_batch, dim_hidden)
        # self.c_t1 = torch.zeros(dim_batch, dim_hidden)
        # self.h_t2 = torch.zeros(dim_batch, dim_hidden)
        # self.c_t2 = torch.zeros(dim_batch, dim_hidden)

    def forward(self, inputs):
        epcho, t, n = inputs.shape
        outputs = torch.zeros(epcho, n, self.dim_out)
        for i in range(0, epcho):
            for j in range(0,n):
            #self.h_t1, self.c_t1 = self.linreg1(inputs[i, :, :], self.h_t1, self.c_t1))
                outputs[i, j, :] = self.linear(inputs[i, :, j])
        #outputs = self.time_linear.forward(outputs)
        return outputs





def train(inputs, targets, epcho, linreg_lr, threshold):
    
    t, m1, n = inputs.shape
    m2 = targets.shape[2]
    dim_hidden = 64

    # series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
    # X = series.values
    # size = int(len(X) * 0.66)
    # train, test = inputs, targets
    # #TODO: Have to add loops for the features, banks, outputs.
    # history = [x for x in train]
    # predictions = list()
    # for t in range(len(test)):
    #     model = ARIMA(history, order=(5, 1, 0))
    #     model_fit = model.fit(disp=0)
    #     output = model_fit.forecast()
    #     yhat = output[0]
    #     predictions.append(yhat)
    #     obs = test[t]
    #     history.append(obs)
    #     print('predicted=%f, expected=%f' % (yhat, obs))
    # error = mean_squared_error(test, predictions)
    # print('Test MSE: %.3f' % error)
    # # plot
    # pyplot.plot(test)
    # pyplot.plot(predictions, color='red')
    # pyplot.show()

    #m_LSTM = mLSTM(m1, n, m2, dim_hidden)
    #model = mLinReg(m1, n, m2, dim_hidden)
    # model = mLinReg(m1, n, m2, dim_hidden)
    # m_loss = torch.nn.MSELoss()
    # m_loss_list = []
    # #print(m_loss)
    # m_optimizer = torch.optim.SGD(model.parameters(), lr=linreg_lr)
    # t_loss = np.inf
    # t_loss_rmse = np.inf
    # for i in range(0, epcho):
    #     m_optimizer.zero_grad()
    #     outputs = model.forward(inputs)
    #     loss = m_loss(outputs, targets)
    #     loss_rmse = torch.sqrt(m_loss(outputs, targets))
    #     loss_rmse.backward(retain_graph=True)
    #     loss.backward(retain_graph=True)
    #     m_optimizer.step()
    #     m_optimizer.zero_grad()
    #     m_loss_list.append(loss.item())
    #     print("LSTM Training Loss at Epoch:",i,"Loss:",str(loss.item()))
    #
    #     if t_loss > loss.data and np.abs(t_loss - loss.data) > threshold:
    #         t_loss = loss.data
    #         t_loss_rmse = loss_rmse
    #     else:
    #         print(loss.item())
    #         print("Done!")
    #         break
    # training_hist = pd.DataFrame(m_loss_list)
    # training_hist.index.name = "EPOCH"
    # training_hist.columns = ["MSE_Loss"]

    return model, loss.data, loss_rmse, training_hist


def predict(model, inputs):
    
    outputs = model.forward(inputs)
    
    return outputs




from sklearn.ensemble import RandomForestRegressor

X, y = make_regression(n_features=4, n_informative=2,
...                        random_state=0, shuffle=False)
>>> regr = RandomForestRegressor(max_depth=2, random_state=0,
...                              n_estimators=100)
>>> regr.fit(X, y)
RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=2,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
           oob_score=False, random_state=0, verbose=0, warm_start=False)
>>> print(regr.feature_importances_)
[0.18146984 0.81473937 0.00145312 0.00233767]
>>> print(regr.predict([[0, 0, 0, 0]]))
[-8.32987858]


def model_random_forecast(Xtrain, Xtest, ytrain):
    X_train = Xtrain
    y_train = ytrain
    rfr = RandomForestRegressor(n_jobs=1, random_state=0)
    param_grid = {'n_estimators': [1000]}
    # 'n_estimators': [1000], 'max_features': [10,15,20,25], 'max_depth':[20,20,25,25,]}
    model = GridSearchCV(estimator=rfr, param_grid=param_grid, n_jobs=1, cv=10, scoring=RMSE)
    model.fit(X_train, y_train)
    print('Random forecast regression...')
    print('Best Params:')
    print(model.best_params_)
    print('Best CV Score:')
    print(-model.best_score_)

    y_pred = model.predict(Xtest)
    return y_pred, -model.best_score_

