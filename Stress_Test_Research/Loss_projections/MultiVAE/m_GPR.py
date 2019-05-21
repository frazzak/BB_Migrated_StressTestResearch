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
from statsmodels.tsa.ar_model import AR
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


print(__doc__)

# Author: Vincent Dubourg <vincent.dubourg@gmail.com>
#         Jake Vanderplas <vanderplas@astro.washington.edu>
#         Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>s
# License: BSD 3 clause

import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

np.random.seed(1)


def f(x):
    """The function to predict."""
    return x * np.sin(x)

# ----------------------------------------------------------------------
#  First the noiseless case
X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T

# Observations
y = f(X).ravel()

# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
x = np.atleast_2d(np.linspace(0, 10, 1000)).T

# Instantiate a Gaussian Process model
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(x, return_std=True)

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
plt.figure()
plt.plot(x, f(x), 'r:', label=r'$f(x) = x\,\sin(x)$')
plt.plot(X, y, 'r.', markersize=10, label='Observations')
plt.plot(x, y_pred, 'b-', label='Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')

# ----------------------------------------------------------------------
# now the noisy case
X = np.linspace(0.1, 9.9, 20)
X = np.atleast_2d(X).T

# Observations and noise
y = f(X).ravel()
dy = 0.5 + 1.0 * np.random.random(y.shape)
noise = np.random.normal(0, dy)
y += noise

# Instantiate a Gaussian Process model
gp = GaussianProcessRegressor(kernel=kernel, alpha=dy ** 2,
                              n_restarts_optimizer=10)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(x, return_std=True)

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
plt.figure()
plt.plot(x, f(x), 'r:', label=r'$f(x) = x\,\sin(x)$')
plt.errorbar(X.ravel(), y, dy, fmt='r.', markersize=10, label='Observations')
plt.plot(x, y_pred, 'b-', label='Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')

plt.show()