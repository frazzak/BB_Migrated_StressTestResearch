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




# dim_batch = m1
# dim_inputs = n
# dim_out = m2
# dim_hidden
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
                print("LinReg Bank Count:", j, "Quarter Number:",i)
                outputs[i, j, :] = self.linear(inputs[i, :, j])
        #outputs = self.time_linear.forward(outputs)
        return outputs





def train(inputs, targets, epcho, linreg_lr, threshold):
    
    t, m1, n = inputs.shape
    m2 = targets.shape[2]
    dim_hidden = 64
       
    #m_LSTM = mLSTM(m1, n, m2, dim_hidden)
    #model = mLinReg(m1, n, m2, dim_hidden)
    model = mLinReg(m1, n, m2, dim_hidden)
    m_loss = torch.nn.MSELoss()
    m_loss_list = []
    #print(m_loss)
    m_optimizer = torch.optim.SGD(model.parameters(), lr=linreg_lr)
    t_loss = np.inf
    t_loss_rmse = np.inf
    for i in range(0, epcho):
        print("LinReg Training Epoch:", i)
        m_optimizer.zero_grad()
        outputs = model.forward(inputs)
        loss = m_loss(outputs, targets)
        loss_rmse = torch.sqrt(m_loss(outputs, targets))
        loss_rmse.backward(retain_graph=True)
        loss.backward(retain_graph=True)
        m_optimizer.step()
        m_optimizer.zero_grad()
        m_loss_list.append(loss.item())
        print("LinReg Training Loss at Epoch:",i,"Loss:",str(loss.item()))

        if t_loss > loss.data and np.abs(t_loss - loss.data) > threshold:
            t_loss = loss.data
            t_loss_rmse = loss_rmse
        else:
            print(loss.item())
            print("Done!")
            break
    training_hist = pd.DataFrame(m_loss_list)
    training_hist.index.name = "EPOCH"
    training_hist.columns = ["MSE_Loss"]

    return model, loss.data, loss_rmse, training_hist


def predict(model, inputs):
    
    outputs = model.forward(inputs)
    
    return outputs



# from sklearn.linear_model import LinearRegression
# >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# >>> # y = 1 * x_0 + 2 * x_1 + 3
# >>> y = np.dot(X, np.array([1, 2])) + 3
# >>> reg = LinearRegression().fit(X, y)
# >>> reg.score(X, y)
# 1.0
# >>> reg.coef_
# array([1., 2.])
# >>> reg.intercept_
# 3.0000...
# >>> reg.predict(np.array([[3, 5]]))
# array([16.])