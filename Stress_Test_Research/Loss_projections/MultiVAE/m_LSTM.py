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


class mLSTM(nn.Module):
    
    # inputs.shape: t * n, t = dim_batch, n = dim_inputs
    def __init__(self, dim_batch, dim_inputs, dim_out, dim_hidden):
        
        super(mLSTM, self).__init__()
        self.lstm1 = nn.LSTMCell(dim_inputs, dim_hidden)
        self.lstm2 = nn.LSTMCell(dim_hidden, dim_hidden)
        self.hidden_linear = nn.Linear(dim_hidden, dim_inputs)
        self.time_linear = nn.Linear(dim_batch, dim_out)
        
        t_constant = 1e-2
        Init.constant_(self.lstm1.weight_hh.data, t_constant)
        Init.constant_(self.lstm1.weight_ih.data, t_constant)
        Init.constant_(self.lstm2.weight_hh.data, t_constant)
        Init.constant_(self.lstm2.weight_ih.data, t_constant)
        Init.constant_(self.hidden_linear.weight.data, t_constant)
        Init.constant_(self.time_linear.weight.data, t_constant)
        
        self.h_t1 = torch.zeros(dim_batch, dim_hidden)
        self.c_t1 = torch.zeros(dim_batch, dim_hidden)
        self.h_t2 = torch.zeros(dim_batch, dim_hidden)
        self.c_t2 = torch.zeros(dim_batch, dim_hidden)
        
        
    def forward(self, inputs):
        
        epcho, t, n = inputs.shape
        outputs = torch.zeros(epcho, n, t)
        for i in range(0, epcho):
            self.h_t1, self.c_t1 = self.lstm1(inputs[i,:,:], (self.h_t1, self.c_t1))
            self.h_t2, self.c_t2 = self.lstm2(self.h_t1, (self.h_t2, self.c_t2))
            outputs[i, :, :] = torch.t(self.hidden_linear.forward(self.h_t2))
        outputs = self.time_linear.forward(outputs)
        return outputs
    
def train(inputs, targets, epcho, lstm_lr, threshold):
    
    t, m1, n = inputs.shape
    m2 = targets.shape[2]
    dim_hidden = 64
       
    m_LSTM = mLSTM(m1, n, m2, dim_hidden)
    m_loss = torch.nn.MSELoss()
    #print(m_loss)
    m_optimizer = torch.optim.SGD(m_LSTM.parameters(), lr=lstm_lr)
    t_loss = np.inf
    
    for i in range(0, epcho):
        m_optimizer.zero_grad()
        outputs = m_LSTM.forward(inputs)
        loss = m_loss(outputs, targets)
        loss.backward(retain_graph=True)
        m_optimizer.step()
    #    print(loss.data)
        if t_loss > loss.data and np.abs(t_loss - loss.data) > threshold:
            t_loss = loss.data
        else:
            #print(loss.data)
            print("Done!")
            break
    
    return m_LSTM, loss.data


# def plot(m_lstm,inputs_eval, targets_eval):
#     with torch.no_grad():
#         prediction = m_lstm.forward(inputs_eval).view(-1)
#         loss = nn.MSELoss(prediction, targets_eval)
#         plt.title("MESLoss: {:.5f}".format(loss))
#         plt.plot(prediction.detach().numpy(), label="pred")
#         plt.plot(targets_eval.detach().numpy(), label="true")
#         plt.legend()
#         plt.show()

def predict(model, inputs):
    
    outputs = model.forward(inputs)
    
    return outputs
    