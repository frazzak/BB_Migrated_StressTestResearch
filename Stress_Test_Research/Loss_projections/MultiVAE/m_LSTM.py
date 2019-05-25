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

class mLSTM(nn.Module):
    
    # inputs.shape: t * n, t = dim_batch, n = dim_inputs
    def __init__(self, dim_batch, dim_inputs, dim_out, dim_hidden):
        
        super(mLSTM, self).__init__()
        #super(mLSTM, lstminit).__init__()
        self.lstm1 = nn.LSTMCell(dim_inputs, dim_hidden)
        self.lstm2 = nn.LSTMCell(dim_hidden, dim_hidden)
        self.hidden_linear = nn.Linear(dim_hidden, dim_inputs)
        self.time_linear = nn.Linear(dim_batch, dim_out)

        # lstm1 = nn.LSTMCell(dim_inputs, dim_hidden)
        # lstm2 = nn.LSTMCell(dim_hidden, dim_hidden)
        # hidden_linear = nn.Linear(dim_hidden, dim_inputs)
        # time_linear = nn.Linear(dim_batch, dim_out)

        t_constant = 1e-2
        Init.constant_(self.lstm1.weight_hh.data, t_constant)
        Init.constant_(self.lstm1.weight_ih.data, t_constant)
        Init.constant_(self.lstm2.weight_hh.data, t_constant)
        Init.constant_(self.lstm2.weight_ih.data, t_constant)
        Init.constant_(self.hidden_linear.weight.data, t_constant)
        Init.constant_(self.time_linear.weight.data, t_constant)
        #
        # Init.constant_(lstm1.weight_hh.data, t_constant)
        # Init.constant_(lstm1.weight_ih.data, t_constant)
        # Init.constant_(lstm2.weight_hh.data, t_constant)
        # Init.constant_(lstm2.weight_ih.data, t_constant)
        # Init.constant_(hidden_linear.weight.data, t_constant)
        # Init.constant_(time_linear.weight.data, t_constant)
        #
        self.h_t1 = torch.zeros(dim_batch, dim_hidden)
        self.c_t1 = torch.zeros(dim_batch, dim_hidden)
        self.h_t2 = torch.zeros(dim_batch, dim_hidden)
        self.c_t2 = torch.zeros(dim_batch, dim_hidden)

        # h_t1 = torch.zeros(dim_batch, dim_hidden)
        # c_t1 = torch.zeros(dim_batch, dim_hidden)
        # h_t2 = torch.zeros(dim_batch, dim_hidden)
        # c_t2 = torch.zeros(dim_batch, dim_hidden)

    def forward(self, inputs):
        
        epcho, t, n = inputs.shape
        outputs = torch.zeros(epcho, n, t)
        for i in range(0, epcho):
            self.h_t1, self.c_t1 = self.lstm1(inputs[i,:,:], (self.h_t1, self.c_t1))
            self.h_t2, self.c_t2 = self.lstm2(self.h_t1, (self.h_t2, self.c_t2))
            outputs[i, :, :] = torch.t(self.hidden_linear.forward(self.h_t2))
        outputs = self.time_linear.forward(outputs)



        for i in range(0, epcho):
            print(inputs[i,:,:].shape, h_t1.shape, c_t1.shape)
            h_t1, c_t1 = lstm1(inputs[i,:,:], (h_t1, c_t1))
            h_t2, c_t2 = lstm2(h_t1, (h_t2, c_t2))
            print(h_t2.shape , hidden_linear.forward(h_t2).shape, torch.t(hidden_linear.forward(h_t2)).shape)
            outputs[i, :, :] = torch.t(hidden_linear.forward(h_t2))

        print(outputs.shape)
        outputs = time_linear.forward(outputs)
        return outputs
    





def train(inputs, targets, epcho, lstm_lr, threshold):
    
    t, m1, n = inputs.shape
    m2 = targets.shape[2]
    dim_hidden = 64
    print(m1, n,m2,dim_hidden)
    m_LSTM = mLSTM(dim_batch= m1, dim_inputs= n,dim_out= m2,dim_hidden= dim_hidden)
    m_loss = torch.nn.MSELoss()
    m_loss_list = []
    #print(m_loss)
    m_optimizer = torch.optim.ASGD(m_LSTM.parameters(), lr=lstm_lr)
    t_loss = np.inf
    t_loss_rmse = np.inf
    for i in range(0, epcho):
        m_optimizer.zero_grad()
        outputs = m_LSTM.forward(inputs)
        loss = m_loss(outputs, targets)
        loss_rmse = torch.sqrt(m_loss(outputs, targets))
        loss_rmse.backward(retain_graph=True)
        loss.backward(retain_graph=True)
        m_optimizer.step()
        m_loss_list.append(loss.item())
        print("LSTM Training Loss at Epoch:",i,"Loss:",str(loss.item()))

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

    return m_LSTM, loss.data, loss_rmse, training_hist


def plot(m_lstm, inputs_eval, targets_eval, figname = None):
     with torch.no_grad():
         prediction = m_lstm.forward(inputs_eval).view(-1)
         loss = torch.nn.MSELoss(prediction, targets_eval)
         plt.title("MESLoss: {:.5f}".format(loss))
         plt.plot(prediction.detach().numpy(), label="pred")
         plt.plot(targets_eval.detach().numpy(), label="true")
         plt.legend()
         if figname is not None:
             plt.savefig("_".join(["./Images/m_LSTM_Plot",figname,".png"]), format = 'png')
         else:
             plt.savefig("_".join(["./Images/m_LSTM_Plot", "temp", ".png"]), format='png')
         plt.show()
         plt.close()
     return("Completed")
def predict(model, inputs):
    
    outputs = model.forward(inputs)
    
    return outputs

