# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 21:21:26 2019

@author: yifei
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import tensorflow as tf
from torch.nn import Parameter
from torch import Tensor
import torch.nn.functional as F
import pandas as pd
import math
import numpy as np
# class mLSTM(nn.Module):
#
#     # inputs.shape: t * n, t = dim_batch, n = dim_inputs
#     def __init__(self, dim_batch, dim_inputs, dim_out, dim_hidden):
#
#         super(mLSTM, self).__init__()
#         self.lstm1 = nn.LSTMCell(dim_inputs, dim_hidden)
#         self.lstm2 = nn.LSTMCell(dim_hidden, dim_hidden)
#         self.hidden_linear = nn.Linear(dim_hidden, dim_inputs)
#         self.time_linear = nn.Linear(dim_batch, dim_out)
#
#         t_constant = 1e-2
#         Init.constant_(self.lstm1.weight_hh.data, t_constant)
#         Init.constant_(self.lstm1.weight_ih.data, t_constant)
#         Init.constant_(self.lstm2.weight_hh.data, t_constant)
#         Init.constant_(self.lstm2.weight_ih.data, t_constant)
#         Init.constant_(self.hidden_linear.weight.data, t_constant)
#         Init.constant_(self.time_linear.weight.data, t_constant)
#
#         self.h_t1 = torch.zeros(dim_batch, dim_hidden)
#         self.c_t1 = torch.zeros(dim_batch, dim_hidden)
#         self.h_t2 = torch.zeros(dim_batch, dim_hidden)
#         self.c_t2 = torch.zeros(dim_batch, dim_hidden)
#
#
#     def forward(self, inputs):
#
#         epcho, t, n = inputs.shape
#         outputs = torch.zeros(epcho, n, t)
#         for i in range(0, epcho):
#             self.h_t1, self.c_t1 = self.lstm1(inputs[i,:,:], (self.h_t1, self.c_t1))
#             self.h_t2, self.c_t2 = self.lstm2(self.h_t1, (self.h_t2, self.c_t2))
#             outputs[i, :, :] = torch.t(self.hidden_linear.forward(self.h_t2))
#         outputs = self.time_linear.forward(outputs)
#         return outputs

#
# class GRUCell(nn.Module):
#     """
#     An implementation of GRUCell.
#
#     """
#
#     def __init__(self, input_size, hidden_size, bias=True):
#         super(GRUCell, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.bias = bias
#         self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
#         self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         std = 1.0 / math.sqrt(self.hidden_size)
#         for w in self.parameters():
#             w.data.uniform_(-std, std)
#
#     def forward(self, x, hidden):
#         x = x.view(-1, x.size(1))
#
#         gate_x = self.x2h(x)
#         gate_h = self.h2h(hidden)
#
#         gate_x = gate_x.squeeze()
#         gate_h = gate_h.squeeze()
#
#         i_r, i_i, i_n = gate_x.chunk(3, 1)
#         h_r, h_i, h_n = gate_h.chunk(3, 1)
#
#         resetgate = torch.sigmoid(i_r + h_r)
#         inputgate = torch.sigmoid(i_i + h_i)
#         newgate = torch.tanh(i_n + (resetgate * h_n))
#
#         hy = newgate + inputgate * (hidden - newgate)
#
#         return hy
#
#
# class GRUModel(nn.Module):
#     def __init__(self, layer_dim, input_dim, output_dim, hidden_dim, bias=True):
#         super(GRUModel, self).__init__()
#         self.output_dim = output_dim
#         # Hidden dimensions
#         self.hidden_dim = hidden_dim
#
#         # Number of hidden layers
#         self.layer_dim = layer_dim
#
#         self.gru_cell = GRUCell(input_dim, hidden_dim, layer_dim)
#
#         self.fc = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, x):
#
#         y, t, n = x.shape
#         outputs = torch.zeros(y, n, self.output_dim)
#         print("expected output chape",outputs.shape)
#         #for i in range(0, epcho):
#         #     self.h_t1, self.c_t1 = self.lstm1(inputs[i,:,:], (self.h_t1, self.c_t1))
#         #     self.h_t2, self.c_t2 = self.lstm2(self.h_t1, (self.h_t2, self.c_t2))
#         #     outputs[i, :, :] = torch.t(self.hidden_linear.forward(self.h_t2))
#         #outputs = self.time_linear.forward(outputs)
#         # Initialize hidden state with zeros
#         #######################
#         #  USE GPU FOR MODEL  #
#         #######################
#         # print(x.shape,"x.shape")100, 28, 28
#
#         for i in range(0, y):
#             x_tmp = x[i,:,:]
#             if torch.cuda.is_available():
#                 h0 = Variable(torch.zeros(self.layer_dim, x_tmp.size(0), self.hidden_dim).cuda())
#             else:
#                 h0 = Variable(torch.zeros(self.layer_dim, x_tmp.size(0), self.hidden_dim))
#             outs = []
#             print("hidden, initialized", h0.shape)
#             hn = h0[0, :, :]
#             print("hidden before cell",hn.shape)
#             # print(x[:,:,cnt].shape,x[cnt,:,:].size(0))
#             #print("shape of input", x[i,:,:].shape)
#             for seq in range(x_tmp.size(1)):
#                 input_x = x_[:, seq]
#                 print("Input before GRU Cell", input_x.shape)
#                 hn = self.gru_cell(input_x, hn)
#                 outs.append(hn)
#             out = outs[-1].squeeze()
#             print("hidden", out.shape)
#             out = self.fc(out)
#             print("output",out.shape)
#         #outputs[i, :, :] = torch.t(out)
#
#         # out.size() --> 100, 10
#         return out


class RNN(nn.Module):
    #item_set_size, store_set_size, fam_set_size,cluster_set_size,
    def __init__(self, bs, input_size, output_size, hidden_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.batch_size = bs

#        self.item_encoder = nn.Embedding(item_set_size, 40)
#        self.store_encoder = nn.Embedding(store_set_size, 5)
#        self.fam_encoder = nn.Embedding(fam_set_size, 5)
#        self.cluster_encoder = nn.Embedding(cluster_set_size, 5)
#        self.day_encoder = nn.Embedding(32, 10)
#        self.month_encoder = nn.Embedding(13, 3)

#        self.gru = nn.GRU(input_size + 40 + 5 + 5 + 5 + 10 + 3, hidden_size, n_layers, batch_first=True, dropout=0.2)
        self.gru = nn.GRU(input_size , hidden_size, n_layers, batch_first=True, dropout=0.2)
        self.hidden_regressor = nn.Linear(hidden_size, input_size)
        self.regressor = nn.Linear(bs, output_size)
#####
    # gru = nn.GRU(input_size, hidden_size, n_layers, batch_first=True, dropout=0.2)
    # hidden_regressor = nn.Linear(hidden_size, input_size)
    # regressor = nn.Linear(bs, output_size)

    #input = inputs
####
    def forward(self, input, hidden):#item_tensor, store_tensor, fam_tensor, cluster_tensor, day, month, hidden):
        # embedding = torch.cat((self.item_encoder(item_tensor.squeeze()),
        #                        self.day_encoder(day),
        #                        self.month_encoder(month),
        #                        self.store_encoder(store_tensor.squeeze()), self.fam_encoder(fam_tensor.squeeze()),
        #                        self.cluster_encoder(cluster_tensor.squeeze())), 1)

        #input = torch.cat((input, embedding), 1).unsqueeze(1)
        epcho, t_1, n = input.shape
        outputs = torch.zeros(epcho, n, t_1)
        #input.shape
        #outputs.shape

        #hidden = Variable(torch.zeros(n_layers, bs, hidden_size))
        #hidden.shape
        for i in range(0, epcho):
            #print(input.shape, input[0,:,:].shape, hidden.shape, hidden[0,:,:].shape)
            #output, hidden = self.gru(input, hidden)
            input_tmp = input[i,:,:].unsqueeze(0).transpose(0,1)
            #print(input[0,:,:].shape, input_tmp.shape)
            #input_tmp.shape
            #hidden.shape
            output, hidden = self.gru(input_tmp, hidden)
            #output.shape
            #hidden.shape
            outputs[i,:,:]= torch.t(self.hidden_regressor(hidden)[0,:,:])
        #outputs.shape
        output = self.regressor(outputs)
        #output.shape
        return output#, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, self.batch_size, self.hidden_size))






#
# inputs = inputs_train
# targets = targets_train
# epcho = epoch

def train(inputs, targets, epcho, lstm_lr, threshold):
    
    t, m1, n = inputs.shape
    m2 = targets.shape[2]
    dim_hidden = 64
    #print(t, n, m2, dim_hidden)
    model = RNN(bs = m1, input_size = n, output_size = m2, hidden_size = dim_hidden)
    hidden = model.init_hidden()
    #hidden.shape
    model.zero_grad()

    m_loss = torch.nn.MSELoss()
    m_loss_list = []
    #print(m_loss)
    m_optimizer = torch.optim.ASGD(model.parameters(), lr=lstm_lr)
    t_loss = np.inf
    t_loss_rmse = np.inf

    #inputs = Variable(inputs.view(-1, seq_dim, input_dim))


    for i in range(0, epcho):
        m_optimizer.zero_grad()
        outputs = model(inputs, hidden)
        #print(outputs.shape, targets.shape)
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

    return model, loss.data, loss_rmse, training_hist


def predict(model, inputs):
    hidden = model.init_hidden()
    outputs = model.forward(inputs, hidden)
    
    return outputs

