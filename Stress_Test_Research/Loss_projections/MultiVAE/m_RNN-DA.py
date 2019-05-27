# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 21:21:26 2019

@author: yifei
"""

# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import torch.nn as nn
# import torch.nn.init as Init
# import pandas as pd
#
# class mLSTM(nn.Module):
#
#     # inputs.shape: t * n, t = dim_batch, n = dim_inputs
#     def __init__(self, dim_batch, dim_inputs, dim_out, dim_hidden):
#
#         super(mLSTM, self).__init__()
#         #super(mLSTM, lstminit).__init__()
#         self.lstm1 = nn.LSTMCell(dim_inputs, dim_hidden)
#         self.lstm2 = nn.LSTMCell(dim_hidden, dim_hidden)
#         self.hidden_linear = nn.Linear(dim_hidden, dim_inputs)
#         self.time_linear = nn.Linear(dim_batch, dim_out)
#
#         # lstm1 = nn.LSTMCell(dim_inputs, dim_hidden)
#         # lstm2 = nn.LSTMCell(dim_hidden, dim_hidden)
#         # hidden_linear = nn.Linear(dim_hidden, dim_inputs)
#         # time_linear = nn.Linear(dim_batch, dim_out)
#
#         t_constant = 1e-2
#         Init.constant_(self.lstm1.weight_hh.data, t_constant)
#         Init.constant_(self.lstm1.weight_ih.data, t_constant)
#         Init.constant_(self.lstm2.weight_hh.data, t_constant)
#         Init.constant_(self.lstm2.weight_ih.data, t_constant)
#         Init.constant_(self.hidden_linear.weight.data, t_constant)
#         Init.constant_(self.time_linear.weight.data, t_constant)
#         #
#         # Init.constant_(lstm1.weight_hh.data, t_constant)
#         # Init.constant_(lstm1.weight_ih.data, t_constant)
#         # Init.constant_(lstm2.weight_hh.data, t_constant)
#         # Init.constant_(lstm2.weight_ih.data, t_constant)
#         # Init.constant_(hidden_linear.weight.data, t_constant)
#         # Init.constant_(time_linear.weight.data, t_constant)
#         #
#         self.h_t1 = torch.zeros(dim_batch, dim_hidden)
#         self.c_t1 = torch.zeros(dim_batch, dim_hidden)
#         self.h_t2 = torch.zeros(dim_batch, dim_hidden)
#         self.c_t2 = torch.zeros(dim_batch, dim_hidden)
#
#         # h_t1 = torch.zeros(dim_batch, dim_hidden)
#         # c_t1 = torch.zeros(dim_batch, dim_hidden)
#         # h_t2 = torch.zeros(dim_batch, dim_hidden)
#         # c_t2 = torch.zeros(dim_batch, dim_hidden)
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
#
#
#
#         for i in range(0, epcho):
#             print(inputs[i,:,:].shape, h_t1.shape, c_t1.shape)
#             h_t1, c_t1 = lstm1(inputs[i,:,:], (h_t1, c_t1))
#             h_t2, c_t2 = lstm2(h_t1, (h_t2, c_t2))
#             print(h_t2.shape , hidden_linear.forward(h_t2).shape, torch.t(hidden_linear.forward(h_t2)).shape)
#             outputs[i, :, :] = torch.t(hidden_linear.forward(h_t2))
#
#         print(outputs.shape)
#         outputs = time_linear.forward(outputs)
#         return outputs
    
import torch
#from torch import nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

# %import matplotlib
# # matplotlib.use('Agg')
# %matplotlib inline

import datetime as dt, itertools, pandas as pd, matplotlib.pyplot as plt, numpy as np

import utility as util

global logger

util.setup_log()
util.setup_path()
logger = util.logger
text_process.logger = logger

use_cuda = torch.cuda.is_available()
logger.info("Is CUDA available? %s.", use_cuda)
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as tf


def init_hidden(x, hidden_size: int):
    """
    Train the initial value of the hidden state:
    https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
    """
    return Variable(torch.zeros(1, x.size(0), hidden_size))


class Encoder(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, T: int):
        """
        input size: number of underlying factors (81)
        T: number of time steps (10)
        hidden_size: dimension of the hidden state
        """
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T

        self.lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        self.attn_linear = nn.Linear(in_features=2 * hidden_size + T - 1, out_features=1)

    def forward(self, input_data):
        # input_data: (batch_size, T - 1, input_size)
        input_weighted = Variable(torch.zeros(input_data.size(0), self.T - 1, self.input_size))
        input_encoded = Variable(torch.zeros(input_data.size(0), self.T - 1, self.hidden_size))
        # hidden, cell: initial states with dimension hidden_size
        hidden = init_hidden(input_data, self.hidden_size)  # 1 * batch_size * hidden_size
        cell = init_hidden(input_data, self.hidden_size)

        for t in range(self.T - 1):
            # Eqn. 8: concatenate the hidden states with each predictor
            x = torch.cat((hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           input_data.permute(0, 2, 1)), dim=2)  # batch_size * input_size * (2*hidden_size + T - 1)
            # Eqn. 8: Get attention weights
            x = self.attn_linear(x.view(-1, self.hidden_size * 2 + self.T - 1))  # (batch_size * input_size) * 1
            # Eqn. 9: Softmax the attention weights
            attn_weights = tf.softmax(x.view(-1, self.input_size), dim=1)  # (batch_size, input_size)
            # Eqn. 10: LSTM
            weighted_input = torch.mul(attn_weights, input_data[:, t, :])  # (batch_size, input_size)
            # Fix the warning about non-contiguous memory
            # see https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            self.lstm_layer.flatten_parameters()
            _, lstm_states = self.lstm_layer(weighted_input.unsqueeze(0), (hidden, cell))
            hidden = lstm_states[0]
            cell = lstm_states[1]
            # Save output
            input_weighted[:, t, :] = weighted_input
            input_encoded[:, t, :] = hidden

        return input_weighted, input_encoded


class Decoder(nn.Module):

    def __init__(self, encoder_hidden_size: int, decoder_hidden_size: int, T: int, out_feats=1):
        super(Decoder, self).__init__()

        self.T = T
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.attn_layer = nn.Sequential(nn.Linear(2 * decoder_hidden_size + encoder_hidden_size,
                                                  encoder_hidden_size),
                                        nn.Tanh(),
                                        nn.Linear(encoder_hidden_size, 1))
        self.lstm_layer = nn.LSTM(input_size=out_feats, hidden_size=decoder_hidden_size)
        self.fc = nn.Linear(encoder_hidden_size + out_feats, out_feats)
        self.fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, out_feats)

        self.fc.weight.data.normal_()

    def forward(self, input_encoded, y_history):
        # input_encoded: (batch_size, T - 1, encoder_hidden_size)
        # y_history: (batch_size, (T-1))
        # Initialize hidden and cell, (1, batch_size, decoder_hidden_size)
        hidden = init_hidden(input_encoded, self.decoder_hidden_size)
        cell = init_hidden(input_encoded, self.decoder_hidden_size)
        context = Variable(torch.zeros(input_encoded.size(0), self.encoder_hidden_size))

        for t in range(self.T - 1):
            # (batch_size, T, (2 * decoder_hidden_size + encoder_hidden_size))
            x = torch.cat((hidden.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           input_encoded), dim=2)
            # Eqn. 12 & 13: softmax on the computed attention weights
            x = tf.softmax(
                    self.attn_layer(
                        x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size)
                    ).view(-1, self.T - 1),
                    dim=1)  # (batch_size, T - 1)

            # Eqn. 14: compute context vector
            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :]  # (batch_size, encoder_hidden_size)

            # Eqn. 15
            y_tilde = self.fc(torch.cat((context, y_history[:, t]), dim=1))  # (batch_size, out_size)
            # Eqn. 16: LSTM
            self.lstm_layer.flatten_parameters()
            _, lstm_output = self.lstm_layer(y_tilde.unsqueeze(0), (hidden, cell))
            hidden = lstm_output[0]  # 1 * batch_size * decoder_hidden_size
            cell = lstm_output[1]  # 1 * batch_size * decoder_hidden_size

        # Eqn. 22: final output
        return self.fc_final(torch.cat((hidden[0], context), dim=1))



# Train the model
class da_rnn:
    def __init__(self, file_data, logger, encoder_hidden_size = 64, decoder_hidden_size = 64, T = 10,
                 learning_rate = 0.01, batch_size = 128, parallel = True, debug = False):
        self.T = T
        dat = pd.read_csv(file_data, nrows = 100 if debug else None)
        self.logger = logger
        self.logger.info("Shape of data: %s.\nMissing in data: %s.", dat.shape, dat.isnull().sum().sum())

        self.X = dat.loc[:, [x for x in dat.columns.tolist() if x != 'NDX']].as_matrix()
        self.y = np.array(dat.NDX)
        self.batch_size = batch_size

        self.encoder = encoder(input_size = self.X.shape[1], hidden_size = encoder_hidden_size, T = T,
                              logger = logger).cuda()
        self.decoder = decoder(encoder_hidden_size = encoder_hidden_size,
                               decoder_hidden_size = decoder_hidden_size,
                               T = T, logger = logger).cuda()

        if parallel:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)

        self.encoder_optimizer = optim.Adam(params = itertools.ifilter(lambda p: p.requires_grad, self.encoder.parameters()),
                                           lr = learning_rate)
        self.decoder_optimizer = optim.Adam(params = itertools.ifilter(lambda p: p.requires_grad, self.decoder.parameters()),
                                           lr = learning_rate)
        # self.learning_rate = learning_rate

        self.train_size = int(self.X.shape[0] * 0.7)
        self.y = self.y - np.mean(self.y[:self.train_size]) # Question: why Adam requires data to be normalized?
        self.logger.info("Training size: %d.", self.train_size)

    def train(self, n_epochs = 10):
        iter_per_epoch = int(np.ceil(self.train_size * 1. / self.batch_size))
        logger.info("Iterations per epoch: %3.3f ~ %d.", self.train_size * 1. / self.batch_size, iter_per_epoch)
        self.iter_losses = np.zeros(n_epochs * iter_per_epoch)
        self.epoch_losses = np.zeros(n_epochs)

        self.loss_func = nn.MSELoss()

        n_iter = 0

        learning_rate = 1.

        for i in range(n_epochs):
            perm_idx = np.random.permutation(self.train_size - self.T)
            j = 0
            while j < self.train_size:
                batch_idx = perm_idx[j:(j + self.batch_size)]
                X = np.zeros((len(batch_idx), self.T - 1, self.X.shape[1]))
                y_history = np.zeros((len(batch_idx), self.T - 1))
                y_target = self.y[batch_idx + self.T]

                for k in range(len(batch_idx)):
                    X[k, :, :] = self.X[batch_idx[k] : (batch_idx[k] + self.T - 1), :]
                    y_history[k, :] = self.y[batch_idx[k] : (batch_idx[k] + self.T - 1)]

                loss = self.train_iteration(X, y_history, y_target)
                self.iter_losses[i * iter_per_epoch + j / self.batch_size] = loss
                #if (j / self.batch_size) % 50 == 0:
                #    self.logger.info("Epoch %d, Batch %d: loss = %3.3f.", i, j / self.batch_size, loss)
                j += self.batch_size
                n_iter += 1

                if n_iter % 10000 == 0 and n_iter > 0:
                    for param_group in self.encoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9
                    for param_group in self.decoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9

            self.epoch_losses[i] = np.mean(self.iter_losses[range(i * iter_per_epoch, (i + 1) * iter_per_epoch)])
            if i % 10 == 0:
                self.logger.info("Epoch %d, loss: %3.3f.", i, self.epoch_losses[i])

            if i % 10 == 0:
                y_train_pred = self.predict(on_train = True)
                y_test_pred = self.predict(on_train = False)
                y_pred = np.concatenate((y_train_pred, y_test_pred))
                plt.figure()
                plt.plot(range(1, 1 + len(self.y)), self.y, label = "True")
                plt.plot(range(self.T , len(y_train_pred) + self.T), y_train_pred, label = 'Predicted - Train')
                plt.plot(range(self.T + len(y_train_pred) , len(self.y) + 1), y_test_pred, label = 'Predicted - Test')
                plt.legend(loc = 'upper left')
                plt.show()

    def train_iteration(self, X, y_history, y_target):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_weighted, input_encoded = self.encoder(Variable(torch.from_numpy(X).type(torch.FloatTensor).cuda()))
        y_pred = self.decoder(input_encoded, Variable(torch.from_numpy(y_history).type(torch.FloatTensor).cuda()))

        y_true = Variable(torch.from_numpy(y_target).type(torch.FloatTensor).cuda())
        loss = self.loss_func(y_pred, y_true)
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.data[0]

    def predict(self, on_train = False):
        if on_train:
            y_pred = np.zeros(self.train_size - self.T + 1)
        else:
            y_pred = np.zeros(self.X.shape[0] - self.train_size)

        i = 0
        while i < len(y_pred):
            batch_idx = np.array(range(len(y_pred)))[i : (i + self.batch_size)]
            X = np.zeros((len(batch_idx), self.T - 1, self.X.shape[1]))
            y_history = np.zeros((len(batch_idx), self.T - 1))
            for j in range(len(batch_idx)):
                if on_train:
                    X[j, :, :] = self.X[range(batch_idx[j], batch_idx[j] + self.T - 1), :]
                    y_history[j, :] = self.y[range(batch_idx[j],  batch_idx[j]+ self.T - 1)]
                else:
                    X[j, :, :] = self.X[range(batch_idx[j] + self.train_size - self.T, batch_idx[j] + self.train_size - 1), :]
                    y_history[j, :] = self.y[range(batch_idx[j] + self.train_size - self.T,  batch_idx[j]+ self.train_size - 1)]

            y_history = Variable(torch.from_numpy(y_history).type(torch.FloatTensor).cuda())
            _, input_encoded = self.encoder(Variable(torch.from_numpy(X).type(torch.FloatTensor).cuda()))
            y_pred[i:(i + self.batch_size)] = self.decoder(input_encoded, y_history).cpu().data.numpy()[:, 0]
            i += self.batch_size
        return y_pred



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

