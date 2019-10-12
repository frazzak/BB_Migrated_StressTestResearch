import typing
from typing import Tuple
import json
import os
import gc
import torch
#from torch import nn
from torch.autograd import Variable
from torch.nn import functional as tf


import torch
from torch import nn
from torch import optim
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import da_rnn_master.utils as utils
#from da_rnn_master.modules import Encoder, Decoder
from da_rnn_master.custom_types import DaRnnNet, TrainData, TrainConfig
from da_rnn_master.utils import numpy_to_tvar
from da_rnn_master.constants import device

logger = utils.setup_log()
logger.info(f"Using computation device: {device}")






def preprocess_data(dat, col_names) -> Tuple[TrainData, StandardScaler]:
    scale = StandardScaler().fit(dat)
    proc_dat = scale.transform(dat)

    mask = np.ones(proc_dat.shape[1], dtype=bool)
    dat_cols = list(dat.columns)
    for col_name in col_names:
        mask[dat_cols.index(col_name)] = False

    feats = proc_dat[:, mask]
    targs = proc_dat[:, ~mask]

    return TrainData(feats, targs), scale

# inputs  = inputs_train
# targets = targets_train
# n_epochs = epoch
# lr = lstm_lr
#threshold

def train(inputs, targets, n_epochs, lr, threshold,save_plots=False, debug = False):
    #Iniit
    #logger = utils.setup_log(tag = "Dual-Attention RNN")
    #logger.info(f"Using computation device: {device}")
    #epoch = 10
    learning_rate = lr

    logger.info(f"Shape of train data: {inputs.shape}.\nShape of target data: {targets.shape}.")
    data = TrainData(inputs, targets)
    t, m1, n = data.feats.shape
    m2 = data.targs.shape[2]
    dim_hidden = 64
    da_rnn_kwargs = {"batch_size": m1, "T": 4}
    #da_rnn_kwargs = {"batch_size": n, "T": 4}
    config, model = da_rnn(data, n_targs=m2, learning_rate=learning_rate, **da_rnn_kwargs)

    ##Train
    net = model
    train_data = data
    t_cfg = config


    iter_per_epoch = 1 #int(np.ceil(t_cfg.train_size * 1. / t_cfg.batch_size))
    iter_losses = np.zeros(n_epochs * iter_per_epoch)
    epoch_losses = np.zeros(n_epochs)
    logger.info(f"Iterations per epoch: {t_cfg.train_size * 1. / t_cfg.batch_size:3.3f} ~ {iter_per_epoch:d}.")
    t_loss = np.inf
    n_iter = 0
    #e_i = 0
    for e_i in range(n_epochs):
        batch_idx = range(0,train_data.feats.shape[0])
        feats = np.zeros((len(batch_idx), train_data.feats.shape[1],train_data.feats.shape[2]))
        y_history = np.zeros((len(batch_idx), train_data.targs.shape[1],train_data.targs.shape[2]))
        y_target = train_data.targs
        feats[:, :, :] = train_data.feats[:, :, :]
        y_history[:,:] = train_data.targs

        # y_history.shape
        # feats.shape
        # y_target.shape
        loss = train_iteration(net, t_cfg.loss_func, feats, y_history, y_target)

        iter_losses[e_i * iter_per_epoch // t_cfg.batch_size] = loss
        n_iter += 1
        adjust_learning_rate(net, n_iter)


        epoch_losses[e_i] = np.mean(iter_losses[range(e_i * iter_per_epoch, (e_i + 1) * iter_per_epoch)])
        #logger.info(f"Epoch {e_i:d}, train loss: {epoch_losses[e_i]:4.8f}")
        # if e_i % 10 == 0:
        y_test_pred = predict(net, train_data,t_cfg.train_size, t_cfg.batch_size, t_cfg.T, on_train=False)
        #     # TODO: make this MSE and make it work for multiple inputs
        #val_loss = y_test_pred - train_data.targs[t_cfg.train_size:]
        #y_test_pred = numpy_to_tvar(y_test_pred)
        val_loss = y_test_pred - train_data.targs
        pred_loss = t_cfg.loss_func(y_test_pred, train_data.targs)
        #pred_loss
        logger.info(f"Epoch {e_i:d}, train loss: {epoch_losses[e_i]:4.8f},pred loss: {pred_loss:4.8f},pred_loss_rmse: {torch.sqrt(pred_loss):4.8f}, val loss: {np.abs(val_loss).mean()}.")
        y_train_pred = predict(net, train_data,
                               t_cfg.train_size, t_cfg.batch_size, t_cfg.T,
                               on_train=True)

        if t_loss > loss.data and np.abs(t_loss - loss.data) > threshold:
            t_loss = loss.data

        else:
            print(loss.item())
            print("Done!")
            break

    train_loss = pred_loss
    train_rmse = torch.sqrt(pred_loss)
    training_hist = pd.DataFrame(epoch_losses)
    training_hist.index.name = "EPOCH"
    training_hist.columns = ["MSE_Loss"]

    return model,train_loss, train_rmse, training_hist


def prep_train_data(batch_idx: np.ndarray, t_cfg: TrainConfig, train_data: TrainData):
    #feats = np.zeros((len(batch_idx), t_cfg.T - 1, train_data.feats.shape[1], train_data.feats.shape[2]))
    feats = np.zeros((len(batch_idx), t_cfg.T - 1, train_data.feats.shape[1], train_data.feats.shape[2]))
    #y_history = np.zeros((len(batch_idx), t_cfg.T - 1, train_data.targs.shape[1], train_data.targs.shape[2]))
    y_history = np.zeros((len(batch_idx), t_cfg.T - 1, train_data.targs.shape[1], train_data.targs.shape[2]))
    #y_target = train_data.targs[batch_idx + t_cfg.T]
    y_target = train_data.targs[batch_idx + t_cfg.T]

    for b_i, b_idx in enumerate(batch_idx):
        # print(b_i, b_idx)
        #b_slc = slice(b_idx, b_idx + t_cfg.T - 1)
        #train_data.feats.shape
        feats[b_i, :, :, :] = train_data.feats[b_slc, :, :]
        y_history[b_i, :] = train_data.targs[b_slc]



    return feats, y_history, y_target


def adjust_learning_rate(net: DaRnnNet, n_iter: int):
    # TODO: Where did this Learning Rate adjustment schedule come from?
    # Should be modified to use Cosine Annealing with warm restarts https://www.jeremyjordan.me/nn-learning-rate/
    if n_iter % 10000 == 0 and n_iter > 0:
        for enc_params, dec_params in zip(net.enc_opt.param_groups, net.dec_opt.param_groups):
            enc_params['lr'] = enc_params['lr'] * 0.9
            dec_params['lr'] = dec_params['lr'] * 0.9



def train_iteration(t_net: DaRnnNet, loss_func: typing.Callable, X, y_history, y_target):
    t_net.enc_opt.zero_grad()
    t_net.dec_opt.zero_grad()

    #X.shape
    #numpy_to_tvar(X).shape
    input_weighted, input_encoded = t_net.encoder(numpy_to_tvar(X))
    #input_encoded[4][4][63]
    y_pred = t_net.decoder(input_encoded, numpy_to_tvar(y_history))


    #y_true = numpy_to_tvar(y_target)
    y_true = y_target
    loss = loss_func(y_pred, y_true)
    loss.backward()

    t_net.enc_opt.step()
    t_net.dec_opt.step()

    return loss

#

def predict(model: DaRnnNet, t_dat: TrainData = None, train_size: int = None, batch_size: int = None, T: int = None, inputs = None, targets = None, on_train=False):

    t_net = model
    if t_dat is not None and type(t_dat) == TrainData:
        0+0
        #print("Using TrainData Class")
    elif inputs is not None and targets is not None:
        t_dat = TrainData(inputs,targets)
    else:
        print("Data Not in proper format")
        return

    bank_size = t_dat.targs.shape[1]
    out_dim = t_dat.targs.shape[2]
    if on_train:
        #y_pred = np.zeros((train_size - T + 1, out_size))
        y_pred = np.zeros(tuple(t_dat.targs.shape))
    else:
        #y_pred = np.zeros((t_dat.feats.shape[0] - train_size, out_size))
        #y_pred = np.zeros((t_dat.feats.shape[0] - train_size, bank_size, out_dim))
        y_pred = np.zeros(tuple(t_dat.targs.shape))

        #y_pred.shape

    #for y_i in range(0, len(y_pred), batch_size):
        #y_slc = slice(y_i, y_i + batch_size)
        #batch_idx = range(len(y_pred))[y_slc]
        #b_len = len(batch_idx)
    X = np.zeros(tuple(t_dat.feats.shape))
    y_history = np.zeros(tuple(t_dat.targs.shape))


    #    for b_i, b_idx in enumerate(batch_idx):
    # if on_train:
    #     idx = range(b_idx, b_idx + T - 1)
    # else:
    #     idx = range(b_idx + train_size - T, b_idx + train_size - 1)

    X = t_dat.feats
    y_history = t_dat.targs

    #y_history = numpy_to_tvar(y_history)
    #_, input_encoded = t_net.encoder(numpy_to_tvar(X))
    _, input_encoded = t_net.encoder(X)
    y_pred = t_net.decoder(input_encoded, y_history).cpu().data.numpy()

    return numpy_to_tvar(y_pred)



def init_hidden(x, hidden_size: int):
    """
    Train the initial value of the hidden state:
    https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
    """
    return Variable(torch.zeros(1, x.size(0), hidden_size))


class Encoder(nn.Module):

    def __init__(self, input_size: int, input_size_attn: int, hidden_size: int, T: int):
        """
        input size: number of underlying factors (81)
        T: number of time steps (10)
        hidden_size: dimension of the hidden state
        """
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T

#       input_size = train_data.feats.shape[2]
#        input_size_attn = train_data.feats.shape[1]
#        hidden_size = 64
        #T = 10
        self.lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)

        #self.attn_linear = nn.Linear(in_features=2 * hidden_size + T - 1, out_features=1)
        self.attn_linear = nn.Linear(in_features=2 * hidden_size + input_size_attn, out_features=1)

        #lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        #attn_linear = nn.Linear(in_features= 2 * hidden_size + input_size_attn , out_features=1)
        #del attn_linear

    #input_data = X_tvar
    #T = 10
    #input_data.shape
    def forward(self, input_data):
        # input_data: (batch_size, T - 1, input_size)
        # input_weighted = Variable(torch.zeros(input_data.size(0), self.T - 1, self.input_size))
        # input_encoded = Variable(torch.zeros(input_data.size(0), self.T - 1, self.hidden_size))

#        else:
        input_weighted = Variable(torch.zeros(input_data.size(0), self.input_size,input_data.shape[1]))

        input_encoded = Variable(torch.zeros(input_data.size(0),self.input_size, self.hidden_size))

        #if input_data.shape[1] != input_encoded.shape[1]:
        #    input_data =  input_data.permute(0, 2, 1)



        #input_weighted = Variable(torch.zeros(input_data.size(0), input_size ,input_data.shape[1]))
        #input_encoded = Variable(torch.zeros(input_data.size(0),input_size, hidden_size))

        #input_weighted.shape

        hidden = init_hidden(input_data, self.hidden_size)  # 1 * batch_size * hidden_size
        cell = init_hidden(input_data, self.hidden_size)

        #hidden = init_hidden(input_data, hidden_size)  # 1 * batch_size * hidden_size
        #cell = init_hidden(input_data, hidden_size)

        #cell.shape

        # hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2),
        # cell.repeat(self.input_size, 1, 1).permute(1, 0, 2),
        # input_data.permute(0, 2, 1)), dim = 2



# #        for t in range(self.T - 1):
        for t in range(0,min([input_data.shape[1],input_encoded.shape[1]])):
            #print(t)
            #Eqn. 8: concatenate the hidden states with each predictor
            #TODO, handling for quarterly, banksplit.  Find proper logic to handle.
            x = torch.cat((hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           input_data.permute(0, 2, 1)), dim=2)  # batch_size * input_size * (2*hidden_size + T - 1)

            # x = torch.cat((hidden.repeat(input_size, 1, 1).permute(1, 0, 2),
            #                cell.repeat(input_size, 1, 1).permute(1, 0, 2),
            #                input_data.permute(0, 2, 1)), dim=2)  # batch_size * input_size * (2*hidden_size + T - 1)


            #print("x:", x.shape)
            #Eqn. 8: Get attention weights
            x = self.attn_linear(x.view(-1, self.hidden_size * 2 + input_data.size(1) * 1))  # (batch_size * input_size) * 1

            #x = attn_linear(x.view(-1, hidden_size * 2 + input_data.size(1) * 1))  # (batch_size * input_size) * 1


            #print("attn_linear x:", x.shape)
            # Eqn. 9: Softmax the attention weights
            attn_weights = tf.softmax(x.view(-1, self.input_size), dim=1)  # (batch_size, input_size)
            #attn_weights = tf.softmax(x.view(-1, input_size), dim=1)
            #print("attn_weights:", attn_weights.shape)
            # Eqn. 10: LSTM
            #t = 0
            weighted_input = torch.mul(attn_weights, input_data[:,t,:])  # (batch_size, input_size)
            #print("weighted_input:", weighted_input.shape)
            del attn_weights, x
            # Fix the warning about non-contiguous memory
            # see https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            self.lstm_layer.flatten_parameters()
            _, lstm_states = self.lstm_layer(weighted_input.unsqueeze(0), (hidden, cell))
            #lstm_layer.flatten_parameters()
            #_, lstm_states = lstm_layer(weighted_input.unsqueeze(0), (hidden, cell))


            hidden = lstm_states[0]
            #print("hidden:", hidden.shape)
            cell = lstm_states[1]
            #print("cell:", cell.shape)
            del _, lstm_states
            # Save output



            input_weighted[:, :, t] = weighted_input
            #print(t, input_data.shape,input_encoded.shape, hidden.shape)
            input_encoded[:, t, :] = hidden
            gc.collect()
            torch.cuda.empty_cache()
        return input_weighted, input_encoded


class Decoder(nn.Module):

    def __init__(self, encoder_hidden_size: int, decoder_hidden_size: int, T: int, out_feats=1):
        super(Decoder, self).__init__()
        self.n_targs = out_feats
        self.T = T
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.attn_layer = nn.Sequential(nn.Linear(2 * decoder_hidden_size + encoder_hidden_size,
                                                  encoder_hidden_size),
                                        nn.Tanh(),
                                        nn.Linear(encoder_hidden_size, 1))
        # #attn_layer = nn.Sequential(nn.Linear(2 * decoder_hidden_size + encoder_hidden_size,
        #                                      encoder_hidden_size),
        #                            nn.Tanh(),
        #                            nn.Linear(encoder_hidden_size, 1))

        self.lstm_layer = nn.LSTM(input_size=out_feats, hidden_size=decoder_hidden_size)
        #lstm_layer = nn.LSTM(input_size=out_feats, hidden_size=decoder_hidden_size)

        self.fc = nn.Linear(encoder_hidden_size + out_feats, out_feats)
        #fc1 = nn.Linear(encoder_hidden_size + out_feats, input_size)



        self.fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, out_feats)
        #fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, out_feats)

        self.fc.weight.data.normal_()

    # input_encoded,
    # Ytvar = numpy_to_tvar(y_history)
    #dec_kwargs = {"encoder_hidden_size": encoder_hidden_size,
    #              "decoder_hidden_size": decoder_hidden_size, "T": T, "out_feats": n_targs}
    #encoder_hidden_size = 64
    #decoder_hidden_size = 64
    #out_feats = n_targs


    def forward(self, input_encoded, y_history):
        # input_encoded: (batch_size, T - 1, encoder_hidden_size)
        # y_history: (batch_size, (T-1))
        # Initialize hidden and cell, (1, batch_size, decoder_hidden_size)
        # hidden = init_hidden(input_encoded, self.decoder_hidden_size)
        # cell = init_hidden(input_encoded, self.decoder_hidden_size)
        # context = Variable(torch.zeros(input_encoded.size(0), self.encoder_hidden_size))
        #
        final_output = Variable(torch.zeros(input_encoded.size(0), input_encoded.size(1), self.n_targs))
        hidden = Variable(torch.zeros(input_encoded.size(1), self.decoder_hidden_size))
        cell = Variable(torch.zeros(input_encoded.size(1), self.decoder_hidden_size))
        context = Variable(torch.zeros(input_encoded.size(0), self.encoder_hidden_size))

        # for t in range(self.T - 1):
        #     # (batch_size, T, (2 * decoder_hidden_size + encoder_hidden_size))
        #     x = torch.cat((hidden.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
        #                    cell.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
        #                    input_encoded), dim=2)


        #t = 0
        #input_encoded.shape
        #t = 0


        for t in range(0,input_encoded.shape[0]):
            #print(t)
            # (batch_size, T, (2 * decoder_hidden_size + encoder_hidden_size))

            x = torch.cat((hidden.repeat(1, 1),
                           cell.repeat(1, 1),
                           input_encoded[t,:,:]), dim=1)
            #print("Cell, Input, Hidden merge:", x.shape)
            #x.shape
            # Eqn. 12 & 13: softmax on the computed attention weights
            # x = tf.softmax(
            #         self.attn_layer(
            #             x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size)
            #         ).view(-1, self.T - 1),
            #         dim=1)

            x = tf.softmax(self.attn_layer(x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size)).view(-1,input_encoded.shape[1]), dim = 1)
            #print("Softmax on computed weights", x.shape)
            # x = tf.softmax(
            #    attn_layer(
            #         x.view(-1, 2 * decoder_hidden_size + encoder_hidden_size)
            #     ).view(-1, input_encoded.shape[2]),
            #     dim=1)                        # (batch_size, T - 1)
            #x.shape

            # Eqn. 14: compute context vector

            #context = torch.bmm(x.unsqueeze(1), input_encoded[t,:,:])[:, 0, :]  # (batch_size, encoder_hidden_size)
            context = torch.bmm(x.unsqueeze(1), input_encoded[t,:,:].unsqueeze(0)) [:,0,:]
            #print("Softmax on context vector", context.shape)

            #context.shape
            # Eqn. 15

            #y_tilde = self.fc(torch.cat((context, y_history[:, t]), dim=1))  # (batch_size, out_size)
            y_tilde = self.fc(torch.cat((context.repeat(input_encoded.shape[1],1), y_history[t, :]), dim=1))  # (batch_size, out_size)
            #print("Y_tilde", y_tilde.shape)
            #y_tilde1 = fc1(torch.cat((context.repeat(input_encoded.shape[1],1), Ytvar[t, :]), dim=1))
            #y_tilde.shape
            #y_tilde1.shape
            #fc(torch.cat((context, Ytvar[:, t]), dim=1)).shape
            # Eqn. 16: LSTM
            #self.lstm_layer.flatten_parameters()
            self.lstm_layer.flatten_parameters()
            _, lstm_output = self.lstm_layer(y_tilde.unsqueeze(0), (hidden.unsqueeze(0), cell.unsqueeze(0)))
            hidden = lstm_output[0]  # 1 * batch_size * decoder_hidden_size
            cell = lstm_output[1]
            #print("Hidden Output LSTM:",hidden.shape)
            # Eqn. 22: final output
            final_output[t,:,:] = self.fc_final(torch.cat((hidden[0], context.repeat(input_encoded.shape[1],1)), dim=1))
            #print("Final Output Entered")
            hidden = hidden.squeeze(0)
            cell = cell.squeeze(0)
            gc.collect()


        #return self.fc_final(torch.cat((hidden[0], context), dim=1))
        return final_output




# train_data = data
# batch_size = m1

def da_rnn(train_data: TrainData, n_targs: int, encoder_hidden_size=64, decoder_hidden_size=64,
           T=10, learning_rate=0.01, batch_size=128):


    train_cfg = TrainConfig(T, int(train_data.feats.shape[0] * 0.7), batch_size, nn.MSELoss())
    #logger.info(f"Training size: {train_cfg.train_size:d}.")

    #TODO: Need switch to handle bank split quarterly vs. timesplits.
    enc_kwargs = {"input_size": train_data.feats.shape[2], "input_size_attn": train_data.feats.shape[1],"hidden_size": encoder_hidden_size, "T": T}
    encoder = Encoder(**enc_kwargs).to(device)
    with open(os.path.join("data", "enc_kwargs.json"), "w") as fi:
        json.dump(enc_kwargs, fi, indent=4)

    dec_kwargs = {"encoder_hidden_size": encoder_hidden_size,
                  "decoder_hidden_size": decoder_hidden_size, "T": T, "out_feats": n_targs}
    decoder = Decoder(**dec_kwargs).to(device)

    with open(os.path.join("data", "dec_kwargs.json"), "w") as fi:
        json.dump(dec_kwargs, fi, indent=4)

    encoder_optimizer = optim.Adam(
        params=[p for p in encoder.parameters() if p.requires_grad],
        lr=learning_rate)

    decoder_optimizer = optim.Adam(
        params=[p for p in decoder.parameters() if p.requires_grad],
        lr=learning_rate)

    da_rnn_net = DaRnnNet(encoder, decoder, encoder_optimizer, decoder_optimizer)

    return train_cfg, da_rnn_net



class TrainData(typing.NamedTuple):
    feats: np.ndarray
    targs: np.ndarray




# def main():
#     logger = utils.setup_log()
#     logger.info(f"Using computation device: {device}")
#     epoch = 10
#     learning_rate = lstm_lr
#     threshold
#
#
#     save_plots = True
#     debug = False
#
#     #raw_data = pd.read_csv(os.path.join("./data/nasdaq", "nasdaq100_padding_test.csv"), nrows=100 if debug else None)
#     #logger.info(f"Shape of data: {raw_data.shape}.\nMissing in data: {raw_data.isnull().sum().sum()}.")
#     #targ_cols = ("NDX",)
#     #data, scaler = preprocess_data(raw_data, targ_cols)
#
#     logger.info(f"Shape of train data: {inputs_train.shape}.\nShape of target data: {targets_train.shape}.")
#
#     da_rnn_kwargs = {"batch_size": 50, "T": 10}
#
#     test_dat = TrainData(inputs_train, targets_train)
#
#     t, m1, n = test_dat.feats.shape
#     m2 = test_dat.targs.shape[2]
#     dim_hidden = 64
#
#     da_rnn_kwargs = {"batch_size": m1, "T": 4}
#     config, model = da_rnn(test_dat, n_targs=test_dat.targs.shape[2], learning_rate=learning_rate, **da_rnn_kwargs)
#
#     #net: DaRnnNet, train_data: TrainData, t_cfg: TrainConfig
#     iter_loss, epoch_loss = train(net = model, train_data=test_dat, t_cfg= config, n_epochs=epoch, save_plots=save_plots)
#
#
#     testing_data = TrainData(inputs_test , targets_test)
#     final_y_pred = predict(model, testing_data, config.train_size, config.batch_size, config.T)
#
#     final_y_pred_train = predict(model, test_dat, config.train_size, config.batch_size, config.T)
#
#     final_y_pred_test = predict(model, testing_data, config.train_size, config.batch_size, config.T)
#
#


    # for y in range(0,final_y_pred.shape[1]):
    #     loss_tmp = config.loss_func(final_y_pred[:,y,:], test_dat.targs[:,y,:])
    #     if loss_tmp <1e-3:
    #         print(loss_tmp.item(), torch.sqrt(loss_tmp).item(),"Bank",y)


    # plt.figure()
    # plt.plot(final_y_pred[:,90,:].numpy(), label='Predicted')
    # plt.plot(test_dat.targs[:,85,:].numpy(), label="True")
    # plt.legend(loc='upper left')
    #
    #
    # plt.figure()
    # plt.semilogy(range(len(iter_loss)), iter_loss)
    # utils.save_or_show_plot("iter_loss.png", save_plots)
    #
    # plt.figure()
    # plt.semilogy(range(len(epoch_loss)), epoch_loss)
    # utils.save_or_show_plot("epoch_loss.png", save_plots)
    #
    #
    # plt.figure()
    # plt.plot(final_y_pred[:,418,:].numpy(), label='Predicted')
    # plt.plot(testing_data.targs[:,418,:].numpy(), label="True")
    # plt.legend(loc='upper left')
    # utils.save_or_show_plot("final_predicted.png", save_plots)
    #
    #
    #
    # #Quickplots
    #
    # data_hist = test_dat
    # data_hist_pred = final_y_pred_train
    # data_target = testing_data
    # data_target_pred = final_y_pred_test
    # bank_id = 418
    # plotbank_y(data_hist,data_hist_pred, data_target,data_target_pred, bank_id)
    # def plotbank_y(data_hist,data_hist_pred, data_target,data_target_pred, bank_id, saveplot = False):
    #     plt.figure()
    #     plt.plot(range(1, len(data_hist.targs) + 1), data_hist.targs[:,bank_id,:].numpy(),label="True - History")
    #     plt.plot(range(len(data_hist.targs) + 1, len(data_hist.targs) + len(data_target.targs) ),data_target.targs[:, bank_id, :].numpy()[1:], label='True - Target History')
    #     plt.plot(range(1, len(data_hist_pred) + 1), data_hist_pred[:,bank_id,:], label='Predicted - Train')
    #     plt.plot(range(len(data_hist.targs) + 1, len(data_hist.targs) +  len(data_target_pred)), data_target_pred[:,bank_id,:][1:], label='Predicted - Test')
    #     plt.legend(loc='upper left')
    #     plt.show()
    #     utils.save_or_show_plot(f"pred_{bank_id}.png", saveplot)
    #
    #
    # #Save model settings.
    # with open(os.path.join("data","models" "da_rnn_kwargs.json"), "w") as fi:
    #     json.dump(da_rnn_kwargs, fi, indent=4)
    #
    # joblib.dump(scaler, os.path.join("data","models", "scaler.pkl"))
    # torch.save(model.encoder.state_dict(), os.path.join("data","models", "encoder.torch"))
    # torch.save(model.decoder.state_dict(), os.path.join("data","models", "decoder.torch"))
    #
    # plt.figure()
    # plt.plot(range(1, 1 + len(train_data.targs)), train_data.targs,
    #          label="True")
    # plt.plot(range(t_cfg.T, len(y_train_pred) + t_cfg.T), y_train_pred,
    #          label='Predicted - Train')
    # plt.plot(range(t_cfg.T + len(y_train_pred), len(train_data.targs) + 1), y_test_pred,
    #          label='Predicted - Test')
    # plt.legend(loc='upper left')
    # utils.save_or_show_plot(f"pred_{e_i}.png", save_plots)