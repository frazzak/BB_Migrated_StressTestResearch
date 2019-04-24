# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 18:07:34 2019

@author: yifei
"""
import os, sys
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

#from CGAN import CGAN
#from GAN import GAN .
from MultiVAE import m_MCVAE
from MultiVAE import m_LSTM




def elbo_loss(output_modalities, output_estimations, mu, logvar):

    MSE = 0
    for i in range(0, len(output_modalities)):
        if output_modalities[i] is not None:
            # since our outputs are not in range 0~1, we dont apply Binary_cross_entropy here
            temp_MSE = torch.nn.functional.mse_loss(output_estimations[i],
                                                    output_modalities[i])
            MSE = MSE + temp_MSE
    MSE = MSE / len(output_modalities)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    ELBO = torch.mean(MSE + KLD)

    return ELBO, MSE, KLD

def build_fake_modalities(train_size):

    # construct fake modalities using existing modalities
    # here I just manully separate raw data into four modalities
    path = os.getcwd()
    mod_dome = np.load(path + '/data/modality_domestic.npy')
    mod_inter = np.load(path + '/data/modality_international.npy')

    # since some data is missing, I only use data start from 1999, which is the 92nd row
    mod_dome = mod_dome[92:, :]
    mod_inter = mod_inter[92:, :]

    mod_dome_a = torch.from_numpy(mod_dome[:, 0:8]).float()
    mod_dome_b = torch.from_numpy(mod_dome[:, 8:]).float()
    mod_inter_a = torch.from_numpy(mod_inter[:, 0:6]).float()
    mod_inter_b = torch.from_numpy(mod_inter[:, 6:]).float()

    fake_modalities_train = [mod_dome_a[0:train_size, :],
                             mod_dome_b[0:train_size, :],
                             mod_inter_a[0:train_size, :],
                             mod_inter_b[0:train_size, :]]
    fake_modalities_test = [mod_dome_a[train_size:, :],
                             mod_dome_b[train_size:, :],
                             mod_inter_a[train_size:, :],
                             mod_inter_b[train_size:, :]]

    return fake_modalities_train, fake_modalities_test

def separate_cond_from_mod(conditional_id, modality_train, modality_test):

    cond_train = modality_train[conditional_id]
    cond_test = modality_test[conditional_id]
    num_cond = cond_train.shape[1]

    new_mod_train = []
    new_mod_test = []
    for i in range(0, len(modality_train)):
        if i == conditional_id:
            continue
        else:
            new_mod_train.extend([modality_train[i]])
            new_mod_test.extend([modality_test[i]])

    return num_cond, cond_train, cond_test, new_mod_train, new_mod_test

def inference_error(mod_real, mod_estimation):

    mse = 0
    for i in range(0, len(mod_real)):
        temp_mse = torch.nn.functional.mse_loss(mod_real[i], mod_estimation[i])
        mse = mse + temp_mse
    mse = mse / len(mod_real)
    return mse


#TODO: Modify and configure CGAN and GAN models to fit our problem , then adjust function accordingly.

# def train_GAN(num_cond, cond_train, cond_test, mod_train, mod_test, learning_rate, conditional):
#
#     # build GAN/CGAN for each modality separately to evaluate the performance
#     # for comparison
#
#     # set parameters
#     use_cuda = False
#     epcho = 1000
#     #learning_rate = 1e-3
#     latent_size = 10
#     modality_num = len(mod_train)
#     #conditional = False
#     layer_size = [128, 64, 32]
#     modality_size = 1 # here refers to single modality
#
#
#     m_error = 0
#     for i in range(0, modality_num):
#         # training procedure
#         print("evaluating # %d modality" % i)
#         mod_input_size = [mod_train[i].shape[1]]
#
#         if conditional:
#             print("CGAN running")
#             mGAN = CGAN(latent_size, modality_size, conditional, num_cond,
#                                  mod_input_size, layer_size, use_cuda)
#             m_optimizer = torch.optim.Adam(mGAN.parameters(), lr=learning_rate)
#             mGAN.train()
#             for j in range(0, epcho):
#                 m_optimizer.zero_grad()
#                 outputs, mu, logvar = mGAN.forward([mod_train[i]], cond_train)
#                 m_loss, MSE, KLD = elbo_loss([mod_train[i]], outputs, mu, logvar)
#                 m_loss.backward()
#                 m_optimizer.step()
#         else:
#             print("GAN running")
#             mGAN = GAN(latent_size, modality_size, conditional, num_cond,
#                         mod_input_size, layer_size, use_cuda)
#             m_optimizer = torch.optim.Adam(mGAN.parameters(), lr=learning_rate)
#             mGAN.train()
#             for j in range(0, epcho):
#                 m_optimizer.zero_grad()
#                 outputs, mu, logvar = mGAN.forward([mod_train[i]], cond_train)
#                 m_loss, MSE, KLD = elbo_loss([mod_train[i]], outputs, mu, logvar)
#                 m_loss.backward()
#                 m_optimizer.step()
#
#             #if j%100 == 0:
#             #    print("loss:%.2f\tMSE:%.2f\tKLD:%.2f" % (m_loss.data[0], MSE, KLD))
#         #print("Training Done!")
#
#         # testing procedure
#         mGAN.test()
#         batch_size = mod_test[i].shape[0]
#         estimations = mGAN.inference(n=batch_size, cond=cond_test)[0] # select the only one result
#
#
#         t_error = torch.nn.functional.mse_loss(estimations, mod_test[i])
#        # print("Forcing NAN values to zero")
#        # t_error[t_error != t_error] = 0
#         #print(m_error, t_error)
#         m_error = m_error + t_error
#         #print(m_error)
#         #print("Testing_error (mse):%.2f" % t_error)
#
#     #print("Diag NAN for CGAN", m_error, modality_num, m_error / modality_num)
#
#     m_error = m_error / modality_num
#     if conditional:
#         print("CGAN testing_error (mse):%.2f" % m_error)
#     else:
#         print("GAN testing_error (mse):%.2f" % m_error)
#
#     return m_error



#TODO: Consolidate VAE, CVAE and MCVAE into one function for comparasion.
def train_CVAE(num_cond, cond_train, cond_test, mod_train, mod_test, learning_rate, conditional, use_cuda = False, epcho = 1000, layer_size = [128, 64, 32]):

    # build GAN/CGAN for each modality separately to evaluate the performance
    # for comparison

    # set parameters
    #use_cuda = False
    #epcho = 1000
    #learning_rate = 1e-3
    #latent_size = 10
    modality_num = len(mod_train)
    #conditional = False
    #layer_size = [128, 64, 32]
    modality_size = 1 # here refers to single modality
    m_error = 0
    t_error = 0
    for i in range(0, modality_num):
        # training procedure
        print("evaluating # %d modality" % i)
        mod_input_size = [mod_train[i].shape[1]]
        mVAE = m_MCVAE.MCVAE(latent_size, modality_size, conditional, num_cond,
                             mod_input_size, layer_size, use_cuda)
        m_optimizer = torch.optim.Adam(mVAE.parameters(), lr=learning_rate)
        mVAE.train()
        for j in range(0, epcho):
            m_optimizer.zero_grad()
            outputs, mu, logvar = mVAE.forward([mod_train[i]], cond_train)
            m_loss, MSE, KLD = elbo_loss([mod_train[i]], outputs, mu, logvar)
            m_loss.backward()
            m_optimizer.step()
            #if j%100 == 0:
            #    print("loss:%.2f\tMSE:%.2f\tKLD:%.2f" % (m_loss.data[0], MSE, KLD))
        #print("Training Done!")

        # testing procedure
        mVAE.test()
        batch_size = mod_test[i].shape[0]
        estimations = mVAE.inference(n=batch_size, cond=cond_test)[0] # select the only one result

        #TODO:Some issue here, getting NAN for t_error and sometimes getting nan for m_error Causing overall m_error to be nan

        # if t_error != 0:
        #     t_error_prev = t_error
        # else:
        #     t_error_prev =
        t_error = torch.nn.functional.mse_loss(estimations, mod_test[i])
        #print("Removing  NAN values to zero")
        #t_error[t_error != t_error] = 0
        #print(m_error, t_error)
        # if torch.isnan(t_error):
        #     print("t_error value is NAN, using previous value")
        #     m_error = m_error + t_error_prev
        #     #t_errornan = t_errornan + 1
        # else:
        m_error = m_error + t_error
        #print(m_error)
        #print("Testing_error (mse):%.2f" % t_error)

    #print("Diag NAN for CGAN", m_error, modality_num, m_error / modality_num)
    #modality_num = modality_num - t_errornan
    m_error = m_error / modality_num
    # if conditional:
    #     print("CVAE testing_error (mse):%.2f" % m_error)
    # else:
    #     print("VAE testing_error (mse):%.2f" % m_error)

    return m_error
def train_MCVAE(num_cond, cond_train, cond_test, mod_train, mod_test, learning_rate, conditional,use_cuda = False,epcho = 1000,layer_size = [128, 64, 32]):
    #conditional = True
    # parameters for mcvae
    print("Model parameter initialization")
    #use_cuda = False
    #epcho = 1000
    #learning_rate = 1e-6
    #latent_size = 10
    modality_size = len(mod_train)
    #conditional = True
    #layer_size = [128, 64, 32]
    mod_input_sizes = []
    for i in range(0, modality_size):
        print(mod_train[i].shape)
        mod_input_sizes.extend([mod_train[i].shape[1]])


    mcvae = m_MCVAE.MCVAE(latent_size, modality_size, conditional, num_cond, mod_input_sizes, layer_size, use_cuda)
    m_optimizer = torch.optim.Adam(mcvae.parameters(), lr=learning_rate)
    print("Model parameter initialization Done!")
    print("Training Started!")

    #t_loss = -np.inf
    mcvae.train()
    #i = 0
    #TODO: Model training giving nan loss. Could be related to the additonal data.
    for i in range(0, epcho):
        m_optimizer.zero_grad()
        outputs, mu, logvar = mcvae.forward(mod_train, cond_train)
        m_loss, MSE, KLD = elbo_loss(mod_train, outputs, mu, logvar)
        m_loss.backward()
        m_optimizer.step()
        if i%100 == 0: # print loss at every 100 epochs #TODO: Good potential training visualization, loss decresing every 100 epochs..
           print("epoch:",i)
           print("loss:%.2f\tMSE:%.2f\tKLD:%.2f" % (m_loss.data, MSE, KLD))
    print("Training Done!")

    print("Testing Started")
    mcvae.test()
    batch_size = mod_test[0].shape[0]
    estimations = mcvae.inference(n=batch_size, cond=cond_test)
    error_mse = inference_error(mod_test, estimations)

    # if conditional:
    #     print("MCVAE testing_error (mse):%.2f" % error_mse)
    # else:
    #     print("MVAE testing_error (mse):%.2f" % error_mse)
    print("Testing Ended")

    return error_mse, estimations

def reform_data(data):

    dim = data.shape
    if len(dim) == 3:
        print("Scanning Through 3 Dimensional Rows and Columns")
        for i in range(0, dim[0]):
            for j in range(0, dim[1]):
                for k in range(0, dim[2]):
                    if data[i, j, k] > 1:
                        print("Normailizing values above 1 to log value")
                        data[i, j, k] = np.log(data[i, j, k])
                    elif data[i, j, k] < 0:
                        print("Normalizing values below 1 to Exponential value")
                        data[i, j, k] = np.exp(data[i, j, k])
    if len(dim) == 2:
        print("Scanning Through 3 Dimensional Rows and Columns")
        for i in range(0, dim[0]):
            for j in range(0, dim[1]):
                if data[i, j] > 1:
                    print("Normailizing values above 1 to log value")
                    data[i, j] = np.log(data[i, j])
                elif data[i, j] < 0:
                    print("Normalizing values below 1 to Exponential value")
                    data[i, j] = np.exp(data[i, j])

    return data

#TODO: make load_quarter_based_data more dynamic and able to load additional modalities.
#TODO: Ensure all quarters and modalities are ingested.
#TODO: Create format and design so all quarters are ingested and modelled properly.

def load_all_quarter_data(cond_name,
                            path_dict = {"path_root" : os.path.join(os.getcwd(),"data/"),
                                         "X" : "data_X.npy",
                                         "Y" : "data_Y.npy",
                                         "XYCap" : "data_XYCap.npy",
                                         "Moda_prefix":"data_moda_",
                                         "Moda_suffix":".npy"

                                         },
                            path_qtr_dict = {"path_root" : os.path.join(os.getcwd(),"data/quarter_based/"),
                                         "X" : "data_X_quarter.npy",
                                         "Y" : "data_Y_quarter.npy",
                                         "XYCap" : "data_XYCap_quarter.npy",
                                         "Moda_prefix":"data_moda_",
                                         "Moda_suffix":"_quarter.npy"

                                         }
                            , modality_names = ['SBidx', 'zmicro', 'domestic', 'international', 'Sectidx'], data_names = ["X","Y","XYCap"], quarter_ID = None):
    print("Initialize Result Dict")
    data_dict = dict()
    for data in data_names:
        print("Loading Bank Data Variable:",data)
        if quarter_ID is None:
            data_tmp_numpy = np.load( path_dict["path_root"] + path_dict[data])
        else:
            data_tmp_numpy = np.load(path_qtr_dict["path_root"] + path_qtr_dict[data])[quarter_ID]
        print("Convert nan to num")
        data_tmp_numpy = np.nan_to_num(data_tmp_numpy)
        print("Converting to Torch format")
        data_tmp_numpy = torch.from_numpy(data_tmp_numpy)
        print("Saving to Dictionary")
        if quarter_ID is None:
            key_tmp = "_".join(["data",data])
        else:
            key_tmp = "_".join(["data",data,"quarter",str(quarter_ID)])
        data_dict[key_tmp] = data_tmp_numpy
        print(data_dict[key_tmp].shape)

    print(data_dict.keys())

    print("Loading Modalities")
    print("Initializing Modality List Object")

    data_moda_lst = []
    for names in modality_names:
        if names == cond_name:
            print("Loading Conditional Modality:", names)
            if quarter_ID is None:
                data_moda_cond_tmp = np.load( path_dict["path_root"] +  path_dict["Moda_prefix"] + cond_name + path_dict["Moda_suffix"])
            else:
                data_moda_cond_tmp = np.load(path_qtr_dict["path_root"] + path_qtr_dict["Moda_prefix"] + cond_name + path_qtr_dict["Moda_suffix"])[quarter_ID]

            data_moda_cond_tmp = np.nan_to_num(data_moda_cond_tmp)
            data_moda_cond_tmp = torch.from_numpy(data_moda_cond_tmp).float()
            if quarter_ID is None:
                key_tmp = "_".join(["cond", names])
            else:
                key_tmp = "_".join(["cond", names, "quarter", str(quarter_ID)])
            data_dict[key_tmp] = data_moda_cond_tmp
            print(data_dict[key_tmp].shape)
        else:
            print("Loading Regular Modality:", names)
            if quarter_ID is None:
                temp_moda = np.load(path_dict["path_root"] +  path_dict["Moda_prefix"] + names + path_dict["Moda_suffix"])
            else:
                temp_moda = np.load(path_qtr_dict["path_root"] + path_qtr_dict["Moda_prefix"] + names + path_qtr_dict["Moda_suffix"])[quarter_ID]
            temp_moda = np.nan_to_num(temp_moda)
            temp_moda = torch.from_numpy(temp_moda).float()
            print(temp_moda.shape)
            data_moda_lst.extend([temp_moda])
    print("Load Modality List into Dict")
    data_dict["modalities_lst"] = data_moda_lst
    return data_dict



def load_quarter_based_data(quarter_ID, cond_name,
                            path_dict = {"path_root" : os.path.join(os.getcwd(),"data/quarter_based/"),
                                         "X_qtr" : "data_X_quarter.npy",
                                         "Y_qtr" : "data_Y_quarter.npy",
                                         "XYCap_qtr" : "data_XYCap_quarter.npy",
                                         "Moda_prefix":"data_moda_",
                                         "Moda_suffix":"_quarter.npy"

                                         }
                            , modality_names = ['SBidx', 'zmicro', 'domestic', 'international', 'Sectidx']):

    print("Loading Raw Bank Characteristics, X and Performance Data, Y")
    # if path_dict["path_root"] is None:
    #     path = os.join.path(os.getcwd(), "data/quarter_based/")
    #     print("Loading X Variable")
    #     data_X_quarter = np.load( path_dict["path_root"] + "data_X_quarter.npy")[quarter_ID]
    #     print("Loading Y Variable")
    #     data_Y_quarter = np.load( path_dict["path_root"] + "data_Y_quarter.npy")[quarter_ID]
    #     #data_X_quarter = reform_data(data_X_quarter)
    #     #data_Y_quarter = reform_data(data_Y_quarter)
    # else:
    #quarter_ID = 3
    print("Loading X Variable")
    data_X_quarter = np.load( path_dict["path_root"] + path_dict["X_qtr"])[quarter_ID]
    data_X_quarter = np.nan_to_num(data_X_quarter)
    print("Loading Y Variable")
    data_Y_quarter = np.load( path_dict["path_root"] + path_dict["Y_qtr"])[quarter_ID]
    data_Y_quarter = np.nan_to_num(data_Y_quarter)
    print("Loading XYCap Variable")
    data_XYCap_quarter = np.load(path_dict["path_root"] + path_dict["XYCap_qtr"])[quarter_ID]
    data_XYCap_quarter = np.nan_to_num(data_XYCap_quarter)

    print("Converting to Torch format")
    data_X_quarter = torch.from_numpy(data_X_quarter)
    data_Y_quarter = torch.from_numpy(data_Y_quarter)
    data_XYCap_quarter = torch.from_numpy(data_XYCap_quarter)
#    data_X_quarter.shape

    # with four modalities, one conditional and other three for inputs
    #TODO: Static search for modality names.
    #TODO: Improve to dynamic based on file naming.

    print("Loading Modalities")
    #modality_names = ['SBidx', 'zmicro', 'domestic', 'international']
    print("Initializing Modality List Object")
    data_moda_quarter = []
    for names in modality_names:
        if names == cond_name:
            print("Loading Conditional Modality:", names)
            data_moda_cond_quarter = np.load( path_dict["path_root"] +  path_dict["Moda_prefix"] + cond_name + path_dict["Moda_suffix"])[quarter_ID]
            #data_moda_cond = reform_data(data_moda_cond)
            data_moda_cond_quarter = np.nan_to_num(data_moda_cond_quarter)
            data_moda_cond_quarter = torch.from_numpy(data_moda_cond_quarter).float()
            print(data_moda_cond_quarter.shape)
        else:
            print("Loading Regular Modality:", names)
            temp_moda = np.load(path_dict["path_root"] +  path_dict["Moda_prefix"] + names + path_dict["Moda_suffix"])[quarter_ID]
            temp_moda = np.nan_to_num(temp_moda)
            #temp_moda = reform_data(temp_moda)
            temp_moda = torch.from_numpy(temp_moda).float()
            print(temp_moda.shape)
            data_moda_quarter.extend([temp_moda])

    return data_X_quarter, data_Y_quarter, data_XYCap_quarter, data_moda_quarter, data_moda_cond_quarter


def build_datasets(data_dict, train_window, test_window):
    '''
    the model requires inputs as X_t, Y_t-1, mod_t, and cond_t
    cond_t is used to generate mod_t
    normally train_windwo[1] = test_window[0], in a consecutive manner
    '''
    #modality = moda
    #TODO: Add logic to check the training and testing window lengths.
    print('Building datasets for expermiment and models')
    print("Initalize Dataset Dict")
    dataset_dict = dict()

    print("Generating test and training set for Conditionaly Modality")
    for condkey in [x for x in data_dict.keys() if x.startswith("cond_")]:
        cond_train = data_dict[condkey][train_window[0]:train_window[1], :]
        cond_test = data_dict[condkey][test_window[0]:test_window[1], :]

    print("Initialize Modality Lists")
    mod_train = []
    mod_test = []
    print('Building datasets for Generative Models component')
    for modlstkey in [x for x in data_dict.keys() if x.startswith("modalities_")]:
        for mod in data_dict[modlstkey]:
            mod_train.extend([mod[train_window[0]:train_window[1], :]])
            mod_test.extend([mod[test_window[0]:test_window[1], :]])
        dataset_dict["trainset_mod"] = mod_train
        dataset_dict["testset_mod"] = mod_test
    print('Building datasets for LSTM component')
    for datakey in [x for x in data_dict.keys() if x.startswith("data_")]:
        print(datakey.split('_')[1])
        tmp_data_obj = datakey.split('_')[1]
        print("Generating Training and Testing for:", tmp_data_obj)
        dataset_dict["_".join(["trainset",datakey])] = data_dict[datakey][:, train_window[0]:train_window[1], :]
        dataset_dict["_".join(["testset",datakey])] = data_dict[datakey][:, test_window[0]:test_window[1], :]
        dataset_dict["_".join(["trainset",datakey,"tminus1"])] = data_dict[datakey][:, train_window[0] - 1:train_window[1] - 1, :]
        dataset_dict["_".join(["testset",datakey,"tminus1"])] = data_dict[datakey][:, test_window[0] - 1:test_window[1] - 1, :]

        #TODO: Also consider the tranining and testing window validitly

    print("Summary of objects.")
    for key in dataset_dict.keys():
        print("key:", key)
        if type(dataset_dict[key]) == list:
            for listkey in dataset_dict[key]:
                #  print(listkey.size())
                print(listkey.shape)
        else:
            print(dataset_dict[key].shape)

    return dataset_dict






def build_train_eval_data(X, Y, XYCap, modality, cond, train_window, test_window):
    '''
    the model requires inputs as X_t, Y_t-1, mod_t, and cond_t
    cond_t is used to generate mod_t
    normally train_windwo[1] = test_window[0], in a consecutive manner
    '''
    #modality = moda

    print('Building data for generative model')
    mod_train = []
    mod_test = []
    print("Generating test and training set for Conditionaly Modality")
    cond_train = cond[train_window[0]:train_window[1], :]
    cond_test = cond[test_window[0]:test_window[1], :]

    print("Generating test and training sets for other modalities")
    for mod in modality:
        mod_train.extend([mod[train_window[0]:train_window[1], :]])
        mod_test.extend([mod[test_window[0]:test_window[1], :]])

 #   X.shape
    #TODO: Consolidate entry into dict
    print('Building data for LSTM component')
    print("Generating Training and Testing for X")
    X_train = X[:, train_window[0]:train_window[1], :]
    X_train_t_1 = X[:, train_window[0] - 1:train_window[1] - 1, :]

    X_test = X[:, test_window[0]:test_window[1], :]
    X_test_t_1 = X[:, test_window[0] - 1:test_window[1] - 1, :]
    print("Generating Training and Testing for Y,Y_t-1")
    Y_train_t_1 = Y[:, train_window[0]-1:train_window[1]-1, :]
    Y_train_t = Y[:, train_window[0]:train_window[1], :]
    Y_test_t_1 = Y[:, test_window[0]-1:test_window[1]-1, :]
    Y_test_t = Y[:, test_window[0]:test_window[1], :]

    print("Generating Training and Testing for XYCap")
    XYCap_train_t_1 = XYCap[:, train_window[0] - 1:train_window[1] - 1, :]
    XYCap_train_t = XYCap[:, train_window[0]:train_window[1], :]
    XYCap_test_t_1 = XYCap[:, test_window[0] - 1:test_window[1] - 1, :]
    XYCap_test_t = XYCap[:, test_window[0]:test_window[1], :]

    #TODO:Loop this and automate
    traintest_sets_dict = {"modTrain":mod_train,
                  "condTrain": cond_train,
                  "Xtminus1Train": X_train_t_1,
                  "XTrain": X_train,
                  "Ytminus1Train": Y_train_t_1,
                  "YTrain" : Y_train_t,
                  "XYCapTminus1Train": XYCap_train_t_1,
                  "XYCapTrain" : XYCap_train_t,
                  "modTest": mod_test,
                  "condTest" : cond_test,
                  "Xtminus1Test": X_test_t_1,
                  "XTest" : X_test,
                  "Ytminus1Test": Y_test_t_1,
                  "YTest" : Y_test_t,
                  "XYCapTminus1Test": XYCap_test_t_1,
                  "XYCapTest":XYCap_test_t}
    #test_sets = {mod_test, cond_test, X_test, Y_test_t_1, Y_test_t, XYCap_test_t_1, XYCap_test_t}

    print("Summary of objects.")
    for key in traintest_sets_dict.keys():
        print("key:", key)
        if type(traintest_sets_dict[key]) == list:
            for listkey in traintest_sets_dict[key]:
                #  print(listkey.size())
                print(listkey.shape)
        else:
            print(traintest_sets_dict[key].shape)

    return traintest_sets_dict



def get_raw_train_test_data(moda_names = ['SBidx', 'zmicro', 'domestic', 'international', 'Sectidx'], quarter_ID = 0, cond_name = 'Sectidx',
                            train_window = [1, 19], test_window = [20, 26]):

    #TODO: Set if condition to get synthetic modalities.

    #    '''
    #    obtain fake modaliaties to evaluate
    #    '''
    # train_size = 76
    # modality_train, modality_test = build_fake_modalities(train_size)

    # we shall take one modality as conditional factor, and others as inputs
    # we can change the value of conditional_id from 0 to 3 (here in this case)
    # to find the most representative modality in training as well as testing
    # that is the modality can achieve lowest error on testing data
    # conditional_id = 1
    # num_cond, cond_train, cond_test, mod_train, mod_test = separate_cond_from_mod(conditional_id, modality_train, modality_test)

    print('build real modalities to evaluate')
    print("retrieving data...")
    #TODO: Address Quarter ID only taking first Quarter Issue
    #quarter_ID = 0
    print("Modalities to be loaded ", moda_names, "for quarter_ID ", str(quarter_ID))


    #cond_name = moda_names[2]  # this can be changed to see different conditional effects from differnt modalities
    print("setting conditional modality to ", cond_name)



    #TODO: Address the static loading of the modalities available
    print("Loading Data and generating X,Y, Modality and Conditional Objects")

    #TODO: Explore ways to extract the modalities properly as well as the X and Y and conditional
    if quarter_ID is not None:
        print("Running Load Quarter Based Data Function")
    #X, Y, XYCap, moda, cond = load_quarter_based_data(quarter_ID, cond_name, modality_names = moda_names)
        data_dict = load_all_quarter_data(quarter_ID=quarter_ID, cond_name = cond_name, modality_names= moda_names)
        t = data_dict["data_Y_quarter_"+ str(quarter_ID)].shape[0]
    else:
        data_dict = load_all_quarter_data(quarter_ID=None, cond_name = cond_name, modality_names= moda_names)
        t = data_dict["data_Y"].shape[0]

    # TODO: Try to use full historical Bank Data rather than just 10 years for testing and 3 for training.
    print("Create Training and Testing Sets")
    #train_window = [1, 11]  # indicating ten years
    #test_window = [11, 14]  # use three years to evaluate
    print("Running Train Eval Data Function")
    traintest_sets_dict= build_datasets(data_dict, train_window, test_window)
    traintest_sets_dict= build_train_eval_data(X, Y, XYCap, moda, cond, train_window, test_window)
    print("Done!")

    #TODO: Address the Static aspect of the outputs
#    cond_train = traintest_sets_dict["condTrain"] #train_sets[1]
#    cond_test = traintest_sets_dict["condTest"]#test_sets[1]
#    mod_train = traintest_sets_dict["modTrain"] #train_sets[0]
#    mod_test = traintest_sets_dict["modTest"]  #test_sets[0]
#    num_cond = traintest_sets_dict["condTrain"].shape[1]

    return traintest_sets_dict["condTrain"], traintest_sets_dict["condTest"], traintest_sets_dict["modTrain"] , traintest_sets_dict["modTest"], traintest_sets_dict["condTrain"].shape[1], cond_name, traintest_sets_dict



def GenerativeModelCompare(num_cond, cond_train, cond_test, mod_train, mod_test, cond_name, learning_rate = 1e-4, times = 3, epcho = 1000):
    print("Comparison on Generative models")
    print("Iterations: ", times, "Learning Rate:", learning_rate, "Conditionality Name:", cond_name)
    print("Initializing Results Objects")
    mcvae = []
    cvae = []
    vae = []
    #cgan = []
    #gan = []
    #tmp = 0

    print("Running Iterative Loop of the Models")
    for i in range(0, times):
        print("iteration:", i)
        print("MCVAE Modelling")
        MCVAE_error, pred_moda = train_MCVAE(num_cond, cond_train, cond_test, mod_train,
                                             mod_test, learning_rate, epcho = epcho, conditional=True)

        print("MCVAE testing_error (mse):%.2f" % MCVAE_error)
        #mcvae = mcvae + MCVAE_error
        mcvae.append(MCVAE_error)


        print("CVAE Modelling")
        CVAE_error = train_CVAE(num_cond, cond_train, cond_test, mod_train, mod_test, learning_rate,epcho = epcho, conditional=True)
        print("CVAE testing_error (mse):%.2f" % CVAE_error)
        #cvae = cvae + CVAE_error
        cvae.append(CVAE_error)


        print("VAE Modelling")
        VAE_error = train_CVAE(num_cond, cond_train, cond_test, mod_train, mod_test, learning_rate,epcho = epcho, conditional=False)
        #vae = vae + VAE_error
        print("VAE testing_error (mse):%.2f" % VAE_error)
        vae.append(VAE_error)

    print("Calculating Average Performance")
    mcvae = torch.stack(mcvae)
    cvae = torch.stack(cvae)
    vae = torch.stack(vae)
    #cgan = torch.stack(cgan)
    #gan = torch.stack(gan)
    #    gan = gan / times
    #    cgan = cgan / times
    print("conditional modality:\t%s" % cond_name)
    print("Averaged performance at %d iterations:" % times)
#    print("MCVAE testing_error (mse):\t%.2f" % mcvae.mean())
#    print("CVAE testing_error (mse):\t%.2f" % cvae.mean())
#    print("VAE testing_error (mse):\t%.2f" % vae.mean())
    #    print("GAN testing_error (mse):\t%.2f" % gan)
    #    print("CGAN testing_error (mse):\t%.2f" % cgan)

    print("Creating Resutls Object")
    labels = ["MCVAE","CVAE","VAE"]
    results = pd.DataFrame([mcvae.data,cvae.data,vae.data], index = labels).transpose()
    results.loc["mean"] = results.mean()
    results.loc["iterations"] = times
    print(results)
    return(results, pred_moda)


def LSTM_BankPrediction( pred_moda ,traintest_sets_dict,learn_types = ["Only_Yminus1", "Only_Xminus1","Yminus1&Xminus1","Yminus1&X",
                                                                       "Yminus1&Xminus1&moda","Yminus1&X&moda","Yminus1&Xminus1&XYCapminus1",
                                                                       "Yminus1&X&XYCapminus1","Yminus1&Xminus1&XYCapminus1&moda","Yminus1&X&XYCapminus1&moda"], lstm_lr = 1e-2, threshold = 1e-3, modelTarget = "Y"):
    print("Comparison on LSTM models")
    #learn_types = ["Only_Y", "Y&X", "Y&X&moda"]
    rmse_train_list = []
    rmse_lst = []
    #ids = 0
    #TODO: Iterate through learn types list rather than a range list counter.
    for ids in range(0, len(learn_types)):
        m_learn_type = learn_types[ids]

        print("Setting Raw Inputs and Raw Evaluation Inputs")
        #TODO: Address the Static nature of the learn type to raw inputs mapping
        if m_learn_type == "Only_Yminus1":
            print(m_learn_type)
            raw_inputs = traintest_sets_dict["Ytminus1Train"]#train_sets[3]
            print(raw_inputs.shape)
            raw_eval_inputs = traintest_sets_dict["Ytminus1Test"]#test_sets[3]
            print(raw_eval_inputs.shape)

        if m_learn_type == "Only_Xminus1":
            print(m_learn_type)
            raw_inputs = traintest_sets_dict["Xtminus1Train"]#train_sets[3]
            print(raw_inputs.shape)
            raw_eval_inputs = traintest_sets_dict["Xtminus1Test"]#test_sets[3]
            print(raw_eval_inputs.shape)
        #TODO: Address the Static nature of the learn type to raw inputs mapping
        if m_learn_type == "Yminus1&Xminus1":
            print(m_learn_type)
            raw_inputs = torch.cat((traintest_sets_dict["Ytminus1Train"], traintest_sets_dict["Xtminus1Train"]), dim=2)
            raw_eval_inputs = torch.cat((traintest_sets_dict["Ytminus1Test"], traintest_sets_dict["Xtminus1Test"]), dim=2)

        if m_learn_type == "Yminus1&X":
            print(m_learn_type)
            raw_inputs = torch.cat((traintest_sets_dict["Ytminus1Train"], traintest_sets_dict["XTrain"]), dim=2)
            raw_eval_inputs = torch.cat((traintest_sets_dict["Ytminus1Test"], traintest_sets_dict["XTest"]), dim=2)

        #TODO: Address the Static nature of the learn type to raw inputs mapping
        if m_learn_type == "Yminus1&Xminus1&moda":
            print(m_learn_type)
            raw_inputs = torch.cat((traintest_sets_dict["Ytminus1Train"], traintest_sets_dict["Xtminus1Train"]), dim=2)
            raw_eval_inputs = torch.cat((traintest_sets_dict["Ytminus1Test"], traintest_sets_dict["Xtminus1Test"]), dim=2)
            print("in testing stage the modality is applied from the predicted modality from previous stage")
            #TODO: May need to consider capturing other generative models predictions rather than just MCVAE
            temp_eval_moda = pred_moda[0]
            temp_moda = traintest_sets_dict["modTrain"][0]#train_sets[0][0]
            #TODO: Need additional detail to this part to understand what exactly it is doing.
            for i in range(1, len(traintest_sets_dict["modTrain"])):
                temp_moda = torch.cat((temp_moda, traintest_sets_dict["modTrain"][i]), dim=1)
                temp_eval_moda = torch.cat((temp_eval_moda, pred_moda[i]), dim=1)
            raw_moda = temp_moda.expand_as(torch.zeros([raw_inputs.shape[0], temp_moda.shape[0], temp_moda.shape[1]]))
            raw_eval_moda = temp_eval_moda.expand_as(torch.zeros([raw_eval_inputs.shape[0], temp_eval_moda.shape[0], temp_eval_moda.shape[1]]))
            raw_inputs = torch.cat((raw_inputs, raw_moda.double()), dim=2)
            raw_eval_inputs = torch.cat((raw_eval_inputs, raw_eval_moda.double()), dim=2)
        if m_learn_type == "Yminus1&X&moda":
            print(m_learn_type)
            raw_inputs = torch.cat((traintest_sets_dict["Ytminus1Train"], traintest_sets_dict["XTrain"]), dim=2)
            raw_eval_inputs = torch.cat((traintest_sets_dict["Ytminus1Test"], traintest_sets_dict["XTest"]), dim=2)
            print("in testing stage the modality is applied from the predicted modality from previous stage")
            #TODO: May need to consider capturing other generative models predictions rather than just MCVAE
            temp_eval_moda = pred_moda[0]
            temp_moda = traintest_sets_dict["modTrain"][0]#train_sets[0][0]
            #TODO: Need additional detail to this part to understand what exactly it is doing.
            for i in range(1, len(traintest_sets_dict["modTrain"])):
                temp_moda = torch.cat((temp_moda, traintest_sets_dict["modTrain"][i]), dim=1)
                temp_eval_moda = torch.cat((temp_eval_moda, pred_moda[i]), dim=1)
            raw_moda = temp_moda.expand_as(torch.zeros([raw_inputs.shape[0], temp_moda.shape[0], temp_moda.shape[1]]))
            raw_eval_moda = temp_eval_moda.expand_as(torch.zeros([raw_eval_inputs.shape[0], temp_eval_moda.shape[0], temp_eval_moda.shape[1]]))
            raw_inputs = torch.cat((raw_inputs, raw_moda.double()), dim=2)
            raw_eval_inputs = torch.cat((raw_eval_inputs, raw_eval_moda.double()), dim=2)

        if m_learn_type == "Yminus1&Xminus1&XYCapminus1":
            print(m_learn_type)
            raw_inputs = torch.cat((traintest_sets_dict["Ytminus1Train"], traintest_sets_dict["Xtminus1Train"]), dim=2)
            raw_inputs = torch.cat((raw_inputs, traintest_sets_dict["XYCapTminus1Train"]), dim=2)
            raw_eval_inputs = torch.cat((traintest_sets_dict["Ytminus1Test"], traintest_sets_dict["Xtminus1Test"]), dim=2)
            raw_eval_inputs = torch.cat((raw_eval_inputs , traintest_sets_dict["XYCapTminus1Test"]), dim=2)



        if m_learn_type == "Yminus1&X&XYCapminus1":
            print(m_learn_type)
            raw_inputs = torch.cat((traintest_sets_dict["Ytminus1Train"], traintest_sets_dict["XTrain"]), dim=2)
            raw_inputs = torch.cat((raw_inputs, traintest_sets_dict["XYCapTminus1Train"]), dim=2)
            raw_eval_inputs = torch.cat((traintest_sets_dict["Ytminus1Test"], traintest_sets_dict["XTest"]), dim=2)
            raw_eval_inputs = torch.cat((raw_eval_inputs , traintest_sets_dict["XYCapTminus1Test"]), dim=2)


        if m_learn_type == "Yminus1&Xminus1&XYCapminus1&moda":
            print(m_learn_type)
            raw_inputs = torch.cat((traintest_sets_dict["Ytminus1Train"], traintest_sets_dict["Xtminus1Train"]), dim=2)
            raw_inputs = torch.cat((raw_inputs, traintest_sets_dict["XYCapTminus1Train"]), dim=2)
            raw_eval_inputs = torch.cat((traintest_sets_dict["Ytminus1Test"], traintest_sets_dict["Xtminus1Test"]), dim=2)
            raw_eval_inputs = torch.cat((raw_eval_inputs , traintest_sets_dict["XYCapTminus1Test"]), dim=2)
            for i in range(1, len(traintest_sets_dict["modTrain"])):
                temp_moda = torch.cat((temp_moda, traintest_sets_dict["modTrain"][i]), dim=1)
                temp_eval_moda = torch.cat((temp_eval_moda, pred_moda[i]), dim=1)
            raw_moda = temp_moda.expand_as(torch.zeros([raw_inputs.shape[0], temp_moda.shape[0], temp_moda.shape[1]]))
            raw_eval_moda = temp_eval_moda.expand_as(torch.zeros([raw_eval_inputs.shape[0], temp_eval_moda.shape[0], temp_eval_moda.shape[1]]))
            raw_inputs = torch.cat((raw_inputs, raw_moda.double()), dim=2)
            raw_eval_inputs = torch.cat((raw_eval_inputs, raw_eval_moda.double()), dim=2)


        if m_learn_type == "Yminus1&X&XYCapminus1&moda":
            print(m_learn_type)
            raw_inputs = torch.cat((traintest_sets_dict["Ytminus1Train"], traintest_sets_dict["XTrain"]), dim=2)
            raw_inputs = torch.cat((raw_inputs, traintest_sets_dict["XYCapTminus1Train"]), dim=2)
            raw_eval_inputs = torch.cat((traintest_sets_dict["Ytminus1Test"], traintest_sets_dict["XTest"]), dim=2)
            raw_eval_inputs = torch.cat((raw_eval_inputs , traintest_sets_dict["XYCapTminus1Test"]), dim=2)
            for i in range(1, len(traintest_sets_dict["modTrain"])):
                temp_moda = torch.cat((temp_moda, traintest_sets_dict["modTrain"][i]), dim=1)
                temp_eval_moda = torch.cat((temp_eval_moda, pred_moda[i]), dim=1)
            raw_moda = temp_moda.expand_as(torch.zeros([raw_inputs.shape[0], temp_moda.shape[0], temp_moda.shape[1]]))
            raw_eval_moda = temp_eval_moda.expand_as(torch.zeros([raw_eval_inputs.shape[0], temp_eval_moda.shape[0], temp_eval_moda.shape[1]]))
            raw_inputs = torch.cat((raw_inputs, raw_moda.double()), dim=2)
            raw_eval_inputs = torch.cat((raw_eval_inputs, raw_eval_moda.double()), dim=2)


        print("Setting Inputs and Target Parameters for Training")
        # TODO: Investigate how to resolve the static nature of setting the target
        # TODO: May need to add the Capital Ratios part as targets.

        if modelTarget == "Y":
            raw_targets = traintest_sets_dict["YTrain"]
            raw_eval_targets = traintest_sets_dict["YTest"]
        elif modelTarget == "XYCap":
            raw_targets = traintest_sets_dict["XYCapTrain"]
            raw_eval_targets = traintest_sets_dict["XYCapTest"]
        elif modelTarget == "X":
            raw_targets = traintest_sets_dict["XTrain"]
            raw_eval_targets = traintest_sets_dict["XTest"]
        else:
            print("Setting Default Target as Y")
            raw_targets = traintest_sets_dict["YTrain"]
            raw_eval_targets = traintest_sets_dict["YTest"]

        n, t, m1 = raw_inputs.shape
        m2 = raw_targets.shape[2]
        inputs = torch.zeros([t, m1, n]).float()
        targets = torch.zeros([t, n, m2]).float()
        for i in range(0, n):
            inputs[:, :, i] = raw_inputs[i, :, :]
            targets[:, i, :] = raw_targets[i, :, :]


        print("Training LSTM model")
        m_lstm, train_loss = m_LSTM.train(inputs, targets, 50, lstm_lr, threshold)

        print("Calculating Training RMSE")
        rmse_train_list.append(train_loss)
        print("%s\terror:\t%.5f" % (m_learn_type, train_loss))

        print("Setting Inputs and Target Parameters for Testing")
        #raw_eval_inputs = raw_inputs = torch.cat((test_sets[3], test_sets[2]), dim=2)
        #TODO: Investigate how to resolve the static nature of setting the target
        #raw_eval_targets = traintest_sets_dict["YTest"] #test_sets[4]
        n, t, m1 = raw_eval_inputs.shape
        m2 = raw_eval_targets.shape[2]
        inputs = torch.zeros([t, m1, n]).float()
        targets = torch.zeros([t, n, m2]).float()
        for i in range(0, n):
            inputs[:, :, i] = raw_eval_inputs[i, :, :]
            targets[:, i, :] = raw_eval_targets[i, :, :]


        print("Running Predictions on Inputs using Trained Model: Testing Error")
        pred = m_LSTM.predict(m_lstm, inputs)


        print("Calculating Testing RMSE")
        rmse = torch.nn.functional.mse_loss(pred, targets)
        rmse_lst.append(rmse)
        print("%s\terror:\t%.5f" % (m_learn_type, rmse))

    rmse_train_lst_sk = torch.stack(rmse_train_list)
    rmse_lst_sk = torch.stack(rmse_lst)
    rmse_list_final = [rmse_train_lst_sk, rmse_lst_sk.data]
    result_obj = pd.DataFrame(rmse_list_final, columns = learn_types, index = ["TrainErr","TestErr"])
    return(result_obj)



os.chdir("/Users/phn1x/icdm2018_research_BB/Stress_Test_Research/Loss_projections/")


#TODO: fix so it checks for miniums and maxiums for the datasets so no error occurs at LSTM
#TODO:Train window must be above 0. Add logic to get min and max for the window.
#'SBidx'
#'Sectidx'
cond_train, cond_test, mod_train, mod_test, num_cond, cond_name,traintest_sets_dict  = get_raw_train_test_data(quarter_ID = 1, cond_name = "domestic"
                                                                                                                 , moda_names = ['zmicro', 'domestic', 'international'],
                                                                                                               train_window = [1,21], test_window = [22,25])





results, pred_moda = GenerativeModelCompare(num_cond, cond_train, cond_test, mod_train, mod_test, cond_name, learning_rate = 1e-3 , times = 1, epcho = 1000)



#TODO: Need to fix the handling of the different time slices.
#TODO: Add Capital ratio features as well.
#TODO: Add Model Target Target variable for Testing
BankPredEval = LSTM_BankPrediction(pred_moda, traintest_sets_dict, lstm_lr=1e-2, threshold=1e-3, modelTarget= "Y")


#Graphical LSTM MSE representaiton.
# with torch.no_grad():
#     prediction = model.forward(xtest).view(-1)
#     loss = criterion(prediction, ytest)
#     plt.title("MESLoss: {:.5f}".format(loss))
#     plt.plot(prediction.detach().numpy(), label="pred")
#     plt.plot(ytest.detach().numpy(), label="true")
#     plt.legend()
#     plt.show()










#Previous Code.
0

#if __name__=="__main__":

#    '''
#    obtain fake modaliaties to evaluate
#    '''
    #train_size = 76
    #modality_train, modality_test = build_fake_modalities(train_size)

    # we shall take one modality as conditional factor, and others as inputs
    # we can change the value of conditional_id from 0 to 3 (here in this case)
    # to find the most representative modality in training as well as testing
    # that is the modality can achieve lowest error on testing data
    #conditional_id = 1
    #num_cond, cond_train, cond_test, mod_train, mod_test = separate_cond_from_mod(conditional_id, modality_train, modality_test)



    # print('build real modalities to evaluate')
    # print("retriving data...")
    # #TODO: Correct and include all modalities
    # moda_names = ['SBidx', 'zmicro', 'domestic', 'international']
    #
    # #TODO:Fix all time slices and handle more than just 1 quarter.
    # quarter_ID = 0
    #
    # print("Modalities to be loaded " , moda_names, "for quarter_ID ", str(quarter_ID) )
    #
    #
    # #TODO:Set the correct Conditional modality, Net-Charge-offss
    # print("setting conditional modality to ", moda_names[2])
    # cond_name = moda_names[2] # this can be changed to see different conditional effects from differnt modalities
    #
    # X, Y, moda, cond = load_quarter_based_data(quarter_ID, cond_name)
    # t = Y.shape[0]

    #TODO: Try to use full historical Bank Data rather than just 10 years for testing and 3 for training.

    # print("Create Training and Testing Sets")
    # train_window = [1, 11]  # indicating ten years
    # test_window = [11, 14]  # use three years to evaluate
    # train_sets, test_sets = build_train_eval_data(X, Y, moda, cond, train_window, test_window)
    # print("Done!")
    #
    # cond_train = train_sets[1]
    # cond_test = test_sets[1]
    # mod_train = train_sets[0]
    # mod_test = test_sets[0]
    # num_cond = cond_train.shape[1]


#print("Comparison on Generative models")
#    learning_rate = 1e-4
#    mcvae = 0
#    cvae = 0
#    vae = 0
#    cgan = 0
#    gan = 0
#    times = 3
#    for i in range(0, times):
#        print("iteration:",i)
#        print("MCVAE Modelling")
#        MCVAE_error, pred_moda = train_MCVAE(num_cond, cond_train, cond_test, mod_train,
#                                  mod_test, learning_rate, conditional=True)

#        print("MCVAE testing_error (mse):%.2f" % MCVAE_error)
#        mcvae = mcvae + MCVAE_error



#        print("CVAE Modelling")
#        CVAE_error = train_CVAE(num_cond, cond_train, cond_test, mod_train, mod_test, learning_rate, conditional=True)
#        cvae = cvae + CVAE_error
#        print("CVAE testing_error (mse):%.2f" % CVAE_error)


#        print("VAE Modelling")
#        VAE_error = train_CVAE(num_cond, cond_train, cond_test, mod_train,mod_test, learning_rate, conditional=False)
#        print("VAE testing_error (mse):%.2f" % VAE_error)
#        vae = vae + VAE_error

#        print("GAN Modelling")
#        GAN_error = train_GAN(num_cond, cond_train, cond_test, mod_train, mod_test, learning_rate, conditional=False)
#        print("GAN testing_error (mse):%.2f" % GAN_error)
#       gan = gan + GAN_error


#        print("CGAN Modelling")
#        CGAN_error = train_GAN(num_cond, cond_train, cond_test, mod_train,mod_test, learning_rate, conditional=True)
#        print("CGAN testing_error (mse):%.2f" % CGAN_error)
#        CGAN = cgan + CGAN_error

#    mcvae = mcvae / times
#    cvae = cvae / times
#    vae = vae / times
#    gan = gan / times
#    cgan = cgan / times
#    print("conditional modality:\t%s" % cond_name)
#    print("Averaged performance:")
#    print("MCVAE testing_error (mse):\t%.2f" % mcvae)
#    print("CVAE testing_error (mse):\t%.2f" % cvae)
#    print("VAE testing_error (mse):\t%.2f" % vae)
#TODO: Setup GAN and CGAN
#    print("GAN testing_error (mse):\t%.2f" % gan)
#    print("CGAN testing_error (mse):\t%.2f" % cgan)










#TODO: Make this part for LSTM into a seperate function



#      print("comparison on LSTM models")
#     learn_types = ["Only_Y", "Y&X", "Y&X&moda"]
#     for ids in range(0, 3):
#         m_learn_type = learn_types[ids]
#         lstm_lr = 1e-2
#         threshold = 1e-3
#         if m_learn_type == learn_types[0]:
#             raw_inputs = train_sets[3]
#             raw_eval_inputs = test_sets[3]
#         if m_learn_type == learn_types[1]:
#             raw_inputs = torch.cat((train_sets[3], train_sets[2]), dim=2)
#             raw_eval_inputs = torch.cat((test_sets[3], test_sets[2]), dim=2)
#         if m_learn_type == learn_types[2]:
#             raw_inputs = torch.cat((train_sets[3], train_sets[2]), dim=2)
#             raw_eval_inputs = torch.cat((test_sets[3], test_sets[2]), dim=2)
#             print("in testing stage the modality is applied from the predicted modality from previous stage")
#             temp_eval_moda = pred_moda[0]
#             temp_moda = train_sets[0][0]
#             for i in range(1, len(train_sets[0])):
#                 temp_moda = torch.cat((temp_moda, train_sets[0][i]), dim=1)
#                 temp_eval_moda = torch.cat((temp_eval_moda, pred_moda[i]), dim=1)
#             raw_moda = temp_moda.expand_as(torch.zeros([raw_inputs.shape[0], temp_moda.shape[0], temp_moda.shape[1]]))
#             raw_eval_moda = temp_eval_moda.expand_as(torch.zeros([raw_eval_inputs.shape[0], temp_eval_moda.shape[0], temp_eval_moda.shape[1]]))
#             raw_inputs = torch.cat((raw_inputs, raw_moda.double()), dim=2)
#             raw_eval_inputs = torch.cat((raw_eval_inputs, raw_eval_moda.double()), dim=2)
#
#
#         print("Setting Inputs and Target Parameters")
#         raw_targets = train_sets[4]
#         n, t, m1 = raw_inputs.shape
#         m2 = raw_targets.shape[2]
#         inputs = torch.zeros([t, m1, n]).float()
#         targets = torch.zeros([t, n, m2]).float()
#         for i in range(0, n):
#             inputs[:, :, i] = raw_inputs[i, :, :]
#             targets[:, i, :] = raw_targets[i, :, :]
#
#
#         print("Training LSTM model")
#         m_lstm = m_LSTM.train(inputs, targets, 50, lstm_lr, threshold)
#
#         #raw_eval_inputs = raw_inputs = torch.cat((test_sets[3], test_sets[2]), dim=2)
#         raw_eval_targets = test_sets[4]
#         n, t, m1 = raw_eval_inputs.shape
#         m2 = raw_eval_targets.shape[2]
#         inputs = torch.zeros([t, m1, n]).float()
#         targets = torch.zeros([t, n, m2]).float()
#         for i in range(0, n):
#             inputs[:, :, i] = raw_eval_inputs[i, :, :]
#             targets[:, i, :] = raw_eval_targets[i, :, :]
#
#
#         print("Running Predictions on Inputs using Trained Model")
#         pred = m_LSTM.predict(m_lstm, inputs)
#
#
#         print("Calculating RMSE")
#         rmse = torch.nn.functional.mse_loss(pred, targets)
#         print("%s\terror:\t%.5f" % (m_learn_type, rmse))






# with torch.no_grad():
#     prediction = model.forward(xtest).view(-1)
#     loss = criterion(prediction, ytest)
#     plt.title("MESLoss: {:.5f}".format(loss))
#     plt.plot(prediction.detach().numpy(), label="pred")
#     plt.plot(ytest.detach().numpy(), label="true")
#     plt.legend()
#     plt.show()









