# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 18:07:34 2019

@author: yifei
"""
import os
import numpy as np
import pandas as pd
import itertools
import copy
import pickle

import torch
import torch.nn.functional as F
#from CGAN import CGAN
#from GAN import GAN .
from MultiVAE import m_MCVAE, m_CGAN, m_LSTM,m_GRU,m_DA_RNN
# from MultiVAE import m_LSTM
# from MultiVAE import m_CGAN
# from MultiVAE import m_GRU
#
# import importlib
# importlib.reload(m_MCVAE)
# importlib.reload(m_CGAN)
# importlib.reload(m_LSTM)
# importlib.reload(m_GRU)
# importlib.reload(m_DA_RNN)


from BDMC_master import ais
from BDMC_master import simulate
from BDMC_master import bdmc
from BDMC_master import hmc
#from BDMC_master import vae



import logging

def setup_log(tag='VOC_TOPICS'):
    # create logger
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = logging.getLogger(tag)
    logger.handlers = []
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    # logger.handlers = []
    logger.addHandler(ch)
    return logger


def elbo_loss(output_modalities, output_estimations, mu, logvar):

    MSE = 0
    for i in range(0, len(output_modalities)):
        if output_modalities[i] is not None:
            # since our outputs are not in range 0~1, we dont apply Binary_cross_entropy here
            temp_MSE = torch.nn.functional.mse_loss(output_estimations[i],
                                                    output_modalities[i])
            MSE = MSE + temp_MSE
    MSE = MSE / len(output_modalities)
    RMSE = torch.sqrt(MSE)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    ELBO = torch.mean(MSE + KLD)

    return ELBO, MSE, KLD, RMSE

# def build_fake_modalities(train_size):
#
#     # construct fake modalities using existing modalities
#     # here I just manully separate raw data into four modalities
#     path = os.getcwd()
#     mod_dome = np.load(path + '/data/modality_domestic.npy')
#     mod_inter = np.load(path + '/data/modality_international.npy')
#
#     # since some data is missing, I only use data start from 1999, which is the 92nd row
#     mod_dome = mod_dome[92:, :]
#     mod_inter = mod_inter[92:, :]
#
#     mod_dome_a = torch.from_numpy(mod_dome[:, 0:8]).float()
#     mod_dome_b = torch.from_numpy(mod_dome[:, 8:]).float()
#     mod_inter_a = torch.from_numpy(mod_inter[:, 0:6]).float()
#     mod_inter_b = torch.from_numpy(mod_inter[:, 6:]).float()
#
#     fake_modalities_train = [mod_dome_a[0:train_size, :],
#                              mod_dome_b[0:train_size, :],
#                              mod_inter_a[0:train_size, :],
#                              mod_inter_b[0:train_size, :]]
#     fake_modalities_test = [mod_dome_a[train_size:, :],
#                              mod_dome_b[train_size:, :],
#                              mod_inter_a[train_size:, :],
#                              mod_inter_b[train_size:, :]]
#
#     return fake_modalities_train, fake_modalities_test

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

#
# def reform_data(data):
#
#     dim = data.shape
#     if len(dim) == 3:
#         print("Scanning Through 3 Dimensional Rows and Columns")
#         for i in range(0, dim[0]):
#             for j in range(0, dim[1]):
#                 for k in range(0, dim[2]):
#                     if data[i, j, k] > 1:
#                         print("Normailizing values above 1 to log value")
#                         data[i, j, k] = np.log(data[i, j, k])
#                     elif data[i, j, k] < 0:
#                         print("Normalizing values below 1 to Exponential value")
#                         data[i, j, k] = np.exp(data[i, j, k])
#     if len(dim) == 2:
#         print("Scanning Through 3 Dimensional Rows and Columns")
#         for i in range(0, dim[0]):
#             for j in range(0, dim[1]):
#                 if data[i, j] > 1:
#                     print("Normailizing values above 1 to log value")
#                     data[i, j] = np.log(data[i, j])
#                 elif data[i, j] < 0:
#                     print("Normalizing values below 1 to Exponential value")
#                     data[i, j] = np.exp(data[i, j])
#
#     return data


#
# def load_quarter_based_data(quarter_ID, cond_name,
#                             path_dict = {"path_root" : os.path.join(os.getcwd(),"data/quarter_based/"),
#                                          "X_qtr" : "data_X_quarter.npy",
#                                          "Y_qtr" : "data_Y_quarter.npy",
#                                          "XYCap_qtr" : "data_XYCap_quarter.npy",
#                                          "Moda_prefix":"data_moda_",
#                                          "Moda_suffix":"_quarter.npy"
#
#                                          }
#                             , modality_names = ['SBidx', 'zmicro', 'domestic', 'international', 'Sectidx']):
#
#     print("Loading Raw Bank Characteristics, X and Performance Data, Y")
#     # if path_dict["path_root"] is None:
#     #     path = os.join.path(os.getcwd(), "data/quarter_based/")
#     #     print("Loading X Variable")
#     #     data_X_quarter = np.load( path_dict["path_root"] + "data_X_quarter.npy")[quarter_ID]
#     #     print("Loading Y Variable")
#     #     data_Y_quarter = np.load( path_dict["path_root"] + "data_Y_quarter.npy")[quarter_ID]
#     #     #data_X_quarter = reform_data(data_X_quarter)
#     #     #data_Y_quarter = reform_data(data_Y_quarter)
#     # else:
#     #quarter_ID = 3
#     print("Loading X Variable")
#     data_X_quarter = np.load( path_dict["path_root"] + path_dict["X_qtr"])[quarter_ID]
#     data_X_quarter = np.nan_to_num(data_X_quarter)
#     print("Loading Y Variable")
#     data_Y_quarter = np.load( path_dict["path_root"] + path_dict["Y_qtr"])[quarter_ID]
#     data_Y_quarter = np.nan_to_num(data_Y_quarter)
#     print("Loading XYCap Variable")
#     data_XYCap_quarter = np.load(path_dict["path_root"] + path_dict["XYCap_qtr"])[quarter_ID]
#     data_XYCap_quarter = np.nan_to_num(data_XYCap_quarter)
#
#     print("Converting to Torch format")
#     data_X_quarter = torch.from_numpy(data_X_quarter)
#     data_Y_quarter = torch.from_numpy(data_Y_quarter)
#     data_XYCap_quarter = torch.from_numpy(data_XYCap_quarter)
# #    data_X_quarter.shape
#
#     # with four modalities, one conditional and other three for inputs
#     #TODO: Static search for modality names.
#     #TODO: Improve to dynamic based on file naming.
#
#     print("Loading Modalities")
#     #modality_names = ['SBidx', 'zmicro', 'domestic', 'international']
#     print("Initializing Modality List Object")
#     data_moda_quarter = []
#     for names in modality_names:
#         if names == cond_name:
#             print("Loading Conditional Modality:", names)
#             data_moda_cond_quarter = np.load( path_dict["path_root"] +  path_dict["Moda_prefix"] + cond_name + path_dict["Moda_suffix"])[quarter_ID]
#             #data_moda_cond = reform_data(data_moda_cond)
#             data_moda_cond_quarter = np.nan_to_num(data_moda_cond_quarter)
#             data_moda_cond_quarter = torch.from_numpy(data_moda_cond_quarter).float()
#             print(data_moda_cond_quarter.shape)
#         else:
#             print("Loading Regular Modality:", names)
#             temp_moda = np.load(path_dict["path_root"] +  path_dict["Moda_prefix"] + names + path_dict["Moda_suffix"])[quarter_ID]
#             temp_moda = np.nan_to_num(temp_moda)
#             #temp_moda = reform_data(temp_moda)
#             temp_moda = torch.from_numpy(temp_moda).float()
#             print(temp_moda.shape)
#             data_moda_quarter.extend([temp_moda])
#
#     return data_X_quarter, data_Y_quarter, data_XYCap_quarter, data_moda_quarter, data_moda_cond_quarter
#


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


#
# def GenerativeModelCompare(num_cond, cond_train, cond_test, mod_train, mod_test, cond_name, learning_rate = 1e-4, times = 3, epcho = 1000):
#     print("Comparison on Generative models")
#     print("Iterations: ", times, "Learning Rate:", learning_rate, "Conditionality Name:", cond_name)
#     print("Initializing Results Objects")
#     mcvae = []
#     cvae = []
#     vae = []
#     #cgan = []
#     #gan = []
#     #tmp = 0
#
#     print("Running Iterative Loop of the Models")
#     for i in range(0, times):
#         print("iteration:", i)
#         print("MCVAE Modelling")
#         MCVAE_error, pred_moda = train_MCVAE(num_cond, cond_train, cond_test, mod_train,
#                                              mod_test, learning_rate, epcho = epcho, conditional=True)
#
#         print("MCVAE testing_error (mse):%.2f" % MCVAE_error)
#         #mcvae = mcvae + MCVAE_error
#         mcvae.append(MCVAE_error)
#
#
#         print("CVAE Modelling")
#         CVAE_error = train_CVAE(num_cond, cond_train, cond_test, mod_train, mod_test, learning_rate,epcho = epcho, conditional=True)
#         print("CVAE testing_error (mse):%.2f" % CVAE_error)
#         #cvae = cvae + CVAE_error
#         cvae.append(CVAE_error)
#
#
#         print("VAE Modelling")
#         VAE_error = train_CVAE(num_cond, cond_train, cond_test, mod_train, mod_test, learning_rate,epcho = epcho, conditional=False)
#         #vae = vae + VAE_error
#         print("VAE testing_error (mse):%.2f" % VAE_error)
#         vae.append(VAE_error)
#
#     print("Calculating Average Performance")
#     mcvae = torch.stack(mcvae)
#     cvae = torch.stack(cvae)
#     vae = torch.stack(vae)
#     #cgan = torch.stack(cgan)
#     #gan = torch.stack(gan)
#     #    gan = gan / times
#     #    cgan = cgan / times
#     print("conditional modality:\t%s" % cond_name)
#     print("Averaged performance at %d iterations:" % times)
# #    print("MCVAE testing_error (mse):\t%.2f" % mcvae.mean())
# #    print("CVAE testing_error (mse):\t%.2f" % cvae.mean())
# #    print("VAE testing_error (mse):\t%.2f" % vae.mean())
#     #    print("GAN testing_error (mse):\t%.2f" % gan)
#     #    print("CGAN testing_error (mse):\t%.2f" % cgan)
#
#     print("Creating Resutls Object")
#     labels = ["MCVAE","CVAE","VAE"]
#     results = pd.DataFrame([mcvae.data,cvae.data,vae.data], index = labels).transpose()
#     results.loc["mean"] = results.mean()
#     results.loc["iterations"] = times
#     print(results)
#     return(results, pred_moda)
#
#
#

def build_datasets(data_dict, data_train_window, data_test_window,  moda_train_window, moda_test_window, splittype = "timesplit", traintest_pct = [.8,.2]):
    '''
    the model requires inputs as X_t, Y_t-1, mod_t, and cond_t
    cond_t is used to generate mod_t
    normally train_windwo[1] = test_window[0], in a consecutive manner
    '''

    print('Building datasets for experiment and models')
    print("Initialize Dataset Dict")
    dataset_dict = dict()

    print("Generating Datasets for Scenario Generation")
    if splittype == "timesplit":
        print("Generating test and training set for Conditionaly Modality")
        for condkey in [x for x in data_dict.keys() if x.startswith("data_cond_")]:
            dataset_dict["_".join(["trainset",condkey,"timesplit"])] = data_dict[condkey][moda_train_window[0]:moda_train_window[1] + 1, :]
            dataset_dict["_".join(["testset",condkey,"timesplit"])] = data_dict[condkey][moda_test_window[0]:moda_test_window[1] + 1, :]

        print("Initialize Modality Lists")
        mod_train = []
        mod_test = []
        print('Building datasets for Generative Models component')
        for modlstkey in [x for x in data_dict.keys() if x.startswith("modalities_")]:

            for mod in data_dict[modlstkey]:
                mod_train.extend([mod[moda_train_window[0]:moda_train_window[1] + 1, :]])
                mod_test.extend([mod[moda_test_window[0]:moda_test_window[1] + 1, :]])
            dataset_dict["trainset_data_mod_timesplit"] = mod_train
            dataset_dict["testset_data_mod_timesplit"] = mod_test
    elif splittype == "banksplit":

        for condkey in [x for x in data_dict.keys() if x.startswith("data_cond_")]:
            dataset_dict["_".join(["trainset", condkey, "banksplit"])] = data_dict[condkey][moda_train_window[0]:moda_train_window[1] + 1, :]
            dataset_dict["_".join(["testset", condkey, "banksplit"])] = data_dict[condkey][moda_train_window[0]:moda_train_window[1] + 1, :]

        print("Initialize Modality Lists")
        mod_train = []
        mod_test = []
        print('Building datasets for Generative Models component')
        for modlstkey in [x for x in data_dict.keys() if x.startswith("modalities_")]:

            for mod in data_dict[modlstkey]:
                mod_train.extend([mod[moda_train_window[0]:moda_train_window[1] + 1, :]])
                mod_test.extend([mod[moda_train_window[0]:moda_train_window[1] + 1, :]])
            dataset_dict["trainset_data_mod_banksplit"] = mod_train
            dataset_dict["testset_data_mod_banksplit"] = mod_test

    print('Building datasets for Forecasting component')

    for datakey in [x for x in data_dict.keys() if x.startswith("data_") if not x.startswith(("data_cond", "data_mod"))]:
        print(datakey.split('_')[1])
        tmp_data_obj = datakey.split('_')[1]
        #TODO: Also consider the tranining and testing window validitly
        if splittype == 'timesplit':
            print("Train Test Windows:",data_train_window,data_test_window)
            if data_train_window is None and data_test_window is None and len(traintest_pct) == 2:
                data_train_window = [1,round(traintest_pct[0] * data_dict[datakey].shape[1]) + 1]
                data_test_window = [train_window[1] + 1, data_dict[datakey].shape[1] - 1]

            print("Generating Training and Testing for Time Split:", tmp_data_obj,data_train_window,data_test_window)
            print(data_dict[datakey].shape)
            dataset_dict["_".join(["trainset",datakey,"timesplit"])] = data_dict[datakey][:, data_train_window[0]:data_train_window[1] + 1, :]
            dataset_dict["_".join(["testset",datakey,"timesplit"])] = data_dict[datakey][:, data_test_window[0]:data_test_window[1] + 1, :]
            dataset_dict["_".join(["trainset",datakey,"tminus1","timesplit"])] = data_dict[datakey][:, data_train_window[0] - 1:data_train_window[1] , :]
            dataset_dict["_".join(["testset",datakey,"tminus1","timesplit"])] = data_dict[datakey][:, data_test_window[0] - 1:data_test_window[1], :]

        elif splittype == 'banksplit':
            #np.random.shuffle(arr)
            train_window_bs = [1,round(traintest_pct[0] * data_dict[datakey].shape[0]) + 1]
            test_window_bs = [train_window_bs[1] + 1, data_dict[datakey].shape[0] - 1]
            print("Generating Training and Testing for Bank Split:", tmp_data_obj)
            dataset_dict["_".join(["trainset",datakey,"banksplit"])] = data_dict[datakey][train_window_bs[0]:train_window_bs[1], data_train_window[0]:data_train_window[1] + 1 , :]
            dataset_dict["_".join(["testset",datakey,"banksplit"])] = data_dict[datakey][test_window_bs[0] :test_window_bs[1],  data_train_window[0]:data_train_window[1] + 1 ,  :]



    print("Summary of objects.")
    for key in dataset_dict.keys():
        print("key:", key)
        if isinstance(dataset_dict[key],list):
            for listkey in dataset_dict[key]:
                #  print(listkey.size())
                print(listkey.shape)
        else:
            print(dataset_dict[key].shape)

    return dataset_dict


def load_all_quarter_data(cond_name,
                            path_dict = {"path_root" : os.path.join(os.getcwd(),"data/"),
                                         "X" : "data_X.npy",
                                         "Y" : "data_Y.npy",
                                         #"XYCap" : "data_XYCap.npy",
                                         "CapRatios":"data_CapRatios.npy",
                                         "Moda_prefix":"data_moda_",
                                         "Moda_suffix":".npy"

                                         },
                            path_qtr_dict = {"path_root" : os.path.join(os.getcwd(),"data/quarter_based/"),
                                         "X" : "data_X_quarter.npy",
                                         "Y" : "data_Y_quarter.npy",
                                         #"XYCap" : "data_XYCap_quarter.npy",
                                         "CapRatios":"data_CapRatios_quarter.npy",
                                         "Moda_prefix":"data_moda_",
                                         "Moda_suffix":"_quarter.npy"

                                         }
                            , modality_names = ['sbidx', 'zmicro', 'zmacro_domestic', 'zmacro_international', 'Sectidx'], data_names = ["X","Y","CapRatios"], quarter_ID = None, datadir = "data/"):

    print("Setting Path Directories")
    path_dict["path_root"] = os.path.join(os.getcwd(), datadir)
    path_qtr_dict["path_root"] = os.path.join(os.getcwd(), "data/quarter_based/")
    print("Initialize Result Dict")
    data_dict = dict()
    for data in data_names:
        print("Loading Bank Data Variable:",data)
        if quarter_ID is None:
            data_tmp_numpy = np.load( path_dict["path_root"] + path_dict[data])
        else:
            data_tmp_numpy = np.load(path_qtr_dict["path_root"] + path_qtr_dict[data])[quarter_ID]
        #print("Convert nan to num")
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
                data_moda_cond_tmp = np.load( path_dict["path_root"] +  path_dict["Moda_prefix"] + cond_name + path_dict["Moda_suffix"])[0]
            else:
                data_moda_cond_tmp = np.load(path_qtr_dict["path_root"] + path_qtr_dict["Moda_prefix"] + cond_name + path_qtr_dict["Moda_suffix"])[quarter_ID]

            data_moda_cond_tmp = np.nan_to_num(data_moda_cond_tmp)
            data_moda_cond_tmp = torch.from_numpy(data_moda_cond_tmp).float()
            if quarter_ID is None:
                key_tmp = "_".join(["data","cond", names])
            else:
                key_tmp = "_".join(["data","cond", names, "quarter", str(quarter_ID)])
            data_dict[key_tmp] = data_moda_cond_tmp
            print(data_dict[key_tmp].shape)
        else:
            print("Loading Regular Modality:", names)
            if quarter_ID is None:
                temp_moda = np.load(path_dict["path_root"] +  path_dict["Moda_prefix"] + names + path_dict["Moda_suffix"])[0]
            else:
                temp_moda = np.load(path_qtr_dict["path_root"] + path_qtr_dict["Moda_prefix"] + names + path_qtr_dict["Moda_suffix"])[quarter_ID]
            temp_moda = np.nan_to_num(temp_moda)
            temp_moda = torch.from_numpy(temp_moda).float()
            print(temp_moda.shape)
            data_moda_lst.extend([temp_moda])
    print("Load Modality List into Dict")
    data_dict["modalities_lst"] = data_moda_lst
    return data_dict

def get_raw_train_test_data(moda_names=['sbidx', 'zmicro', 'zmacro_domestic', 'z_macrointernational', 'Sectidx'], quarter_ID=0,
                            cond_name='Sectidx',
                            data_names = ["X","Y","CapRatios"],
                            path_dict = {"path_root" : os.path.join(os.getcwd(),"data/"),
                                         "X" : "data_X.npy",
                                         "Y" : "data_Y.npy",
                                         #"XYCap" : "data_XYCap.npy",
                                         "CapRatios":"data_CapRatios.npy",
                                         "Moda_prefix":"data_moda_",
                                         "Moda_suffix":".npy"

                                         },
                            path_qtr_dict = {"path_root" : os.path.join(os.getcwd(),"data/quarter_based/"),
                                         "X" : "data_X_quarter.npy",
                                         "Y" : "data_Y_quarter.npy",
                                         #"XYCap" : "data_XYCap_quarter.npy",
                                         "CapRatios":"data_CapRatios_quarter.npy",
                                         "Moda_prefix":"data_moda_",
                                         "Moda_suffix":"_quarter.npy"

                                         },
                            splittype = "timesplit", modaDateIdx = None, dataDateIdx = None
                            ,moda_train_date = ["1976 Q1","2007 Q4"],data_train_date = ["1990 Q1","2007 Q4"], datamoda_test_date = ["2008 Q1","2016 Q4"]
):
    # TODO: Set if condition to get synthetic modalities.

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
    if modaDateIdx is None and dataDateIdx is None:
        print("Data Index References Required")
        return
    else:
        print("Setting Mod Date Index Params")
        if moda_train_date is None:
            moda_train_date = data_train_date
        elif data_train_date is None:
            data_train_date = moda_train_date

        moda_train_window = list(modaDateIdx[modaDateIdx.isin(moda_train_date)].index)
        moda_test_window = list(modaDateIdx[modaDateIdx.isin(datamoda_test_date)].index)
        print("Modalities:\tTrain:",moda_train_date,moda_train_window,"\tTest:",datamoda_test_date,moda_test_window)
        data_train_window =   list(dataDateIdx[dataDateIdx.isin(data_train_date)].index)
        data_test_window =    list(dataDateIdx[dataDateIdx.isin(datamoda_test_date)].index)
        print("Data:\tTrain:", data_train_date, data_train_window,"\tTest:", datamoda_test_date, data_test_window)

        print('build real modalities to evaluate')
        print("retrieving data...")

        print("Modalities to be loaded ", moda_names, "for quarter_ID ", str(quarter_ID))

        # cond_name = moda_names[2]  # this can be changed to see different conditional effects from differnt modalities
        print("setting conditional modality to ", cond_name)

        # TODO: Address the static loading of the modalities available
        print("Loading Data and generating X,Y, Modality and Conditional Objects")

        # TODO: Explore ways to extract the modalities properly as well as the X and Y and conditional
        if quarter_ID is not None:
            print("Running Load Quarter Based Data Function")
            data_dict = load_all_quarter_data(quarter_ID=quarter_ID, cond_name=cond_name, modality_names=moda_names, data_names = data_names,path_dict = path_dict, path_qtr_dict = path_qtr_dict)
        else:
            #TODO: Need model target for t shape.
            data_dict = load_all_quarter_data(quarter_ID=None, cond_name=cond_name, modality_names=moda_names, data_names = data_names,path_dict = path_dict, path_qtr_dict = path_qtr_dict)
            #t = data_dict["data_Y"].shape[0]


        # TODO: Try to use full historical Bank Data rather than just 10 years for testing and 3 for training.
        print("Create Training and Testing Sets")
        print("Running Train Eval Data Function")
        #TODO: Add logic to make dataset appropirate for bank split rather than time split.
        traintest_sets_dict = build_datasets(data_dict, data_train_window = data_train_window, data_test_window = data_test_window,
                                             moda_train_window = moda_train_window , moda_test_window = moda_test_window,splittype = splittype)
        # traintest_sets_dict= build_train_eval_data(X, Y, XYCap, moda, cond, train_window, test_window)


        print("Storing Condition Name and number")
        traintest_sets_dict["cond_name"] = cond_name
        for trainset_cond in [x for x in traintest_sets_dict.keys() if x.startswith("trainset_data_cond")]:
            traintest_sets_dict["cond_num"] = traintest_sets_dict[trainset_cond].shape[1]
        print("Done!")





    return traintest_sets_dict




#importlib.reload(simulate)
def ais_bdmc_lld(model, mod, latent_dim, cond=None, batch_size=None, n_batch=5, chain_length=100, iwae_samples=1,
                 modeltype="vae"):
    print("Initializing Parameters for:", modeltype)


    print("Paramters\t latent_dim:%s\tn_batch:%s\tchain_length:%s\tiwae_samples:%s\tbatch_size:%s\tmodeltype:%s" % (
    latent_dim, str(n_batch), str(chain_length), str(iwae_samples), str(batch_size), modeltype))
    forward_schedule = np.linspace(0., 1., chain_length)

    if modeltype not in ["mcvae"]:
        latent_dim = None


    print("Setting Model to Evaluation Mode")
    model.test()
    # bdmc uses simulated data from the model
    print("Generating Simulated Data from the Model:", modeltype)
    loader = simulate.simulate_data(model, mod = mod, batch_size=batch_size, n_batch=n_batch, cond=cond, modeltype=modeltype, mod_dim = latent_dim)
    # run bdmc
    print("Running Bi-Directional Monte Carlo with AIS Sampling")
    forward_logws, backward_logws, lower_bounds, upper_bounds = bdmc.bdmc(model=model, loader=loader,
                                                                          forward_schedule=forward_schedule,
                                                                          n_sample=iwae_samples
                                                                          , cond=cond, modeltype=modeltype)
    print("Lower/Upper Bounds LLD Average on Simulated Data: %.4f,%.4f" % (lower_bounds, upper_bounds))
    return forward_logws, backward_logws, lower_bounds, upper_bounds


#TODO: Add logic to get best EPOCH, Parameters and Validation Loss


def save_checkpoint(state, is_best, filename='/output/checkpoint.pth.tar'):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print ("=> Saving a new best")
        torch.save(state, filename)  # save checkpoint
    else:
        print ("=> Validation RMSE did not improve")


def train_CVAE(num_cond, cond_train, cond_test, mod_train, mod_test, learning_rate, conditional, use_cuda=False,
               epoch=1000, latent_size=10, layer_size=[128, 64, 32], verbose=False,
               bdmc_run=False, n_batch=5, chain_length=100, iwae_samples=1):
    print("Number of Modalities:", len(mod_train))
    modality_num = len(mod_train)
    modality_size = 1  # here refers to single modality
    results_dict = dict()
    estimations_lst = list()
    training_history = list()
    dfs_train = list()
    dfs_test = list()
    m_error = 0
    m_train_error = 0
    print("Training Started!")
    # i = 0
    for i in range(0, modality_num):
        # training procedure
        training_history_tmp = list()
        print("evaluating # %d modality" % i)
        mod_input_size = [mod_train[i].shape[1]]
        batch_size_train = mod_train[i].shape[0]
        latent_size = mod_input_size[0]
        mVAE = m_MCVAE.MCVAE(latent_size, modality_size, conditional, num_cond, mod_input_size, layer_size, use_cuda)
        m_optimizer = torch.optim.Adamax(mVAE.parameters(), lr=learning_rate)
        mVAE.train()
        for j in range(0, epoch):
            m_optimizer.zero_grad()
            outputs, mu, logvar, train_pdf = mVAE.forward([mod_train[i]], cond_train)
            m_loss, MSE, KLD, RMSE = elbo_loss([mod_train[i]], outputs, mu, logvar)
            try:
                best_loss
            except NameError:
                best_loss = m_loss.detach()
            else:
                if verbose:
                    print("Best Loss Exists:", best_loss)
            is_best = bool(m_loss.detach().numpy() < best_loss.numpy())
            best_loss = torch.FloatTensor(min(m_loss.detach().numpy(), best_loss.numpy()))
            if is_best:
                train_best_checkpoint = {
                    'epoch': 0 + j,
                    'state_dict': mVAE.state_dict(),
                    'best_loss': best_loss
                }


            m_loss.backward()
            m_optimizer.step()
            training_history_tmp.append(m_loss.data)

            if j % 100 == 0 and verbose is True:
                print("epoch:", j)
                print("loss:%.2f\tMSE:%.2f\tKLD:%.2f\tRMSE:%.2f" % (m_loss.data, MSE, KLD, RMSE))



        training_history.append(training_history_tmp)
        estimations_train, train_pdf = mVAE.inference(n=batch_size_train, cond=cond_train)  # select the only one result
        estimations_train = estimations_train[0]
        t_train_error = torch.nn.functional.mse_loss(estimations_train, mod_train[i])

        m_train_error = m_train_error + t_train_error
        print("Mod Training MSE:", str(t_train_error.data), "Total Training Error:", str(m_train_error.data))
        print("Training Done!")

        # testing procedure

        print("Testing Started!")

        mVAE.test()
        batch_size = mod_test[i].shape[0]
        estimations, test_pdf = mVAE.inference(n=batch_size, cond=cond_test)  # select the only one result
        estimations = estimations[0]
        t_error = torch.nn.functional.mse_loss(estimations, mod_test[i])
        m_error = m_error + t_error
        # Return Estimations per modalitiy.
        # Save so it can be used correspondingly in LSTM part.
        print("Testing Done!")
        print("Saving Modality:", i, "Estimations")
        estimations_lst.append(estimations)
        if bdmc_run:
            if conditional:
                modeltype = "cvae"
                # forward_logws_train, backward_logws_train, lower_bounds_train, upper_bounds_train = ais_bdmc_lld(mVAE,
                #                                                                                                  mod_train[
                #                                                                                                      i],
                #                                                                                                  latent_size,
                #                                                                                                  cond=cond_train,
                #                                                                                                  n_batch=n_batch,
                #                                                                                                  chain_length=chain_length,
                #                                                                                                  iwae_samples=iwae_samples,
                #                                                                                                  batch_size=batch_size_train,
                #                                                                                                  modeltype=modeltype)

                forward_logws_test, backward_logws_test, lower_bounds_test, upper_bounds_test = ais_bdmc_lld(mVAE,
                                                                                                             [mod_test[i]],
                                                                                                             latent_size,
                                                                                                             cond=cond_test,
                                                                                                             n_batch=n_batch,
                                                                                                             chain_length=chain_length,
                                                                                                             iwae_samples=iwae_samples,
                                                                                                             batch_size=batch_size,
                                                                                                             modeltype=modeltype)


            else:
                modeltype = "vae"
                # forward_logws_train, backward_logws_train, lower_bounds_train, upper_bounds_train = ais_bdmc_lld(mVAE,
                #                                                                                                  [
                #                                                                                                      mod_train[
                #                                                                                                          i]],
                #                                                                                                  latent_dim=latent_size,
                #                                                                                                  cond=cond_train,
                #                                                                                                  n_batch=n_batch,
                #                                                                                                  chain_length=chain_length,
                #                                                                                                  iwae_samples=iwae_samples,
                #                                                                                                  batch_size=batch_size_train,
                #                                                                                                  modeltype=modeltype)

                forward_logws_test, backward_logws_test, lower_bounds_test, upper_bounds_test = ais_bdmc_lld(mVAE,
                                                                                                             [mod_test[i]],
                                                                                                             latent_size,
                                                                                                             cond=cond_test,
                                                                                                             n_batch=n_batch,
                                                                                                             chain_length=chain_length,
                                                                                                             iwae_samples=iwae_samples,
                                                                                                             batch_size=batch_size,
                                                                                                             modeltype=modeltype)

            print("Appending Modality Results to Average")
            # train_df = pd.DataFrame([lower_bounds_train, upper_bounds_train]).transpose()
            # dfs_train.append(train_df)

            test_df = pd.DataFrame([lower_bounds_test, upper_bounds_test]).transpose()
            dfs_test.append(test_df)

    if bdmc_run:
        print("Getting Mean LLD across all modalities")
        #lld_ais_train = pd.concat(dfs_train).mean()
        lld_ais_test = pd.concat(dfs_test).mean()
        # print("Average LLD Train Bounds:\t.2f\t.2f\tAverage LLD Test Bounds:\t.2f\t.2f" % (lld_ais_train[1], lld_ais_train[0], lld_ais_test[1], lld_ais_test[0]))
        #print(lld_ais_train[1], lld_ais_train[0], lld_ais_test[1], lld_ais_test[0])
        print("Test LLD:",lld_ais_test[0], lld_ais_test[1])
        results_dict['AIS_BDMC_LLD'] = pd.DataFrame([tuple([lld_ais_test[0], lld_ais_test[1]])])
        results_dict['AIS_BDMC_LLD'].columns = ["lower_bounds_test", "upper_bounds_test"]

    training_history_df = pd.DataFrame(training_history).transpose().astype('float')
    training_history_df.index.names = ['EPOCH']
    training_history_df.columns = ["_".join(["modality", str(x)]) for x in
                                   range(0, training_history_df.columns.__len__())]
    if conditional:
        training_history_df['cvae_modality_total_loss'] = training_history_df.mean(axis=1)
    else:
        training_history_df['vae_modality_total_loss'] = training_history_df.mean(axis=1)

    print("Calculating Testing Error")
    print("m_error:", m_error)
    m_error = m_error / modality_num
    # training_history = pd.DataFrame(training_history, columns=['epoch', 'loss', 'MSE', 'KLD', 'RMSE'])
    # training_history = pd.concat([])
    rmse_error = torch.sqrt(m_error)
    results_dict["TrainHist"] = training_history_df
    results_dict['TrainBestModel'] = train_best_checkpoint

    return m_error, estimations_lst, rmse_error, train_pdf, test_pdf, results_dict


def train_MCVAE(num_cond, cond_train, cond_test, mod_train, mod_test, learning_rate, conditional=True, use_cuda=False,
                epoch=1000, layer_size=[128, 64, 32], latent_size=10, verbose=False, bdmc_run=False, n_batch=5,
                chain_length=100, iwae_samples=1):
    if conditional:
        print("Set to Conditonal Modeling")
        print("Number of Modalities:", len(mod_train))
    else:
        print("Set to Non-Conditional Modeling")
        print("Appending Modality List with Conditional Modality")
        print("Number of Modalities:", len(mod_train))
        print("Saving Original Mod")
        mod_train_orig = mod_train
        mod_test_orig = mod_test
        mod_train.append(cond_train)
        mod_test.append(cond_test)
        print("Number of Modalities after Append:", len(mod_train))
    print("Model parameter initialization")
    results_dict = dict()
    training_history = list()
    modality_size = len(mod_train)
    mod_input_sizes = []
   #modesl_dict_list = []

    # i = 0
    for i in range(0, modality_size):
        print(mod_train[i].shape)
        mod_input_sizes.extend([mod_train[i].shape[1]])

    # conditional = True
    latent_size = mod_input_sizes[0]
    mcvae = m_MCVAE.MCVAE(latent_size, modality_size, conditional, num_cond, mod_input_sizes, layer_size, use_cuda)
    m_optimizer = torch.optim.Adamax(mcvae.parameters(), lr=learning_rate)
    print("Model parameter initialization Done!")
    print("Training Started!")
    training_history_tmp = list()
    #models_dict_tmp = list()
    mcvae.train()
    batch_size_train = mod_train[0].shape[0]
    for i in range(0, epoch):
        m_optimizer.zero_grad()
        outputs, mu, logvar, train_pdf = mcvae.forward(mod_train, cond_train)
        m_loss, MSE, KLD, RMSE = elbo_loss(mod_train, outputs, mu, logvar)

        try:
            best_loss
        except NameError:
            best_loss = m_loss.detach()
        else:
            if verbose:
                print("Best Loss Exists:", best_loss)

        is_best = bool(m_loss.detach().numpy() < best_loss.numpy())
        best_loss = torch.FloatTensor(min(m_loss.detach().numpy(), best_loss.numpy()))
        if is_best:
            train_best_checkpoint = {
                'epoch': 0 + i,
                'state_dict': mcvae.state_dict(),
                'best_loss': best_loss
            }


        m_loss.backward()
        m_optimizer.step()
        training_history_tmp.append(m_loss)

        if i % 100 == 0 and verbose is True:
            print("epoch:", i)
            print("Training\tloss:%.2f\tMSE:%.2f\tKLD:%.2f\tRMSE:%.2f" % (m_loss.data, MSE, KLD, RMSE))
    training_history.append(training_history_tmp)


    # print("Get Predicted Probability Distribution")
    # print(train_pdf.shape)

    print("Training and Validation Done!")
    training_history_df = pd.DataFrame(training_history).transpose().astype('float')
    training_history_df.index.names = ['EPOCH']
    training_history_df.columns = ["mcvae_modality_total_loss"]

    results_dict["TrainHist"] = training_history_df
    results_dict['TrainBestModel'] = train_best_checkpoint
    print("Testing of Estimations Started")

    mcvae.test()
    batch_size = mod_test[0].shape[0]
    estimations, test_pdf = mcvae.inference(n=batch_size, cond=cond_test)
    error_mse = inference_error(mod_test, estimations)
    error_rmse = torch.sqrt(error_mse)
    # t_error = torch.nn.functional.mse_loss(estimations, mod_test)
    # print("T_Error:", t_error)
    # m_error = m_error + t_error
    print("Testing Ended")
    if bdmc_run:
        # forward_logws_train, backward_logws_train, lower_bounds_train, upper_bounds_train = ais_bdmc_lld(mcvae,
        #                                                                                                  mod_train,
        #                                                                                                  latent_dim= mod_input_sizes,
        #                                                                                                  cond=cond_train,
        #                                                                                                  n_batch=n_batch,
        #                                                                                                  chain_length=chain_length,
        #                                                                                                  iwae_samples=iwae_samples,
        #                                                                                                  batch_size=batch_size_train,
        #                                                                                                  modeltype="mcvae")

        forward_logws_test, backward_logws_test, lower_bounds_test, upper_bounds_test = ais_bdmc_lld(mcvae, mod_test,
                                                                                                     latent_dim= mod_input_sizes,
                                                                                                     cond=cond_test,
                                                                                                     n_batch=n_batch,
                                                                                                     chain_length=chain_length,
                                                                                                     iwae_samples=iwae_samples,
                                                                                                     batch_size=batch_size,
                                                                                                     modeltype="mcvae")
        # print("Average LLD Train Bounds:\t.2f\t.2f\tAverage LLD Test Bounds:\t.2f\t.2f" % (lower_bounds_train, upper_bounds_train, lower_bounds_test, upper_bounds_test))
        #print(lower_bounds_train, upper_bounds_train, lower_bounds_test, upper_bounds_test)
        #results_dict['AIS_BDMC_LLD'] = pd.DataFrame([tuple([lower_bounds_train, upper_bounds_train, lower_bounds_test, upper_bounds_test])])
        #results_dict['AIS_BDMC_LLD'].columns = ["lower_bounds_train", "upper_bounds_train", "lower_bounds_test", "upper_bounds_test"]
        print(lower_bounds_test, upper_bounds_test)
        results_dict['AIS_BDMC_LLD'] = pd.DataFrame([tuple([lower_bounds_test, upper_bounds_test])])
        results_dict['AIS_BDMC_LLD'].columns = ["lower_bounds_test", "upper_bounds_test"]
        #results_dict['TrainBestModel'] = train_best_checkpoint
    # results_dict["Train"]
    if not conditional:
        print("Reassigning Mods")
        print("Saving Original Mod")
        mod_train = mod_train_orig
        mod_test = mod_test_orig

    return error_mse, estimations, error_rmse, train_pdf, test_pdf, results_dict


def train_GAN(num_cond, cond_train, cond_test, mod_train, mod_test, learning_rate, conditional=True, latent_size=10,
              layer_size=[32, 64, 128], valid_dim=1, epoch=1000, verbose = False, bdmc_run = False,
              n_batch=5, chain_length=100, iwae_samples=1):

    print("Number of Modalities:",len(mod_train))

    modality_num = len(mod_train)
    #modality_size = 1  # here refers to single modality
    estimations_lst = list()
    training_history_loss = list()
    testing_history_loss = list()
    training_history_gen = list()
    training_history_dis = list()
    training_history_disreal = list()
    training_history_disfake = list()
    training_history_distotal = list()
    results_dict = dict()
    m_error = 0
    dfs_train = list()
    dfs_test = list()
    valid_dim = 1
    #epcho = 1000
    # Loss functions
    adversarial_loss = torch.nn.MSELoss()

    print("Training Started!")
    for j in range(0, modality_num):

        # training procedure
        print('''
                Train
                ''')
        #testing_history_loss_tmp = list()
        #training_history_loss_tmp = list()
        training_history_gen_tmp = list()
        training_history_dis_tmp = list()
        training_history_disreal_tmp = list()
        training_history_disfake_tmp = list()
        training_history_distotal_tmp = list()
        print("evaluating # %d modality" % j)

        # only train cgan on single modality
        batch_size = mod_train[j].shape[0]
        mod_dim = mod_train[j].shape[1]
        cond_dim = cond_train.shape[1]
        latent_size = mod_dim
        # Loss functions
        adversarial_loss = torch.nn.MSELoss()

        print("Initialize generator and discriminator")
        generator = m_CGAN.Generator(latent_size, layer_size, conditional, cond_dim, mod_dim)
        D_layer_size = layer_size.copy()
        D_layer_size.reverse()
        discriminator = m_CGAN.Discriminator(mod_dim, D_layer_size, valid_dim)

        optimizer_G = torch.optim.Adamax(generator.parameters(), lr=learning_rate)
        optimizer_D = torch.optim.Adamax(discriminator.parameters(), lr=learning_rate)


        valid = torch.from_numpy(np.ones([batch_size, valid_dim])).float()
        fake = torch.from_numpy(np.zeros([batch_size, valid_dim])).float()

        print('''
                training generator
                ''')
        # print("Sample noise as generator input")
        # print("Generate a batch of modality")
        # print("Calculating Loss measures generator's ability to fool the discriminator")
        # print('''
        # training discriminator
        # ''')
        # print("Calculating Loss for real modality")
        # print("Calculating Loss for fake modality")
        # print("Calculating Total discriminator loss")
        for i in range(epoch):
            optimizer_G.zero_grad()

            # print("Sample noise as generator input")
            z = torch.from_numpy(np.random.normal(0, 1, (batch_size, latent_size))).float()
            # print("Generate a batch of modality")
            gen_mod , train_gen_mu, train_gen_logvar= generator(z, cond_train)
            m_loss_gen, MSE_gen, KLD_gen, RMSE_gen = elbo_loss([mod_train[j]], gen_mod, train_gen_mu, train_gen_logvar)
            training_history_gen_tmp.append(m_loss_gen.data)
            # print("Calculating Loss measures generator's ability to fool the discriminator")
            validity,train_dis_mu, train_dis_logvar = discriminator(gen_mod)

            m_loss_dis, MSE_dis, KLD_dis, RMSE_dis = elbo_loss([mod_train[j]], validity, train_dis_mu, train_dis_logvar)
            training_history_dis_tmp.append(m_loss_dis.data)
            g_loss = adversarial_loss(validity, valid)
            g_loss.backward()
            optimizer_G.step()


            '''
            training discriminator
            '''
            optimizer_D.zero_grad()

            # print("Calculating Loss for real modality")
            validity_real, train_disRealMod_mu, train_disRealMod_logvar  = discriminator(mod_train[j])
            d_real_loss = adversarial_loss(validity_real, valid)

            m_loss_disreal, MSE_disreal, KLD_disreal, RMSE_disreal = elbo_loss([mod_train[j]], validity_real, train_disRealMod_mu,train_disRealMod_logvar)

            training_history_disreal_tmp.append(m_loss_disreal.data)
            # print("Calculating Loss for fake modality")

            validity_fake, train_disFakeMod_mu, train__disFakeMod_logvar  = discriminator(gen_mod.detach())
            d_fake_loss = adversarial_loss(validity_fake, fake)
            m_loss_disfake, MSE_disfake, KLD_disfake, RMSE_disfake = elbo_loss([mod_train[j]], validity_real,
                                                                               train_disRealMod_mu,
                                                                               train_disRealMod_logvar)
            training_history_disfake_tmp.append(m_loss_disfake.data)
            # print("Calculating Total discriminator loss")
            d_loss = (d_real_loss + d_fake_loss) / 2
            try:
                best_loss
            except NameError:
                best_loss = d_loss.detach()
            else:
                if verbose:
                    print("Best Loss Exists:", best_loss)
            is_best = bool(d_loss.detach().numpy() < best_loss.numpy())
            best_loss = torch.FloatTensor(min(d_loss.detach().numpy(), best_loss.numpy()))
            if is_best:
                train_discrim_best_checkpoint = {
                    'epoch': 0 + i,
                    'state_dict': discriminator.state_dict(),
                    'best_loss': best_loss
                }

            d_loss.backward()
            optimizer_D.step()
            training_history_distotal_tmp.append(d_loss.data)
            if i%100 == 0 and verbose is True:
                print("epoch:", i)
                print("Training Generator \tloss:%.2f\tMSE:%.2f\tKLD:%.2f\tRMSE:%.2f" % (
                m_loss_gen.data, MSE_gen, KLD_gen, RMSE_gen))
                print("Training Generator ability to fool Discriminator \tloss:%.2f\tMSE:%.2f\tKLD:%.2f\tRMSE:%.2f" % (
                m_loss_dis.data, MSE_dis, KLD_dis, RMSE_dis))
                print("Adversarial Loss:\tMSE_Loss:%.2f" % g_loss.data)
                print("Training Discriminator Real Modality \tloss:%.2f\tMSE:%.2f\tKLD:%.2f\tRMSE:%.2f" % ( m_loss_disreal.data, MSE_disreal, KLD_disreal, RMSE_disreal))
                print("Training Discriminator Fake Modality \tloss:%.2f\tMSE:%.2f\tKLD:%.2f\tRMSE:%.2f" % ( m_loss_disfake.data, MSE_disfake, KLD_disfake, RMSE_disfake))
                print("Training Discriminator Total Loss \tloss:%.2f" % (d_loss.data))
            #estimations, test_mu, test_logvar = generator(mod_test[j], cond_test)
            #tepoch_error = torch.nn.functional.mse_loss(estimations, mod_test[j])
            #training_history_loss_tmp.append(tepoch_error)
            #results_dict['TrainBestModel'] = train_best_checkpoint


        training_history_gen.append(training_history_gen_tmp)
        training_history_dis.append(training_history_dis_tmp)
        training_history_disreal.append(training_history_disreal_tmp)
        training_history_disfake.append(training_history_disfake_tmp)
        training_history_distotal.append(training_history_disfake_tmp)

        print("Training Done")
        print('''
            evaluation
            ''')
        test_batch_size = cond_test.shape[0]
        #z = torch.from_numpy(np.random.normal(0, 1, (test_batch_size, latent_size))).float()
        #Use Discriminator ?
        estimations, test_mu, test_logvar = generator(mod_test[j], cond_test)
        # t_loss_gen, MSE_gen_test, KLD_gen_test, RMSE_gen_test = elbo_loss([mod_test[j]], estimations, test_mu, test_logvar)
        t_error = torch.nn.functional.mse_loss(estimations, mod_test[j])
        m_error = m_error + t_error
        #testing_history_loss_tmp.append(t_error)
        print("Testing Done!")
        print("Saving Modality:", j, "Estimations")
        estimations_lst.append(estimations)

        if bdmc_run:
            if conditional:
                modeltype = "cgan"
                # forward_logws_train, backward_logws_train, lower_bounds_train, upper_bounds_train = ais_bdmc_lld(
                #     generator,
                #     [mod_train[j]],
                #     latent_size,
                #     cond=cond_train,
                #     n_batch=n_batch,
                #     chain_length=chain_length,
                #     iwae_samples=iwae_samples,
                #     batch_size= batch_size,
                #     modeltype=modeltype)


                forward_logws_test, backward_logws_test, lower_bounds_test, upper_bounds_test = ais_bdmc_lld(
                    discriminator,
                    mod_test[j],
                    latent_size,
                    cond=cond_test,
                    n_batch=n_batch,
                    chain_length=chain_length,
                    iwae_samples=iwae_samples,
                    batch_size=test_batch_size,
                    modeltype=modeltype)

            else:

                modeltype = "gan"
                # forward_logws_train, backward_logws_train, lower_bounds_train, upper_bounds_train = ais_bdmc_lld(
                #     generator,
                #     [mod_train[j]],
                #     latent_size,
                #     cond=cond_train,
                #     n_batch=n_batch,
                #     chain_length=chain_length,
                #     iwae_samples=iwae_samples,
                #     batch_size=batch_size,
                #     modeltype=modeltype)


                forward_logws_test, backward_logws_test, lower_bounds_test, upper_bounds_test = ais_bdmc_lld(
                    discriminator,
                    mod_test[j],
                    latent_size,
                    cond=cond_test,
                    n_batch=n_batch,
                    chain_length=chain_length,
                    iwae_samples=iwae_samples,
                    batch_size=test_batch_size,
                    modeltype=modeltype)

            # train_df = pd.DataFrame([lower_bounds_train, upper_bounds_train]).transpose()
            # dfs_train.append(train_df)

            test_df = pd.DataFrame([lower_bounds_test, upper_bounds_test]).transpose()
            dfs_test.append(test_df)

    if bdmc_run:
        print("Getting Mean LLD across all modalities")
        #lld_ais_train = pd.concat(dfs_train).mean()
        lld_ais_test = pd.concat(dfs_test).mean()
        #print("Average LLD Train Bounds:\t.2f\t.2f\tAverage LLD Test Bounds:\t.2f\t.2f" % (lld_ais_train[1], lld_ais_train[0], lld_ais_test[1], lld_ais_test[0]))
        #print(lld_ais_train[1], lld_ais_train[0], lld_ais_test[1], lld_ais_test[0])
        print(lld_ais_test[0], lld_ais_test[1])
        results_dict['AIS_BDMC_LLD'] = pd.DataFrame([tuple([lld_ais_test[0], lld_ais_test[1]])])
        results_dict['AIS_BDMC_LLD'].columns = ["lower_bounds_test", "upper_bounds_test"]
    training_history_dict = dict()

    # training_history_loss = pd.DataFrame(training_history_loss_tmp).transpose().astype('float')
    # training_history_loss.index.names = ['EPOCH']
    # training_history_loss.columns = ["_".join(["modality", str(x)]) for x in
    #                                 range(0, training_history_loss.columns.__len__())]
    #
    # if conditional:
    #     training_history_dict['cgan_total_modality_loss'] = training_history_loss.mean(axis=1)
    # else:
    #     training_history_dict['gan_total_modality_loss'] = training_history_loss.mean(axis=1)


    training_history_gen = pd.DataFrame(training_history_gen).transpose().astype('float')
    training_history_gen.index.names = ['EPOCH']
    training_history_gen.columns = ["_".join(["modality", str(x)]) for x in
                                   range(0, training_history_gen.columns.__len__())]

    if conditional:
        training_history_dict['cgan_TrainingGenerator_loss'] = training_history_gen.mean(axis=1)
    else:
        training_history_dict['gan_TrainingGenerator_loss'] = training_history_gen.mean(axis=1)


    


    training_history_dis = pd.DataFrame(training_history_dis).transpose().astype('float')
    training_history_dis.index.names = ['EPOCH']
    training_history_dis.columns = ["_".join(["modality", str(x)]) for x in
                                    range(0, training_history_dis.columns.__len__())]
    if conditional:
        training_history_dict['cgan_TrainingGeneratorFoolDiscrim_loss'] = training_history_dis.mean(axis = 1)
    else:
        training_history_dict['gan_TrainingGeneratorFoolDiscrim_loss'] = training_history_dis.mean(axis = 1)



    training_history_disreal = pd.DataFrame(training_history_disreal).transpose().astype('float')
    training_history_disreal.index.names = ['EPOCH']
    training_history_disreal.columns = ["_".join(["modality", str(x)]) for x in
                                    range(0, training_history_disreal.columns.__len__())]

    if conditional:
        training_history_dict['cgan_TrainingDiscriminator_RealMod_loss'] =  training_history_disreal.mean(axis = 1)
    else:
        training_history_dict['gan_TrainingDiscriminator_RealMod_loss'] =  training_history_disreal.mean(axis = 1)


    training_history_disfake = pd.DataFrame(training_history_disfake).transpose().astype('float')
    training_history_disfake.index.names = ['EPOCH']
    training_history_disfake.columns = ["_".join(["modality", str(x)]) for x in
                                    range(0, training_history_disfake.columns.__len__())]


    if conditional:
        training_history_dict['cgan_TrainingDiscriminator_FakeMod_loss'] =  training_history_disfake.mean(axis = 1)
    else:
        training_history_dict['gan_TrainingDiscriminator_FakeMod_loss'] =  training_history_disfake.mean(axis = 1)




    training_history_distotal = pd.DataFrame(training_history_distotal).transpose().astype('float')
    training_history_distotal.index.names = ['EPOCH']
    training_history_distotal.columns = ["_".join(["modality", str(x)]) for x in
                                        range(0, training_history_distotal.columns.__len__())]

    if conditional:
        training_history_dict['cgan_TrainingDiscriminator_TotalLoss'] = training_history_distotal.mean(axis = 1)
    else:
        training_history_dict['gan_TrainingDiscriminator_TotalLoss'] = training_history_distotal.mean(axis = 1)

    # testing_history_gen = pd.DataFrame(testing_history_gen).transpose().astype('float')
    # testing_history_gen.index.names = ['EPOCH']
    # testing_history_gen.columns = ["_".join(["modality", str(x)]) for x in
    #                                 range(0, testing_history_gen.columns.__len__())]
    #
    # if conditional:
    #     training_history_dict['cgan_TestingGenerator_loss'] = testing_history_gen.mean(axis=1)
    # else:
    #     training_history_dict['gan_TestingGenerator_loss'] = testing_history_gen.mean(axis=1)


    if conditional:
        training_history_df = pd.concat([training_history_dict['cgan_TrainingGenerator_loss'],training_history_dict['cgan_TrainingGeneratorFoolDiscrim_loss'],training_history_dict['cgan_TrainingDiscriminator_RealMod_loss']
                             ,training_history_dict['cgan_TrainingDiscriminator_FakeMod_loss'],training_history_dict['cgan_TrainingDiscriminator_TotalLoss']], axis = 1)
        training_history_df.columns = ['cgan_TrainingGenerator_loss','cgan_TrainingGeneratorFoolDiscrim_loss','cgan_TrainingDiscriminator_RealMod_loss','cgan_TrainingDiscriminator_FakeMod_loss','cgan_TrainingDiscriminator_TotalLoss']
    else:
        training_history_df = pd.concat([training_history_dict['gan_TrainingGenerator_loss'],
                                         training_history_dict['gan_TrainingGeneratorFoolDiscrim_loss'],
                                         training_history_dict['gan_TrainingDiscriminator_RealMod_loss']
                                            , training_history_dict['gan_TrainingDiscriminator_FakeMod_loss'],
                                         training_history_dict['gan_TrainingDiscriminator_TotalLoss']], axis=1)
        training_history_df.columns = ['gan_TrainingGenerator_loss', 'gan_TrainingGeneratorFoolDiscrim_loss',
                                   'gan_TrainingDiscriminator_RealMod_loss', 'gan_TrainingDiscriminator_FakeMod_loss',
                                   'gan_TrainingDiscriminator_TotalLoss']

    print("Calculating Testing Error")
    #print("m_error:", m_error)
    m_error = m_error / modality_num
    rmse_error = torch.sqrt(m_error)
    #print("Saving Last Training History")

    if verbose:
        if conditional:
            print("CGAN testing_error (mse):%.2f\t(rmse):%.2f" % (m_error, rmse_error))
        else:
            print("GAN testing_error (mse):%.2f\t(rmse):%.2f" % (m_error, rmse_error))


    results_dict["TrainHist"] = training_history_df
    results_dict['TrainBestModel'] = train_discrim_best_checkpoint
    return m_error, estimations_lst, rmse_error, results_dict




#TODO: Need to incorporate training and testing visualizaitons

#traintest_sets_dict = traintest_sets_dict_timesplit

def GenerativeModels_ScenarioGen(traintest_sets_dict,cond_name = None,learning_rate = 1e-4, iterations = 3, epoch = 1000, models = ["MCVAE", "CVAE", "VAE"]
                                 , splittype = "timesplit", verbose = True,
                                 bdmc_run = False,n_batch = 5, chain_length = 100, iwae_samples = 1):

    print("Comparison on Generative models")
    if cond_name is None:
        cond_name = traintest_sets_dict["cond_name"]
    elif cond_name is None and "cond_name" not in traintest_sets_dict:
        print("Missing Conditional Modality name parameter.")

    mod_train = []
    mod_test = []
    cond_train = None
    cond_test = None
    print("Initializing  Objects")
    print("Setting number of conditional dimensions")
    num_cond = traintest_sets_dict['cond_num']

    print("Assigning conditional modality training set")
    for key in [x for x in traintest_sets_dict.keys() if x.startswith("trainset_data_cond") if
                x.endswith(splittype)]:
        cond_train = traintest_sets_dict[key]
    print("Assigning conditional modality testing set")
    for key in [x for x in traintest_sets_dict.keys() if x.startswith("testset_data_cond") if
                x.endswith(splittype)]:
        cond_test = traintest_sets_dict[key]
    print("Assigning modality training set")
    for key in [x for x in traintest_sets_dict.keys() if x.startswith("trainset_data_mod") if
                x.endswith(splittype)]:
        mod_train = traintest_sets_dict[key]
        print(len(mod_train))
    print("Assigning conditional modality testing set")
    for key in [x for x in traintest_sets_dict.keys() if x.startswith("testset_data_mod") if
                x.endswith(splittype)]:
        mod_test = traintest_sets_dict[key]
        print(len(mod_test))

    print("Running Iterative Loop of the Models")
    print("Initalize Results List")
    result_dict = dict()
    results_lst = list()
    results_rmse_lst = list()
    results_lld_lst = list()
    for model in models:
        tmp_result_obj = []
        tmp_error_obj = []
        tmp_rmse_error_obj = []
        tmp_lld_obj = []

        for i in range(0, iterations):
            print("Model:", model,"Iteration:", i + 1,  " of ",
                  iterations, "Learning Rate:", learning_rate, "Epochs:", epoch,
                  "Conditionality Name:", cond_name)
            print(model, " Modeling")
            #error = 0
            #rmse_error = 0
            #estimation_lst = []

            if model.lower() == "mcvae":
                conditional_tmp = True
                print(model.lower(), "Conditional:", conditional_tmp)
                error, pred_moda, rmse_error, tmp_train_pdf, tmp_test_pdf, tmp_results_dict = train_MCVAE(num_cond, cond_train, cond_test, mod_train,mod_test, learning_rate = learning_rate, epoch = epoch, conditional=True, verbose = verbose, bdmc_run =bdmc_run,n_batch = n_batch, chain_length = chain_length, iwae_samples = iwae_samples)


            elif model.lower() == "cvae":
                conditional_tmp = True
                print(model.lower(),"Conditional:",conditional_tmp)
                error, pred_moda, rmse_error, tmp_train_pdf, tmp_test_pdf, tmp_results_dict = train_CVAE(num_cond,
                                                                                                         cond_train,
                                                                                                         cond_test,
                                                                                                         mod_train,
                                                                                                         mod_test,
                                                                                                         learning_rate,
                                                                                                         epoch=epoch,
                                                                                                         conditional=conditional_tmp,
                                                                                                         verbose=verbose,
                                                                                                         bdmc_run=bdmc_run,
                                                                                                         n_batch=n_batch,
                                                                                                         chain_length=chain_length,
                                                                                                         iwae_samples=iwae_samples)
            elif model.lower() == "vae":
                conditional_tmp = False
                print(model.lower(), "Conditional:", conditional_tmp)
                mod_train_tmp = mod_train.copy()
                mod_train_tmp.append(cond_train)
                mod_test_tmp = mod_test.copy()
                mod_test_tmp.append(cond_test)
                error, pred_moda, rmse_error, tmp_train_pdf, tmp_test_pdf, tmp_results_dict = train_CVAE(num_cond, cond_train,
                                                                                             cond_test, mod_train_tmp,
                                                                                             mod_test_tmp,
                                                                                             learning_rate,
                                                                                             epoch=epoch,
                                                                                             conditional=conditional_tmp,
                                                                                             verbose=verbose,
                                                                                             bdmc_run=bdmc_run,
                                                                                             n_batch=n_batch,
                                                                                             chain_length=chain_length,
                                                                                             iwae_samples=iwae_samples)

            elif model.lower() == "cgan":
                conditional_tmp = True
                print(model.lower(), "Conditional:", conditional_tmp)
                error, pred_moda, rmse_error, tmp_results_dict = train_GAN(num_cond, cond_train, cond_test, mod_train,
                                                                           mod_test, learning_rate, epoch=epoch,
                                                                           conditional=conditional_tmp, bdmc_run=bdmc_run,
                                                                           n_batch=n_batch, chain_length=chain_length,
                                                                           iwae_samples=iwae_samples, verbose=verbose)
                # tmp_trainhist_df = None
                tmp_train_pdf = None
                tmp_test_pdf = None

            elif model.lower() == "gan":
                conditional_tmp = False
                print(model.lower(), "Conditional:", conditional_tmp)
                mod_train_tmp = mod_train.copy()
                mod_train_tmp.append(cond_train)
                mod_test_tmp = mod_test.copy()
                mod_test_tmp.append(cond_test)
                error, pred_moda, rmse_error, tmp_results_dict = train_GAN(num_cond, cond_train, cond_test, mod_train_tmp,
                                                                           mod_test_tmp, learning_rate, epoch=epoch,
                                                                           conditional=conditional_tmp, bdmc_run=bdmc_run,
                                                                           n_batch=n_batch, chain_length=chain_length,
                                                                           iwae_samples=iwae_samples, verbose=verbose)
                # tmp_trainhist_df = None
                tmp_train_pdf = None
                tmp_test_pdf = None

            else:
                print("Model not Found")
                break
                #conditional_tmp = False
            print(model," testing_error (mse):%.2f\t(rmse):%.2f" % (error.item(), rmse_error.item()))
            print("Saving Error from iteration")
            tmp_error_obj.append(error)
            tmp_rmse_error_obj.append(rmse_error)
            if bdmc_run:
                tmp_lld_obj.append(tmp_results_dict['AIS_BDMC_LLD']["upper_bounds_test"][0])
            print("Saving Estimations")
            result_dict["_".join([model.lower(),"pred_moda"])] = pred_moda
            print("Saving Training and Testing Probability Distributions")
            result_dict["_".join([model.lower(), "train_pdf"])] = tmp_train_pdf
            result_dict["_".join([model.lower(), "test_pdf"])] = tmp_test_pdf
            result_dict["_".join([model.lower(), "results_dict"])] = tmp_results_dict

        print("Adding to Results List")
        tmp_result_obj = torch.stack(tmp_error_obj)
        results_lst.append(tmp_result_obj)
        tmp_result_rmse_obj = torch.stack(tmp_rmse_error_obj)
        results_rmse_lst.append(tmp_result_rmse_obj)
        #tmp_lld_obj = torch.stack(tmp_lld_obj)
        if bdmc_run:
            results_lld_lst.append(tmp_lld_obj)
    print("Calculating Average Performance and adding to Results Object")


    results_mse  = pd.DataFrame(results_lst, index = models).astype(float)
    results_mse = results_mse.transpose()
    results_mse.loc["mean"] = results_mse.mean()
    results_mse.loc["std"] = results_mse.iloc[:iterations].std()
    results_mse.loc["iterations"] = iterations
    results_rmse = pd.DataFrame(results_rmse_lst, index=models).astype(float)
    results_rmse = results_rmse.transpose()
    results_rmse.loc["mean"] = results_rmse.mean()
    results_rmse.loc["std"] = results_rmse.iloc[:iterations].std()
    results_rmse.loc["iterations"] = iterations
    if bdmc_run:
        results_lld = pd.DataFrame(results_lld_lst, index=models).astype(float)
        results_lld = results_lld.transpose()
        results_lld.loc["mean"] = results_lld.mean()
        results_lld.loc["std"] = results_lld.iloc[:iterations].std()
        results_lld.loc["iterations"] = iterations
    print("Saving and Printing results")
    result_dict["results_mse"] = results_mse
    print("MSE:\n",result_dict["results_mse"])
    result_dict["results_rmse"] = results_rmse
    print("RMSE:\n",result_dict["results_rmse"])
    if bdmc_run:
        result_dict["results_lld"] = results_lld
        print("LLD:\n",result_dict["results_lld"])
    return(result_dict)

#TODO: Consolidate VAE, CVAE and MCVAE into one function for comparasion.

#
# def LSTM_BankPrediction( pred_moda ,traintest_sets_dict,learn_types = ["Only_Yminus1", "Only_Xminus1","Yminus1&Xminus1","Yminus1&X",
#                                                                        "Yminus1&Xminus1&moda","Yminus1&X&moda","Yminus1&Xminus1&XYCapminus1",
#                                                                        "Yminus1&X&XYCapminus1","Yminus1&Xminus1&XYCapminus1&moda","Yminus1&X&XYCapminus1&moda"], lstm_lr = 1e-2, threshold = 1e-3, modelTarget = "Y"):
#     print("Comparison on LSTM models")
#     #learn_types = ["Only_Y", "Y&X", "Y&X&moda"]
#     rmse_train_list = []
#     rmse_lst = []
#     #ids = 0
#
#     for ids in range(0, len(learn_types)):
#         m_learn_type = learn_types[ids]
#
#         print("Setting Raw Inputs and Raw Evaluation Inputs")
#         #TODO: Address the Static nature of the learn type to raw inputs mapping
#         if m_learn_type == "Only_Yminus1":
#             print(m_learn_type)
#             raw_inputs = traintest_sets_dict["Ytminus1Train"]#train_sets[3]
#             print(raw_inputs.shape)
#             raw_eval_inputs = traintest_sets_dict["Ytminus1Test"]#test_sets[3]
#             print(raw_eval_inputs.shape)
#
#         if m_learn_type == "Only_Xminus1":
#             print(m_learn_type)
#             raw_inputs = traintest_sets_dict["Xtminus1Train"]#train_sets[3]
#             print(raw_inputs.shape)
#             raw_eval_inputs = traintest_sets_dict["Xtminus1Test"]#test_sets[3]
#             print(raw_eval_inputs.shape)
#         #TODO: Address the Static nature of the learn type to raw inputs mapping
#         if m_learn_type == "Yminus1&Xminus1":
#             print(m_learn_type)
#             raw_inputs = torch.cat((traintest_sets_dict["Ytminus1Train"], traintest_sets_dict["Xtminus1Train"]), dim=2)
#             raw_eval_inputs = torch.cat((traintest_sets_dict["Ytminus1Test"], traintest_sets_dict["Xtminus1Test"]), dim=2)
#
#         if m_learn_type == "Yminus1&X":
#             print(m_learn_type)
#             raw_inputs = torch.cat((traintest_sets_dict["Ytminus1Train"], traintest_sets_dict["XTrain"]), dim=2)
#             raw_eval_inputs = torch.cat((traintest_sets_dict["Ytminus1Test"], traintest_sets_dict["XTest"]), dim=2)
#
#         #TODO: Address the Static nature of the learn type to raw inputs mapping
#         if m_learn_type == "Yminus1&Xminus1&moda":
#             print(m_learn_type)
#             raw_inputs = torch.cat((traintest_sets_dict["Ytminus1Train"], traintest_sets_dict["Xtminus1Train"]), dim=2)
#             raw_eval_inputs = torch.cat((traintest_sets_dict["Ytminus1Test"], traintest_sets_dict["Xtminus1Test"]), dim=2)
#             print("in testing stage the modality is applied from the predicted modality from previous stage")
#             #TODO: May need to consider capturing other generative models predictions rather than just MCVAE
#             temp_eval_moda = pred_moda[0]
#             temp_moda = traintest_sets_dict["modTrain"][0]#train_sets[0][0]
#             #TODO: Need additional detail to this part to understand what exactly it is doing.
#             for i in range(1, len(traintest_sets_dict["modTrain"])):
#                 temp_moda = torch.cat((temp_moda, traintest_sets_dict["modTrain"][i]), dim=1)
#                 temp_eval_moda = torch.cat((temp_eval_moda, pred_moda[i]), dim=1)
#             raw_moda = temp_moda.expand_as(torch.zeros([raw_inputs.shape[0], temp_moda.shape[0], temp_moda.shape[1]]))
#             raw_eval_moda = temp_eval_moda.expand_as(torch.zeros([raw_eval_inputs.shape[0], temp_eval_moda.shape[0], temp_eval_moda.shape[1]]))
#             raw_inputs = torch.cat((raw_inputs, raw_moda.double()), dim=2)
#             raw_eval_inputs = torch.cat((raw_eval_inputs, raw_eval_moda.double()), dim=2)
#         if m_learn_type == "Yminus1&X&moda":
#             print(m_learn_type)
#             raw_inputs = torch.cat((traintest_sets_dict["Ytminus1Train"], traintest_sets_dict["XTrain"]), dim=2)
#             raw_eval_inputs = torch.cat((traintest_sets_dict["Ytminus1Test"], traintest_sets_dict["XTest"]), dim=2)
#             print("in testing stage the modality is applied from the predicted modality from previous stage")
#             #TODO: May need to consider capturing other generative models predictions rather than just MCVAE
#             temp_eval_moda = pred_moda[0]
#             temp_moda = traintest_sets_dict["modTrain"][0]#train_sets[0][0]
#             #TODO: Need additional detail to this part to understand what exactly it is doing.
#             for i in range(1, len(traintest_sets_dict["modTrain"])):
#                 temp_moda = torch.cat((temp_moda, traintest_sets_dict["modTrain"][i]), dim=1)
#                 temp_eval_moda = torch.cat((temp_eval_moda, pred_moda[i]), dim=1)
#             raw_moda = temp_moda.expand_as(torch.zeros([raw_inputs.shape[0], temp_moda.shape[0], temp_moda.shape[1]]))
#             raw_eval_moda = temp_eval_moda.expand_as(torch.zeros([raw_eval_inputs.shape[0], temp_eval_moda.shape[0], temp_eval_moda.shape[1]]))
#             raw_inputs = torch.cat((raw_inputs, raw_moda.double()), dim=2)
#             raw_eval_inputs = torch.cat((raw_eval_inputs, raw_eval_moda.double()), dim=2)
#
#         if m_learn_type == "Yminus1&Xminus1&XYCapminus1":
#             print(m_learn_type)
#             raw_inputs = torch.cat((traintest_sets_dict["Ytminus1Train"], traintest_sets_dict["Xtminus1Train"]), dim=2)
#             raw_inputs = torch.cat((raw_inputs, traintest_sets_dict["XYCapTminus1Train"]), dim=2)
#             raw_eval_inputs = torch.cat((traintest_sets_dict["Ytminus1Test"], traintest_sets_dict["Xtminus1Test"]), dim=2)
#             raw_eval_inputs = torch.cat((raw_eval_inputs , traintest_sets_dict["XYCapTminus1Test"]), dim=2)
#
#
#
#         if m_learn_type == "Yminus1&X&XYCapminus1":
#             print(m_learn_type)
#             raw_inputs = torch.cat((traintest_sets_dict["Ytminus1Train"], traintest_sets_dict["XTrain"]), dim=2)
#             raw_inputs = torch.cat((raw_inputs, traintest_sets_dict["XYCapTminus1Train"]), dim=2)
#             raw_eval_inputs = torch.cat((traintest_sets_dict["Ytminus1Test"], traintest_sets_dict["XTest"]), dim=2)
#             raw_eval_inputs = torch.cat((raw_eval_inputs , traintest_sets_dict["XYCapTminus1Test"]), dim=2)
#
#
#         if m_learn_type == "Yminus1&Xminus1&XYCapminus1&moda":
#             print(m_learn_type)
#             raw_inputs = torch.cat((traintest_sets_dict["Ytminus1Train"], traintest_sets_dict["Xtminus1Train"]), dim=2)
#             raw_inputs = torch.cat((raw_inputs, traintest_sets_dict["XYCapTminus1Train"]), dim=2)
#             raw_eval_inputs = torch.cat((traintest_sets_dict["Ytminus1Test"], traintest_sets_dict["Xtminus1Test"]), dim=2)
#             raw_eval_inputs = torch.cat((raw_eval_inputs , traintest_sets_dict["XYCapTminus1Test"]), dim=2)
#             for i in range(1, len(traintest_sets_dict["modTrain"])):
#                 temp_moda = torch.cat((temp_moda, traintest_sets_dict["modTrain"][i]), dim=1)
#                 temp_eval_moda = torch.cat((temp_eval_moda, pred_moda[i]), dim=1)
#             raw_moda = temp_moda.expand_as(torch.zeros([raw_inputs.shape[0], temp_moda.shape[0], temp_moda.shape[1]]))
#             raw_eval_moda = temp_eval_moda.expand_as(torch.zeros([raw_eval_inputs.shape[0], temp_eval_moda.shape[0], temp_eval_moda.shape[1]]))
#             raw_inputs = torch.cat((raw_inputs, raw_moda.double()), dim=2)
#             raw_eval_inputs = torch.cat((raw_eval_inputs, raw_eval_moda.double()), dim=2)
#
#
#         if m_learn_type == "Yminus1&X&XYCapminus1&moda":
#             print(m_learn_type)
#             raw_inputs = torch.cat((traintest_sets_dict["Ytminus1Train"], traintest_sets_dict["XTrain"]), dim=2)
#             raw_inputs = torch.cat((raw_inputs, traintest_sets_dict["XYCapTminus1Train"]), dim=2)
#             raw_eval_inputs = torch.cat((traintest_sets_dict["Ytminus1Test"], traintest_sets_dict["XTest"]), dim=2)
#             raw_eval_inputs = torch.cat((raw_eval_inputs , traintest_sets_dict["XYCapTminus1Test"]), dim=2)
#             for i in range(1, len(traintest_sets_dict["modTrain"])):
#                 temp_moda = torch.cat((temp_moda, traintest_sets_dict["modTrain"][i]), dim=1)
#                 temp_eval_moda = torch.cat((temp_eval_moda, pred_moda[i]), dim=1)
#             raw_moda = temp_moda.expand_as(torch.zeros([raw_inputs.shape[0], temp_moda.shape[0], temp_moda.shape[1]]))
#             raw_eval_moda = temp_eval_moda.expand_as(torch.zeros([raw_eval_inputs.shape[0], temp_eval_moda.shape[0], temp_eval_moda.shape[1]]))
#             raw_inputs = torch.cat((raw_inputs, raw_moda.double()), dim=2)
#             raw_eval_inputs = torch.cat((raw_eval_inputs, raw_eval_moda.double()), dim=2)
#
#
#         print("Setting Inputs and Target Parameters for Training")
#         # TODO: Investigate how to resolve the static nature of setting the target
#         # TODO: May need to add the Capital Ratios part as targets.
#
#         if modelTarget == "Y":
#             raw_targets = traintest_sets_dict["YTrain"]
#             raw_eval_targets = traintest_sets_dict["YTest"]
#         elif modelTarget == "XYCap":
#             raw_targets = traintest_sets_dict["XYCapTrain"]
#             raw_eval_targets = traintest_sets_dict["XYCapTest"]
#         elif modelTarget == "X":
#             raw_targets = traintest_sets_dict["XTrain"]
#             raw_eval_targets = traintest_sets_dict["XTest"]
#         else:
#             print("Setting Default Target as Y")
#             raw_targets = traintest_sets_dict["YTrain"]
#             raw_eval_targets = traintest_sets_dict["YTest"]
#
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
#         m_lstm, train_loss = m_LSTM.train(inputs, targets, 50, lstm_lr, threshold)
#
#         # # Graphical LSTM MSE representaiton.
#         # with torch.no_grad():
#         #      prediction = m_lstm.forward(xtest).view(-1)
#         #      loss = criterion(prediction, ytest)
#         #      plt.title("MESLoss: {:.5f}".format(loss))
#         #      plt.plot(prediction.detach().numpy(), label="pred")
#         #      plt.plot(ytest.detach().numpy(), label="true")
#         #      plt.legend()
#         #      plt.show()
#
#         print("Calculating Training RMSE")
#         rmse_train_list.append(train_loss)
#         print("%s\terror:\t%.5f" % (m_learn_type, train_loss))
#
#         print("Setting Inputs and Target Parameters for Testing")
#         #raw_eval_inputs = raw_inputs = torch.cat((test_sets[3], test_sets[2]), dim=2)
#         #TODO: Investigate how to resolve the static nature of setting the target
#         #raw_eval_targets = traintest_sets_dict["YTest"] #test_sets[4]
#         n, t, m1 = raw_eval_inputs.shape
#         m2 = raw_eval_targets.shape[2]
#         inputs = torch.zeros([t, m1, n]).float()
#         targets = torch.zeros([t, n, m2]).float()
#         for i in range(0, n):
#             inputs[:, :, i] = raw_eval_inputs[i, :, :]
#             targets[:, i, :] = raw_eval_targets[i, :, :]
#
#
#         print("Running Predictions on Inputs using Trained Model: Testing Error")
#         pred = m_LSTM.predict(m_lstm, inputs)
#
#
#         print("Calculating Testing RMSE")
#         rmse = torch.nn.functional.mse_loss(pred, targets)
#         rmse_lst.append(rmse)
#         print("%s\terror:\t%.5f" % (m_learn_type, rmse))
#
#     rmse_train_lst_sk = torch.stack(rmse_train_list)
#     rmse_lst_sk = torch.stack(rmse_lst)
#     rmse_list_final = [rmse_train_lst_sk, rmse_lst_sk.data]
#     result_obj = pd.DataFrame(rmse_list_final, columns = learn_types, index = ["TrainErr","TestErr"])
#     return(result_obj)
#
#
#

#TODO: Need to incorporate training and testing visualizaitons
#if __name__=="__main__":

#TODO: fix so it checks for minimums and maximumss for the datasets so no error occurs at LSTM
#TODO: Train window must be above 0. Add logic to get min and max for the window.


def BankPerfPred(ScenarioGenResults_dict , traintest_sets_dict, generativemodel = ["mcvae"]
                      , lstm_lr = 1e-2, threshold = 1e-4, modelTarget = "Y",
                      exclude = ["trainset_data_cond"], basetraindataset = "trainset_data_X" , basetestdataset = "testset_data_X"
                      , epoch = 1000,splittype = "timesplit", include = None, predictmodel = 'LSTM'):
    # traintest_sets_dict = traintest_sets_dict_timesplit
    # ScenarioGenResults_dict = ScenarioGenResults_dict_timesplit
    #exclude = ["CapRatios"]
    print("Comparison on Prediction models")
    rmse_train_list = []
    rmse_lst = []
    mse_train_list = []
    mse_lst = []
    learn_types_list = []
    train_trainhist_dict = dict()

    #exclude = []
    print("Getting All combinations of Data for LSTM")

    print("Checking to convert Conditional Modality into 3 dimensions.")

    cond_data_sets = list(filter(None, [x for x in traintest_sets_dict.keys() if "_data_cond_" in x]))


    for condkeyname in cond_data_sets:
        if condkeyname.startswith("trainset_"):
            n = traintest_sets_dict[basetraindataset].shape[0]
        elif condkeyname.startswith("testset_"):
            n = traintest_sets_dict[basetestdataset].shape[0]
        if len(traintest_sets_dict[condkeyname].shape) == 2:
            print("Found two dimensional Modality Converting to 3 dimensional with N based on dataset: %s" % (condkeyname))
            cond_tmp = np.zeros([n,
                                 traintest_sets_dict[condkeyname].shape[0],
                                 traintest_sets_dict[condkeyname].shape[1]])

            # cond_tmp.shape
            for i in range(0, n):
                cond_tmp[i, :, :] = np.expand_dims(traintest_sets_dict[condkeyname], axis=0)




            print("Appending to exclusion list")
            exclude.append(condkeyname)
            print("Converting to Tensor object")
            cond_tmp = np.nan_to_num(cond_tmp)
            cond_tmp = torch.from_numpy(cond_tmp).float()
            traintest_sets_dict["_".join([condkeyname,"3D"])] = cond_tmp
            print(traintest_sets_dict["_".join([condkeyname,"3D"])].shape)


    print("Updating TrainTest Exclusion list: Adding ModelTarget:", modelTarget)
    exclude.append(modelTarget)
    print("Converting Exclusion list to tuple")
    exclude = tuple(exclude)
    print("Exclusion List:", exclude)
    trainsets = list(filter(None,[x for x in traintest_sets_dict.keys() if x.startswith(("trainset_data","trainset_mod")) if x.endswith(splittype) if not x.endswith(exclude)]))
    dataset_subsets = list()
    for L in range(0, len(trainsets) + 1):
        for subset in itertools.combinations(trainsets, L):
            dataset_subsets.append(list(subset))
    dataset_subsets = [x for x in dataset_subsets if x !=[]]


    if include is not None:
        print("Filtering out Datasubsets based on include logic:", include)
        dataset_subsets = [x for x in dataset_subsets if set(include) <= set(x)]



    #genmodel = "mcvae"
    for genmodel in generativemodel:

        for subset in dataset_subsets:
            tmp_dict_name = "&".join(["_".join(x.split("_")[2:]) for x in subset])
            tmp_dict_name = tmp_dict_name + "&GenModel_" + genmodel
            print("Features to be used:", tmp_dict_name)
            print("Current Subset:", subset)
            subset_test_tmp = [x.replace("trainset", "testset") for x in subset]
            print("Create testset names from subset", subset_test_tmp)

            print("Set Raw Inputs")
            print("Setting Training Input")
            if len(subset) == 1 and not any("data_mod" in x for x in subset):
                raw_inputs  = traintest_sets_dict[subset[0]]
                print("Raw Training Set Input Shape:", raw_inputs.shape)
                print("Setting Testing\Eval Input")
                raw_eval_inputs = traintest_sets_dict[subset_test_tmp[0]]
            elif len(subset) > 1 and not any("data_mod" in x for x in subset):
                print("Setting Initial Training Input")
                raw_inputs  = traintest_sets_dict[subset[0]]
                print("Setting Initial Testing\Eval Input")
                raw_eval_inputs = traintest_sets_dict[subset_test_tmp[0]]
                for subcnt in range(1,len(subset)):
                    print("Setting Training Input:", subcnt)
                    raw_inputs = torch.cat((raw_inputs.double(), traintest_sets_dict[subset[subcnt]].double()), dim=2)
                    print("Setting Testing\Eval Input:", subcnt)
                    raw_eval_inputs = torch.cat((raw_eval_inputs.double(), traintest_sets_dict[subset_test_tmp[subcnt]].double()), dim=2)
            elif len(subset) > 1 and any("data_mod" in x for x in subset):

                print("Create subset without modality dataset")
                subset_train_tmp = [x for x in subset if not "data_mod" in x]
                print("Setting Initial Training Input")
                raw_inputs = traintest_sets_dict[subset_train_tmp[0]]
                print("Setting Initial Testing\Eval Input")
                subset_test_tmp = [x.replace("trainset", "testset") for x in subset_train_tmp]
                raw_eval_inputs = traintest_sets_dict[subset_test_tmp[0]]
                for subcnt in range(1, len(subset_train_tmp)):
                    print("Setting Training Input:", subcnt)
                    raw_inputs = torch.cat((raw_inputs, traintest_sets_dict[subset_train_tmp[subcnt]]), dim=2)
                    print("Setting Testing\Eval Input:", subcnt)
                    raw_eval_inputs = torch.cat((raw_eval_inputs, traintest_sets_dict[subset_test_tmp[subcnt]]), dim=2)

                print("Append Modality Data to Tensor")
                #Maybe consider running with all different gen models at once
                print("Assigning Predicted Modality Estimates")
                pred_moda = ScenarioGenResults_dict["_".join([genmodel,"pred","moda"])]
                print("Assigning Modality Training and Testing Key names")
                modTrain = [x for x in subset if "data_mod" in x][0]
                modTest = [x.replace("trainset","testset") for x in subset if "data_mod" in x][0]

                if genmodel in ["vae", "gan"]:
                    print("Combining Conditional Modality with Regular Modalities:",genmodel)
                    modTrainkeyname = [x for x in traintest_sets_dict_timesplit.keys() if
                                       x.startswith("trainset_data_mod")]
                    condTrainkeyname = [x for x in traintest_sets_dict_timesplit.keys() if
                                        x.startswith("trainset_data_cond")]

                    modTrain_tmp = traintest_sets_dict_timesplit[modTrainkeyname[0]].copy()
                    modTrain_tmp.append(traintest_sets_dict_timesplit[condTrainkeyname[0]])

                    print("Initalizing Modality objects")
                    temp_eval_moda = pred_moda[0]
                    temp_moda = modTrain_tmp[0]  # train_sets[0][0]
                    print("Iterating Modalities and adding to Tensor")
                    for i in range(1, len(modTrain_tmp)):
                        print("Appending Modality:", i, "to modality training and testing objects")
                        temp_moda = torch.cat((temp_moda, modTrain_tmp[i]), dim=1)
                        temp_eval_moda = torch.cat((temp_eval_moda, pred_moda[i]), dim=1)

                else:
                    print("Initalizing Modality objects")
                    temp_eval_moda = pred_moda[0]
                    temp_moda = traintest_sets_dict[modTrain][0]  # train_sets[0][0]
                    print("Iterating Modalities and adding to Tensor")
                    for i in range(1, len(traintest_sets_dict[modTrain])):
                        print("Appending Modality:", i, "to modality training and testing objects")
                        temp_moda = torch.cat((temp_moda, traintest_sets_dict[modTrain][i]), dim=1)
                        temp_eval_moda = torch.cat((temp_eval_moda, pred_moda[i]), dim=1)

                raw_moda = temp_moda.expand_as(torch.zeros([raw_inputs.shape[0], temp_moda.shape[0], temp_moda.shape[1]]))
                raw_eval_moda = temp_eval_moda.expand_as(
                    torch.zeros([raw_eval_inputs.shape[0], temp_eval_moda.shape[0], temp_eval_moda.shape[1]]))
                raw_inputs = torch.cat((raw_inputs.double(), raw_moda.double()), dim=2)
                raw_eval_inputs = torch.cat((raw_eval_inputs.double(), raw_eval_moda.double()), dim=2)
            else:
                print("Skipping this scenario")
                continue
            print("Setting Inputs and Target Parameters for Training")
            model_raw_target_keyname = [x for x in traintest_sets_dict.keys() if x.startswith("trainset") if x.endswith(modelTarget)][0]
            model_raw_eval_target_keyname = [x for x in traintest_sets_dict.keys() if x.startswith("testset") if x.endswith(modelTarget)][0]
            raw_targets = traintest_sets_dict[model_raw_target_keyname]
            raw_eval_targets = traintest_sets_dict[model_raw_eval_target_keyname]

            print("Setting up Dimensions for inputs and targets for Training model:", predictmodel)

            if splittype == "timesplit":
                n, t, m1 = raw_inputs.shape
                m2 = raw_targets.shape[2]
                inputs_train = torch.zeros([t, m1, n]).float()
                targets_train = torch.zeros([t, n, m2]).float()
                for i in range(0, n):
                    inputs_train[:, :, i] = raw_inputs[i, :, :]
                    targets_train[:, i, :] = raw_targets[i, :, :]

            elif splittype == "banksplit":
                n, t, m1 = raw_inputs.shape
                m2 = raw_targets.shape[2]
                inputs_train = torch.zeros([n, m1, t]).float()
                targets_train = torch.zeros([n, t, m2]).float()
                for i in range(0, t):
                    inputs_train[:, :, i] = raw_inputs[:, i, :]
                    targets_train[:, i, :] = raw_targets[:, i, :]

######### Add more Prediction Models
            print("Training:", predictmodel.lower())
            if predictmodel.lower() in ['lstm']:

                model, train_loss, train_rmse, train_trainhist = m_LSTM.train(inputs_train, targets_train, epoch, lstm_lr, threshold)
                train_trainhist_dict["_".join([tmp_dict_name,"trainhist",predictmodel,modelTarget])] = train_trainhist
                model_trained = m_LSTM
            elif predictmodel.lower() in ['lr','linearregression','linreg']:

                model, train_loss, train_rmse, train_trainhist = m_LinReg.train(inputs_train, targets_train, epoch,lstm_lr, threshold)
                #train_trainhist_dict["_".join([tmp_dict_name, "trainhist", predictmodel])] = train_trainhist
                model_trained = m_LinReg
            elif predictmodel.lower() in ['gru']:

                model, train_loss, train_rmse, train_trainhist = m_GRU.train(inputs_train, targets_train, epoch,
                                                                                lstm_lr, threshold)
                train_trainhist_dict["_".join([tmp_dict_name, "trainhist", predictmodel,modelTarget])] = train_trainhist
                model_trained = m_GRU


            elif predictmodel.lower() in ['darnn',"da-rnn","dualstagelstm","dual-stage-lstm"]:
                #importlib.reload(m_DA_RNN)
                model, train_loss, train_rmse, train_trainhist = m_DA_RNN.train(inputs_train, targets_train, epoch, lstm_lr , threshold)
                train_trainhist_dict["_".join([tmp_dict_name, "trainhist", predictmodel,modelTarget])] = train_trainhist
                model_trained = m_DA_RNN


            else:
                print("No Prediction Model Found:",predictmodel)
                return

            print("Calculating Training RMSE:\t",modelTarget)
            mse_train_list.append(train_loss)
            rmse_train_list.append(train_rmse)
            print("%s\tMSEerror:\t%.5f\tRMSEerror:\t%.5f" % (tmp_dict_name, train_loss, train_rmse))
            #print("Calculating Training Accuracy")
            #acc_train = accuracy(m_lstm, inputs_train, targets_train, .05)
            #print("%s\tMSEerror:\t%.5f\tRMSEerror:\t%.5f\tAccuracy:\t%.5f" % (tmp_dict_name, train_loss, train_rmse,acc_train))



            print("Setting up Dimensions for inputs and targets for Testing model:", predictmodel.lower())
            if splittype == "timesplit":
                n, t, m1 = raw_eval_inputs.shape
                m2 = raw_eval_targets.shape[2]
                inputs_test = torch.zeros([t, m1, n]).float()
                targets_test = torch.zeros([t, n, m2]).float()
                for i in range(0, n):
                    inputs_test[:, :, i] = raw_eval_inputs[i, :, :]
                    targets_test[:, i, :] = raw_eval_targets[i, :, :]

            elif splittype == "banksplit":
                n, t, m1 = raw_eval_inputs.shape
                m2 = raw_eval_targets.shape[2]
                #t2 = raw_eval_targets.shape[1]
                #n2 = raw_eval_targets.shape[1]
                inputs_test = torch.zeros([n, m1, t]).float()
                targets_test = torch.zeros([n, t, m2]).float()
                for i in range(0, t):
                    inputs_test[:, :, i] = raw_eval_inputs[ :,i, :]
                    targets_test[:, i, :] = raw_eval_targets[:, i, :]
######### Add more Prediction Models
            #TODO: Consolidate the models into a loop for training and testing.

            # if predictmodel in ['LSTM']:
            #     print("Running Predictions on Inputs using Trained Model: Testing Error")
            #     pred = m_LSTM.predict(model, inputs_test)
            # elif predictmodel.lower() in ['lr', 'linearregression', 'linreg']:
            #     print("Running Predictions on Inputs using Trained Model: Testing Error")
            #     pred = m_LinReg.predict(model, inputs_test)
            # elif predictmodel.lower() in ['gru']:
            #     pred = m_GRU.predict(model, inputs_test)
            # else:
            #     print("No Prediction Model Found:", predictmodel)
            #     return
            if predictmodel.lower() in ['darnn', "da-rnn", "dualstagelstm", "dual-stage-lstm"]:
                pred = model_trained.predict(model, inputs = inputs_test, targets = targets_train)
            else:
                pred = model_trained.predict(model, inputs_test)
            print("Calculating Testing RMSE:\t",modelTarget)
            #print(pred.shape, targes_test.shape)
            mse = torch.nn.functional.mse_loss(pred, targets_test)
            rmse = torch.sqrt(torch.nn.functional.mse_loss(pred, targets_test))
            rmse_lst.append(rmse)
            mse_lst.append(mse)
            learn_types_list.append(tmp_dict_name)
            #acc_test = accuracy(m_lstm, inputs_test, targets_test, .05)
            #print("%s\tMSEerror:\t%.5f\tRMSEerror:\t%.5f\tAccuracy:\t%.5f" % (
            #tmp_dict_name, train_loss, train_rmse, acc_test))
            print("%s\tMSEerror:\t%.5f\tRMSEerror:\t%.5f" % (tmp_dict_name, mse, rmse))
            #print(accuracy_score(targets_test, pred))


            # print("Graphical Visualization")
            # lstm_plot = m_LSTM.plot(m_lstm, inputs_test,targets_test,figname=tmp_dict_name)
            # print(lstm_plot)
            # # # Graphical LSTM MSE representaiton.
            # with torch.no_grad():
            #     #prediction = m_lstm.forward(targets_test).view(-1)
            #     for i in range(0,pred.shape[1]):
            #
            #         mloss = torch.nn.MSELoss()
            #         loss = mloss(pred[:,i,:], targets_test[:,i,:])
            #         if loss < 100:
            #
                #         # i = 664
                #         print(i,"MSELoss: {:.5f}".format(loss))
                #         plt.title("MSELoss: {:.5f}".format(loss))
                #         plt.plot(pred[:,i,:].detach().numpy(), label="pred")
                #         plt.plot(targets_test[:,i,:].detach().numpy(), label="true")
                #         plt.legend()
                #         plt.show()
                # plt.close()


    rmse_train_lst_sk = torch.stack(rmse_train_list)
    rmse_lst_sk = torch.stack(rmse_lst)
    rmse_list_final = [rmse_train_lst_sk, rmse_lst_sk.data]
    result_obj = pd.DataFrame(rmse_list_final, columns = learn_types_list, index = ["TrainErr","TestErr"])


    print("Target:",modelTarget,"\nPredictModel:",predictmodel,"\nMinimum Training Error:",result_obj.astype(float).idxmin(axis = 1)[0]
        , result_obj.min(axis=1)[0], "\nMinimum Testing Error:",result_obj.astype(float).idxmin(axis = 1)[1], result_obj.min(axis=1)[1])
    return result_obj,train_trainhist_dict, model




#TODO: Should Consider Bank Split training and testing sets. Randomize index.
#TODO: The paper predicts for Loss Loss Rate as a combined aggregate
#TODO: Paper gets probability build_datasetsdistribution for scenarios, not predictions.
#TODO: They are able to get a capital ratio for case study part per scenario per year.
#TODO: May need to run for all scenarios, then join to ground truth data based on proximity from prediction to bank.
#TODO: Consider T - 1, T -2, T- 3, T - 4, to help
#TODO: Consider additional Modalities to help with LSTM [CDX index]




#For quarter_id Quarterly and Yearly
#For splittype in ['Time', 'Bank']
#IMBSTP Experiment 1 Modality Train 1976Q1-2007Q4, Test 2008Q1-2016Q4
#                    BankData Train 1990Q1-2007Q4, Test 2008Q1-2016Q4


#IMBSTP Experiment 2 Modality Train 1976Q1-2015Q4, Test 2008Q1-2016Q4
#                    BankData Train 1990Q1-2015Q4, Test 2016Q1-2017Q1




#ECE Experiments.
# Experiment 1    1976 Q1 - 2007 Q4 2008 Q1 - 2009 Q4    Financial Crisis
# Experiment 2    1976 Q1 - 2015 Q4 2016 Q1 - 2017 Q4    All Econ Data
# Experiment 3 1990 Q1 - 2007 Q4    2008 Q1 - 2009 Q4   Aligned Bank Data & Financial Crisis
# Experiment 4    1990 Q1 - 2015 Q4 2016 Q1 - 2017 Q4    Aligned Bank & Econ Data
#
# #Load PreprocessDict
os.chdir("/Users/phn1x/icdm2018_research_BB/Stress_Test_Research/Loss_projections/")
tmp_name = os.path.join(os.getcwd(),"data","BankPerf_Preprocess.pkl")
f = open(tmp_name, "rb")
Preprocess_Dict = pickle.load(f)
f.close()
#del Preprocess_Dict1

path_dict = {"path_root": os.path.join(os.getcwd(), "data/"),
             "X": "data_X.npy",
             "X_loancat": "data_X_loancat.npy",
             "X_ppnr": "data_X_ppnr.npy",
             "Y": "data_Y.npy",
             "Y_nco": "data_Y_nco.npy",
             "Y_Cap": "data_Y_Cap.npy",
             "Y_target_CapRatio" : "data_Y_target_CapRatio.npy",
             "Y_target_ncoR" : "data_Y_target_ncoR.npy",
              "X_Y_NCO_norm": 'data_X_Y_NCO_norm.npy',
             "X_Y_NCO_pca": 'data_X_Y_NCO_pca.npy',
             "X_Y_NCO_all_pca": 'data_X_Y_NCO_all_pca.npy',
             "CapRatios": "data_CapRatios.npy",
             "NCO": "data_NCO.npy",
             "Moda_prefix": "data_moda_",
             "Moda_suffix": "_pca_all.npy"

             }
path_qtr_dict = {"path_root": os.path.join(os.getcwd(), "data/quarter_based/"),
                 "X": "data_X_quarter.npy",
                 "X_loancat": "data_X_loancat_quarter.npy",
                 "X_ppnr": "data_X_ppnr_quarter.npy",
                 "Y": "data_Y_quarter.npy",
                 "Y_nco": "data_Y_nco_quarter.npy",
                 "Y_Cap": "data_Y_Cap_quarter.npy",
                 "Y_target_CapRatio": "data_Y_target_CapRatio_quarter.npy",
                 "Y_target_ncoR": "data_Y_target_ncoR_quarter.npy",
                 "X_Y_NCO_norm_quarter": 'data_X_Y_NCO_norm_quarter.npy',
                 "X_Y_NCO_pca": 'data_X_Y_NCO_pca_quarter.npy',
                 "X_Y_NCO_all_pca": 'data_X_Y_NCO_all_pca_quarter.npy',
                 "CapRatios": "data_CapRatios_quarter.npy",
                 "NCO": "data_NCO_quarter.npy",
                 "Moda_prefix": "data_moda_",
                 "Moda_suffix": "_pca_all_quarter.npy"

                 }

experiments = {
                'ECE_Experiment 1' : {
                    'Experiment_Info' : {'Experiment_Type' :'ECE','ExperimentNum' : 1
                                         ,'Description' : "Financial Crisis"},
                    'get_raw_train_data' : {'moda_train_date' : ["1976 Q1","2007 Q4"]
                                            ,'datamoda_test_date' : ["2008 Q1","2009 Q4"]}},

                'ECE_Experiment 2' : {
                    'Experiment_Info' : {'Experiment_Type' :'ECE','ExperimentNum' : 2
                                        ,'Description' : "All Econ Data"},
                    'get_raw_train_data' : {'moda_train_date' : ["1976 Q1","2015 Q4"], 'datamoda_test_date' : ["2016 Q1","2017 Q4"]}},

                'ECE_Experiment 3' : {
                    'Experiment_Info' : {'Experiment_Type' :'ECE','ExperimentNum' : 3,'Description' : "Aligned Bank Data & Financial Crisis"},
                    'get_raw_train_data' : {'moda_train_date' : ["1990 Q1","2007 Q4"],'datamoda_test_date' : ["2008 Q1","2009 Q4"]}},

                'ECE_Experiment 4': {
                    'Experiment_Info': {'Experiment_Type' :'ECE','ExperimentNum': 4,'Description': "Aligned Bank & Econ Data"},
                    'get_raw_train_data': {'moda_train_date': ["1990 Q1", "2015 Q4"],'datamoda_test_date': ["2016 Q1", "2017 Q4"]}},

                # 'BCLP_Experiment 1': {
                #     'Experiment_Info': {'Experiment_Type' :'BCLP','ExperimentNum': 1,'Description': "Projections After Financial Crisis"},
                #     'get_raw_train_data': {'moda_train_date': ["1990 Q1", "2007 Q4"],'data_train_date': ["1990 Q1", "2007 Q4"], 'datamoda_test_date': ["2008 Q1", "2016 Q4"], 'data_names': ["X_loancat", "X_ppnr", "Y_nco", "CapRatios","Y_target_CapRatio","Y_target_ncoR"]},
                #     'GenerativeModels_ScenarioGen' : {'models':['mcvae'],'bdmc_run' : False },
                #     'BankPerfPred' : {'generativemodel':['mcvae'],"predictmodels" : ['darnn'],'epoch': 5, 'modelTargets': ["Y_target_CapRatio","Y_target_ncoR"],'lstm_lr': 1e-3,
                #      'threshold': 1e-4}},
                'BCLP_Experiment 2': {
                    'Experiment_Info': {'Experiment_Type' :'BCLP','ExperimentNum': 2,'Description': "Projections After Financial Crisis"},
                    'get_raw_train_data': {'moda_train_date': ["1990 Q1", "2015 Q4"],'data_train_date': ["1990 Q1", "2015 Q4"], 'datamoda_test_date': ["2016 Q1", "2017 Q1"],'data_names': ["X_loancat", "X_ppnr", "Y_nco", "CapRatios","Y_target_CapRatio","Y_target_ncoR"]},
                    'BankPerfPred' : {'generativemodel':['mcvae'], "predictmodels" : ['darnn'],'epoch': 5, 'modelTargets': ["Y_target_CapRatio","Y_target_ncoR"], 'lstm_lr': 1e-3,
                     'threshold': 1e-4}}
                }




#
# experiment_base_dict_orig = {
#         'Experiment_Info': {'Experiment_Type': "ECE", "ExperimentNum": 0, "Description": 'Financial Crisis',
#                             "Interval": 'Quarterly', 'savedir': "./Images/Table_Data"},
#         'get_raw_train_data': {
#             'quarter_ID': None,
#             'cond_name': 'zmacro_domestic',
#             'moda_names': ['zmacro_domestic', 'zmacro_international', "zmicro", "sectsbidx"],
#             'data_names': ["X_loancat", "X_ppnr", "Y_nco", "CapRatios"],
#             'path_dict': path_dict,
#             'path_qtr_dict': path_qtr_dict,
#             'modaDateIdx': Preprocess_Dict['moda_DateIdx'],
#             'dataDateIdx': Preprocess_Dict['data_DateIdx'],
#             'moda_train_date': ["1976 Q1", "2007 Q4"],
#             'datamoda_test_date': ["2008 Q1", "2009 Q4"],
#             'data_train_date': ["1990 Q1", "2015 Q4"],
#             'splittype': 'timesplit'},
#         'GenerativeModels_ScenarioGen': {
#             'learning_rate': 1e-3,
#             'iterations': 1,
#             'epoch': 1000,
#             'data_names': ["X_loancat", "X_ppnr", "Y_nco", "CapRatios"],
#             'models': ["mcvae", "cvae", "vae", "cgan", "gan"],
#             'verbose': False,
#             'bdmc_run': True,
#             'chain_length': 50,
#             'n_batch': 1,
#             'splittype': 'timesplit'}
#     }
#

experiment_base_dict_orig = {
    'Experiment_Info': {'Experiment_Type': "ECE", "ExperimentNum": 0, "Description": 'Financial Crisis',
                        "Interval": 'Quarterly', 'savedir': "./Images/Table_Data"},
    'get_raw_train_data': {
        'quarter_ID': None,
        'cond_name': 'zmacro_domestic',
        'moda_names': ['zmacro_domestic', 'zmacro_international', "zmicro", "sectsbidx"],
        'data_names': ["X_loancat", "X_ppnr", "Y_nco", "CapRatios"],
        'path_dict': path_dict,
        'path_qtr_dict': path_qtr_dict,
        'modaDateIdx': Preprocess_Dict['moda_DateIdx'],
        'dataDateIdx': Preprocess_Dict['data_DateIdx'],
        'moda_train_date': ["1976 Q1", "2007 Q4"],
        'datamoda_test_date': ["2008 Q1", "2009 Q4"],
        'data_train_date': ["1990 Q1", "2015 Q4"],
        'splittype': 'timesplit'},
    'GenerativeModels_ScenarioGen': {
        'learning_rate': 1e-3,
        'iterations': 1,
        'epoch': 1000,
        'models': ["mcvae", "cvae", "vae", "cgan", "gan"],
        'verbose': False,
        'bdmc_run': True,
        'chain_length': 50,
        'n_batch': 1,
        'splittype': 'timesplit'},
    'BankPerfPred': {'ScenarioGenResults_dict': None,
                     'traintest_sets_dict': None,
                     'generativemodel': None,
                     'modelTarget': "CapRatios_timesplit",
                     'exclude': ["trainset_data_X_loancat_timesplit", "trainset_data_X_ppnr_timesplit",
                                 "trainset_data_Y_nco_timesplit"],
                     'epoch': 10,
                     'basetraindataset': "trainset_data_X_loancat_tminus1_timesplit",
                     'basetestdataset': "testset_data_X_loancat_tminus1_timesplit",
                     'splittype': "timesplit",
                     'include': ["trainset_data_mod_timesplit", "trainset_data_Y_nco_tminus1_timesplit",
                                 "trainset_data_X_ppnr_tminus1_timesplit", "trainset_data_X_loancat_tminus1_timesplit",
                                 "trainset_data_CapRatios_tminus1_timesplit"],
                     'lstm_lr': 1e-2,
                     'threshold': 1e-10,
                     'predictmodel': None,
                     "predictmodels" : ['gru','lstm','darnn']

                     }
}


#Setup All Experiments.
def ECE_Experiment_Wrapper(experiment_base_dict_orig = {
    'Experiment_Info': {'Experiment_Type': "ECE", "ExperimentNum": 0, "Description": 'Financial Crisis',
                        "Interval": 'Quarterly', 'savedir': "./Images/Table_Data"},
    'get_raw_train_data': {
        'quarter_ID': None,
        'cond_name': 'zmacro_domestic',
        'moda_names': ['zmacro_domestic', 'zmacro_international', "zmicro", "sectsbidx"],
        'data_names': ["X_loancat", "X_ppnr", "Y_nco", "CapRatios"],
        'path_dict': path_dict,
        'path_qtr_dict': path_qtr_dict,
        'modaDateIdx': Preprocess_Dict['moda_DateIdx'],
        'dataDateIdx': Preprocess_Dict['data_DateIdx'],
        'moda_train_date': ["1976 Q1", "2007 Q4"],
        'datamoda_test_date': ["2008 Q1", "2009 Q4"],
        'data_train_date': ["1990 Q1", "2015 Q4"],
        'splittype': 'timesplit'},
    'GenerativeModels_ScenarioGen': {
        'learning_rate': 1e-3,
        'iterations': 1,
        'epoch': 1000,
        'models': ["mcvae", "cvae", "vae", "cgan", "gan"],
        'verbose': False,
        'bdmc_run': True,
        'chain_length': 50,
        'n_batch': 1,
        'splittype': 'timesplit'},
    'BankPerfPred': {'ScenarioGenResults_dict': None,
                     'traintest_sets_dict': None,
                     'generativemodel': None,
                     'modelTarget': "CapRatios_timesplit",
                     'exclude': ["trainset_data_X_loancat_timesplit", "trainset_data_X_ppnr_timesplit",
                                 "trainset_data_Y_nco_timesplit"],
                     'epoch': 10,
                     'basetraindataset': "trainset_data_X_loancat_tminus1_timesplit",
                     'basetestdataset': "testset_data_X_loancat_tminus1_timesplit",
                     'splittype': "timesplit",
                     'include': ["trainset_data_mod_timesplit", "trainset_data_Y_nco_tminus1_timesplit",
                                 "trainset_data_X_ppnr_tminus1_timesplit", "trainset_data_X_loancat_tminus1_timesplit",
                                 "trainset_data_CapRatios_tminus1_timesplit"],
                     'lstm_lr': 1e-2,
                     'threshold': 1e-10,
                     'predictmodel': None,
                     "predictmodels" : ['gru','lstm','darnn']

                     }
}, experiments = {
                'Experiment 1' : {
                    'Experiment_Info' : {'ExperimentNum' : 1
                                         ,'Description' : "Financial Crisis"},
                    'get_raw_train_data' : {'moda_train_date' : ["1976 Q1","2007 Q4"]
                                            ,'datamoda_test_date' : ["2008 Q1","2009 Q4"]}},

                'Experiment 2' : {
                    'Experiment_Info' : {'ExperimentNum' : 2
                                        ,'Description' : "All Econ Data"},
                    'get_raw_train_data' : {'moda_train_date' : ["1976 Q1","2015 Q4"], 'datamoda_test_date' : ["2016 Q1","2017 Q4"]}},

                'Experiment 3' : {
                    'Experiment_Info' : {'ExperimentNum' : 3,'Description' : "Aligned Bank Data & Financial Crisis"},
                    'get_raw_train_data' : {'moda_train_date' : ["1990 Q1","2007 Q4"],'datamoda_test_date' : ["2008 Q1","2009 Q4"]}},

                'Experiment 4': {
                    'Experiment_Info': {'ExperimentNum': 4,'Description': "Aligned Bank & Econ Data"},
                    'get_raw_train_data': {'moda_train_date': ["1990 Q1", "2015 Q4"],'datamoda_test_date': ["2016 Q1", "2017 Q4"]}}

                }
                          ,iteration_num = 1, DataInterval = ["Quarterly","Yearly"]
                         , epoch = 1000, models = ["mcvae", "cvae", "vae", "cgan", "gan"], chain_length =50,bdmc_run = True, extype  = "BCLP" , splittype= 'timesplit'):
    logger = setup_log("Economic Conditons Estimator Wrapper")

    result_dict = dict()
    result_dict['Results'] = pd.DataFrame()
    result_dict['Models'] = pd.DataFrame()
    for interval in DataInterval:
        for key in [x for x in experiments.keys() if x.startswith(extype)]:
            print(key)
            print("Resetting Experiment")
            experiment_base_dict = copy.deepcopy(experiment_base_dict_orig)
            experiment_base_dict['Experiment_Info']['Interval'] = interval

            #experiment_base_dict = dict((k,v) for k,v in experiment_base_dict_orig.items())
            for key1 in experiments[key].keys():
                #print(key1)
                #print(experiments[key][key1].keys())
                for key2 in experiments[key][key1].keys():
                    #print(key2)
                    #print(experiments[key][key1][key2])
                    logger.info(f"Updating with Experiment:{key} Parameters")
                    experiment_base_dict[key1][key2] = experiments[key][key1][key2]

            logger.info(f"Setting Generative Experiment Parameters  \tEpoch:{epoch}, \titerations:{iteration_num}\tmodels{models}")
            experiment_base_dict["GenerativeModels_ScenarioGen"]["epoch"] = epoch
            experiment_base_dict["GenerativeModels_ScenarioGen"]["iterations"] = iteration_num
            experiment_base_dict["GenerativeModels_ScenarioGen"]["chain_length"] = chain_length
            experiment_base_dict["GenerativeModels_ScenarioGen"]["bdmc_run"] = bdmc_run
            experiment_base_dict["GenerativeModels_ScenarioGen"]["models"] = models

            experiment_base_dict["get_raw_train_data"]["splittype"] = splittype
            experiment_base_dict["GenerativeModels_ScenarioGen"]["splittype"] = splittype
            experiment_base_dict["BankPerfPred"]["splittype"] = splittype
            if interval == "Yearly":
                for qtr_num in range(0,4):
#                    qtr_num = 2
                    experiment_base_dict_qtr = copy.deepcopy(experiment_base_dict)
                    experiment_base_dict_qtr['get_raw_train_data']['quarter_ID'] = qtr_num
                    logger.info(f"Running with Data Interval:{interval} \t Quarter {qtr_num + 1}")

                    experiment_base_dict_qtr['get_raw_train_data']['modaDateIdx'] = \
                    experiment_base_dict['get_raw_train_data']['modaDateIdx'][
                        experiment_base_dict['get_raw_train_data']['modaDateIdx'].str.endswith(str(qtr_num + 1))].reset_index(drop=True)

                    experiment_base_dict_qtr['BankPerfPred']['modelTargets'] = ["_".join([x, 'quarter', str(qtr_num)]) for x in
                                                                                experiment_base_dict_qtr['BankPerfPred']['modelTargets']]

                    experiment_base_dict_qtr['get_raw_train_data']['dataDateIdx'] = \
                        experiment_base_dict['get_raw_train_data']['dataDateIdx'][
                            experiment_base_dict['get_raw_train_data']['dataDateIdx'].str.endswith(
                                str(qtr_num + 1))].reset_index(drop=True)
                    # print(experiment_base_dict_qtr['get_raw_train_data']['modaDateIdx'])

                    experiment_base_dict_qtr['get_raw_train_data']['moda_train_date'] = [
                        x.split('Q')[0] + 'Q' + str(qtr_num + 1) for x in
                        experiment_base_dict_qtr['get_raw_train_data']['moda_train_date']]

                    experiment_base_dict_qtr['get_raw_train_data']['data_train_date'] = [
                        x.split('Q')[0] + 'Q' + str(qtr_num + 1) for x in
                        experiment_base_dict_qtr['get_raw_train_data']['moda_train_date']]

                    experiment_base_dict_qtr['get_raw_train_data']['datamoda_test_date'] = [
                        x.split('Q')[0] + 'Q' + str(qtr_num + 1) for x in
                        experiment_base_dict_qtr['get_raw_train_data']['datamoda_test_date']]

                    experiment = ECE_Experiments(experiment_base_dict_qtr.copy())
                    experiment_base_dict_qtr.clear()
                    if extype in ['BCLP']:
                        result_dict['Results'] = pd.concat([result_dict['Results'], experiment['BankPred']['Results']], axis=0)
                        result_dict['Models'] = experiment['BankPred']['ModelTrained']
            #                    experiment_base_dict_qtr['get_raw_train_data']['dataDateIdx'] = experiment_base_dict['get_raw_train_data']['dataDateIdx'][experiment_base_dict['get_raw_train_data']['dataDateIdx'].str.endswith(str(qtr_num + 1))].reset_index(drop=True)
            else:
                experiment_base_dict['get_raw_train_data']['quarter_ID'] = None
                experiment = ECE_Experiments(experiment_base_dict)
                if extype in ['BCLP']:
                    result_dict['Results'] = pd.concat([result_dict['Results'], experiment['BankPred']['Results']],
                                                       axis=0)
                    result_dict['Models'] = experiment['BankPred']['ModelTrained']

            # tmp_df = pd.DataFrame([experiment_base_dict['Experiment_Info']['Experiment_Type'],
            #                        experiment_base_dict['Experiment_Info']['ExperimentNum'],
            #                        experiment_base_dict['Experiment_Info']['Interval'],
            #                        experiment_base_dict["get_raw_train_data"]["quarter_ID"], predictmodel,
            #                        Bank_PredEval.loc['TestErr'].item().item()]).T

            # tmp_df.columns = ['ExperimentType', 'ExpNum', 'Interval', 'QuarterID', 'Model', 'TestRMSE']


            experiment_base_dict.clear()
    return(result_dict)


experiment_base_dict = {
        'Experiment_Info': {'Experiment_Type': "BCLP", "ExperimentNum": 0, "Description": 'Financial Crisis',
                            "Interval": 'Quarterly', 'savedir': "./Images/Table_Data"},
        'get_raw_train_data': {
            'quarter_ID': None,
            'cond_name': 'zmacro_domestic',
            'moda_names': ['zmacro_domestic', 'zmacro_international', "zmicro", "sectsbidx"],
            'data_names': ["X_loancat", "X_ppnr", "Y_nco", "CapRatios"],
            'path_dict': path_dict,
            'path_qtr_dict': path_qtr_dict,
            'modaDateIdx': Preprocess_Dict['moda_DateIdx'],
            'dataDateIdx': Preprocess_Dict['data_DateIdx'],
            'moda_train_date': ["1990 Q1", "2015 Q4"],
            'datamoda_test_date': ["2016 Q1", "2017 Q1"],
            'data_train_date': ["1990 Q1", "2015 Q4"],
            'splittype': 'timesplit'},
        'GenerativeModels_ScenarioGen': {
            'learning_rate': 1e-3,
            'iterations': 1,
            'epoch': 1000,
            'data_names': ["X_loancat", "X_ppnr", "Y_nco", "CapRatios"],
            'models': ["mcvae"],
            'verbose': False,
            'bdmc_run': False,
            'chain_length': 50,
            'n_batch': 1,
            'splittype': 'timesplit'},
    'BankPerfPred': {'ScenarioGenResults_dict': None,
                     'traintest_sets_dict': None,
                     'generativemodel': ['mcvae'],
                     'modelTargets': ["CapRatios"],
                     'exclude': ["trainset_data_X_loancat_timesplit", "trainset_data_X_ppnr_timesplit",
                                 "trainset_data_Y_nco_timesplit"],
                     'epoch': 10,
                     'basetraindataset': "trainset_data_X_loancat_tminus1_timesplit",
                     'basetestdataset': "testset_data_X_loancat_tminus1_timesplit",
                     'splittype': "timesplit",
                     'include': ["trainset_data_mod_timesplit", "trainset_data_Y_nco_tminus1_timesplit",
                                 "trainset_data_X_ppnr_tminus1_timesplit", "trainset_data_X_loancat_tminus1_timesplit",
                                 "trainset_data_CapRatios_tminus1_timesplit"],
                     'lstm_lr': 1e-2,
                     'threshold': 1e-10,
                     'predictmodel': None,
                     "predictmodels": ['lstm']

                     }
    }

def ECE_Experiments(experiment_base_dict = {
        'Experiment_Info': {'Experiment_Type': "ECE", "ExperimentNum": 0, "Description": 'Financial Crisis',
                            "Interval": 'Quarterly', 'savedir': "./Images/Table_Data"},
        'get_raw_train_data': {
            'quarter_ID': None,
            'cond_name': 'zmacro_domestic',
            'moda_names': ['zmacro_domestic', 'zmacro_international', "zmicro", "sectsbidx"],
            'data_names': ["X_loancat", "X_ppnr", "Y_nco", "CapRatios"],
            'path_dict': path_dict,
            'path_qtr_dict': path_qtr_dict,
            'modaDateIdx': Preprocess_Dict['moda_DateIdx'],
            'dataDateIdx': Preprocess_Dict['data_DateIdx'],
            'moda_train_date': ["1976 Q1", "2007 Q4"],
            'datamoda_test_date': ["2008 Q1", "2009 Q4"],
            'data_train_date': ["1990 Q1", "2015 Q4"],
            'splittype': 'timesplit'},
        'GenerativeModels_ScenarioGen': {
            'learning_rate': 1e-3,
            'iterations': 1,
            'epoch': 1000,
            'data_names': ["X_loancat", "X_ppnr", "Y_nco", "CapRatios"],
            'models': ["mcvae", "cvae", "vae", "cgan", "gan"],
            'verbose': False,
            'bdmc_run': True,
            'chain_length': 50,
            'n_batch': 1,
            'splittype': 'timesplit'}
    }):
    #experiment_base_dict = experiment_base_dict.copy()
    os.chdir("/Users/phn1x/icdm2018_research_BB/Stress_Test_Research/Loss_projections/")
    logger = setup_log("Economic Conditons Estimator")
    logger.info(f"Using computation device: {device}")
    logger.info(f"Experiment Type:{experiment_base_dict['Experiment_Info']['Experiment_Type']}\tExperiment Number:{experiment_base_dict['Experiment_Info']['ExperimentNum']}\tData Interval:{experiment_base_dict['Experiment_Info']['Interval']}")
    logger.info(f"Experiment Description:{experiment_base_dict['Experiment_Info']['Description']}\tModality Training:{experiment_base_dict['get_raw_train_data']['moda_train_date']}\tModality Testing:{experiment_base_dict['get_raw_train_data']['datamoda_test_date']}\tSplit:{experiment_base_dict['get_raw_train_data']['splittype']}")


    logger.info(f"Started: Raw Data and generating Training and Testing Datasets")
    traintest_sets_dict= get_raw_train_test_data(quarter_ID=experiment_base_dict["get_raw_train_data"]["quarter_ID"]
                                                            ,cond_name=experiment_base_dict["get_raw_train_data"]["cond_name"]
                                                            ,moda_names=experiment_base_dict["get_raw_train_data"]["moda_names"]
                                                            ,data_names=experiment_base_dict["get_raw_train_data"]["data_names"]
                                                            ,path_dict=experiment_base_dict["get_raw_train_data"]["path_dict"]
                                                            ,path_qtr_dict=experiment_base_dict["get_raw_train_data"]["path_qtr_dict"]
                                                            ,modaDateIdx=experiment_base_dict["get_raw_train_data"]["modaDateIdx"]
                                                            ,dataDateIdx=experiment_base_dict["get_raw_train_data"]["dataDateIdx"]
                                                            , moda_train_date=experiment_base_dict["get_raw_train_data"]["moda_train_date"]
                                                            ,data_train_date=experiment_base_dict["get_raw_train_data"]["data_train_date"]
                                                            ,datamoda_test_date=experiment_base_dict["get_raw_train_data"]["datamoda_test_date"]
                                                            , splittype=experiment_base_dict["get_raw_train_data"]["splittype"])
    logger.info(f"Completed: Raw Data and generating Training and Testing Datasets")

    logger.info(f"Started: Generative Model Scenario Estimation")
    ScenarioGenResults_dict = GenerativeModels_ScenarioGen(traintest_sets_dict=traintest_sets_dict
                                                                     ,learning_rate=experiment_base_dict["GenerativeModels_ScenarioGen"]["learning_rate"]
                                                                     ,iterations=experiment_base_dict["GenerativeModels_ScenarioGen"]["iterations"]
                                                                     , epoch=experiment_base_dict["GenerativeModels_ScenarioGen"]["epoch"]
                                                                     ,models=experiment_base_dict["GenerativeModels_ScenarioGen"]["models"]
                                                                     ,splittype=experiment_base_dict["GenerativeModels_ScenarioGen"]["splittype"]
                                                                     ,verbose=experiment_base_dict["GenerativeModels_ScenarioGen"]["verbose"]
                                                                     ,bdmc_run=experiment_base_dict["GenerativeModels_ScenarioGen"]["bdmc_run"]
                                                                     ,chain_length=experiment_base_dict["GenerativeModels_ScenarioGen"]["chain_length"]
                                                                     ,n_batch=experiment_base_dict["GenerativeModels_ScenarioGen"]["n_batch"]
                                                                     )
    logger.info(f"Completed: Generative Model Scenario Estimation")

    if experiment_base_dict['Experiment_Info']['Experiment_Type'] in ['BCLP']:
        PredHist = dict()
        PredHist['Results'] = pd.DataFrame()
        PredHist['EvalHist'] = dict()
        PredHist['TrainHist'] = dict()
        PredHist['ModelTrained'] = dict()
        logger.info(f"Started: Banking Capital and Loss Prediction")
        if isinstance(experiment_base_dict["BankPerfPred"]["predictmodels"], list) and \
                experiment_base_dict["BankPerfPred"]["predictmodel"] is None:

            for modeltarget in experiment_base_dict["BankPerfPred"]["modelTargets"]:

                if experiment_base_dict["BankPerfPred"]["splittype"] in ['timesplit']:
                    tmp_model_target = "_".join(
                        [modeltarget, experiment_base_dict["BankPerfPred"]["splittype"]])
                    # prepare include, exclude, basesets
                    tmp_include = [x for x in traintest_sets_dict.keys() if x.startswith('trainset_') if
                                   x.endswith("_".join(
                                       ['tminus1', experiment_base_dict["BankPerfPred"]["splittype"]]))] \
                                  + [x for x in traintest_sets_dict.keys() if
                                     x.startswith('trainset_data_mod')]

                    tmp_exclude = [x for x in traintest_sets_dict.keys() if x.startswith('trainset_') if not
                    x.endswith("_".join(['tminus1', experiment_base_dict["BankPerfPred"]["splittype"]])) if
                                   not x.split('_')[2] in ['cond', 'mod']]

                    tmp_basetraindataset = [x for x in tmp_include if x.split('_')[2] == 'X'][0]
                    tmp_basetestdataset = tmp_basetraindataset.replace('trainset', 'testset')
                elif experiment_base_dict["BankPerfPred"]["splittype"] in ['banksplit']:
                    tmp_model_target = "_".join(
                        [modeltarget, experiment_base_dict["BankPerfPred"]["splittype"]])
                    # prepare include, exclude, basesets
                    tmp_include = [x for x in traintest_sets_dict.keys() if x.startswith('trainset_data_X') if
                                   x.endswith(experiment_base_dict["BankPerfPred"]["splittype"]) if
                                   not x.split('_')[2] in ['cond', 'mod']] \
                                  + [x for x in traintest_sets_dict.keys() if
                                     x.startswith('trainset_data_mod')]

                    tmp_exclude = [x for x in traintest_sets_dict.keys() if x.startswith('trainset_') if
                                   not x.startswith('trainset_data_X') if
                                   not x.split('_')[2] in ['cond', 'mod']]

                    tmp_basetraindataset = [x for x in tmp_include if x.split('_')[2] == 'X'][0]
                    tmp_basetestdataset = tmp_basetraindataset.replace('trainset', 'testset')

                else:
                    print("Incorrect Split type or missing.")
                    break
                for predictmodel in experiment_base_dict["BankPerfPred"]["predictmodels"]:

                    logger.info(f"{experiment_base_dict['Experiment_Info']['Experiment_Type']}\t{experiment_base_dict['Experiment_Info']['ExperimentNum']}\t{experiment_base_dict['Experiment_Info']['Interval']}\t{experiment_base_dict['get_raw_train_data']['quarter_ID']}\tTarget:{tmp_model_target}\t{predictmodel}")


                    Bank_PredEval, BankPredEval_trainhist, BankPredModel = BankPerfPred(
                        ScenarioGenResults_dict=ScenarioGenResults_dict,
                        traintest_sets_dict=traintest_sets_dict,
                        generativemodel=experiment_base_dict["BankPerfPred"]["generativemodel"],
                        modelTarget=tmp_model_target,
                        exclude=tmp_exclude,
                        epoch=experiment_base_dict["BankPerfPred"]["epoch"],
                        basetraindataset=tmp_basetraindataset,
                        basetestdataset=tmp_basetestdataset,
                        splittype=experiment_base_dict["BankPerfPred"]["splittype"],
                        include=tmp_include,
                        lstm_lr=experiment_base_dict["BankPerfPred"]["lstm_lr"],
                        threshold=experiment_base_dict["BankPerfPred"]["threshold"],
                        predictmodel=predictmodel)
                    PredHist['EvalHist']["_".join([experiment_base_dict['Experiment_Info']['Experiment_Type'],str(experiment_base_dict['Experiment_Info']['ExperimentNum']), 'PredEval'])] = Bank_PredEval
                    PredHist['TrainHist']["_".join([experiment_base_dict['Experiment_Info']['Experiment_Type'],str(experiment_base_dict['Experiment_Info']['ExperimentNum']), 'trainhist'])] = BankPredEval_trainhist
                    PredHist['ModelTrained']["_".join([experiment_base_dict['Experiment_Info']['Experiment_Type'],str(experiment_base_dict['Experiment_Info']['ExperimentNum']),predictmodel,'trained'])] = BankPredModel

                    model_tmp_name = "_".join([experiment_base_dict['Experiment_Info']['Experiment_Type'], "Experiment",
                                               str(experiment_base_dict['Experiment_Info']['ExperimentNum']),
                                               experiment_base_dict['Experiment_Info']['Interval'], predictmodel + ".pkl"])

                    #print('Saving Model:',predictmodel, model_tmp_name)
                    #torch.save(BankPredModel.state_dict(), "_".join(['model',model_tmp_name]))
                    #print("Saving State")
                    #torch.save(BankPredModel.state_dict(), model_tmp_name)

                    tmp_df = pd.DataFrame([experiment_base_dict['Experiment_Info']['Experiment_Type'],experiment_base_dict['Experiment_Info']['ExperimentNum'], experiment_base_dict['Experiment_Info']['Interval'], experiment_base_dict["get_raw_train_data"]["quarter_ID"], predictmodel, Bank_PredEval.loc['TestErr'].item().item()]).T

                    tmp_df.columns = ['ExperimentType', 'ExpNum','Interval','QuarterID','Model','TestRMSE']
                    PredHist['Results'] = pd.concat([PredHist['Results'],tmp_df], axis = 0)
                    logger.info(f"{experiment_base_dict['Experiment_Info']['Experiment_Type']}\t{experiment_base_dict['Experiment_Info']['ExperimentNum']}\t{experiment_base_dict['Experiment_Info']['Interval']}\t{experiment_base_dict['get_raw_train_data']['quarter_ID']}\tTarget:{tmp_model_target}\t{predictmodel}\t{ Bank_PredEval.loc['TestErr'].item().item()}")

        logger.info(f"Completed: Banking Capital and Loss Prediction")


    logger.info(f"Saving Evaluation Metrics")

    for key in [x for x in ScenarioGenResults_dict.keys() if x.startswith("results_")]:
        if experiment_base_dict["get_raw_train_data"]["quarter_ID"] is None:
            tmp_name = "_".join([experiment_base_dict['Experiment_Info']['Experiment_Type'],"Experiment",str(experiment_base_dict['Experiment_Info']['ExperimentNum']),experiment_base_dict['Experiment_Info']['Interval'],key + ".csv"])

        else:
            tmp_name = "_".join([experiment_base_dict['Experiment_Info']['Experiment_Type'], "Experiment",
                                 str(experiment_base_dict['Experiment_Info']['ExperimentNum']),
                                 experiment_base_dict['Experiment_Info']['Interval'], key, "qtr",str(experiment_base_dict["get_raw_train_data"]["quarter_ID"]) + ".csv"])
        tmp_name  = os.path.join(experiment_base_dict['Experiment_Info']['savedir'], tmp_name)
        ScenarioGenResults_dict[key].to_csv(tmp_name)
    logger.info(f"Saving Evaluation Metrics to Dict")
    Final_Dict = {"Scenario_Gen" : ScenarioGenResults_dict, "TrainingTesting" : traintest_sets_dict}
    if experiment_base_dict['Experiment_Info']['Experiment_Type'] in ['BCLP']:
        Final_Dict['BankPred'] = PredHist

    if experiment_base_dict["get_raw_train_data"]["quarter_ID"] is None:
        tmp_name = "_".join([experiment_base_dict['Experiment_Info']['Experiment_Type'], "Experiment",
                  str(experiment_base_dict['Experiment_Info']['ExperimentNum']),
                  experiment_base_dict['Experiment_Info']['Interval'],".pkl"])
    else:
        tmp_name = "_".join([experiment_base_dict['Experiment_Info']['Experiment_Type'], "Experiment",
                             str(experiment_base_dict['Experiment_Info']['ExperimentNum']),
                             experiment_base_dict['Experiment_Info']['Interval'], "qtr",str(experiment_base_dict["get_raw_train_data"]["quarter_ID"]) + ".pkl"])

    tmp_name = os.path.join(experiment_base_dict['Experiment_Info']['savedir'], tmp_name)
    print("Saving Final Results Dictionary")
    f = open(tmp_name, "wb")
    pickle.dump(Final_Dict, f)
    f.close()
    logger.info(f"Completed")
    return Final_Dict





#importlib.reload(m_DA_RNN)

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#models = ['cgan','gan']
models = ['mcvae']

experiments_transferlearning = {
                # 'ECE_Experiment 1' : {
                #     'Experiment_Info' : {'Experiment_Type' :'ECE','ExperimentNum' : 1
                #                          ,'Description' : "Financial Crisis"},
                #     'get_raw_train_data' : {'moda_train_date' : ["1976 Q1","2007 Q4"]
                #                             ,'datamoda_test_date' : ["2008 Q1","2009 Q4"]}},

                # 'ECE_Experiment 2' : {
                #     'Experiment_Info' : {'Experiment_Type' :'ECE','ExperimentNum' : 2
                #                         ,'Description' : "All Econ Data"},
                #     'get_raw_train_data' : {'moda_train_date' : ["1976 Q1","2015 Q4"], 'datamoda_test_date' : ["2016 Q1","2017 Q4"]}},

                # 'ECE_Experiment 3' : {
                #     'Experiment_Info' : {'Experiment_Type' :'ECE','ExperimentNum' : 3,'Description' : "Aligned Bank Data & Financial Crisis"},
                #     'get_raw_train_data' : {'moda_train_date' : ["1990 Q1","2007 Q4"],'datamoda_test_date' : ["2008 Q1","2009 Q4"]}},

                'ECE_Experiment 4': {
                    'Experiment_Info': {'Experiment_Type' :'ECE','ExperimentNum': 4,'Description': "Aligned Bank & Econ Data"},
                    'get_raw_train_data': {'moda_train_date': ["1990 Q1", "2015 Q4"],'datamoda_test_date': ["2016 Q1", "2017 Q4"]}},

                # 'BCLP_Experiment 1': {
                #     'Experiment_Info': {'Experiment_Type' :'BCLP','ExperimentNum': 1,'Description': "Projections After Financial Crisis"},
                #     'get_raw_train_data': {'moda_train_date': ["1990 Q1", "2007 Q4"],'data_train_date': ["1990 Q1", "2007 Q4"], 'datamoda_test_date': ["2008 Q1", "2016 Q4"], 'data_names': ["X_loancat", "X_ppnr", "Y_nco", "CapRatios","Y_target_CapRatio","Y_target_ncoR"]},
                #     'GenerativeModels_ScenarioGen' : {'models':['mcvae'],'bdmc_run' : False },
                #     'BankPerfPred' : {'generativemodel':['mcvae'],"predictmodels" : ['darnn'],'epoch': 5, 'modelTargets': ["Y_target_CapRatio","Y_target_ncoR"],'lstm_lr': 1e-3,
                #      'threshold': 1e-4}},
                'BCLP_Experiment 2': {
                    'Experiment_Info': {'Experiment_Type' :'BCLP','ExperimentNum': 2,'Description': "Projections After Financial Crisis"},
                    'get_raw_train_data': {'moda_train_date': ["1990 Q1", "2015 Q4"],'data_train_date': ["1990 Q1", "2015 Q4"], 'datamoda_test_date': ["2016 Q1", "2017 Q1"],'data_names': ["X_loancat", "X_ppnr", "Y_nco", "CapRatios","Y_target_CapRatio","Y_target_ncoR"]},
                    'BankPerfPred' : {'generativemodel':['mcvae'], "predictmodels" : ['darnn'],'epoch': 5, 'modelTargets': ["Y_target_CapRatio"], 'lstm_lr': 1e-3,
                     'threshold': 1e-4}}
                }



test = ECE_Experiment_Wrapper(DataInterval = ["Quarterly"], experiments = experiments_transferlearning, iteration_num = 1, epoch = 1000, models = ["mcvae"], chain_length =30, bdmc_run = False,extype = "BCLP", splittype = 'timesplit' )


#test['Results'].to_csv('BCLP_Experiments_2_darnn_banksplit.csv', index = False)
#test['Results'].to_csv('BCLP_Experiments_gru_lstm_banksplit.csv', index = False)
# DataInterval = ["Quarterly"]
# splittype = "timesplit"
# #DataInterval.reverse()
# extype = "ECE"
# logger = setup_log("Economic Conditons Estimator Wrapper")
# PredHist = dict()
# for interval in DataInterval:
#     for key in [x for x in experiments.keys() if x.startswith(extype)]:
#         print(key, interval)
#         #key = "BCLP_Experiment 2"
#         #interval = "Quarterly"
#         print("Resetting Experiment")
#         experiment_base_dict = copy.deepcopy(experiment_base_dict_orig)
#         experiment_base_dict['Experiment_Info']['Interval'] = interval
#
#         # experiment_base_dict = dict((k,v) for k,v in experiment_base_dict_orig.items())
#         for key1 in experiments[key].keys():
#             # print(key1)
#             # print(experiments[key][key1].keys())
#             for key2 in experiments[key][key1].keys():
#                 # print(key2)
#                 # print(experiments[key][key1][key2])
#                 #logger.info(f"Updating with Experiment:{key} Parameters")
#                 experiment_base_dict[key1][key2] = experiments[key][key1][key2]
#
#
#
#        # logger.info(f"Setting Generative Experiment Parameters  \tEpoch:{epoch}, \titerations:{iteration_num}\tmodels{models}")
#         experiment_base_dict["GenerativeModels_ScenarioGen"]["epoch"] = epoch
#         experiment_base_dict["GenerativeModels_ScenarioGen"]["iterations"] = iteration_num
#         experiment_base_dict["GenerativeModels_ScenarioGen"]["chain_length"] = chain_length
#         experiment_base_dict["GenerativeModels_ScenarioGen"]["bdmc_run"] = bdmc_run
#         experiment_base_dict["GenerativeModels_ScenarioGen"]["models"] = models
#         #experiment_base_dict["BankPerfPred"]["epoch"] = bankperf_epoch
#         experiment_base_dict["get_raw_train_data"]["splittype"] = splittype
#         experiment_base_dict["GenerativeModels_ScenarioGen"]["splittype"] = splittype
#         experiment_base_dict["BankPerfPred"]["splittype"] = splittype
#
#
#
#         if interval == "Yearly":
#             for qtr_num in range(0, 4):
#                 #                    qtr_num = 2
#                 experiment_base_dict_qtr = copy.deepcopy(experiment_base_dict)
#                 experiment_base_dict_qtr['get_raw_train_data']['quarter_ID'] = qtr_num
#                 logger.info(f"Running with Data Interval:{interval} \t Quarter {qtr_num + 1}")
#
#                 experiment_base_dict_qtr['get_raw_train_data']['modaDateIdx'] = experiment_base_dict['get_raw_train_data']['modaDateIdx'][
#                     experiment_base_dict['get_raw_train_data']['modaDateIdx'].str.endswith(str(qtr_num + 1))].reset_index(drop=True)
#
#                 experiment_base_dict_qtr['BankPerfPred']['modelTargets'] = ["_".join([x,'quarter',str(qtr_num)]) for x in experiment_base_dict_qtr['BankPerfPred']['modelTargets']]
#
#                 experiment_base_dict_qtr['get_raw_train_data']['dataDateIdx'] = \
#                 experiment_base_dict['get_raw_train_data']['dataDateIdx'][
#                     experiment_base_dict['get_raw_train_data']['dataDateIdx'].str.endswith(
#                         str(qtr_num + 1))].reset_index(drop=True)
#                 # print(experiment_base_dict_qtr['get_raw_train_data']['modaDateIdx'])
#
#                 experiment_base_dict_qtr['get_raw_train_data']['moda_train_date'] = [
#                     x.split('Q')[0] + 'Q' + str(qtr_num + 1) for x in
#                     experiment_base_dict_qtr['get_raw_train_data']['moda_train_date']]
#
#                 experiment_base_dict_qtr['get_raw_train_data']['data_train_date'] = [
#                     x.split('Q')[0] + 'Q' + str(qtr_num + 1) for x in
#                     experiment_base_dict_qtr['get_raw_train_data']['moda_train_date']]
#
#                 experiment_base_dict_qtr['get_raw_train_data']['datamoda_test_date'] = [
#                     x.split('Q')[0] + 'Q' + str(qtr_num + 1) for x in
#                     experiment_base_dict_qtr['get_raw_train_data']['datamoda_test_date']]
#
#                 experiment = ECE_Experiments(experiment_base_dict_qtr.copy())
#
#                 experiment_base_dict_qtr.clear()
#         #experiment_base_dict_qtr['get_raw_train_data']['dataDateIdx'] = experiment_base_dict['get_raw_train_data']['dataDateIdx'][experiment_base_dict['get_raw_train_data']['dataDateIdx'].str.endswith(str(qtr_num + 1))].reset_index(drop=True)
#         else:
#             experiment_base_dict['get_raw_train_data']['quarter_ID'] = None
#             experiment = ECE_Experiments(experiment_base_dict)
#             # os.chdir("/Users/phn1x/icdm2018_research_BB/Stress_Test_Research/Loss_projections/")
#         experiment_base_dict.clear()
#
#
#
#            #experiment_base_dict = experiment_base_dict_qtr.copy()
#
#
#             traintest_sets_dict = get_raw_train_test_data(quarter_ID=experiment_base_dict["get_raw_train_data"]["quarter_ID"]
#                                                       , cond_name=experiment_base_dict["get_raw_train_data"]["cond_name"]
#                                                       , moda_names=experiment_base_dict["get_raw_train_data"]["moda_names"]
#                                                       , data_names=experiment_base_dict["get_raw_train_data"]["data_names"]
#                                                       , path_dict=experiment_base_dict["get_raw_train_data"]["path_dict"]
#                                                       , path_qtr_dict=experiment_base_dict["get_raw_train_data"]["path_qtr_dict"]
#                                                       , modaDateIdx=experiment_base_dict["get_raw_train_data"]["modaDateIdx"]
#                                                       , dataDateIdx=experiment_base_dict["get_raw_train_data"]["dataDateIdx"]
#                                                       , moda_train_date=experiment_base_dict["get_raw_train_data"][ "moda_train_date"]
#                                                       , data_train_date=experiment_base_dict["get_raw_train_data"]["data_train_date"]
#                                                       , datamoda_test_date=experiment_base_dict["get_raw_train_data"][ "datamoda_test_date"]
#                                                       , splittype=experiment_base_dict["get_raw_train_data"]["splittype"])
#
#                 ScenarioGenResults_dict = GenerativeModels_ScenarioGen(traintest_sets_dict=traintest_sets_dict
#                                                                    , learning_rate=experiment_base_dict["GenerativeModels_ScenarioGen"]["learning_rate"]
#                                                                    ,iterations=experiment_base_dict["GenerativeModels_ScenarioGen"]["iterations"]
#                                                                    , epoch=1000#experiment_base_dict["GenerativeModels_ScenarioGen"]["epoch"]
#                                                                    , models=experiment_base_dict["GenerativeModels_ScenarioGen"]["models"]
#                                                                    , splittype=experiment_base_dict["GenerativeModels_ScenarioGen"]["splittype"]
#                                                                    , verbose=experiment_base_dict["GenerativeModels_ScenarioGen"]["verbose"]
#                                                                    , bdmc_run=experiment_base_dict["GenerativeModels_ScenarioGen"]["bdmc_run"]
#                                                                    , chain_length=experiment_base_dict["GenerativeModels_ScenarioGen"][ "chain_length"]
#                                                                    , n_batch=experiment_base_dict["GenerativeModels_ScenarioGen"]["n_batch"]
#                                                                    )
#
#
#
#                 if experiment_base_dict['Experiment_Info']['Experiment_Type'] in ['BCLP']:
#                     if isinstance(experiment_base_dict["BankPerfPred"]["predictmodels"], list) and experiment_base_dict["BankPerfPred"]["predictmodel"] is None:
#                         for modeltarget in experiment_base_dict["BankPerfPred"]["modelTargets"]:
#
#                             if experiment_base_dict["BankPerfPred"]["splittype"] in ['timesplit']:
#                                 tmp_model_target = "_".join(
#                                     [modeltarget, experiment_base_dict["BankPerfPred"]["splittype"]])
#                                 # prepare include, exclude, basesets
#                                 tmp_include = [x for x in traintest_sets_dict.keys() if x.startswith('trainset_') if
#                                                x.endswith("_".join(
#                                                    ['tminus1', experiment_base_dict["BankPerfPred"]["splittype"]]))] \
#                                               + [x for x in traintest_sets_dict.keys() if
#                                                  x.startswith('trainset_data_mod')]
#
#                                 tmp_exclude = [x for x in traintest_sets_dict.keys() if x.startswith('trainset_') if not
#                                 x.endswith("_".join(['tminus1', experiment_base_dict["BankPerfPred"]["splittype"]])) if
#                                                not x.split('_')[2] in ['cond', 'mod']]
#
#                                 tmp_basetraindataset = [x for x in tmp_include if x.split('_')[2] == 'X'][0]
#                                 tmp_basetestdataset = tmp_basetraindataset.replace('trainset', 'testset')
#                             elif experiment_base_dict["BankPerfPred"]["splittype"] in ['banksplit']:
#                                 #modeltarget = "Y_target_CapRatio_quarter_0"
#
#                                 tmp_model_target = "_".join(
#                                     [modeltarget, experiment_base_dict["BankPerfPred"]["splittype"]])
#                                 # prepare include, exclude, basesets
#                                 tmp_include = [x for x in traintest_sets_dict.keys() if x.startswith('trainset_data_X') if
#                                                x.endswith(experiment_base_dict["BankPerfPred"]["splittype"]) if
#                                                not x.split('_')[2] in ['cond', 'mod']] \
#                                               + [x for x in traintest_sets_dict.keys() if
#                                                  x.startswith('trainset_data_mod')]
#
#                                 tmp_exclude = [x for x in traintest_sets_dict.keys() if  x.startswith('trainset_') if not x.startswith('trainset_data_X') if
#                                                not x.split('_')[2] in ['cond', 'mod']]
#
#                                 tmp_basetraindataset = [x for x in tmp_include if x.split('_')[2] == 'X'][0]
#                                 tmp_basetestdataset = tmp_basetraindataset.replace('trainset', 'testset')
#
#                             else:
#                                 print("Incorrect Split type or missing.")
#                                 break
#
#
#
#                             for predictmodel in experiment_base_dict["BankPerfPred"]["predictmodels"]:
#                                 #predictmodel = 'darnn'
#
#                                 #importlib.reload(m_DA_RNN)
#                                 print('*******************',predictmodel,':\t',tmp_model_target)
#
#                                 Bank_PredEval, BankPredEval_trainhist = BankPerfPred(ScenarioGenResults_dict = ScenarioGenResults_dict,
#                                                                                                         traintest_sets_dict = traintest_sets_dict,
#                                                                                                          generativemodel = experiment_base_dict["BankPerfPred"]["generativemodel"],
#                                                                                                         modelTarget= tmp_model_target,
#                                                                                                         exclude = tmp_exclude,
#                                                                                                         epoch = experiment_base_dict["BankPerfPred"]["epoch"],
#                                                                                                        basetraindataset = tmp_basetraindataset,
#                                                                                                        basetestdataset = tmp_basetestdataset,
#                                                                                                        splittype= experiment_base_dict["BankPerfPred"]["splittype"],
#                                                                                                        include = tmp_include,
#                                                                                                         lstm_lr =experiment_base_dict["BankPerfPred"]["lstm_lr"],
#                                                                                                         threshold = experiment_base_dict["BankPerfPred"]["threshold"],
#                                                                                                         predictmodel = predictmodel)
#                                 PredHist["_".join([key, 'PredEval'])] = Bank_PredEval
#                                 PredHist["_".join([key, 'trainhist'])] = BankPredEval_trainhist
#
#
#  experiment_base_dict_qtr['get_raw_train_data']['data_train_date']
#
#         # if interval == "Yearly":
#         #     for qtr_num in range(0, 4):
#         #         #                    qtr_num = 2
#         #         experiment_base_dict_qtr = copy.deepcopy(experiment_base_dict)
#         #         experiment_base_dict_qtr['get_raw_train_data']['quarter_ID'] = qtr_num
#         #         logger.info(f"Running with Data Interval:{interval} \t Quarter {qtr_num + 1}")
#         #         experiment_base_dict_qtr['get_raw_train_data']['modaDateIdx'] = \
#         #         experiment_base_dict['get_raw_train_data']['modaDateIdx'][
#         #             experiment_base_dict['get_raw_train_data']['modaDateIdx'].str.endswith(
#         #                 str(qtr_num + 1))].reset_index(drop=True)
#         #         # print(experiment_base_dict_qtr['get_raw_train_data']['modaDateIdx'])
#         #
#         #         experiment_base_dict_qtr['get_raw_train_data']['moda_train_date'] = [
#         #             x.split('Q')[0] + 'Q' + str(qtr_num + 1) for x in
#         #             experiment_base_dict_qtr['get_raw_train_data']['moda_train_date']]
#         #         experiment_base_dict_qtr['get_raw_train_data']['datamoda_test_date'] = [
#         #             x.split('Q')[0] + 'Q' + str(qtr_num + 1) for x in
#         #             experiment_base_dict_qtr['get_raw_train_data']['datamoda_test_date']]
#         #
#         #     os.chdir("/Users/phn1x/icdm2018_research_BB/Stress_Test_Research/Loss_projections/")
#         #     traintest_sets_dict = get_raw_train_test_data(quarter_ID=experiment_base_dict["get_raw_train_data"]["quarter_ID"]
#         #                                               , cond_name=experiment_base_dict["get_raw_train_data"]["cond_name"]
#         #                                               , moda_names=experiment_base_dict["get_raw_train_data"]["moda_names"]
#         #                                               , data_names=experiment_base_dict["get_raw_train_data"]["data_names"]
#         #                                               , path_dict=experiment_base_dict["get_raw_train_data"]["path_dict"]
#         #                                               , path_qtr_dict=experiment_base_dict["get_raw_train_data"]["path_qtr_dict"]
#         #                                               , modaDateIdx=experiment_base_dict["get_raw_train_data"]["modaDateIdx"]
#         #                                               , dataDateIdx=experiment_base_dict["get_raw_train_data"]["dataDateIdx"]
#         #                                               , moda_train_date=experiment_base_dict["get_raw_train_data"][ "moda_train_date"]
#         #                                               , data_train_date=experiment_base_dict["get_raw_train_data"]["data_train_date"]
#         #                                               , datamoda_test_date=experiment_base_dict["get_raw_train_data"][ "datamoda_test_date"]
#         #                                               , splittype=experiment_base_dict["get_raw_train_data"]["splittype"])
#         #
#         #         ScenarioGenResults_dict = GenerativeModels_ScenarioGen(traintest_sets_dict=traintest_sets_dict
#         #                                                            , learning_rate=experiment_base_dict["GenerativeModels_ScenarioGen"]["learning_rate"]
#         #                                                            ,iterations=experiment_base_dict["GenerativeModels_ScenarioGen"]["iterations"]
#         #                                                            , epoch=experiment_base_dict["GenerativeModels_ScenarioGen"]["epoch"]
#         #                                                            , models=experiment_base_dict["GenerativeModels_ScenarioGen"]["models"]
#         #                                                            , splittype=experiment_base_dict["GenerativeModels_ScenarioGen"]["splittype"]
#         #                                                            , verbose=experiment_base_dict["GenerativeModels_ScenarioGen"]["verbose"]
#         #                                                            , bdmc_run=experiment_base_dict["GenerativeModels_ScenarioGen"]["bdmc_run"]
#         #                                                            , chain_length=experiment_base_dict["GenerativeModels_ScenarioGen"][ "chain_length"]
#         #                                                            , n_batch=experiment_base_dict["GenerativeModels_ScenarioGen"]["n_batch"]
#         #                                                            )
#         #
#         #
#         #
#         #         if experiment_base_dict['Experiment_Info']['Experiment_Type'] in ['BCLP']:
#         #             if isinstance(experiment_base_dict["BankPerfPred"]["predictmodels"], list) and experiment_base_dict["BankPerfPred"]["predictmodel"] is None:
#         #                 for modeltarget in experiment_base_dict["BankPerfPred"]["modelTargets"]:
#         #                     tmp_model_target = "_".join([modeltarget,experiment_base_dict["BankPerfPred"]["splittype"]])
#         #                     #prepare include, exclude, basesets
#         #                     tmp_include = [ x for x in traintest_sets_dict.keys() if x.startswith('trainset_') if x.endswith("_".join(['tminus1',experiment_base_dict["BankPerfPred"]["splittype"]]))]\
#         #                     + [ x for x in traintest_sets_dict.keys() if x.startswith('trainset_data_mod')]
#         #
#         #                     tmp_exclude = [x for x in traintest_sets_dict.keys() if x.startswith('trainset_') if not
#         #                      x.endswith("_".join(['tminus1', experiment_base_dict["BankPerfPred"]["splittype"]])) if not x.split('_')[2] in ['cond','mod'] ]
#         #
#         #                     tmp_basetraindataset = [x for x in tmp_include if x.split('_')[2] == 'X'][0]
#         #                     tmp_basetestdataset = tmp_basetraindataset.replace('trainset','testset')
#         #
#         #                     for predictmodel in experiment_base_dict["BankPerfPred"]["predictmodels"]:
#         #                         print('*******************',predictmodel,':\t',tmp_model_target)
#         #                         Bank_PredEval, BankPredEval_trainhist = BankPerfPred(ScenarioGenResults_dict = ScenarioGenResults_dict,
#         #                                                                                                 traintest_sets_dict = traintest_sets_dict,
#         #                                                                                                  generativemodel = experiment_base_dict["BankPerfPred"]["generativemodel"],
#         #                                                                                                 modelTarget= tmp_model_target,
#         #                                                                                                 exclude = tmp_exclude,
#         #                                                                                                 epoch = experiment_base_dict["BankPerfPred"]["epoch"],
#         #                                                                                                basetraindataset = tmp_basetraindataset,
#         #                                                                                                basetestdataset = tmp_basetestdataset,
#         #                                                                                                splittype= experiment_base_dict["BankPerfPred"]["splittype"],
#         #                                                                                                include = tmp_include,
#         #                                                                                                 lstm_lr =experiment_base_dict["BankPerfPred"]["lstm_lr"],
#         #                                                                                                 threshold = experiment_base_dict["BankPerfPred"]["threshold"],
#         #                                                                                                 predictmodel = predictmodel)
#         #                         PredHist["_".join([key, 'PredEval'])] = Bank_PredEval
#         #                         PredHist["_".join([key, 'trainhist'])] = BankPredEval_trainhist
#         #
#
# #TODO Need to populate these fields dynamically.
# #Will create new inputs that are specific to targets, T1CR, and Loan Loss Rate
# #modelTarget= experiment_base_dict["BankPerfPred"]["modelTarget"]
# #Will need to apply logic to determine include and exclude list
# exclude = experiment_base_dict["BankPerfPred"]["exclude"]
# include = experiment_base_dict["BankPerfPred"]["include"]
# #Will need to apply logic to detemine base trainsets.
# basetraindataset = experiment_base_dict["BankPerfPred"]["basetraindataset"]
# basetestdataset = experiment_base_dict["BankPerfPred"]["basetestdataset"]




#
# ScenarioGenResults_dict["mcvae_results_dict"]['TrainBestModel']['best_loss']
# ScenarioGenResults_dict["mcvae_results_dict"]["TrainHist"].plot()
#
# ScenarioGenResults_dict["vae_results_dict"]["TrainBestModel"]["best_loss"]
# ScenarioGenResults_dict["cgan_results_dict"]["TrainBestModel"]["epoch"]
#
# np.sqrt(ScenarioGenResults_dict["cgan_results_dict"]['TrainHist']).min()
#
# tmp_file_name = "GenModel_RMSE_TrainHist.pdf"
# np.sqrt(ScenarioGenResults_dict["mcvae_results_dict"]["TrainHist"]).plot()
# np.sqrt(ScenarioGenResults_dict["vae_results_dict"]["TrainHist"]['vae_modality_total_loss']).plot()
# np.sqrt(ScenarioGenResults_dict["cgan_results_dict"]['TrainHist']).plot()
# np.sqrt(ScenarioGenResults_dict["cgan_results_dict"]['TrainHist']['cgan_TrainingGeneratorFoolDiscrim_loss']).plot()
# np.sqrt(ScenarioGenResults_dict["gan_results_dict"]['TrainHist']['gan_TrainingGeneratorFoolDiscrim_loss']).plot()
# plt.title("Economic Conditions Generative Model Training History")
# plt.ylabel("RMSE")
# plt.legend()
# plt.savefig(os.path.join(tmp_file_name), dpi=300, format='pdf', bbox_inches='tight')
#
# ScenarioGenResults_dict["cgan_results_dict"]['TrainHist'].plot(subplots = True)
#
#
#
#
#
#
#
#
# traintest_sets_dict_timesplit.keys()

# BankPredEval_timesplit, BankPredEval_timesplit_trainhist = BankPerfPred(ScenarioGenResults_dict = ScenarioGenResults_dict_timesplit,
#                                            traintest_sets_dict = traintest_sets_dict_timesplit,
#                                            generativemodel = ["mcvae"]
#                                            ,modelTarget= "CapRatios_timesplit",
#                                            exclude = ["trainset_data_X_loancat_timesplit"
#                                                       ,"trainset_data_X_ppnr_timesplit","trainset_data_Y_nco_timesplit"],
#
#                                            epoch = 10,
#                                            basetraindataset = "trainset_data_X_loancat_tminus1_timesplit",
#                                            basetestdataset = "testset_data_X_loancat_tminus1_timesplit",
#                                            splittype= "timesplit",
#                                            include = ["trainset_data_mod_timesplit",
#                                                       "trainset_data_Y_nco_tminus1_timesplit",
#                                                       "trainset_data_X_ppnr_tminus1_timesplit",
#                                                       "trainset_data_X_loancat_tminus1_timesplit",
#                                                       "trainset_data_CapRatios_tminus1_timesplit"
#                                                       ],
#                                             lstm_lr = 1e-2, threshold = 1e-10, predictmodel = "darnn")




#Plot predicted probability distribution
# destplot = pd.DataFrame(ScenarioGenResults_dict_timesplit["mcvae_pred_moda"][0].detach().numpy())
# import seaborn as sns
# g = sns.distplot(destplot[0],hist = False, kde = True,
#                   kde_kws = {'shade': True, 'linewidth': 3},
#                    label = "PCA"
#                          )

#TODO: Create Accuracy vs. Loss vs. EPoch training history
#TODO: Create Accuracy metric for Testing.
#TODO: Plot probability distribution of each Model.
#TODO: Get average training history epochs arather than just the last iteration




#Plot for True vs. Predicted Code.
#Find Banks that have the lowest MSE and use these as figures.
# final_y_pred = predict(enc, dec, data, **da_rnn_kwargs)
#
# plt.figure()
# plt.plot(final_y_pred, label='Predicted')
# plt.plot(data.targs[(da_rnn_kwargs["T"]-1):], label="True")
# plt.legend(loc='upper left')
# utils.save_or_show_plot("final_predicted_reloaded.png", save_plots)


#TODO: Make into Model Compare Bar Plot
#ScenarioGenResults_dict_timesplit[keyname][keyname2].plot.bar(subplots = True)
# speed = [0.1, 17.5, 40, 48, 52, 69, 88]
# lifespan = [2, 8, 70, 1.5, 25, 12, 28]
# index = ['snail', 'pig', 'elephant',
# 'rabbit', 'giraffe', 'coyote', 'horse']
# df = pd.DataFrame({'speed': speed,
#                     'lifespan': lifespan}, index=index)
# ax = df.plot.bar(rot=0)

#split columns
#axes = df.plot.bar(rot=0, subplots=True)
#axes[1].legend(loc=2)  # doctest: +SKIP

#selected category
#ax = df.plot.bar(y='speed', rot=0)
#ax = df.plot.bar(x='lifespan', rot=0)
#
# dfs_list = list()
# for keyname in [x for x in ScenarioGenResults_dict_timesplit.keys() if x.endswith("_results_dict")]:
#     for keyname2 in [ x for x in ScenarioGenResults_dict_timesplit[keyname].keys() if x.endswith("_AIS_BDMC_LLD")]:
#         print(keyname2)
#         print(ScenarioGenResults_dict_timesplit[keyname][keyname2][["upper_bounds_train","upper_bounds_test"]])
#         tmp_df = ScenarioGenResults_dict_timesplit[keyname][keyname2]
#         tmp_df.index.names = [keyname2.split("_")[0]]
#         dfs_list.extend(tmp_df)



#TODO: Maybe only able to do MSE or Loss Compare between MOdels.
#TODO: MSE may be sufficient or RMSE for overall eval.
#TODO: Plots to show model fit and predicted vs. true for time split and bank split

##################################################################################################################################################################################################################
# splittype = "banksplit"
# # traintest_sets_dict_banksplit  = get_raw_train_test_data(quarter_ID = None, cond_name = "Sectidx", moda_names = ['sbidx','Sectidx','zmicro', 'zmacro_domestic', 'zmacro_international']
# #                                                ,train_window = train_window, test_window = test_window
# #                                                ,  data_names = ["CapRatios",'X_Y_NCO_all_pca'], path_dict= path_dict, path_qtr_dict= path_qtr_dict, splittype= "banksplit")
#
#
# traintest_sets_dict_banksplit  = get_raw_train_test_data(quarter_ID = None, cond_name = 'zmicro', moda_names = ['zmacro_domestic','zmacro_international',"sectidx","sbidx","zmicro"]
#                                                ,train_window = train_window, test_window = test_window
#                                                , data_names = ["CapRatios","X","Y","NCO"], path_dict= path_dict, path_qtr_dict= path_qtr_dict
#                                                          , splittype= splittype)
#
#
#
# ScenarioGenResults_dict_banksplit = GenerativeModels_ScenarioGen(traintest_sets_dict_banksplit, learning_rate = 1e-4 , iterations = 1, epoch = 1000, models=["mcvae", "cgan"], splittype = splittype)
#
#
#
#
# # BankPredEval_banksplit = LSTM_BankPerfPred(ScenarioGenResults_dict_banksplit,traintest_sets_dict_banksplit, generativemodel = ["mcvae", "cgan",'gan'], modelTarget= "CapRatios_banksplit",
# #                                            epoch = 10,
# #                                            basetraindataset = "trainset_data_X_Y_NCO_all_pca_banksplit",
# #                                            basetestdataset = "testset_data_X_Y_NCO_all_pca_banksplit",
# #                                            splittype= "banksplit",
# #                                            include = ["trainset_data_mod_banksplit","trainset_data_X_Y_NCO_all_pca_banksplit"])
#
# BankPredEval_banksplit, BankPredEval_banksplit_trainhist = LSTM_BankPerfPred(ScenarioGenResults_dict_banksplit,
#                                            traintest_sets_dict_banksplit,
#                                            generativemodel = ["mcvae"]
#                                            ,modelTarget= "Y_banksplit",
#                                            exclude = ["trainset_data_NCO_banksplit"],
#                                            epoch = 10,
#                                            basetraindataset = "trainset_data_X_banksplit",
#                                            basetestdataset = "testset_data_X_banksplit",
#                                            splittype= splittype,
#                                            include = ["trainset_data_mod_banksplit","trainset_data_X_banksplit"
#                                                       ]
#                                                                              )
#
#
#
#

#TODO: Add Probability Distribution of Estimations.
#TODO: Add visualizaitons
#TODO: Implement Accuracy Score measure
#TODO: PLOT MSE VS. ACCURACY

    #Training and testing error each epoch for LSTM and Generative MOdels
    #Iteration based average scatter plot.
    #Feature based bar chart for LSTM
    #LSTM true vs. pred chart.
    #Predicted probability distributions overlayed with loan performances.


#traintest_sets_dict_qtr_timesplit.keys()

#
# splittype = 'banksplit'
# traintest_sets_dict_qtr_banksplit  = get_raw_train_test_data(quarter_ID = quarter_ID, cond_name = 'zmicro', moda_names = ['sbidx','Sectidx','zmicro', 'zmacro_domestic', 'zmacro_international']
#                                                ,train_window = train_window, test_window = test_window
#                                                , data_names = ["CapRatios",'X_Y_NCO_all_pca'], path_dict= path_dict, path_qtr_dict= path_qtr_dict, splittype= splittype)
#
#
# ScenarioGenResults_qtr_banksplit_dict = GenerativeModels_ScenarioGen(traintest_sets_dict_qtr_banksplit, learning_rate = 1e-4 , iterations = 1, epoch = 1000, models=["mcvae", "cgan"], splittype = splittype)
#
# #traintest_sets_dict_qtr_timesplit.keys()
#
#
#
#
#
#
# traintest_sets_dict_banksplit  = get_raw_train_test_data(quarter_ID = None, cond_name = "zmicro", moda_names = ['zmicro', 'zmacro_domestic', 'zmacro_international']
#                                                ,train_window = train_window, test_window = test_window
#                                                , data_names = ["X","Y","CapRatios","NCO"], path_dict= path_dict, path_qtr_dict= path_qtr_dict, splittype= "banksplit")
#
#
# ScenarioGenResults_dict_banksplit = GenerativeModels_ScenarioGen(traintest_sets_dict_banksplit, learning_rate = 1e-4 , iterations = 1, epoch = 1000, models=["mcvae", "cvae", "vae","gan", "cgan"], splittype = "banksplit")
#
#
# BankPredEval_banksplit = LSTM_BankPerfPred(ScenarioGenResults_dict_banksplit,traintest_sets_dict_banksplit, generativemodel = ["mcvae"], modelTarget= "CapRatios_banksplit", epoch = 100, basetraindataset = "trainset_data_X_banksplit", basetestdataset = "testset_data_X_banksplit", splittype = "banksplit")
#
#
#Update to append dictionary Models to get all models.
# torch.save(test['Models']['BCLP_2_lstm_trained'], './test.pth')
# model = torch.load('./test.pth')

# torch.save(test['Models']['BCLP_2_gru_trained'], './test_gru.pth')
# model = torch.load('./test_gru.pth')


torch.save(test['Models']['BCLP_2_darnn_trained'], './test_darnn.pth')
# model = torch.load('./test_darnn.pth')
