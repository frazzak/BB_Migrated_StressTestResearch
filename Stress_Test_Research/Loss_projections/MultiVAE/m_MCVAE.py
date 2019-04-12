# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:26:27 2019

@author: yifei
"""

import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class MCVAE(nn.Module):
    
    def __init__(self, latent_size, modality_size, conditional, num_cond,
                 mod_input_sizes, layer_size, use_cuda):
        
        super(MCVAE, self).__init__()
        self.sub_encoders = nn.ModuleList()
        self.sub_decoders = nn.ModuleList()
        self.modality_size = modality_size
        self.conditional = conditional
        
        decoder_layer_size = layer_size.copy()
        decoder_layer_size.reverse()
        for i in range(0, self.modality_size):
            temp_encoder = m_encoder(mod_input_sizes[i], latent_size, layer_size,
                                     conditional, num_cond)
            temp_decoder = m_decoder(mod_input_sizes[i], latent_size, decoder_layer_size,
                                     conditional, num_cond)
            self.sub_encoders.append(temp_encoder)
            self.sub_decoders.append(temp_decoder)
        self.latent_size = latent_size
        self.use_cuda = use_cuda
        self.training = False
        
    def train(self):
        # set model in train mode
        self.training = True
    
    def test(self):
        # set model in eval mode
        self.training = False
    
    def init_params(self, size):
        """Universal prior expert. Here we use a spherical
        Gaussian: N(0, 1).

        @param size: integer
                     dimensionality of Gaussian
        @param use_cuda: boolean [default: False]
                         cast CUDA on variables
        """
        mu = Variable(torch.zeros(size))
        logvar = Variable(torch.log(torch.ones(size)))
        if self.use_cuda:
            mu, logvar = mu.cuda(), logvar.cuda()
        return mu, logvar
        
    
    def get_params(self, input_modalities, cond=None):
        # obtain mu, logvar for each modality
        for modality in input_modalities:
            if modality is not None:
                batch_size = modality.size(0)
                break
        
        # initialize the universal prior distribution
        # add extra dimension 1 in order to add all params for each modality
        params_size = [1, batch_size, self.latent_size]
        mu, logvar = self.init_params(params_size)
        
        for i in range(0, self.modality_size):
            if input_modalities[i] is not None:
                temp_mu, temp_logvar = self.sub_encoders[i].forward(input_modalities[i], cond)
                mu = torch.cat((mu, temp_mu.unsqueeze(0)), dim=0)
                logvar = torch.cat((logvar, temp_logvar.unsqueeze(0)), dim=0)
        
        # product of params to combine gaussians
        eps = 1e-8
        var = torch.exp(logvar) + eps
        T = 1 / (var + eps) # weights for each modality
        pd_mu = torch.sum(mu*T, dim=0) / torch.sum(T, dim=0) # weighted average
        pd_var = 1 / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + eps)
        
        return pd_mu, pd_logvar
        

    def reparameterize(self, mu, logvar):
        
        batch_size = mu.size(0)
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn([batch_size, self.latent_size])
            z = eps * std + mu
        else: # return mean during inference
            z = mu
        return z
        
    def forward(self, input_modalities, cond=None):
        # make input_modalities as tuple, each dimension represents a modality
        mu, logvar = self.get_params(input_modalities, cond)
        
        # reparametrization trick to sample
        z = self.reparameterize(mu, logvar)
        
        # reconstruct modalities based on sample
        output_estimations = []
        for i in range(0, self.modality_size):
            temp_estimation = self.sub_decoders[i].forward(z, cond)
            output_estimations.extend([temp_estimation])
        
        return output_estimations, mu, logvar
    
    def inference(self, n=1, cond=None):
        
        batch_size = n
        z = torch.randn([batch_size, self.latent_size])
        
        output_estimations = []
        for i in range(0, self.modality_size):
            temp_estimation = self.sub_decoders[i].forward(z, cond)
            output_estimations.extend([temp_estimation])
        
        return output_estimations

class m_encoder(nn.Module):
    
    def __init__(self, input_size, latent_size, layer_size, conditional, num_cond):
        super(m_encoder, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.conditional = conditional
        
        self.MLP = nn.Sequential()
        if self.conditional:
            self.input_size = input_size + num_cond       
        layer_size = [self.input_size] + layer_size
        
        # add a sigmod() layer first to transfer values into range 0~1
        self.MLP.add_module(name="S0", module=nn.Sigmoid())
        for i, (in_size, out_size) in enumerate(zip(layer_size[:-1], layer_size[1:])):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
        self.linear_mu = nn.Linear(layer_size[-1], self.latent_size)
        self.linear_lagvar = nn.Linear(layer_size[-1], self.latent_size)
    
    def forward(self, inputs, cond=None):
        
        if self.conditional:
            inputs = torch.cat((inputs, cond), dim=-1)
        x = self.MLP(inputs)
        
        mu = self.linear_mu(x)
        logvar = self.linear_lagvar(x)
        
        return mu, logvar

class m_decoder(nn.Module):
    
    def __init__(self, input_size, latent_size, layer_size, conditional, num_cond):
        super(m_decoder, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.conditional = conditional
        
        self.MLP = nn.Sequential()
        if self.conditional:
            self.latent_size = self.latent_size + num_cond
        #layer_size.reverse()
        layer_size = [self.latent_size] + layer_size + [self.input_size]
        
        # add a sigmod() layer first to transfer values into range 0~1
        self.MLP.add_module(name="S0", module=nn.Sigmoid())
        for i, (in_size, out_size) in enumerate(zip(layer_size[:-1], layer_size[1:])):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+2 < len(layer_size):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            #else:
            #    self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())
    
    def forward(self, inputs, cond=None):
        
        if self.conditional:
            inputs = torch.cat((inputs, cond), dim=-1)
        
        x = self.MLP(inputs)
        
        return x
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        