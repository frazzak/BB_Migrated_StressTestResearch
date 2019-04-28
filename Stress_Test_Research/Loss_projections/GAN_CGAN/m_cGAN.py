# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 09:55:18 2019

@author: yifei
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
        
class Generator(nn.Module):
    
    def __init__(self, latent_size, layer_size, conditional, num_cond, output_size):
        
        # output_size corresponds to the modality's input dimension (the num of columns)
        super(Generator, self).__init__()
        self.conditional = conditional
        self.input_size = latent_size
        self.layer_size = layer_size
        
        self.MLP = nn.Sequential()
        if self.conditional:
            self.input_size = self.input_size + num_cond
        
        layer_size = [self.input_size] + layer_size + [output_size]
        
        self.MLP.add_module(name="S0", module=nn.Sigmoid())
        for i, (in_size, out_size) in enumerate(zip(layer_size[:-1], layer_size[1:])):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+2 < len(layer_size):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

    def forward(self, noise, cond):
        
        # noise is in dimension batch_size * modality_input_dim
        # cond is in dimesnion batch_size * num_cond, is the real conditional modality
        if self.conditional:
            latent_inputs = torch.cat([noise, cond], 1)
        else:
            latent_inputs = noise
        
        gen_outputs = self.MLP(latent_inputs)
        
        return gen_outputs


class Discriminator(nn.Module):
    
    def __init__(self, input_size, layer_size, output_size):
        
        # normally, output_size = 1, indicating the validity of each row of instance from batch_size
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.MLP = nn.Sequential()
        
        layer_size = [self.input_size] + layer_size + [output_size]

        self.MLP.add_module(name="S0", module=nn.Sigmoid())
        for i, (in_size, out_size) in enumerate(zip(layer_size[:-1], layer_size[1:])):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+2 < len(layer_size):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                # finally layer to transform data into 0~1 ranges
                self.MLP.add_module(name="A{:d}".format(i), module=nn.Sigmoid())
                

    def forward(self, inputs):
        
        # judge the validity of each input
        # real ones will make validity more closer to 1
        # fake ones will have 0-like values of validity
        # validity: shape(batch_size, 1)
        validity = self.MLP(inputs)
        
        return validity

def train_GAN(num_cond, cond_train, cond_test, mod_train, mod_test, learning_rate, conditional=True, latent_size = 10,layer_size = [32, 64, 128], valid_dim = 1,epoch = 1000):
    #mod_train = mod_train[j]
    #mod_test = mod_test[j]
    # only train cgan on single modality
    batch_size = mod_train.shape[0]
    #latent_size = 10
    #layer_size = [32, 64, 128]
    mod_dim = mod_train.shape[1]
    cond_dim = cond_train.shape[1]
    #valid_dim = 1
    #epcho = 1000
    
    # Loss functions
    adversarial_loss = torch.nn.MSELoss()


    print("Initialize generator and discriminator")
    generator = Generator(latent_size, layer_size, conditional, cond_dim, mod_dim)
    D_layer_size = layer_size.copy()
    D_layer_size.reverse()
    discriminator = Discriminator(mod_dim, D_layer_size, valid_dim)
    
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

    print('''
    Train
    ''')
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

        #print("Sample noise as generator input")
        z = torch.from_numpy(np.random.normal(0, 1, (batch_size, latent_size))).float()
        #print("Generate a batch of modality")
        gen_mod = generator(z, cond_train)

        #print("Calculating Loss measures generator's ability to fool the discriminator")
        validity = discriminator(gen_mod)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        '''
        training discriminator
        '''
        optimizer_D.zero_grad()

        #print("Calculating Loss for real modality")
        validity_real = discriminator(mod_train)
        d_real_loss = adversarial_loss(validity_real, valid)

        #print("Calculating Loss for fake modality")
        validity_fake = discriminator(gen_mod.detach())
        d_fake_loss = adversarial_loss(validity_fake, fake)

        #print("Calculating Total discriminator loss")
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        #print("loss:\t%.2f" % d_loss.data)


    print('''
    evaluation
    ''')
    test_batch_size = cond_test.shape[0]
    z = torch.from_numpy(np.random.normal(0, 1, (test_batch_size, latent_size))).float()
    estimations = generator(z, cond_test)
    error = torch.nn.functional.mse_loss(estimations, mod_test)
    if conditional:
        print("CGAN testing_error (mse):%.2f" % error)
    else:
        print("GAN testing_error (mse):%.2f" % error)
        
    return error, estimations