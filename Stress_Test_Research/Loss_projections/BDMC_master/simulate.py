from __future__ import print_function

import numpy as np

import torch
from torch.autograd import Variable
from torch.distributions import Bernoulli
from torch.distributions import Normal

def simulate_data(model, mod = None, batch_size=10, n_batch=1, cond = None, modeltype = 'vae', mod_dim = None):
  """Simulate data from the VAE model. Sample from the 
  joint distribution p(z)p(x|z). This is equivalent to
  sampling from p(x)p(z|x), i.e. z is from the posterior.

  Bidirectional Monte Carlo only works on simulated data,
  where we could obtain exact posterior samples.

  Args:
      model: VAE model for simulation
      batch_size: batch size for simulated data
      n_batch: number of batches

  Returns:
      iterator that loops over batches of torch Tensor pair x, z
  """


  # shorter aliases
  batches = []
  for i in range(n_batch):

    if mod is None:
      if modeltype in ['mcvae','vae','cvae']:
        z = []
        for dim_size in (model.mod_input_sizes):
          z_tmp = torch.randn([batch_size, dim_size])
          z.append(z_tmp)
      else:
        z = torch.randn(batch_size, model.latent_dim)  # .cuda()
    else:
      z = mod

    if modeltype in ['mcvae',"cvae","cgan"]:
      x_logits = model.decode(z, cond)
    else:
      x_logits = model.decode(z)

    if isinstance(x_logits, tuple):
      #This is getting first estimation object.
      x_mu = x_logits[1]
      x_logvar = x_logits[2]
      x_estimates = x_logits[0]
      #print(x_mu.shape, x_logvar.shape, x_logits.shape)

    if isinstance(x_estimates, list):
      x_lst = []
      for i in range(0,len(x_estimates)):
        x_norm_dist_tmp = Normal(x_mu[:, 0:model.latent_dim], x_estimates[i].std())
        x_lst.append(x_norm_dist_tmp.sample().data)
      x = x_lst
    else:
      #x_bernoulli_dist = Bernoulli(probs=x_logits.sigmoid())
      #print(model.latent_dim)
      x_norm_dist = Normal(x_mu[:,0:model.latent_dim],x_estimates.std())
      #x = x_bernoulli_dist.sample().data
      x = x_norm_dist.sample().data
    #print(x.shape, z.shape)
    paired_batch = (x, z)
    batches.append(paired_batch)

  return iter(batches)
