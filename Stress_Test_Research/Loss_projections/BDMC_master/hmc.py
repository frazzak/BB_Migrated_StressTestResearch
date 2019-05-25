from __future__ import print_function

import torch


def hmc_trajectory(current_z, current_v, U, grad_U, epsilon, L=10, cond = None, modeltype = None):
  """This version of HMC follows https://arxiv.org/pdf/1206.1901.pdf.

  Args:
      U: function to compute potential energy/minus log-density
      grad_U: function to compute gradients w.r.t. U
      epsilon: (adaptive) step size
      L: number of leap-frog steps
      current_z: current position
  """
  # if isinstance(current_z, list):
  #   current_z = torch.mean(torch.stack(current_z), dim=0)

  eps = epsilon.view(-1, 1)
  z = current_z
  v = current_v

  if isinstance(v, list) and isinstance(z, list) and modeltype in ["mcvae"]:
    #v = torch.mean(torch.stack(v), dim=0)
    for mod in range(0, len(z)):
      #print(v[mod].shape, z[mod].shape)
      grad_calc = grad_U([z[mod]], cond=cond, modeltype=modeltype)
      v[mod] = v[mod] -  grad_calc.mul(eps).mul_(.5)
  else:
    v = v - grad_U(z, cond = cond, modeltype = modeltype).mul(eps).mul_(.5)


  for i in range(1, L + 1):
    if isinstance(current_z, list) and isinstance(v, list) and modeltype in ["mcvae"]:
      for mod in range(0,len(z)):
        z[mod] = z[mod] + v[mod].mul(eps)
    else:
      z = z + v.mul(eps)

    if i < L:
      if isinstance(current_z, list) and isinstance(v, list) and modeltype in ["mcvae"]:
        for mod in range(0, len(z)):
          v[mod] = v[mod] - grad_U([z[mod]], cond=cond, modeltype=modeltype).mul(eps)
      else:
        v = v - grad_U(z,cond = cond, modeltype = modeltype).mul(eps)


  if isinstance(current_z, list) and isinstance(v, list) and modeltype in ["mcvae"]:
   for mod in range(0, len(z)):
     v[mod] = v[mod] - grad_U([z[mod]], cond = cond, modeltype = modeltype).mul(eps).mul_(.5)
     v[mod] = -v[mod]  # this is not needed; only here to conform to the math
  else:
    v = v - grad_U(z, cond = cond, modeltype = modeltype).mul(eps).mul_(.5)
    v = -v  # this is not needed; only here to conform to the math


  if isinstance(z, list) and isinstance(v, list):
    z_tmp_list = []
    v_tmp_list = []
    for i in range(0,len(z)):
      z_tmp_list.append(z[i].detach())
      v_tmp_list.append(v[i].detach())
    z = z_tmp_list
    v = v_tmp_list
  else:
    z = z.detach()
    v = v.detach()
  return z, v


def accept_reject(current_z, current_v,
                  z, v,
                  epsilon,
                  accept_hist, hist_len,
                  U, K=lambda v: torch.sum(v * v, 1), cond = None, modeltype = None):
  """Accept/reject based on Hamiltonians for current and propose.

  Args:
      current_z: position *before* leap-frog steps
      current_v: speed *before* leap-frog steps
      z: position *after* leap-frog steps
      v: speed *after* leap-frog steps
      epsilon: step size of leap-frog.
      U: function to compute potential energy
      K: function to compute kinetic energy
  """
  if isinstance(current_v, list) and isinstance(current_z, list):
    current_Hamil_tmp_lst = []
    propose_Hamil_tmp_lst = []
    for i in range(0, len(z)):
      current_Hamil_tmp_lst.append(K(current_v[i], modeltype = modeltype) + U([current_z[i]], cond = cond, modeltype = modeltype))
      propose_Hamil_tmp_lst.append(K(v[i],modeltype = modeltype) + U([z[i]], cond = cond, modeltype = modeltype))
    current_Hamil = torch.mean(torch.stack(current_Hamil_tmp_lst), dim=0)
    propose_Hamil = torch.mean(torch.stack(propose_Hamil_tmp_lst), dim=0)
  else:
      current_Hamil = K(current_v, modeltype = modeltype) + U(current_z, cond = cond, modeltype = modeltype)
      propose_Hamil = K(v,modeltype = modeltype) + U(z, cond = cond, modeltype = modeltype)



  prob = torch.exp(current_Hamil - propose_Hamil)

  with torch.no_grad():
    uniform_sample = torch.rand(prob.size())
    #.cuda()
    accept = (prob > uniform_sample).float()
    #.cuda()
    #if isinstance(current_z,list):
      #current_z = torch.mean(torch.stack(current_z), dim=0)
    if isinstance(z,list):
      #z = torch.mean(torch.stack(z), dim=0)
      #z_tmp_list = []
      for i in range(0,len(z)):
        z[i] = z[i].mul(accept.view(-1, 1)) + current_z[i].mul(1. - accept.view(-1, 1))
        z[i].requires_grad_()
    else:
      z = z.mul(accept.view(-1, 1)) + current_z.mul(1. - accept.view(-1, 1))
      z.requires_grad_()
    accept_hist = accept_hist.add(accept)
    criteria = (accept_hist / hist_len > 0.65).float()
    # .cuda()
    adapt = 1.02 * criteria + 0.98 * (1. - criteria)
    epsilon = epsilon.mul(adapt).clamp(1e-4, .5)

  return z, epsilon, accept_hist
