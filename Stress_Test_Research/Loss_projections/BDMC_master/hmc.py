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
  if isinstance(v, list):
    v = torch.mean(torch.stack(v), dim=0)
  v = v - grad_U(z, cond = cond, modeltype = modeltype).mul(eps).mul_(.5)

  for i in range(1, L + 1):
    if isinstance(current_z, list):
      for mod in range(0,len(z)):
        z[mod] = z[mod] + v.mul(eps)
    else:
      z = z + v.mul(eps)
    if i < L:
      v = v - grad_U(z,cond = cond, modeltype = modeltype).mul(eps)

  v = v - grad_U(z, cond = cond, modeltype = modeltype).mul(eps).mul_(.5)
  v = -v  # this is not needed; only here to conform to the math

  if isinstance(z, list):
    z_tmp_list = []
    for i in range(0,len(z)):
      z_tmp_list.append(z[i].detach())
    z = z_tmp_list
  else:
    z = z.detach()
  return z, v.detach()


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
  if isinstance(current_v, list):
    current_v = torch.mean(torch.stack(current_v), dim=0)

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
