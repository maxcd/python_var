# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 16:31:06 2017

@author: mxc13
"""
import numpy as np

c, i, y  = np.loadtxt(r".\data\KPSW_ciy.txt").transpose()
data = np.concatenate([y.reshape(168,1), c.reshape(168,1), i.reshape(168,1)], axis=1)
data.shape

from VECMml import VECM

p = 2
r = 2
var_names = ['y', 'c', 'i']
shocks = ['supply', 'demand', 'investment']
model = VECM(data, p, r, var_names=var_names, shock_names=shocks)

print('\nReduced form residual covariance matrix Sigma:\n',
      model.Sigma_u)
#print(model.beta)
model.normalize()
print('\nComnpanion matrix form of the VAR:\n', model.companion)
#print(model.beta)
model.get_LR_impact()
print('\nshort run estimates Gamma\n:', model.Gamma)

print('\nlong run matrix XI:\n', model.Xi)

''' reproduce restrictions from Helmut
    where 0 means restriced and 1 means unrestricted
'''

SR = np.ones((3,3))
SR[1,2] = 0
SR = SR == 1.

LR = np.zeros((3,3), dtype=int)
LR[:,0] = 1
LR = LR == 1.

model.set_restrictions(SR, LR)

B0inv_guess = np.random.rand(3,3)#np.linalg.cholesky(model.Sigma_u)
#errs = model.restriction_errors(B0inv_guess)
#print(errs)
model.get_B0inv()
#print(model.opt_res)

print('\nResult for B0inv:\n', model.B0inv)

print('\nCompare to cholesky:\n', np.linalg.cholesky(model.Sigma_u))

print('\nResult for Upsilon:\n', model.Xi @ model.B0inv)

irf_fig = model.get_irfs(nsteps=30, plot=True)
irf_fig.savefig('irf_LRSR.pdf')

chol_fig = model.get_irfs(nsteps=30, B='chol', plot=True)
chol_fig.savefig('irf_chol.pdf')
