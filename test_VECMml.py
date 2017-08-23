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
A = model.companion
print('\nComnpanion matrix form of the VAR:\n', model.companion)
#print(model.beta)
model.get_LR_impact()
print('\nshort run estimates Gamma\n:', model.Gamma)

print('\nlong run matrix XI:\n', model.Xi)

print('\nnormalized beta:\n', model.beta)
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

# use the same guess that helmut uses
guess_helmut =np.array([[ 0.979448879095330, -0.0962679550523870, 1.88600200593293],
                        [-0.265611268123836, -1.38067086579540, -2.94138589282478],
                        [-0.548372720057146, -0.728371038269661, 0.980021092332167]])
guess = guess_helmut.flatten()

# it also works with a random guess
rand_guess = np.random.rand(3,3)
guess = rand_guess.flatten()

model.get_B0inv(guess)

print('\nResult for B0inv:\n', model.B0inv)

print('\nResult for Upsilon:\n', model.Xi @ model.B0inv)

irf_fig = model.get_irfs(nsteps=30, plot=True)
irf_fig.savefig('irf_LRSR.pdf')

chol_fig = model.get_irfs(nsteps=30, B='chol', plot=True)
chol_fig.savefig('irf_chol.pdf')
