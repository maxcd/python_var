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
model = VECM(data, p, r)
#print(model.beta)
model.normalize()
#print(model.Gamma_norm)
#print(model.beta)
model.get_LR_impact()
print(model.Xi)
